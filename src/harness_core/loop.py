"""The agent loop: a thin, model-native loop wrapped in explicit, code-enforced controls.

Research basis (build plan "The loop"; reference report sections 2.3, 2.7, 2.8):

* **Canonical predicate.** "Stop only when the last response has no tool calls, no user input is
  pending, and every Stop hook or goal evaluator allows it" (Claude Agent SDK loop; Codex
  ``run_turn``'s ``needs_follow_up = model_needs_follow_up || has_pending_input``).
* **Append-only history.** The assistant turn is appended with its *full* content (thinking,
  compaction and server-tool blocks unchanged); all tool results for one assistant turn go back
  in *one* user message in the native ``tool_result`` role; failures use ``is_error``.
* **Every stop_reason is handled.** ``pause_turn`` is resent to continue; ``max_tokens`` with a
  truncated tool call is retried with a larger ``max_tokens`` (then the call is returned as an
  error), and truncated text gets a continuation message; ``refusal`` ends the run;
  ``model_context_window_exceeded`` signals the compaction hook, with a thrash guard.
* **Tool scheduling.** Only tools whose *trusted* spec says ``read_only`` run concurrently;
  everything else runs sequentially, in order. Denials, unknown tools, executor crashes and
  sandbox deaths all come back to the model as tool results; the loop never crashes on a tool.
* **Budgets** (turns, tokens, wall clock, estimated USD) are checked after each response and after
  each tool batch; when one trips, pending calls are not executed and a Codex-style wrap-up turn
  runs. Consecutive-error, denial and doom-loop limits terminate with typed subtypes.
* **Steering and interrupts.** :meth:`AgentLoop.steer` queues input that is drained between tool
  batches (appended to the same tool-result message) and before any stop; :meth:`interrupt` is a
  hard cancel that records an aborted turn with paired results.
* **Durability.** Every message, tool call, tool result, permission decision, compaction and
  evaluation is an event in the session log. ``run()`` on a woken session repairs a half-finished
  tool batch: completed results are reused, never-started calls run, read-only calls that were
  mid-flight re-run, and mutating calls that were mid-flight return "side effects unknown".
  Each call carries an idempotency key ``hash(session_id, step, tool_use_id)``.
"""

from __future__ import annotations

import hashlib
import os
import random
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Literal, Sequence

from .budgets import Budgets, BudgetTracker, BudgetTrip, CostEstimator
from .context import Compactor, PassthroughAssembler, estimate_tokens, validate_tool_pairing
from .errors import (
    ApprovalRequired,
    CompactionError,
    ContextWindowExceededError,
    ModelError,
    RateLimitError,
    TransientModelError,
)
from .evaluators import Evaluation, StopContext, StopEvaluator, Verdict
from .hooks import HookEvent, HookInput, HookRegistry, HookResult, combine_permission
from .permissions import Approver, ReadOnlyPolicy
from .prompts import (
    BUDGET_WARNING,
    BUDGET_WRAP_UP,
    DOOM_LOOP_NUDGE,
    INTERRUPTED_SIDE_EFFECTS_UNKNOWN,
    MAX_TOKENS_CONTINUATION,
    NOT_EXECUTED_BUDGET,
    NOT_EXECUTED_INTERRUPT,
    TRUNCATED_TOOL_INPUT,
)
from .results import PendingApproval, ResultSubtype, RunResult
from .retry import RetryPolicy
from .session import PendingToolBatch, Session, SessionStatus
from .stuck import StuckConfig, StuckDetector, StuckSignal
from .tools import FunctionToolExecutor, error_outcome, to_outcome
from .types import (
    ContentBlock,
    ContextAssembler,
    Event,
    EventSink,
    EventType,
    Message,
    ModelClient,
    ModelRequest,
    ModelResponse,
    NullTracer,
    PermissionBehavior,
    PermissionDecision,
    PermissionPolicy,
    Role,
    RunContext,
    StopReason,
    TextBlock,
    ToolCall,
    ToolExecutor,
    ToolOutcome,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Tracer,
    Usage,
)


@dataclass
class LoopConfig:
    """Loop behaviour. Values marked "vendor" come from the research; the rest are our choices.

    * ``max_tokens`` 16,000 - Anthropic non-streaming default (vendor).
    * ``max_tokens_ceiling`` 64,000 - escalation cap for truncated tool input (ours).
    * ``max_output_continuations`` 3, ``max_pause_continuations`` 10 (ours).
    * ``max_parallel_tools`` 8 - concurrency cap for read-only tools (ours; "cap concurrency").
    * ``compaction_threshold_fraction`` 0.9 - Codex ``auto_compact_token_limit`` (vendor).
    * ``max_compaction_attempts`` 3 - "stops auto-compacting after a few attempts" (vendor, number ours).
    * ``max_consecutive_stop_blocks`` 8 - Claude Code Stop-hook override (vendor).
    """

    max_tokens: int = 16_000
    max_tokens_ceiling: int = 64_000
    max_output_continuations: int = 3
    max_pause_continuations: int = 10
    parallel_read_only: bool = True
    max_parallel_tools: int = 8
    context_window_tokens: int | None = None
    compaction_threshold_tokens: int | None = None
    compaction_threshold_fraction: float = 0.9
    max_compaction_attempts: int = 3
    max_consecutive_stop_blocks: int = 8
    max_total_stop_blocks: int | None = None
    wrap_up_on_budget: bool = True
    on_evaluator_error: Literal["allow_stop", "block"] = "allow_stop"
    stuck: StuckConfig | None = field(default_factory=StuckConfig)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    model_label: str = ""   # model id for cost estimation when a response does not report one

    @property
    def compaction_threshold(self) -> int | None:
        candidates = [self.compaction_threshold_tokens] if self.compaction_threshold_tokens else []
        if self.context_window_tokens:
            candidates.append(int(self.context_window_tokens * self.compaction_threshold_fraction))
        return min(candidates) if candidates else None


class SteeringQueue:
    """Thread-safe FIFO of user input pushed while a run is in progress."""

    def __init__(self) -> None:
        self._q: deque[str] = deque()
        self._lock = threading.Lock()

    def push(self, text: str) -> None:
        if text and text.strip():
            with self._lock:
                self._q.append(text)

    def drain(self) -> list[str]:
        with self._lock:
            items = list(self._q)
            self._q.clear()
            return items

    def __len__(self) -> int:
        with self._lock:
            return len(self._q)


class _Halt(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _Interrupted(Exception):
    pass


class _Pause(Exception):
    def __init__(self, pending: PendingApproval) -> None:
        super().__init__(pending.reason)
        self.pending = pending


@dataclass
class _RunState:
    tracker: BudgetTracker
    run_usage: Usage = field(default_factory=Usage)
    run_cost: float = 0.0
    run_turns: int = 0
    final_text: str = ""
    stop_reason: StopReason | None = None
    carry: list[str] = field(default_factory=list)
    output_continuations: int = 0
    pause_continuations: int = 0
    max_tokens_override: int | None = None
    consecutive_compactions: int = 0
    last_context_tokens: int = 0
    consecutive_blocks: int = 0
    total_blocks: int = 0
    stop_hook_active: bool = False
    stop_override: bool = False
    nudges: int = 0
    evaluations: list[Evaluation] = field(default_factory=list)
    wrapped_up: bool = False
    using_fallback: bool = False
    approvals: dict[str, Any] = field(default_factory=dict)
    started: float = 0.0


def idempotency_key(session_id: str, step: int, tool_use_id: str) -> str:
    """``hash(session_id, step_idx, tool_call_id)`` from the hosted-durability checklist item."""
    return hashlib.sha256(f"{session_id}:{step}:{tool_use_id}".encode()).hexdigest()[:32]


class AgentLoop:
    """Drive one :class:`~harness_core.session.Session` with a model, tools and controls.

    :param model: the :class:`ModelClient` (e.g. :class:`~harness_core.anthropic_adapter.AnthropicModelClient`).
    :param executor: the "hands"; any :class:`ToolExecutor`. Its ``specs()`` are frozen at the start
        of each run so the cached tool prefix never changes mid-run.
    :param policy: :class:`PermissionPolicy`; default :class:`~harness_core.permissions.ReadOnlyPolicy`.
    :param approver: called for ``ASK`` decisions; without one, ``ASK`` means deny.
    :param evaluators: stop evaluators that may veto a stop and inject feedback.
    :param compactor: called on threshold/overflow; without one, overflow ends the run.
    :param session: defaults to an in-memory session; pass a :class:`SessionStore` session for durability.
    :param sinks: extra :class:`EventSink` s that receive a copy of every event (e.g. a UI stream).
    """

    def __init__(
        self,
        model: ModelClient,
        executor: ToolExecutor | None = None,
        *,
        system: str = "",
        config: LoopConfig | None = None,
        budgets: Budgets | None = None,
        assembler: ContextAssembler | None = None,
        policy: PermissionPolicy | None = None,
        approver: Approver | None = None,
        hooks: HookRegistry | None = None,
        evaluators: Sequence[StopEvaluator] = (),
        compactor: Compactor | None = None,
        session: Session | None = None,
        sinks: Sequence[EventSink] = (),
        tracer: Tracer | None = None,
        cost_estimator: CostEstimator | None = None,
        fallback_model: ModelClient | None = None,
        cwd: str | None = None,
        agent_name: str = "main",
        depth: int = 0,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.model = model
        self.executor: ToolExecutor = executor if executor is not None else FunctionToolExecutor()
        self.system = system
        self.config = config or LoopConfig()
        self.budgets = budgets or Budgets()
        self.assembler: ContextAssembler = assembler or PassthroughAssembler(max_tokens=self.config.max_tokens)
        self.policy: PermissionPolicy = policy or ReadOnlyPolicy()
        self.approver = approver
        self.hooks = hooks or HookRegistry()
        self.evaluators = list(evaluators)
        self.compactor = compactor
        self.session = session or Session.in_memory()
        self.sinks = list(sinks)
        self.tracer: Tracer = tracer or NullTracer()
        self.cost_estimator = cost_estimator
        self.fallback_model = fallback_model
        self.ctx = RunContext(session_id=self.session.id, cwd=cwd or os.getcwd(), agent_name=agent_name, depth=depth)
        self.monotonic = monotonic
        self.sleep = sleep
        self.rng = rng or random.Random()
        self.steering = SteeringQueue()
        self.stuck = StuckDetector(self.config.stuck) if self.config.stuck is not None else None
        self._interrupt = threading.Event()
        self._tools: list[ToolSpec] = []
        self._spec_by_name: dict[str, ToolSpec] = {}
        self._tainted_reported = False

    # ================================================================== public API
    def steer(self, text: str) -> None:
        """Queue user input mid-run; it is read after the current tool batch, within the same turn."""
        self.steering.push(text)

    def interrupt(self) -> None:
        """Hard cancel: stop at the next checkpoint and ask the executor to cancel running tools."""
        self._interrupt.set()
        cancel = getattr(self.executor, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:  # noqa: BLE001 - best effort
                pass

    def run(self, prompt: str | Message | None = None, *, approvals: dict[str, bool | PermissionDecision] | None = None) -> RunResult:
        """Run until the stop predicate holds, a limit trips, or the run pauses for approval.

        ``prompt`` may be ``None`` to resume a woken session (finish a half-done tool batch or
        answer a pending user message). ``approvals`` maps ``tool_use_id`` to the decision for
        calls that paused with ``paused_for_approval``.
        """
        self._interrupt.clear()
        st = _RunState(tracker=BudgetTracker(self.budgets, monotonic=self.monotonic), approvals=dict(approvals or {}))
        st.started = self.monotonic()
        if self.budgets.scope == "session":
            usage, cost, calls = self.session.total_usage()
            st.tracker.usage, st.tracker.cost_usd, st.tracker.turns = usage, cost, calls
        with self.session.lease():
            self.session.set_status(SessionStatus.RUNNING)
            with self.tracer.span("invoke_agent", session_id=self.session.id, agent_name=self.ctx.agent_name):
                try:
                    result = self._run(st, prompt)
                except _Halt as h:
                    result = self._finish(st, ResultSubtype.HALTED_BY_HOOK, reason=h.reason)
                except _Interrupted:
                    result = self._finish(st, ResultSubtype.INTERRUPTED, reason="interrupted by user")
                except _Pause as p:
                    result = self._finish(st, ResultSubtype.PAUSED_FOR_APPROVAL, reason=p.pending.reason, pending=[p.pending])
                except Exception as exc:  # noqa: BLE001 - a harness bug must still end with a typed result
                    result = self._finish(st, ResultSubtype.ERROR_DURING_EXECUTION, error=f"{type(exc).__name__}: {exc}")
            self.session.set_status(_status_for(result.subtype))
        return result

    def end_session(self, reason: str = "") -> None:
        """Fire ``session_end`` and mark the session completed."""
        self._run_hooks(HookInput(HookEvent.SESSION_END, self.session.id, self.ctx, data={"reason": reason}))
        self._emit(EventType.SESSION_END, {"reason": reason})
        self.session.set_status(SessionStatus.COMPLETED)

    # ================================================================== run body
    def _run(self, st: _RunState, prompt: str | Message | None) -> RunResult:
        self._tools = list(self.executor.specs())
        self._spec_by_name = {s.name: s for s in self._tools}
        source = "resume" if self.session.has_event(EventType.SESSION_START) else "startup"
        self._emit(EventType.SESSION_START, {"source": source, "agent_name": self.ctx.agent_name, "depth": self.ctx.depth,
                                             "tools": [s.name for s in self._tools],
                                             "system_sha256": hashlib.sha256(self.system.encode()).hexdigest()})
        for r in self._run_hooks(HookInput(HookEvent.SESSION_START, self.session.id, self.ctx, data={"source": source})):
            if r.additional_context:
                st.carry.append(r.additional_context)

        pending = self.session.pending_tool_batch()
        if pending is not None:
            done = self._resume_batch(st, pending)
            if done is not None:
                return done

        if prompt is not None:
            msg = prompt if isinstance(prompt, Message) else Message.user(prompt)
            results = self._run_hooks(HookInput(HookEvent.USER_PROMPT_SUBMIT, self.session.id, self.ctx, prompt=msg))
            blocked = next((r for r in results if r.block), None)
            if blocked is not None:
                self._emit(EventType.CUSTOM, {"prompt_blocked": True, "reason": blocked.reason})
                return self._finish(st, ResultSubtype.ERROR_PROMPT_BLOCKED, reason=blocked.reason or "prompt blocked by hook")
            extra = [r.additional_context for r in results if r.additional_context]
            context_blocks: list[ContentBlock] = [TextBlock(t) for t in [*st.carry, *extra]]
            st.carry.clear()
            self._append(Message(msg.role, [*context_blocks, *msg.content], id=msg.id, metadata=msg.metadata), kind="prompt")
        else:
            view = self.session.view()
            if not view or view[-1].role is Role.ASSISTANT:
                queued = self.steering.drain()
                if not queued:
                    if view:
                        st.final_text = view[-1].text
                    return self._finish(st, ResultSubtype.SUCCESS, reason="nothing to do")
                self._append_user([], queued, st, kind="steering")
        return self._loop(st)

    def _loop(self, st: _RunState) -> RunResult:
        cfg = self.config
        while True:
            self._check_interrupt()
            threshold = cfg.compaction_threshold
            if self.compactor is not None and threshold and st.last_context_tokens >= threshold:
                done = self._compact(st, "threshold")
                if done is not None:
                    return done

            request = self._prepare_request(st)
            try:
                response = self._call_model(request, st)
            except ContextWindowExceededError as exc:
                self._emit(EventType.ERROR, {"error": "context_window_exceeded", "detail": str(exc)})
                done = self._compact(st, "overflow")
                if done is not None:
                    return done
                continue
            except (_Interrupted, _Halt):
                raise
            except Exception as exc:  # noqa: BLE001 - retries exhausted or terminal model error
                return self._finish(st, ResultSubtype.ERROR_DURING_EXECUTION, error=f"{type(exc).__name__}: {exc}")

            sr = response.stop_reason
            st.stop_reason = sr
            cost = self._account(st, response.usage, response.model)
            post = self._run_hooks(HookInput(HookEvent.POST_MODEL, self.session.id, self.ctx, request=request, response=response))
            st.carry.extend(r.additional_context for r in post if r.additional_context)
            halt = next((r for r in post if not r.continue_), None)

            if sr is StopReason.MODEL_CONTEXT_WINDOW_EXCEEDED:
                self._emit_discarded(response, cost, "model_context_window_exceeded")
                if halt is not None:
                    raise _Halt(halt.stop_reason or "halted by post_model hook")
                done = self._compact(st, "overflow")
                if done is not None:
                    return done
                continue
            st.consecutive_compactions = 0
            st.last_context_tokens = response.usage.total_input + response.usage.output_tokens

            if sr is StopReason.MAX_TOKENS and response.tool_uses and request.max_tokens < cfg.max_tokens_ceiling:
                st.max_tokens_override = min(cfg.max_tokens_ceiling, max(request.max_tokens * 2, request.max_tokens + 1))
                self._emit_discarded(response, cost, "max_tokens_truncated_tool_use", retry_max_tokens=st.max_tokens_override)
                if halt is not None:
                    raise _Halt(halt.stop_reason or "halted by post_model hook")
                continue
            if sr is not StopReason.MAX_TOKENS:
                st.max_tokens_override = None

            assistant = Message.assistant(response.content, stop_reason=sr.value)
            event = self._append(assistant, usage=asdict(response.usage), cost_usd=cost, model=response.model,
                                 stop_reason=sr.value, response_id=response.id)
            if sr is StopReason.REFUSAL:  # checked before reading content
                return self._finish(st, ResultSubtype.ERROR_REFUSAL, reason="the model declined the request")
            if response.text:
                st.final_text = response.text
            if halt is not None:
                raise _Halt(halt.stop_reason or "halted by post_model hook")

            if response.tool_uses:
                done = self._tool_turn(st, event.seq, response.tool_uses, truncated=sr is StopReason.MAX_TOKENS)
                if done is not None:
                    return done
                continue

            if sr is StopReason.PAUSE_TURN:
                st.pause_continuations += 1
                if st.pause_continuations > cfg.max_pause_continuations:
                    return self._finish(st, ResultSubtype.ERROR_DURING_EXECUTION,
                                        reason=f"more than {cfg.max_pause_continuations} pause_turn continuations")
                trip = st.tracker.exceeded(need_another_call=True)
                if trip is not None:
                    return self._finish(st, trip.subtype, reason=trip.reason)
                continue  # resend: the assistant turn is the last message, so the server resumes it

            if sr is StopReason.MAX_TOKENS and st.output_continuations < cfg.max_output_continuations:
                trip = st.tracker.exceeded(need_another_call=True)
                if trip is not None:
                    return self._finish(st, trip.subtype, reason=trip.reason)
                st.output_continuations += 1
                self._append_user([], [MAX_TOKENS_CONTINUATION], st, kind="max_tokens_continuation")
                continue

            # ---- no tool calls: queued input, then stop arbitration
            queued = self.steering.drain()
            if queued:
                trip = st.tracker.exceeded(need_another_call=True)
                if trip is not None:
                    return self._finish(st, trip.subtype, reason=trip.reason, unprocessed=queued)
                self._append_user([], queued, st, kind="steering")
                continue
            verdict = self._arbitrate_stop(st)
            if isinstance(verdict, RunResult):
                return verdict
            if verdict is None:
                return self._finish(st, ResultSubtype.SUCCESS)
            trip = st.tracker.exceeded(need_another_call=True)
            if trip is not None:
                return self._finish(st, trip.subtype, reason=trip.reason)
            self._append_user([], [verdict], st, kind="continuation")

    # ================================================================== model calls
    def _prepare_request(self, st: _RunState) -> ModelRequest:
        request = self.assembler.prepare(self.system, self.session.view(), list(self._tools), self.ctx)
        if st.max_tokens_override and st.max_tokens_override > request.max_tokens:
            request = replace(request, max_tokens=st.max_tokens_override)
        for r in self._run_hooks(HookInput(HookEvent.PRE_MODEL, self.session.id, self.ctx, request=request)):
            if r.updated_request is not None:
                request = r.updated_request
            if not r.continue_:
                raise _Halt(r.stop_reason or r.reason or "halted by pre_model hook")
        self._emit(EventType.MODEL_REQUEST, {"messages": len(request.messages), "tools": len(request.tools),
                                             "max_tokens": request.max_tokens, "fallback": st.using_fallback})
        return request

    def _call_model(self, request: ModelRequest, st: _RunState) -> ModelResponse:
        policy = self.config.retry
        attempt = rl_attempt = 0
        with self.tracer.span("chat", agent_name=self.ctx.agent_name, max_tokens=request.max_tokens) as span:
            while True:
                client = self.fallback_model if (st.using_fallback and self.fallback_model is not None) else self.model
                try:
                    response = client.create(request)
                    _set_attr(span, "retries", attempt + rl_attempt)
                    _set_attr(span, "stop_reason", response.stop_reason.value)
                    return response
                except RateLimitError as exc:
                    rl_attempt += 1
                    if rl_attempt > policy.max_rate_limit_retries:
                        raise
                    delay = policy.rate_limit_delay(rl_attempt, exc.retry_after, self.rng)
                    self._emit(EventType.ERROR, {"error": "rate_limited", "attempt": rl_attempt, "delay_s": delay, "detail": str(exc)})
                except TransientModelError as exc:
                    attempt += 1
                    if attempt > policy.max_retries:
                        if self.fallback_model is not None and not st.using_fallback:
                            st.using_fallback = True
                            attempt = 0
                            self._emit(EventType.ERROR, {"error": "fallback_model", "detail": str(exc)})
                            continue
                        raise
                    delay = policy.backoff(attempt, self.rng)
                    self._emit(EventType.ERROR, {"error": "transient", "attempt": attempt, "delay_s": delay, "detail": str(exc)})
                except (ContextWindowExceededError, ModelError):
                    raise
                self._check_interrupt()
                self.sleep(delay)
                self._check_interrupt()

    def _account(self, st: _RunState, usage: Usage, model: str, *, count_turn: bool = True) -> float:
        cost = 0.0
        if self.cost_estimator is not None:
            try:
                cost = float(self.cost_estimator(usage, model or self.config.model_label))
            except Exception as exc:  # noqa: BLE001
                self._emit(EventType.ERROR, {"error": "cost_estimator", "detail": f"{type(exc).__name__}: {exc}"})
        st.tracker.record_model_call(usage, cost, count_turn=count_turn)
        st.run_usage = st.run_usage + usage
        st.run_cost += cost
        if count_turn:
            st.run_turns += 1
        return cost

    def _emit_discarded(self, response: ModelResponse, cost: float, why: str, **data: Any) -> None:
        self._emit(EventType.MODEL_RESPONSE, {"discarded": why, "stop_reason": response.stop_reason.value,
                                              "usage": asdict(response.usage), "cost_usd": cost, "model": response.model,
                                              "discarded_content": [b.to_api() for b in response.content], **data})

    # ================================================================== tool turns
    def _tool_turn(self, st: _RunState, step: int, uses: list[ToolUseBlock], *, truncated: bool) -> RunResult | None:
        calls = [ToolCall.from_block(u) for u in uses]
        for c in calls:
            self._emit(EventType.TOOL_CALL, {"phase": "requested", "tool_use_id": c.id, "name": c.name, "input": c.input})
        st.consecutive_blocks = 0

        if truncated:
            blocks = [ToolResultBlock(c.id, TRUNCATED_TOOL_INPUT, is_error=True) for c in calls]
            for b in blocks:
                self._record_result(b, executed=False)
            st.tracker.record_tool_turn(all_failed=True)
            trip = st.tracker.exceeded(need_another_call=True)
            if trip is not None:
                return self._budget_stop(st, blocks, [], trip)
            self._append_user(blocks, self.steering.drain(), st, kind="tool_results")
            return None

        trip = st.tracker.exceeded(need_another_call=True)
        if trip is not None:
            return self._budget_stop(st, [], calls, trip)

        if self.stuck is not None:
            signal = self.stuck.observe_calls(calls)
            if signal is not None and self._on_stuck(st, signal):
                return self._finish(st, ResultSubtype.ERROR_DOOM_LOOP, reason=signal.detail)

        return self._complete_batch(st, step, calls, {})

    def _complete_batch(self, st: _RunState, step: int, calls: list[ToolCall], done: dict[str, ToolResultBlock]) -> RunResult | None:
        results = self._execute_batch(st, step, calls, done)
        outcomes = [ToolOutcome(r.content, r.is_error) for r in results]
        st.tracker.record_tool_turn(all_failed=bool(results) and all(r.is_error for r in results))
        doom = None
        if self.stuck is not None:
            signal = self.stuck.observe_results(calls, outcomes)
            if signal is not None and self._on_stuck(st, signal):
                doom = signal

        texts = self.steering.drain()
        texts.extend(BUDGET_WARNING.format(detail=w) for w in st.tracker.new_warnings())
        trip = st.tracker.exceeded(need_another_call=True)
        if trip is not None:
            return self._budget_stop(st, results, [], trip, texts)
        self._append_user(results, texts, st, kind="tool_results")
        if doom is not None:
            return self._finish(st, ResultSubtype.ERROR_DOOM_LOOP, reason=doom.detail)
        limit = st.tracker.error_limit()
        if limit is not None:
            return self._finish(st, limit.subtype, reason=limit.reason)
        return None

    def _on_stuck(self, st: _RunState, signal: StuckSignal) -> bool:
        """Nudge, or return True to terminate (a hard signal after ``max_nudges`` nudges)."""
        assert self.stuck is not None
        self._emit(EventType.CUSTOM, {"stuck": signal.kind, "detail": signal.detail, "nudges": st.nudges})
        if signal.hard:
            if st.nudges >= self.stuck.config.max_nudges:
                return True
            st.nudges += 1
            self.stuck.reset()
        st.carry.append(DOOM_LOOP_NUDGE.format(detail=signal.detail))
        return False

    def _budget_stop(self, st: _RunState, blocks: list[ToolResultBlock], unrun: list[ToolCall], trip: BudgetTrip,
                     texts: list[str] | None = None) -> RunResult:
        """Pair every pending call with a result, optionally run the wrap-up turn, and finish."""
        self._emit(EventType.CUSTOM, {"budget_exhausted": trip.subtype.value, "reason": trip.reason})
        blocks = list(blocks)
        for c in unrun:
            b = ToolResultBlock(c.id, NOT_EXECUTED_BUDGET.format(reason=trip.reason), is_error=True)
            self._record_result(b, executed=False)
            blocks.append(b)
        extra = list(texts or [])
        if self.config.wrap_up_on_budget:
            extra.append(BUDGET_WRAP_UP.format(reason=trip.reason))
        self._append_user(blocks, extra, st, kind="tool_results" if blocks else "wrap_up")
        if self.config.wrap_up_on_budget:
            self._wrap_up(st)
        return self._finish(st, trip.subtype, reason=trip.reason)

    def _wrap_up(self, st: _RunState) -> None:
        try:
            request = self._prepare_request(st)
            response = self._call_model(request, st)
        except (_Halt, _Interrupted):
            raise
        except Exception as exc:  # noqa: BLE001 - the wrap-up is best effort
            self._emit(EventType.ERROR, {"error": "wrap_up_failed", "detail": f"{type(exc).__name__}: {exc}"})
            return
        cost = self._account(st, response.usage, response.model, count_turn=False)
        if response.stop_reason is StopReason.MODEL_CONTEXT_WINDOW_EXCEEDED:
            self._emit_discarded(response, cost, "wrap_up_context_window_exceeded")
            return
        self._append(Message.assistant(response.content, stop_reason=response.stop_reason.value),
                     usage=asdict(response.usage), cost_usd=cost, model=response.model,
                     stop_reason=response.stop_reason.value, response_id=response.id, wrap_up=True)
        st.stop_reason = response.stop_reason
        st.final_text = response.text or st.final_text
        st.wrapped_up = True

    def _execute_batch(self, st: _RunState, step: int, calls: list[ToolCall], done: dict[str, ToolResultBlock]) -> list[ToolResultBlock]:
        results: dict[str, ToolResultBlock] = dict(done)
        todo = [c for c in calls if c.id not in results]
        for group in self._partition(todo):
            self._check_interrupt()
            approved: list[tuple[ToolCall, ToolSpec]] = []
            for call in group:
                spec = self._spec_by_name.get(call.name)
                if spec is None:
                    blk = ToolResultBlock(call.id, f"Error: unknown tool {call.name!r}. Available tools: "
                                          f"{', '.join(sorted(self._spec_by_name)) or '(none)'}", is_error=True)
                    self._record_result(blk, executed=False)
                    results[call.id] = blk
                    continue
                authorized = self._authorize(call, spec, st)
                if isinstance(authorized, ToolResultBlock):
                    self._record_result(authorized, executed=False)
                    results[call.id] = authorized
                else:
                    approved.append((authorized, spec))
            outcomes = self._run_group(approved, step)
            halt_reason = None
            for (call, spec), outcome in zip(approved, outcomes, strict=True):
                outcome = self._post_tool(call, spec, outcome, st)
                if outcome.metadata.get("_halt"):
                    halt_reason = outcome.metadata.pop("_halt")
                self._note_taint(spec, outcome)
                blk = outcome.to_block(call.id)
                self._record_result(blk, executed=True)
                results[call.id] = blk
            if halt_reason is not None:
                raise _Halt(halt_reason)
        return [results[c.id] for c in calls]

    def _partition(self, calls: list[ToolCall]) -> list[list[ToolCall]]:
        """Consecutive read-only calls form one concurrent group; every other call is its own group."""
        groups: list[tuple[bool, list[ToolCall]]] = []
        for c in calls:
            spec = self._spec_by_name.get(c.name)
            parallel = self.config.parallel_read_only and spec is not None and spec.read_only
            if parallel and groups and groups[-1][0]:
                groups[-1][1].append(c)
            else:
                groups.append((parallel, [c]))
        return [g for _, g in groups]

    def _run_group(self, approved: list[tuple[ToolCall, ToolSpec]], step: int) -> list[ToolOutcome]:
        for call, _ in approved:
            self._emit(EventType.TOOL_CALL, {"phase": "executing", "tool_use_id": call.id, "name": call.name, "input": call.input,
                                             "idempotency_key": idempotency_key(self.session.id, step, call.id)})
        if len(approved) <= 1:
            return [self._exec_one(c, s, step) for c, s in approved]
        workers = max(1, min(self.config.max_parallel_tools, len(approved)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="harness-tool") as pool:
            futures = [pool.submit(self._exec_one, c, s, step) for c, s in approved]
            return [f.result() for f in futures]

    def _exec_one(self, call: ToolCall, spec: ToolSpec, step: int) -> ToolOutcome:
        ctx = replace(self.ctx, metadata={**self.ctx.metadata, "tool_use_id": call.id,
                                          "idempotency_key": idempotency_key(self.session.id, step, call.id)})
        with self.tracer.span("execute_tool", tool_name=call.name, tool_use_id=call.id, read_only=spec.read_only):
            try:
                out = self.executor.execute(call, ctx)
            except Exception as exc:  # noqa: BLE001 - tool failures go back to the model
                return error_outcome(f"Error: {type(exc).__name__}: {exc}", exception=type(exc).__name__)
        return out if isinstance(out, ToolOutcome) else to_outcome(out)

    def _authorize(self, call: ToolCall, spec: ToolSpec, st: _RunState) -> ToolCall | ToolResultBlock:
        inp = HookInput(HookEvent.PRE_TOOL, self.session.id, self.ctx, tool_call=call, tool_spec=spec)
        hres = self._run_hooks(inp)
        halt = next((r for r in hres if not r.continue_), None)
        if halt is not None:
            raise _Halt(halt.stop_reason or halt.reason or "halted by pre_tool hook")
        assert inp.tool_call is not None
        call = inp.tool_call
        st.carry.extend(r.additional_context for r in hres if r.additional_context)
        hook_perm, hook_reason = combine_permission(hres)
        if hook_perm is PermissionBehavior.DENY:
            return self._deny(call, st, hook_reason or "denied by pre_tool hook", "hook")

        decision = self.policy.check(call, spec, self.ctx)
        if decision.updated_input is not None:
            call = ToolCall(call.id, call.name, dict(decision.updated_input))
        if decision.behavior is PermissionBehavior.DENY:
            return self._deny(call, st, decision.reason or "denied by policy", "policy")
        need_ask = hook_perm is PermissionBehavior.ASK or (decision.behavior is PermissionBehavior.ASK and hook_perm is not PermissionBehavior.ALLOW)
        source = "hook" if hook_perm is PermissionBehavior.ALLOW and decision.behavior is PermissionBehavior.ASK else "policy"
        if need_ask:
            source = "approver"
            if call.id in st.approvals:
                answer: Any = st.approvals.pop(call.id)
                source = "resumed_approval"
            elif self.approver is None:
                return self._deny(call, st, "approval required but no approver is configured", "default")
            else:
                self._emit(EventType.PERMISSION, {"tool_use_id": call.id, "name": call.name, "behavior": "ask", "reason": decision.reason})
                try:
                    with self.tracer.span("approval_wait", tool_name=call.name, tool_use_id=call.id):
                        answer = self.approver(call, spec, self.ctx, decision)
                except ApprovalRequired as req:
                    pending = PendingApproval(call.id, call.name, dict(call.input), req.reason)
                    self._emit(EventType.PERMISSION, {"tool_use_id": call.id, "name": call.name, "behavior": "ask",
                                                      "status": "pending", "reason": req.reason})
                    raise _Pause(pending) from None
            if isinstance(answer, PermissionDecision):
                if answer.updated_input is not None:
                    call = ToolCall(call.id, call.name, dict(answer.updated_input))
                ok, why = answer.allowed, answer.reason
            else:
                ok, why = bool(answer), ""
            if not ok:
                return self._deny(call, st, why or "denied by approver", source)
        self._emit(EventType.PERMISSION, {"tool_use_id": call.id, "name": call.name, "behavior": "allow", "source": source,
                                          "reason": decision.reason})
        st.tracker.record_permitted()
        return call

    def _deny(self, call: ToolCall, st: _RunState, reason: str, source: str) -> ToolResultBlock:
        self._emit(EventType.PERMISSION, {"tool_use_id": call.id, "name": call.name, "behavior": "deny", "source": source, "reason": reason})
        st.tracker.record_denial()
        return ToolResultBlock(call.id, f"Permission denied: {reason}", is_error=True)

    def _post_tool(self, call: ToolCall, spec: ToolSpec, outcome: ToolOutcome, st: _RunState) -> ToolOutcome:
        event = HookEvent.POST_TOOL_FAILURE if outcome.is_error else HookEvent.POST_TOOL
        if not self.hooks.has(event):
            return outcome
        for r in self._run_hooks(HookInput(event, self.session.id, self.ctx, tool_call=call, tool_spec=spec, outcome=outcome)):
            if r.updated_outcome is not None:
                outcome = r.updated_outcome
            if r.additional_context:
                st.carry.append(r.additional_context)
            if not r.continue_:
                outcome = replace(outcome, metadata={**outcome.metadata, "_halt": r.stop_reason or r.reason or "halted by post_tool hook"})
        return outcome

    def _note_taint(self, spec: ToolSpec, outcome: ToolOutcome) -> None:
        if (spec.open_world or outcome.metadata.get("untrusted")) and not self.ctx.tainted:
            self.ctx.tainted = True
            if not self._tainted_reported:
                self._tainted_reported = True
                self._emit(EventType.CUSTOM, {"tainted": True, "by_tool": spec.name})

    def _record_result(self, block: ToolResultBlock, *, executed: bool) -> None:
        self._emit(EventType.TOOL_RESULT, {"tool_use_id": block.tool_use_id, "is_error": block.is_error, "executed": executed,
                                           "block": block.to_api()})

    def _resume_batch(self, st: _RunState, pending: PendingToolBatch) -> RunResult | None:
        """Finish a tool batch left half-done by a crash or an approval pause."""
        done = dict(pending.completed)
        rerun: set[str] = set()
        for use in pending.tool_uses:
            if use.id not in pending.started:
                continue
            spec = self._spec_by_name.get(use.name)
            if spec is not None and spec.read_only:
                rerun.add(use.id)  # retry-safe by definition
            else:
                blk = ToolResultBlock(use.id, INTERRUPTED_SIDE_EFFECTS_UNKNOWN, is_error=True)
                self._record_result(blk, executed=False)
                done[use.id] = blk
        self._emit(EventType.CUSTOM, {"resume_repair": True, "completed": sorted(pending.completed), "rerun": sorted(rerun),
                                      "unknown_side_effects": sorted(pending.started - rerun),
                                      "not_started": [u.id for u in pending.not_started]})
        calls = [ToolCall.from_block(u) for u in pending.tool_uses]
        return self._complete_batch(st, pending.assistant_seq, calls, done)

    # ================================================================== stop arbitration
    def _arbitrate_stop(self, st: _RunState) -> str | RunResult | None:
        """``None`` = stop allowed; ``str`` = continuation feedback; :class:`RunResult` = terminal."""
        view = self.session.view()
        blocks: list[str] = []
        for r in self._run_hooks(HookInput(HookEvent.STOP, self.session.id, self.ctx, last_assistant_message=st.final_text,
                                           stop_hook_active=st.stop_hook_active)):
            if not r.continue_:
                return self._finish(st, ResultSubtype.HALTED_BY_HOOK, reason=r.stop_reason or r.reason or "halted by stop hook")
            if r.block:
                if r.reason.strip():
                    blocks.append(r.reason.strip())
                else:
                    self._emit(EventType.EVALUATION, {"evaluator": "stop_hook", "verdict": "not_met", "ignored": "block without reason"})
        sc = StopContext(self.session.id, view, st.final_text, self.ctx, st.run_turns, st.run_usage, st.stop_hook_active,
                         st.consecutive_blocks, st.total_blocks, self.budgets.max_total_tokens, st.tracker.tokens_used)
        for ev in self.evaluators:
            name = getattr(ev, "name", type(ev).__name__)
            try:
                with self.tracer.span("evaluate", evaluator=name):
                    e = ev.evaluate(sc)
            except Exception as exc:  # noqa: BLE001
                self._emit(EventType.ERROR, {"error": "evaluator_failed", "evaluator": name, "detail": f"{type(exc).__name__}: {exc}"})
                if self.config.on_evaluator_error == "block":
                    blocks.append(f"The completion check '{name}' could not run ({type(exc).__name__}). Verify your work and "
                                  "show evidence that the task is complete.")
                continue
            e.evaluator = e.evaluator or name
            st.evaluations.append(e)
            if e.usage.total_input or e.usage.output_tokens or e.cost_usd:
                st.tracker.record_model_call(e.usage, e.cost_usd, count_turn=False)
                st.run_cost += e.cost_usd
            ignored = e.verdict is Verdict.NOT_MET and not e.continuation
            self._emit(EventType.EVALUATION, {"evaluator": e.evaluator, "verdict": e.verdict.value, "reason": e.reason,
                                              "ignored": "no feedback" if ignored else None, **e.data})
            if e.verdict is Verdict.IMPOSSIBLE:
                return self._finish(st, ResultSubtype.ERROR_GOAL_IMPOSSIBLE, reason=e.reason)
            if e.verdict is Verdict.NOT_MET and e.continuation:
                blocks.append(e.continuation)
        if not blocks:
            return None
        cfg = self.config
        if st.consecutive_blocks >= cfg.max_consecutive_stop_blocks or (
                cfg.max_total_stop_blocks is not None and st.total_blocks >= cfg.max_total_stop_blocks):
            st.stop_override = True
            self._emit(EventType.EVALUATION, {"evaluator": "loop", "verdict": "override",
                                              "reason": f"stop allowed after {st.consecutive_blocks} consecutive blocks"})
            return None
        st.consecutive_blocks += 1
        st.total_blocks += 1
        st.stop_hook_active = True
        return "\n\n".join(blocks)

    # ================================================================== compaction
    def _compact(self, st: _RunState, reason: str) -> RunResult | None:
        cfg = self.config
        if self.compactor is None:
            if reason == "overflow":
                return self._finish(st, ResultSubtype.ERROR_CONTEXT_WINDOW, reason="context window exceeded and no compactor is configured")
            return None
        if st.consecutive_compactions >= cfg.max_compaction_attempts:
            return self._finish(st, ResultSubtype.ERROR_CONTEXT_WINDOW,
                                reason=f"{st.consecutive_compactions} consecutive compactions did not make the context fit")
        st.consecutive_compactions += 1
        view = self.session.view()
        self._run_hooks(HookInput(HookEvent.PRE_COMPACT, self.session.id, self.ctx, data={"trigger": "auto", "reason": reason}))
        before = estimate_tokens(view, self.system)
        failure = None
        with self.tracer.span("compaction", reason=reason, tokens_before=before):
            try:
                res = self.compactor.compact(view, self.ctx, reason=reason)
                if res is None:
                    failure = "compactor freed nothing"
                else:
                    validate_tool_pairing(res.messages)
                    after = estimate_tokens(res.messages, self.system)
                    if after >= before:
                        failure = f"compaction did not shrink the context ({before} -> {after} est. tokens)"
            except CompactionError as exc:
                failure = f"invalid compaction: {exc}"
            except Exception as exc:  # noqa: BLE001
                failure = f"compactor failed: {type(exc).__name__}: {exc}"
        if failure is not None:
            self._emit(EventType.COMPACTION, {"failed": failure, "reason": reason, "attempt": st.consecutive_compactions})
            if st.consecutive_compactions >= cfg.max_compaction_attempts and reason == "overflow":
                return self._finish(st, ResultSubtype.ERROR_CONTEXT_WINDOW, reason=failure)
            st.last_context_tokens = 0
            return None
        assert res is not None
        self.session.record_compaction(res.messages, summary=res.summary, reason=reason, tokens_before=before,
                                       tokens_after=estimate_tokens(res.messages, self.system), **res.metadata)
        st.last_context_tokens = 0
        self._run_hooks(HookInput(HookEvent.POST_COMPACT, self.session.id, self.ctx, data={"trigger": "auto", "reason": reason,
                                                                                         "summary": res.summary}))
        return None

    # ================================================================== helpers
    def _append_user(self, results: list[ToolResultBlock], texts: list[str], st: _RunState, *, kind: str) -> None:
        """Append ONE user message: tool results first, then carried and given text blocks."""
        text_blocks = [TextBlock(t) for t in [*st.carry, *texts] if t and t.strip()]
        st.carry.clear()
        content: list[ContentBlock] = [*results, *text_blocks]
        if not content:
            return
        self._append(Message(Role.USER, content), kind=kind)

    def _run_hooks(self, inp: HookInput) -> list[HookResult]:
        if not self.hooks.has(inp.event):
            return []
        results = self.hooks.run(inp)
        for err in self.hooks.drain_errors():
            self._emit(EventType.ERROR, {"error": "hook_failed", "hook": err.hook, "event": err.event.value, "detail": err.error})
        return results

    def _append(self, message: Message, **data: Any) -> Event:
        event = self.session.append_message(message, **data)
        for s in self.sinks:
            s.emit(event)
        return event

    def _emit(self, type: EventType, data: dict[str, Any]) -> None:
        event = self.session.emit(type, data)
        for s in self.sinks:
            s.emit(event)

    def _check_interrupt(self) -> None:
        if self._interrupt.is_set():
            raise _Interrupted()

    def _close_dangling(self, reason_text: str) -> None:
        """Keep tool_use/tool_result pairing when a run ends mid-batch (results recorded so far are reused)."""
        pending = self.session.pending_tool_batch()
        if pending is None:
            return
        blocks = []
        for use in pending.tool_uses:
            blk = pending.completed.get(use.id)
            if blk is None:
                text = INTERRUPTED_SIDE_EFFECTS_UNKNOWN if use.id in pending.started else reason_text
                blk = ToolResultBlock(use.id, text, is_error=True)
                self._record_result(blk, executed=False)
            blocks.append(blk)
        self._append(Message(Role.USER, list(blocks)), kind="tool_results_aborted")

    def _finish(self, st: _RunState, subtype: ResultSubtype, *, reason: str = "", error: str | None = None,
                pending: list[PendingApproval] | None = None, unprocessed: list[str] | None = None) -> RunResult:
        if subtype is not ResultSubtype.PAUSED_FOR_APPROVAL:
            note = NOT_EXECUTED_INTERRUPT if subtype is ResultSubtype.INTERRUPTED else f"Error: not executed - the run ended ({subtype.value})."
            try:
                self._close_dangling(note)
            except Exception as exc:  # noqa: BLE001
                self._emit(EventType.ERROR, {"error": "close_dangling_failed", "detail": str(exc)})
        unprocessed = list(unprocessed or []) + self.steering.drain()
        result = RunResult(
            subtype=subtype, session_id=self.session.id, stop_reason=st.stop_reason, final_text=st.final_text,
            num_turns=st.run_turns, usage=st.run_usage, total_cost_usd=st.run_cost, duration_s=self.monotonic() - st.started,
            error=error, reason=reason, messages=self.session.messages(), pending_approvals=list(pending or []),
            unprocessed_input=unprocessed, evaluations=list(st.evaluations), wrapped_up=st.wrapped_up, stop_override=st.stop_override,
        )
        self._emit(EventType.STOP, {"subtype": subtype.value, "stop_reason": st.stop_reason.value if st.stop_reason else None,
                                    "num_turns": st.run_turns, "usage": asdict(st.run_usage), "total_cost_usd": st.run_cost,
                                    "reason": reason, "error": error, "wrapped_up": st.wrapped_up,
                                    "pending_approvals": [p.tool_use_id for p in result.pending_approvals]})
        return result


def _set_attr(span: Any, key: str, value: Any) -> None:
    setter = getattr(span, "set_attribute", None)
    if callable(setter):
        setter(key, value)


def _status_for(subtype: ResultSubtype) -> SessionStatus:
    if subtype is ResultSubtype.PAUSED_FOR_APPROVAL:
        return SessionStatus.PAUSED
    if subtype is ResultSubtype.INTERRUPTED:
        return SessionStatus.INTERRUPTED
    if subtype.is_error:
        return SessionStatus.FAILED
    return SessionStatus.IDLE
