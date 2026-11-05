"""Adapters that connect the separately built packages through their published protocols.

* :class:`SessionBoundAssembler` - the loop's :class:`~harness_core.types.ContextAssembler`: projects
  every window from the session's **canonical** log through
  :class:`harness_context.LayeredContextAssembler`, persists compaction markers into the durable
  session (``append_to_log``), and routes a locally detected overflow into the loop's normal
  overflow path (:data:`OVERFLOW_KEY` + :class:`OverflowGuardClient`).
* :class:`ContextCompactorAdapter` - the loop's :class:`harness_core.context.Compactor`, delegating
  to :meth:`LayeredContextAssembler.compact` so there is exactly one compaction implementation.
* :class:`LoopAgentRunner` - :class:`harness_multiagent.AgentRunner` over :class:`harness_core.AgentLoop`.
* :class:`LoopAgentUnderTest` - :class:`harness_evals.AgentUnderTest` that builds a fresh harness per
  trial; :func:`trajectory_from_session` turns a session log into a :class:`harness_evals.Trajectory`.
* :class:`OpsEventAdapter` / :class:`SessionEventSink` / :class:`StepIndexSink` - event plumbing.
"""

from __future__ import annotations

import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

from harness_context import LayeredContextAssembler
from harness_context.errors import CompactionError as ContextCompactionError
from harness_context.errors import ContextOverflowError, NothingToCompactError
from harness_core.budgets import Budgets, CostEstimator
from harness_core.context import CompactionResult, PassthroughAssembler
from harness_core.errors import CompactionError, ContextWindowExceededError
from harness_core.hooks import HookEvent, HookInput, HookRegistry, HookResult
from harness_core.loop import AgentLoop, LoopConfig
from harness_core.permissions import AllowAllPolicy, Approver
from harness_core.results import ResultSubtype, RunResult
from harness_core.session import Session
from harness_core.types import (
    Event,
    EventSink,
    EventType,
    Message,
    ModelClient,
    ModelRequest,
    ModelResponse,
    PermissionPolicy,
    RunContext,
    ToolCall,
    ToolExecutor,
    ToolOutcome,
    ToolSpec,
    Tracer,
    Usage,
)
from harness_evals import AgentTask, InfraError, Trajectory
from harness_multiagent import SubagentResult, SubagentStatus, SubagentTask, parse_report
from harness_multiagent.errors import StructuredOutputError
from harness_multiagent.result import scan_untrusted
from harness_multiagent.runner import status_for_reason

if TYPE_CHECKING:  # pragma: no cover
    from .config import HarnessConfig

#: ``ModelRequest.extra`` key the assembler sets when the window cannot fit; the guard client turns
#: it into :class:`~harness_core.errors.ContextWindowExceededError` without calling the provider.
OVERFLOW_KEY = "__harness_context_overflow__"


# ============================================================================= context


class SessionBoundAssembler:
    """``ContextAssembler`` bound to one durable :class:`~harness_core.session.Session`.

    ``AgentLoop`` passes ``session.view()`` (a copy, possibly a recorded core compaction view). The
    layered assembler owns projection, so this adapter always hands it the canonical log
    (``session.messages()``), which contains every compaction marker; markers the assembler appends
    go to that list *and* to the durable session via :meth:`append_to_log`.
    """

    def __init__(self, session: Session, *, sinks: Sequence[EventSink] = (), inner: LayeredContextAssembler | None = None) -> None:
        self.session = session
        self.sinks = list(sinks)
        self.inner = inner

    def append_to_log(self, log: list[Message], message: Message) -> None:
        log.append(message)
        event = self.session.append_message(message, kind="context")
        for s in self.sinks:
            s.emit(event)

    def prepare(self, system: str, messages: list[Message], tools: list[ToolSpec], ctx: RunContext) -> ModelRequest:
        assert self.inner is not None, "SessionBoundAssembler.inner is not set"
        log = self.session.messages()
        try:
            return self.inner.prepare(system, log, tools, ctx)
        except ContextOverflowError as exc:
            # prepare() runs outside the loop's overflow handler; signal through the model call instead.
            return ModelRequest(system=system, messages=list(messages), tools=list(tools), max_tokens=self.inner.config.max_tokens,
                                extra={OVERFLOW_KEY: str(exc)})

    def observe_response(self, ctx: RunContext, response: ModelResponse) -> bool:
        assert self.inner is not None
        return self.inner.observe_response(ctx, response)


class OverflowGuardClient:
    """Wraps the real client: a request flagged with :data:`OVERFLOW_KEY` raises the loop's overflow error."""

    def __init__(self, inner: ModelClient) -> None:
        self.inner = inner

    def create(self, request: ModelRequest) -> ModelResponse:
        if OVERFLOW_KEY in request.extra:
            raise ContextWindowExceededError(str(request.extra[OVERFLOW_KEY]))
        return self.inner.create(request)


class ContextCompactorAdapter:
    """:class:`harness_core.context.Compactor` that delegates to :meth:`LayeredContextAssembler.compact`.

    The marker is appended to the durable log by the assembler (through ``append_to_log``); the
    adapter returns the assembler's derived history (``[marker] + kept tail``), which is smaller than
    the view, so the loop's "compaction did not shrink" check passes and ``Session.record_compaction``
    records a view consistent with what the assembler projects next.
    """

    def __init__(self, assembler: LayeredContextAssembler, session: Session, *, system: Callable[[], str] | str,
                 tools: Callable[[], list[ToolSpec]]) -> None:
        self.assembler = assembler
        self.session = session
        self._system = system
        self._tools = tools

    def compact(self, messages: list[Message], ctx: RunContext, *, reason: str) -> CompactionResult | None:
        log = self.session.messages()
        system = self._system() if callable(self._system) else self._system
        try:
            res = self.assembler.compact(system, log, list(self._tools()), ctx, focus=None)
        except NothingToCompactError:
            return None
        except ContextCompactionError as exc:
            raise CompactionError(str(exc)) from exc
        derived = self.assembler.derive_history(log)
        return CompactionResult(messages=derived, summary=res.summary.render(), metadata={
            "delegated_to": "harness_context", "covers_through": res.covers_through, "summarized_count": res.summarized_count,
            "trigger": reason,
        })


# ============================================================================= events


class SessionEventSink:
    """``EventSink`` for components that emit their own events (assembler, compactor, sandbox, sub-agent manager).

    Events for this session go into the durable log; events for other sessions (e.g. a nested
    sub-agent's) are only forwarded. Every event is forwarded to ``forward`` sinks.
    """

    def __init__(self, session: Session, forward: Sequence[EventSink] = ()) -> None:
        self.session = session
        self.forward = list(forward)

    def emit(self, event: Event) -> Event:
        if event.session_id == self.session.id:
            event = self.session.log.emit(event)
        for s in self.forward:
            try:
                s.emit(event)
            except Exception:  # noqa: BLE001 - telemetry never breaks the harness
                pass
        return event


class StepIndexSink:
    """Tracks the seq of the last assistant message so tools see ``ctx.metadata["step_idx"]`` (harness_ops idempotency)."""

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx

    def emit(self, event: Event) -> Event:
        if event.type is EventType.MODEL_RESPONSE and "message" in event.data:
            self.ctx.metadata["step_idx"] = event.seq
        return event


class OpsEventAdapter:
    """Translate loop event payloads to the keys ``harness_ops`` consumers read, then forward a copy.

    * ``TOOL_RESULT`` gets ``tool`` (from the matching ``TOOL_CALL``), ``PERMISSION`` gets ``tool``;
    * ``STOP`` sends ``reason = subtype`` (low-cardinality) and keeps the text as ``detail``;
    * ``SESSION_END`` gets ``outcome`` / ``total_cost_usd`` / ``num_turns`` accumulated from ``STOP`` events;
    * ``ERROR`` gets ``error_class``;
    * ``COMPACTION`` events recorded by the loop for a compaction the context package already
      reported (``delegated_to``) or for a failed attempt are not double-counted.
    The durable event is never modified.
    """

    def __init__(self, sink: EventSink) -> None:
        self.sink = sink
        self._names: dict[str, str] = {}
        self._outcome: str | None = None
        self._cost = 0.0
        self._turns = 0
        self._lock = threading.Lock()

    def emit(self, event: Event) -> Event:
        d = dict(event.data or {})
        t = event.type
        with self._lock:
            if t is EventType.TOOL_CALL and d.get("tool_use_id"):
                self._names[str(d["tool_use_id"])] = str(d.get("name") or "")
            elif t is EventType.TOOL_RESULT:
                d.setdefault("tool", self._names.get(str(d.get("tool_use_id")), "unknown"))
            elif t is EventType.PERMISSION:
                d.setdefault("tool", d.get("name") or "")
            elif t is EventType.STOP:
                self._outcome = str(d.get("subtype") or "unknown")
                self._cost += float(d.get("total_cost_usd") or 0.0)
                self._turns += int(d.get("num_turns") or 0)
                d = {**d, "detail": d.get("reason"), "reason": d.get("subtype")}
            elif t is EventType.SESSION_END:
                d.setdefault("outcome", self._outcome or "unknown")
                d.setdefault("total_cost_usd", self._cost)
                d.setdefault("num_turns", self._turns)
            elif t is EventType.ERROR:
                d.setdefault("error_class", str(d.get("error") or d.get("kind") or "unknown"))
            elif t is EventType.COMPACTION and (d.get("delegated_to") or d.get("failed")):
                return event
        try:
            self.sink.emit(replace(event, data=d))
        except Exception:  # noqa: BLE001
            pass
        return event


# ============================================================================= sub-agents


class _ChargingClient:
    """Charge the sub-agent ledger after *every* response (including retries the loop makes)."""

    def __init__(self, inner: ModelClient, task: SubagentTask, model_id: str) -> None:
        self.inner = inner
        self.task = task
        self.model_id = model_id

    def create(self, request: ModelRequest) -> ModelResponse:
        response = self.inner.create(request)
        self.task.charge(response.usage, response.model or self.model_id)
        return response


class ToolCallCap:
    """``ToolExecutor`` wrapper that enforces a tool-call budget (calls beyond it return an error result)."""

    def __init__(self, inner: ToolExecutor, limit: int | None) -> None:
        self.inner = inner
        self.limit = limit
        self.count = 0
        self._lock = threading.Lock()

    def specs(self) -> list[ToolSpec]:
        return self.inner.specs()

    def execute(self, call: ToolCall, ctx: RunContext) -> ToolOutcome:
        with self._lock:
            over = self.limit is not None and self.count >= self.limit
            if not over:
                self.count += 1
        if over:
            return ToolOutcome(f"Error: tool-call budget ({self.limit}) exhausted; finish with your report now.", is_error=True,
                               metadata={"budget_exhausted": True})
        return self.inner.execute(call, ctx)


class ExcludingExecutor:
    """A view of an executor without some tools (e.g. the memory and spawn tools for sub-agents)."""

    def __init__(self, inner: ToolExecutor, exclude: Iterable[str]) -> None:
        self.inner = inner
        self.exclude = frozenset(exclude)

    def specs(self) -> list[ToolSpec]:
        return [s for s in self.inner.specs() if s.name not in self.exclude]

    def execute(self, call: ToolCall, ctx: RunContext) -> ToolOutcome:
        if call.name in self.exclude:
            return ToolOutcome(f"Error: tool {call.name!r} is not available here", is_error=True)
        return self.inner.execute(call, ctx)


_STATUS_BY_SUBTYPE = {
    ResultSubtype.ERROR_MAX_TURNS: SubagentStatus.ERROR_MAX_TURNS,
    ResultSubtype.ERROR_MAX_BUDGET_USD: SubagentStatus.ERROR_MAX_BUDGET,
    ResultSubtype.ERROR_MAX_TOKEN_BUDGET: SubagentStatus.ERROR_MAX_BUDGET,
    ResultSubtype.ERROR_MAX_WALL_CLOCK: SubagentStatus.ERROR_TIMEOUT,
    ResultSubtype.INTERRUPTED: SubagentStatus.CANCELLED,
}

REPAIR_PROMPT = "Your final message was not a valid report ({error}). Reply with ONLY the report JSON object."


class LoopAgentRunner:
    """:class:`harness_multiagent.AgentRunner` on top of :class:`harness_core.AgentLoop`.

    Per task it builds a fresh loop over an isolated session (the brief is the only user message)
    and honours the runner obligations from ``docs/multiagent/runner-and-results.md``:

    * requests carry ``task.system_prompt``, ``task.initial_messages()``, ``task.tools.specs()`` and
      ``task.request_extra(model_id)`` (model id resolved from the task's model *role*);
    * tools run only through ``task.tools.execute`` (wrapped by a tool-call cap);
    * ``task.charge(usage, model)`` after every response; the run halts as soon as
      ``task.stop_reason()`` is set (and before a model call once the token is cancelled);
    * ``max_turns`` is a loop budget (no wrap-up turn), ``max_tool_calls`` is the cap;
    * the final message goes through ``parse_report`` with up to ``task.max_structured_retries``
      repair prompts; only the report is returned.

    :param model: one client for every role (the id travels in ``extra["model"]``) or a factory ``model_id -> client``.
    :param model_for_role: ``role -> model id`` (mapping or callable); ids come from configuration only.
    :param policy: permission policy for the sub-agent's tool calls (default: allow, because the
        scoped executor already filters tools; pass the harness ``PolicyEngine`` policy to apply the
        permission pipeline too).
    """

    def __init__(
        self,
        model: ModelClient | Callable[[str], ModelClient],
        *,
        model_for_role: Mapping[str, str] | Callable[[str], str],
        policy: PermissionPolicy | None = None,
        approver: Approver | None = None,
        hooks_factory: Callable[[SubagentTask, RunContext], HookRegistry] | None = None,
        tracer: Tracer | None = None,
        sinks: Sequence[EventSink] = (),
        cost_estimator: CostEstimator | None = None,
        loop_config: LoopConfig | None = None,
        session_factory: Callable[[SubagentTask, RunContext], Session] | None = None,
        on_session_start: Callable[[Session, RunContext], None] | None = None,
        on_finish: Callable[[Session, SubagentResult], None] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.model = model
        self.model_for_role = model_for_role
        self.policy = policy
        self.approver = approver
        self.hooks_factory = hooks_factory
        self.tracer = tracer
        self.sinks = list(sinks)
        self.cost_estimator = cost_estimator
        self.loop_config = loop_config or LoopConfig()
        self.session_factory = session_factory
        self.on_session_start = on_session_start
        self.on_finish = on_finish
        self.monotonic = monotonic

    # ------------------------------------------------------------------ resolution
    def _resolve(self, role: str) -> tuple[str, ModelClient]:
        m = self.model_for_role
        model_id = m(role) if callable(m) else m.get(role, "")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(f"no model configured for role {role!r}")
        if hasattr(self.model, "create"):
            return model_id, self.model  # type: ignore[return-value]
        return model_id, self.model(model_id)  # type: ignore[operator]

    # ------------------------------------------------------------------ AgentRunner
    def run(self, task: SubagentTask, ctx: RunContext) -> SubagentResult:
        started = self.monotonic()
        warnings: list[str] = []
        state: dict[str, Any] = {"stop": None, "model": ""}
        session: Session | None = None
        tools = ToolCallCap(task.tools, task.max_tool_calls)

        def finish(status: SubagentStatus, report: Any = None, error: str = "") -> SubagentResult:
            usage, turns = (session.total_usage()[0], session.total_usage()[2]) if session is not None else (Usage(), 0)
            res = SubagentResult(
                agent=task.agent, run_id=task.run_id, status=status, report=report, usage=usage, turns=turns,
                tool_calls=tools.count, wall_clock_s=self.monotonic() - started, model=state["model"], depth=ctx.depth,
                parent_run_id=task.parent_run_id, error=error, warnings=warnings,
            )
            if self.on_finish is not None and session is not None:
                try:
                    self.on_finish(session, res)
                except Exception:  # noqa: BLE001 - observers never fail the run
                    pass
            return res

        try:
            model_id, client = self._resolve(task.model_role)
        except Exception as exc:  # noqa: BLE001
            return finish(SubagentStatus.ERROR_DURING_EXECUTION, error=str(exc))
        state["model"] = model_id

        def before_model(inp: HookInput) -> HookResult | None:
            if task.token.cancelled:
                state["stop"] = status_for_reason(task.token.reason or "")
                return HookResult(continue_=False, stop_reason=f"subagent stopped: {state['stop'].value}")
            return None

        def after_model(inp: HookInput) -> HookResult | None:
            stop = task.stop_reason()
            if stop is not None:
                state["stop"] = stop
                return HookResult(continue_=False, stop_reason=f"subagent stopped: {stop.value}")
            return None

        try:
            session = self.session_factory(task, ctx) if self.session_factory else Session.in_memory(task.run_id)
            hooks = self.hooks_factory(task, ctx) if self.hooks_factory else HookRegistry()
            hooks.on(HookEvent.PRE_MODEL, before_model, name="subagent_cancel")
            hooks.on(HookEvent.POST_MODEL, after_model, name="subagent_budget")
            cfg = replace(self.loop_config, max_tokens=task.max_output_tokens, wrap_up_on_budget=False, model_label=model_id)
            loop = AgentLoop(
                _ChargingClient(client, task, model_id), tools, system=task.system_prompt, config=cfg,
                budgets=Budgets(max_turns=task.max_turns, scope="session"),
                assembler=PassthroughAssembler(max_tokens=task.max_output_tokens, extra=task.request_extra(model_id)),
                policy=self.policy or AllowAllPolicy(), approver=self.approver, hooks=hooks, session=session,
                sinks=self.sinks, tracer=self.tracer, cost_estimator=self.cost_estimator, cwd=task.workspace,
                agent_name=task.agent, depth=ctx.depth,
            )
            loop.ctx.tainted = ctx.tainted
            loop.ctx.metadata.update(ctx.metadata)
            loop.ctx.metadata["parent_session_id"] = ctx.session_id
            if self.on_session_start is not None:
                self.on_session_start(session, loop.ctx)
            initial = task.initial_messages()
            for m in initial[:-1]:  # context: fork - the parent's history precedes the brief
                session.append_message(m, kind="fork_history")
            result = loop.run(initial[-1])
            repairs = 0
            while True:
                if result.subtype is ResultSubtype.SUCCESS:
                    text = result.final_text
                    warnings.extend(w for w in scan_untrusted(text) if w not in warnings)
                    try:
                        report = parse_report(text)
                    except StructuredOutputError as exc:
                        if repairs >= task.max_structured_retries:
                            return finish(SubagentStatus.ERROR_MAX_STRUCTURED_OUTPUT_RETRIES, error=str(exc))
                        repairs += 1
                        result = loop.run(REPAIR_PROMPT.format(error=exc))
                        continue
                    return finish(SubagentStatus.SUCCESS, report=report)
                status = state["stop"] if result.subtype is ResultSubtype.HALTED_BY_HOOK and state["stop"] is not None else (
                    _STATUS_BY_SUBTYPE.get(result.subtype, SubagentStatus.ERROR_DURING_EXECUTION))
                return finish(status, error=result.error or result.reason or result.subtype.value)
        except Exception as exc:  # noqa: BLE001 - API errors are never delivered as a sub-agent's report
            return finish(SubagentStatus.ERROR_DURING_EXECUTION, error=f"{type(exc).__name__}: {exc}")


# ============================================================================= evals / RL


def _counts_as_compaction(d: Mapping[str, Any]) -> bool:
    if d.get("failed") or d.get("delegated_to"):
        return False
    return True


def trajectory_from_session(session: Session, result: RunResult | None = None, *, system: str = "", model: str = "",
                            metadata: Mapping[str, Any] | None = None) -> Trajectory:
    """Convert a harness session log into a :class:`harness_evals.Trajectory`.

    The loop's events are mapped onto the payloads :meth:`Trajectory.from_events` documents:
    message-carrying ``USER_MESSAGE`` / ``MODEL_RESPONSE`` events become ``{"message": ...}`` (tool
    results travel inside the user message, so per-call ``TOOL_RESULT`` events are not replayed a
    second time); discarded responses are not transcript turns but their usage still counts;
    ``COMPACTION`` events are counted once per real compaction.
    """
    converted: list[Event] = []
    for e in session.events():
        d = e.data or {}
        if e.type is EventType.USER_MESSAGE and "message" in d:
            converted.append(Event(e.session_id, e.type, {"message": d["message"]}, seq=e.seq, ts=e.ts, id=e.id))
        elif e.type is EventType.MODEL_RESPONSE and "message" in d:
            converted.append(Event(e.session_id, e.type, {"message": d["message"], "usage": d.get("usage") or {},
                                                          "model": d.get("model") or "", "stop_reason": d.get("stop_reason") or ""},
                                   seq=e.seq, ts=e.ts, id=e.id))
        elif e.type is EventType.COMPACTION and _counts_as_compaction(d):
            converted.append(Event(e.session_id, e.type, {"kind": d.get("kind") or d.get("reason") or "compaction"}, seq=e.seq, ts=e.ts, id=e.id))
    usage, cost, _calls = session.total_usage()
    kw: dict[str, Any] = {"usage": usage, "cost_usd": cost}
    if system:
        kw["system"] = system
    if model:
        kw["model"] = model
    meta = dict(metadata or {})
    if result is not None:
        kw["final_text"] = result.final_text
        if result.stop_reason is not None:
            kw["stop_reason"] = result.stop_reason.value
        if result.is_error:
            kw["error"] = result.error or result.reason or result.subtype.value
        meta.setdefault("result_subtype", result.subtype.value)
        meta.setdefault("num_turns", result.num_turns)
    kw["metadata"] = meta
    return Trajectory.from_events(converted, **kw)


_INFRA_ERRORS = ("TransientModelError", "RateLimitError", "SandboxError", "SandboxCrashedError", "SandboxUnavailableError")


def _safe_session_id(raw: str) -> str:
    import re

    sid = re.sub(r"[^A-Za-z0-9_.\-]", "_", raw or "").strip("._-") or "trial"
    while ".." in sid:
        sid = sid.replace("..", ".")
    return sid[:120]


class LoopAgentUnderTest:
    """:class:`harness_evals.AgentUnderTest` (and RL black-box agent) over a freshly built harness per trial.

    Per trial it: uses ``workspace`` as the harness workspace; ``ctx.metadata["memory_dir"]`` as the
    only memory root (memory is disabled in ``frozen`` memory mode, which must not change);
    ``ctx.metadata["env"]`` as the environment for shell commands; ``ctx.metadata["seed"]`` for the
    loop's RNG; ``ctx.metadata["cancel_event"]`` to interrupt the loop; and, in RL,
    ``ctx.metadata["tools"]`` (the rollout-scoped backend) as the *only* tool executor. Task limits
    map to loop budgets (``max_turns``, wall clock) and a tool-call cap. Provider failures after
    retries are raised as :class:`harness_evals.InfraError`.

    :param config: base harness configuration (workspace and memory paths are overridden per trial).
    :param model_client: shared client, or ``model_client_factory(task, ctx)`` for a fresh one per trial.
    """

    name = "harness-loop"

    def __init__(
        self,
        config: "HarnessConfig",
        *,
        model_client: ModelClient | None = None,
        model_client_factory: Callable[[AgentTask, RunContext], ModelClient] | None = None,
        approver: Approver | None = None,
        interactive: bool = False,
        build_options: Mapping[str, Any] | None = None,
        on_harness: Callable[[Any], None] | None = None,
    ) -> None:
        self.config = config
        self.model_client = model_client
        self.model_client_factory = model_client_factory
        self.approver = approver
        self.interactive = interactive
        self.build_options = dict(build_options or {})
        self.on_harness = on_harness

    def trial_config(self, task: AgentTask, workspace: Path, ctx: RunContext) -> "HarnessConfig":
        cfg = self.config
        md = ctx.metadata
        env = md.get("env") or {}
        state = cfg.state_dir
        if state is None:
            home = env.get("HOME")
            state = str(Path(home) / ".harness-state") if home else tempfile.mkdtemp(prefix="harness-state-")
        features = cfg.features
        if md.get("memory_mode") == "frozen":
            features = replace(features, memory=False)
        limits = task.limits
        budgets = cfg.budgets
        if limits.max_turns is not None:
            budgets = replace(budgets, max_turns=min(limits.max_turns, budgets.max_turns or limits.max_turns))
        if limits.timeout_s:
            budgets = replace(budgets, max_wall_clock_s=min(limits.timeout_s, budgets.max_wall_clock_s or limits.timeout_s))
        return replace(
            cfg, workspace=str(workspace), base_dir=str(md.get("memory_dir") or cfg.base_dir), state_dir=state,
            permissions=replace(cfg.permissions, interactive=self.interactive), features=features, budgets=budgets,
        )

    def run(self, task: AgentTask, workspace: Path, ctx: RunContext) -> Trajectory:
        from .build import build_harness

        cfg = self.trial_config(task, Path(workspace), ctx)
        md = ctx.metadata
        client = self.model_client_factory(task, ctx) if self.model_client_factory else self.model_client
        extra_md = {k: md[k] for k in ("rollout_id", "trial_id", "task_id", "run_id", "profile") if md.get(k) is not None}
        harness = build_harness(
            cfg, model_client=client, approver=self.approver, session_id=_safe_session_id(ctx.session_id),
            tool_executor=md.get("tools"), bash_env=md.get("env"), rng_seed=md.get("seed"),
            max_tool_calls=task.limits.max_tool_calls, extra_metadata=extra_md, **self.build_options,
        )
        if self.on_harness is not None:
            self.on_harness(harness)
        cancel = md.get("cancel_event")
        done = threading.Event()

        def watch() -> None:
            while not done.is_set():
                if cancel.wait(0.05):
                    harness.interrupt()
                    return

        watcher = threading.Thread(target=watch, name=f"cancel-{ctx.session_id}", daemon=True) if cancel is not None else None
        if watcher is not None:
            watcher.start()
        try:
            result = harness.run(task.instruction)
        finally:
            done.set()
            harness.end_session("trial finished")
        if result.subtype is ResultSubtype.ERROR_DURING_EXECUTION and (result.error or "").startswith(_INFRA_ERRORS):
            raise InfraError(result.error or "provider failure")
        return trajectory_from_session(harness.session, result, system=harness.loop.system, model=cfg.model,
                                       metadata={"harness_session_id": harness.session.id})


__all__ = [
    "OVERFLOW_KEY",
    "ContextCompactorAdapter",
    "ExcludingExecutor",
    "LoopAgentRunner",
    "LoopAgentUnderTest",
    "OpsEventAdapter",
    "OverflowGuardClient",
    "SessionBoundAssembler",
    "SessionEventSink",
    "StepIndexSink",
    "ToolCallCap",
    "trajectory_from_session",
]
