"""GenAI instrumentation: the ``invoke_agent`` -> ``chat`` / ``execute_tool`` span tree, with accounting.

Research basis (reference §10.3 minimum span schema; checklist items 1-4, 11, 17):

* ``invoke_agent`` root per run: ``gen_ai.agent.name/id``, ``gen_ai.conversation.id``, harness
  version + prompts/tools SHA, model, effort; on exit the termination subtype
  (``success`` / ``error_max_turns`` / ``error_max_budget_usd`` / ``error_during_execution``),
  total cost and turn count.
* ``chat`` per model call: input / output / cache-read / cache-write tokens, TTFT, latency, stop
  reason, request id, retry count, fallback origin, **computed cost**, and the cache verdict
  (hit / expected rebuild / unexpected miss).
* ``execute_tool`` per tool call: name, call id, **argument hash** (content opt-in), result
  size, truncation flag, error class, duration, idempotency key, sandbox id.
* Custom spans: ``compaction`` (declares an expected cache rebuild), ``subagent_spawn`` (hands the
  child a ``traceparent`` carrier so its root nests under the spawning tool call, also across
  processes or MCP ``_meta``), ``approval_wait`` (decision latency; feeds permission metrics).

Works with any ``Tracer``: with :class:`~harness_ops.tracing.OpsTracer` or
:class:`~harness_ops.otel.OTelTracer` you get span kinds and parents; with ``NullTracer`` the
metrics, cost ledger and cache verdicts still work.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, MutableMapping, Sequence

from harness_core.types import (
    ModelRequest,
    ModelResponse,
    NullTracer,
    PermissionDecision,
    ToolCall,
    ToolOutcome,
    Tracer,
    Usage,
    messages_to_api,
)

from .cache import CacheMonitor, CacheVerdict
from .deploy import HarnessVersion
from .errors import PricingError
from .metrics import HarnessMetrics
from .pricing import CostBreakdown, CostLedger, PriceTable
from .semconv import A, HarnessOp, Op, agent_span_name, chat_span_name, tool_span_name
from .tracing import SpanContext, SpanKind, current_context, inject, set_span_attributes


@dataclass
class AgentRun:
    """Per-run accumulator yielded by :meth:`AgentInstrumentation.invoke_agent`."""

    span: Any
    session_id: str
    agent_name: str
    model: str | None
    effort: str | None
    ledger: CostLedger
    depth: int = 0
    num_turns: int = 0
    termination: str | None = None
    usage: Usage = field(default_factory=Usage)

    def set_termination(self, subtype: str) -> None:
        """Agent-SDK-style result subtype: ``success``, ``error_max_turns``, ``error_max_budget_usd``..."""
        self.termination = subtype

    @property
    def total_cost_usd(self) -> float:
        return self.ledger.total_usd


@dataclass(frozen=True)
class ChatRecord:
    cost: CostBreakdown | None
    cache: CacheVerdict | None
    latency_ms: float


class ChatCall:
    def __init__(self, inst: "AgentInstrumentation", run: AgentRun | None, span: Any, model: str, started: float) -> None:
        self._inst = inst
        self.run = run
        self.span = span
        self.model = model
        self._started = started
        self.record_result: ChatRecord | None = None

    def record(
        self,
        response: ModelResponse,
        *,
        ttft_ms: float | None = None,
        retry_count: int = 0,
        fallback_from: str | None = None,
        request_id: str | None = None,
        iterations: Sequence[Usage] | None = None,
        cache_write_ttl: str = "5m",
        effort: str | None = None,
        replayed: bool = False,
    ) -> ChatRecord:
        """Record the response on the ``chat`` span, metrics, cost ledger and cache tracker.

        ``replayed=True`` (the response came from the durable journal, see
        :meth:`~harness_ops.durable.DurableRun.was_replayed`): the cost still counts toward this
        run's budget ledger, but metrics counters and the cache tracker are skipped so a resumed
        run never double-counts spend that was already reported.
        """
        inst = self._inst
        latency_ms = (inst._monotonic() - self._started) * 1000.0
        usage = response.usage
        billed = usage
        if iterations:
            billed = Usage()
            for it in iterations:
                billed = billed + it
        model = response.model or self.model
        cost: CostBreakdown | None = None
        if inst.prices is not None:
            try:
                cost = inst.prices.cost(model, billed, cache_write_ttl=cache_write_ttl)
            except PricingError:
                cost = None
        verdict: CacheVerdict | None = None
        if inst.cache is not None and self.run is not None and not replayed:
            tracker = inst.cache.tracker(self.run.session_id)
            if fallback_from:
                tracker.expect_rebuild("fallback")
            verdict = tracker.observe(usage, model=model, effort=effort or self.run.effort)
        attrs: dict[str, Any] = {
            A.RESPONSE_MODEL: response.model or None,
            A.RESPONSE_ID: request_id or response.id or None,
            A.RESPONSE_FINISH_REASONS: [response.stop_reason.value],
            A.STOP_REASON: response.stop_reason.value,
            A.USAGE_INPUT_TOKENS: billed.input_tokens,
            A.USAGE_OUTPUT_TOKENS: billed.output_tokens,
            A.USAGE_CACHE_READ_INPUT_TOKENS: billed.cache_read_input_tokens,
            A.USAGE_CACHE_CREATION_INPUT_TOKENS: billed.cache_creation_input_tokens,
            A.TOTAL_INPUT_TOKENS: billed.total_input,
            A.LATENCY_MS: round(latency_ms, 3),
            A.TTFT_MS: ttft_ms,
            A.RETRY_COUNT: retry_count,
            A.FALLBACK_FROM: fallback_from,
            A.COST_USD: cost.total_usd if cost is not None and cost.priced else None,
            A.OUTPUT_MESSAGES: [b.to_api() for b in response.content],
            A.STEP_REPLAYED: True if replayed else None,
        }
        if verdict is not None:
            attrs[A.CACHE_OBSERVATION] = verdict.observation.value
            attrs[A.CACHE_HIT_RATIO] = verdict.hit_ratio
            attrs[A.CACHE_EXPECTED_REBUILD] = verdict.reason or None
        set_span_attributes(self.span, attrs)
        if inst.metrics is not None and not replayed:
            inst.metrics.record_model_call(
                model,
                billed,
                latency_ms=latency_ms,
                ttft_ms=ttft_ms,
                stop_reason=response.stop_reason.value,
                cost_usd=cost.total_usd if cost is not None and cost.priced else None,
                cache_observation=verdict.observation if verdict else None,
                cache_reason=verdict.reason if verdict else "",
                retry_count=retry_count,
            )
        if self.run is not None:
            self.run.num_turns += 1
            self.run.usage = self.run.usage + billed
            if cost is not None:
                self.run.ledger.add(cost, scope=f"depth{self.run.depth}")
        self.record_result = ChatRecord(cost, verdict, latency_ms)
        return self.record_result


class ToolCallSpan:
    def __init__(self, inst: "AgentInstrumentation", call: ToolCall, span: Any, started: float) -> None:
        self._inst = inst
        self.call = call
        self.span = span
        self._started = started
        self.recorded = False

    def mcp_meta(self, meta: MutableMapping[str, Any] | None = None) -> MutableMapping[str, Any]:
        """``traceparent``/``tracestate`` for an MCP request's ``_meta`` (SEP-414)."""
        meta = {} if meta is None else meta
        ctx = getattr(self.span, "context", None)
        return inject(meta, ctx if isinstance(ctx, SpanContext) else None)

    def record(self, outcome: ToolOutcome, *, result_tokens: int | None = None, truncated: bool | None = None) -> None:
        duration_ms = (self._inst._monotonic() - self._started) * 1000.0
        md = outcome.metadata or {}
        error_class = (md.get("error_class") or "tool_error") if outcome.is_error else None
        set_span_attributes(
            self.span,
            {
                A.TOOL_IS_ERROR: outcome.is_error,
                A.ERROR_TYPE: error_class,
                A.TOOL_CALL_RESULT: outcome.content,
                A.TOOL_RESULT_TOKENS: result_tokens if result_tokens is not None else md.get("result_tokens"),
                A.TOOL_TRUNCATED: truncated if truncated is not None else md.get("truncated"),
                A.DURATION_MS: round(duration_ms, 3),
                A.IDEMPOTENCY_KEY: md.get("idempotency_key"),
            },
        )
        if self._inst.metrics is not None:
            self._inst.metrics.record_tool_call(self.call.name, is_error=outcome.is_error, duration_ms=duration_ms)
        self.recorded = True


class ApprovalWait:
    def __init__(self, inst: "AgentInstrumentation", span: Any, tool: str, started: float) -> None:
        self._inst = inst
        self.span = span
        self.tool = tool
        self._started = started

    def decide(self, decision: PermissionDecision | str) -> None:
        behavior = decision.behavior.value if isinstance(decision, PermissionDecision) else str(decision)
        set_span_attributes(self.span, {A.APPROVAL_DECISION: behavior, A.DURATION_MS: round((self._inst._monotonic() - self._started) * 1000.0, 3)})
        if self._inst.metrics is not None:
            self._inst.metrics.record_permission(behavior, self.tool)


class AgentInstrumentation:
    """One object per process; thread-safe as long as each run's spans are used from its own thread/task."""

    def __init__(
        self,
        tracer: Tracer | None = None,
        *,
        metrics: HarnessMetrics | None = None,
        prices: PriceTable | None = None,
        cache: CacheMonitor | None = None,
        provider: str | None = None,
        max_budget_usd: float | None = None,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.tracer: Tracer = tracer or NullTracer()
        self.metrics = metrics
        self.prices = prices if prices is not None else (metrics.prices if metrics is not None else None)
        self.cache = cache if cache is not None else CacheMonitor()
        self.provider = provider
        self.max_budget_usd = max_budget_usd
        self._monotonic = monotonic

    def _span(self, name: str, kind: SpanKind, attrs: Mapping[str, Any], parent: Any = None) -> Any:
        clean = {k: v for k, v in attrs.items() if v is not None}
        if getattr(self.tracer, "supports_span_kind", False):
            return self.tracer.span(name, span_kind=kind, parent=parent, **clean)  # type: ignore[call-arg]
        return self.tracer.span(name, **clean)

    @contextmanager
    def invoke_agent(
        self,
        agent_name: str,
        *,
        session_id: str,
        agent_id: str | None = None,
        model: str | None = None,
        effort: str | None = None,
        version: HarnessVersion | None = None,
        parent: Any = None,
        depth: int = 0,
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[AgentRun]:
        """Root span of one run. ``parent`` may be a ``traceparent`` carrier from :meth:`subagent_spawn`."""
        attrs: dict[str, Any] = {
            A.OPERATION_NAME: Op.INVOKE_AGENT.value,
            A.PROVIDER_NAME: self.provider,
            A.AGENT_NAME: agent_name,
            A.AGENT_ID: agent_id,
            A.CONVERSATION_ID: session_id,
            A.SESSION_ID: session_id,
            A.REQUEST_MODEL: model,
            A.EFFORT: effort,
            A.SUBAGENT_DEPTH: depth or None,
            **(version.span_attributes() if version else {}),
            **dict(attributes or {}),
        }
        with self._span(agent_span_name(agent_name), SpanKind.INTERNAL, attrs, parent) as span:
            run = AgentRun(span, session_id, agent_name, model, effort, CostLedger(self.max_budget_usd), depth)
            try:
                yield run
            except BaseException:
                if run.termination is None:
                    run.termination = "error_during_execution"
                raise
            finally:
                termination = run.termination or "success"
                set_span_attributes(span, {A.TERMINATION: termination, A.TOTAL_COST_USD: run.total_cost_usd, A.NUM_TURNS: run.num_turns})

    @contextmanager
    def chat(self, run: AgentRun | None, model: str, *, request: ModelRequest | None = None) -> Iterator[ChatCall]:
        attrs: dict[str, Any] = {A.OPERATION_NAME: Op.CHAT.value, A.PROVIDER_NAME: self.provider, A.REQUEST_MODEL: model}
        if request is not None:
            attrs[A.REQUEST_MAX_TOKENS] = request.max_tokens
            attrs[A.SYSTEM_INSTRUCTIONS] = request.system
            attrs[A.INPUT_MESSAGES] = messages_to_api(request.messages)
        started = self._monotonic()
        with self._span(chat_span_name(model), SpanKind.CLIENT, attrs) as span:
            yield ChatCall(self, run, span, model, started)

    @contextmanager
    def execute_tool(self, run: AgentRun | None, call: ToolCall, *, idempotency_key: str | None = None, sandbox_id: str | None = None, tool_type: str = "function") -> Iterator[ToolCallSpan]:
        attrs = {
            A.OPERATION_NAME: Op.EXECUTE_TOOL.value,
            A.TOOL_NAME: call.name,
            A.TOOL_CALL_ID: call.id,
            A.TOOL_TYPE: tool_type,
            A.TOOL_CALL_ARGUMENTS: call.input,  # content: hashed unless capture is on (= args_hash)
            A.IDEMPOTENCY_KEY: idempotency_key,
            A.SANDBOX_ID: sandbox_id,
        }
        started = self._monotonic()
        with self._span(tool_span_name(call.name), SpanKind.INTERNAL, attrs) as span:
            handle = ToolCallSpan(self, call, span, started)
            try:
                yield handle
            except BaseException as exc:
                if not handle.recorded and self.metrics is not None:
                    self.metrics.record_tool_call(call.name, is_error=True, duration_ms=(self._monotonic() - started) * 1000.0)
                    handle.recorded = True
                set_span_attributes(span, {A.TOOL_IS_ERROR: True, A.ERROR_TYPE: type(exc).__qualname__})
                raise

    @contextmanager
    def compaction(self, run: AgentRun | None, *, tokens_before: int | None = None, tokens_after: int | None = None, reason: str = "compaction") -> Iterator[Any]:
        """Wrap a compaction. Declares an *expected* cache rebuild for the next call of this session."""
        attrs = {A.OPERATION: HarnessOp.COMPACTION.value, A.COMPACTION_TOKENS_BEFORE: tokens_before, A.COMPACTION_TOKENS_AFTER: tokens_after}
        with self._span(HarnessOp.COMPACTION.value, SpanKind.INTERNAL, attrs) as span:
            yield span
        if run is not None and self.cache is not None:
            self.cache.tracker(run.session_id).expect_rebuild(reason)
        if self.metrics is not None:
            self.metrics.record_compaction()

    @contextmanager
    def subagent_spawn(self, run: AgentRun | None, *, child_agent_name: str, parent_tool_call_id: str | None = None) -> Iterator[dict[str, Any]]:
        """Yields a carrier (``traceparent``) to pass to the child's :meth:`invoke_agent` ``parent=``."""
        attrs = {A.OPERATION: HarnessOp.SUBAGENT_SPAWN.value, A.AGENT_NAME: child_agent_name, A.SUBAGENT_PARENT_TOOL_CALL_ID: parent_tool_call_id}
        with self._span(HarnessOp.SUBAGENT_SPAWN.value, SpanKind.INTERNAL, attrs) as span:
            ctx = getattr(span, "context", None)
            carrier: dict[str, Any] = {}
            inject(carrier, ctx if isinstance(ctx, SpanContext) else current_context())
            yield carrier

    @contextmanager
    def approval_wait(self, run: AgentRun | None, *, tool: str) -> Iterator[ApprovalWait]:
        attrs = {A.OPERATION: HarnessOp.APPROVAL_WAIT.value, A.APPROVAL_TOOL: tool}
        started = self._monotonic()
        with self._span(HarnessOp.APPROVAL_WAIT.value, SpanKind.INTERNAL, attrs) as span:
            yield ApprovalWait(self, span, tool, started)
