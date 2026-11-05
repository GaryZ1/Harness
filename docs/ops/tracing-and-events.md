# Tracing, content policy, exporters and event fan-out

## Purpose

This layer records every run as an OpenTelemetry GenAI span tree carrying token, cache, cost and error attributes, so that any
OTel backend can render, query and grade it. It also gives the session's events a telemetry tap that never leaks content.

## Research basis

* **Span model** (reference §10.3, checklist item 1). The tree is `invoke_agent` (root per run; named `invoke_agent {gen_ai.agent.name}`) → `chat` per model call → `execute_tool` per tool call. On top of that come custom spans for compaction, subagent spawn and approval waits, and `harness.*` attributes for everything the spec does not define. Kind is CLIENT for `chat` and INTERNAL for in-process agents and tools.
* **The spec is still in Development** (v1.41 vs v1.42; cache-token names unverified). Keep "a thin attribute-mapping layer so a spec rename is a one-file change" → `SemConv`.
* **Propagation**: MCP 2026-07-28 (SEP-414) carries `traceparent`/`tracestate` in `_meta`. Link child agents to the parent tool-call span.
* **Content capture is opt-in** (checklist item 2). The default is structure only: tool names, argument hashes, token counts, error classes and timings. Secrets are redacted before export.
* **Pluggable exporters** (checklist item 5). The default is an OTel-native backend (Langfuse or Phoenix self-hosted), with Braintrust and LangSmith reachable via OTLP.

## API

### Tracer

```python
from harness_ops import OpsTracer, JsonlSpanExporter, Sampler, SemConv, ContentPolicy, A

tracer = OpsTracer(
    [JsonlSpanExporter("telemetry/spans.jsonl")],
    policy=ContentPolicy.from_env(),           # HARNESS_CAPTURE_CONTENT / OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT
    semconv=SemConv({A.USAGE_CACHE_READ_INPUT_TOKENS: "gen_ai.usage.cache_read_input_tokens"}),  # if the spec renames
    sampler=Sampler(1.0),
    resource={"service.name": "my-harness"},
)
with tracer.span("invoke_agent main", **{A.AGENT_NAME: "main"}) as root:   # satisfies harness_core.types.Tracer
    with tracer.span("chat test-model", span_kind=SpanKind.CLIENT) as chat:
        chat.set_attribute(A.USAGE_INPUT_TOKENS, 1200)
```

* `span(name, *, span_kind, parent, links, **attributes)` makes the span current through a `contextvars` slot, which is correct across threads and asyncio tasks. On exit it ends the span. An escaping exception sets `error.type` and ERROR status and is re-raised. `parent=` accepts a `SpanContext`, a `Span`, a carrier dict (`{"traceparent": ...}`), or `False` to force a new root.
* `start_span(...)` plus `Span.end()` give manual control. `use_span(span)` adopts a span in a worker thread.
* `inject(carrier)` / `extract(carrier)` / `parse_traceparent(header)` implement W3C trace context. Version `ff`, all-zero ids, and extra fields on v00 are rejected, while future versions may append fields.

### GenAI instrumentation

`AgentInstrumentation(tracer, metrics=, prices=, cache=, provider=, max_budget_usd=)` builds the span tree and does the accounting in one place:

| Call | Span | Attributes (pre-mapping names) |
|---|---|---|
| `invoke_agent(name, session_id=, agent_id=, model=, effort=, version=, parent=, depth=)` → `AgentRun` | `invoke_agent {name}` (INTERNAL) | `gen_ai.operation.name=invoke_agent`, `gen_ai.agent.name/id`, `gen_ai.conversation.id`, `gen_ai.request.model`, `harness.effort`, `harness.version`, `harness.prompts_tools_sha`; on exit `harness.termination`, `harness.total_cost_usd`, `harness.num_turns` |
| `chat(run, model, request=)` → `ChatCall.record(response, ttft_ms=, retry_count=, fallback_from=, request_id=, iterations=, cache_write_ttl=, replayed=)` | `chat {model}` (CLIENT) | `gen_ai.usage.input_tokens/output_tokens`, `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_creation.input_tokens`, `harness.usage.total_input_tokens`, `gen_ai.response.finish_reasons`, `gen_ai.response.id`, `harness.ttft_ms`, `harness.latency_ms`, `harness.retry_count`, `harness.fallback_from`, `harness.cost_usd`, `harness.cache.observation/hit_ratio/expected_rebuild`; content: `gen_ai.system_instructions`, `gen_ai.input.messages`, `gen_ai.output.messages` |
| `execute_tool(run, call, idempotency_key=, sandbox_id=)` → `ToolCallSpan.record(outcome, result_tokens=, truncated=)` | `execute_tool {tool}` | `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.call.arguments` (hashed = args hash), `harness.tool.is_error`, `error.type`, `harness.tool.result_tokens`, `harness.tool.truncated`, `harness.duration_ms`, `harness.idempotency_key`, `harness.sandbox_id` |
| `compaction(run, tokens_before=, tokens_after=)` | `compaction` | declares an expected cache rebuild for the session's next call |
| `subagent_spawn(run, child_agent_name=, parent_tool_call_id=)` → carrier dict | `subagent_spawn` | pass the carrier as the child's `invoke_agent(parent=...)`; it also works across processes |
| `approval_wait(run, tool=)` → `ApprovalWait.decide(decision)` | `approval_wait` | `harness.approval.decision`, wait duration; feeds permission metrics |

`ToolCallSpan.mcp_meta(meta)` writes `traceparent` into an MCP request's `_meta`. With `NullTracer` the metrics, cost ledger
and cache verdicts still work. Pass `replayed=True` for responses that came back from the durable journal: the cost still
counts toward the run's budget, but it is not reported to metrics a second time.

## Content policy

`ContentPolicy(capture_content=False, max_content_chars=16_384, max_attribute_chars=4_096, hash_key=None)`:

* Content attributes (`gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`, `harness.content`) are replaced by `<key>.sha256` and `<key>.length` unless capture is on.
* Event payload keys (`text`, `content`, `prompt`, `messages`, `input`, `output`, `arguments`, `result`, `system`, `completion`, `query`, `body`) become `{"sha256", "chars"}`.
* Every other string, and captured content too, passes `SecretRedactor` and is then truncated.
* Exception messages are recorded only when capture is on, because they often quote content. The exception class is always recorded.
* `hash_key` switches digests to HMAC-SHA256, which defeats dictionary attacks on short content.

## Exporters

* `JsonlSpanExporter(path, fsync=False, max_bytes=None, backups=3)` appends one span per line under a cross-process `file_lock`, with optional size rotation. `read_spans(path)` skips a torn last line, and `span_tree(spans)` groups spans by parent.
* `InMemorySpanExporter` serves tests and in-process sampling.
* `OTelSpanExporter(sdk_exporter)` converts finished spans, **keeping our ids**, into SDK `ReadableSpan`s for any OTel exporter. `otlp_exporter("http/protobuf" | "grpc")` builds one around the OTLP exporter. Both need `opentelemetry-sdk`.
* `OTelTracer(otel_tracer=None)` implements the `Tracer` protocol directly on the OTel API, with the same policy and semconv.
* When the dependency is missing, construction raises `OTelUnavailableError` (an `ImportError`) whose message gives the install command. `otel_available()` checks for it.

## Event fan-out

`FanoutEventSink(exporters, policy=, clock=)` implements `EventSink.emit`. It fills `seq` (per session, monotonic in this
process) and `ts` when they are missing and returns the **caller's** event unchanged, so the durable log keeps full fidelity.
Exporters get a sanitised copy. Any object with `export_event(event)` is an exporter: `HarnessMetrics`, `SLOTracker`,
`ReviewSampler`, `JsonlEventExporter`, `InMemoryEventExporter`, and `SpanEventExporter`, which attaches the event type and
scalar fields to the current span.

Payload keys the built-in consumers read:

| Event type | Keys |
|---|---|
| `MODEL_RESPONSE` | `model`, `usage{input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens}`, `stop_reason`, `latency_ms`, `ttft_ms`, `cost_usd` (else priced from the table), `cache_observation`, `cache_reason`, `retry_count` |
| `TOOL_RESULT` | `tool` (or `name`), `is_error`, `duration_ms` |
| `PERMISSION` | `behavior` (`allow` / `deny` / `ask`), `tool` |
| `STOP` | `reason` (or `stop_reason`) |
| `COMPACTION` | (counted) |
| `ERROR` | `error_class` |
| `SESSION_END` | `outcome` (or `termination`: `success` / `error_*`), `total_cost_usd`, `num_turns`, `duration_ms`, `trace_id`, `flags`, `user_feedback` |

## Defaults

| Setting | Default | Anchor |
|---|---|---|
| Content capture | off | vendor default (OTel GenAI: content opt-in) |
| Capture env vars | `HARNESS_CAPTURE_CONTENT`, `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | the OTel Python GenAI instrumentation variable plus ours |
| Max captured content / attribute | 16,384 / 4,096 chars | our choice |
| Max attributes / events per span | 128 / 128 | vendor default (OTel SDK span limits) |
| Head sampling | 100%, parent-based, trace-id ratio | vendor semantics (OTel `ParentBased(TraceIdRatioBased)`); ratio is our choice |
| `invoke_agent` span kind | INTERNAL | spec allows INTERNAL for in-process agents (CLIENT for remote) |
| Exporter failures | counted (`export_errors`, `FanoutEventSink.errors`), logged by class name | our choice |

## Invariants

* Spans nest by context, never by call order across threads. A new thread starts without a parent unless it calls `use_span`.
* No exporter ever sees an attribute that bypassed `ContentPolicy` and `SemConv`.
* An unsampled trace records nothing but still propagates a valid `traceparent` with flag `00`.
* `emit` never raises because an exporter failed.

## Failure modes and runbook

* **No spans in the backend.** Check `tracer.export_errors`. The OTLP exporter reads its endpoint and headers at construction, so restart after changing them (the Claude Code docs say the same about headers).
* **Content needed for a debugging session.** Enable capture for that environment or tenant only (`ContentPolicy(capture_content=True)`). Secrets are still redacted. Do not enable it globally in multi-tenant production.
* **The spec renamed an attribute.** Add one `SemConv` override and redeploy. Dashboards built on the wire name keep working.
* **Disk growth from JSONL.** Set `max_bytes` and `backups`, or ship the file to the backend and rotate it externally.
