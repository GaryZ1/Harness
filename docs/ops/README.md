# Observability and production operations (`harness_ops`)

This is a production Python implementation of the observability and operations design in
[`reports/SOTA agentic harness reference.md` §10](../../reports/SOTA%20agentic%20harness%20reference.md),
the build plan sections "Evaluation and observability" and "Operations: durable steps, idempotent side effects, rainbow deploys"
([`reports/SOTA agentic harness build plan.md`](../../reports/SOTA%20agentic%20harness%20build%20plan.md)), and
[`research_notes/.../evals_observability_rl.md`](../../research_notes/SOTA%20agentic%20harness%20build%20plan/evals_observability_rl.md).
It needs only the standard library and PyYAML. OpenTelemetry is optional.

| Component | What it does | Modules | Doc |
|---|---|---|---|
| **Tracing** | `Tracer` protocol implementation with nested spans, W3C `traceparent`, the OTel GenAI span tree (`invoke_agent` → `chat` / `execute_tool`), and a semconv rename layer | `tracing.py`, `semconv.py`, `instrument.py` | [tracing-and-events.md](tracing-and-events.md) |
| **Content policy** | Structure-only telemetry by default, opt-in content capture, secrets always redacted | `redaction.py` | [tracing-and-events.md](tracing-and-events.md#content-policy) |
| **Exporters** | JSONL span exporter (stdlib), plus optional OTel/OTLP bridge | `exporters.py`, `otel.py` | [tracing-and-events.md](tracing-and-events.md#exporters) |
| **Event fan-out** | An `EventSink` that sends sanitised copies of every event to every exporter | `events.py` | [tracing-and-events.md](tracing-and-events.md#event-fan-out) |
| **Metrics** | Counters, gauges and histograms, a Prometheus text renderer, the default dashboard set, and the cache verdict (hit, expected rebuild or unexpected miss) | `metrics.py`, `cache.py` | [metrics-and-cost.md](metrics-and-cost.md) |
| **Cost accounting** | A caller-supplied price table (schema + loader), cache read/write multipliers, and a budget ledger | `pricing.py` | [metrics-and-cost.md](metrics-and-cost.md#cost-accounting) |
| **Durable execution** | A step journal, leases with fencing tokens and heartbeats, idempotency keys, and a retry taxonomy | `durable.py`, `leases.py`, `idempotency.py`, `retry.py` | [durable-execution.md](durable-execution.md) |
| **Rainbow deploys** | Per-session version pinning, deterministic traffic split, drain/retire/rollback, per-model feature flags | `deploy.py`, `flags.py` | [deployments.md](deployments.md) |
| **Health, SLOs, alerts, review** | Error budgets, multi-window burn rates, alert rules as data, health checks, stratified transcript sampling | `health.py`, `alerts.py`, `sampling.py` | [health-and-alerts.md](health-and-alerts.md) |

## Why this shape

Each point below comes from the research, and each one is enforced in code.

1. **Instrument against the OTel GenAI conventions, behind a mapping layer.** The research names one span model (`invoke_agent` → `chat` → `execute_tool`) that keeps traces portable across Langfuse, Phoenix, Braintrust and LangSmith. The conventions are still in Development, the v1.41 vs v1.42 sources conflict, and the cache-token names are unverified. All attribute names therefore pass through `SemConv`, so a spec rename is a one-mapping change (pitfall: "pinning dashboards to one OTel release").
2. **Monitor structure, not content.** Anthropic's research agent was debugged from "agent decision patterns and interaction structures" without reading conversations, and OTel treats content as opt-in. Prompts, outputs and tool arguments become SHA-256 digests unless capture is switched on. Secrets are redacted with `harness_memory.SecretRedactor` either way.
3. **Prompt-cache hit rate is the headline KPI.** Manus calls it "the single most important metric", input outnumbers output about 100:1, and cached reads cost 0.1× (or less) of base input. Every model call is classified as a hit, an *expected* rebuild (compaction, model/effort switch, TTL expiry, fallback) or an *unexpected* miss. Only unexpected misses page.
4. **Prices are configuration, not code.** Cache multipliers differ by model (0.1×, 0.05×, 0.025×). The package ships a schema and a loader, not a price list. Compaction iterations are billable (`iterations=`). Budget checks report their post-response overshoot instead of pretending to be a hard cap.
5. **Durable steps, idempotent side effects.** In long runs minor failures compound, so a crashed run resumes from its last committed step (Temporal activities; Managed Agents `wake`). Replay never re-bills a model call. Side-effecting tools carry `sha256(session:step:call)` keys. Leases use fencing tokens, so a zombie worker cannot commit.
6. **Retry only what is transient.** 429/5xx/timeouts/stream-idle errors are retried with full-jitter backoff (Codex: 4 request and 5 stream retries). Auth, validation, context overflow and budget errors fail fast. Usage limits *pause* the session instead of failing it.
7. **Rainbow deploys.** Sessions finish on the version they started on. The version and the prompts/tools SHA are stamped on every root span. New traffic shifts gradually, and a running session's prompt or tools are never hot-patched (feature flags are frozen per session).
8. **Telemetry never breaks the agent.** Exporter, metric and pricing failures are counted and logged by class name only. They are never raised into the loop.

## Architecture

```text
 agent loop ─┬─ AgentInstrumentation.invoke_agent(version=pin) ──────────────────────────────────────────┐
             │     ├─ chat(...)        -> ChatCall.record(response)  ── cost (PriceTable) ── CostLedger   │
             │     │                                                 ── CacheTracker verdict             │
             │     ├─ execute_tool(...) -> ToolCallSpan.record(outcome)     (args hashed, idem key)      │
             │     ├─ compaction / subagent_spawn (traceparent carrier) / approval_wait                  │
             │     ▼                                                                                     │
             │  OpsTracer ── ContentPolicy (hash/redact) ── SemConv (rename) ──┬─ JsonlSpanExporter      │
             │                                                                 ├─ InMemorySpanExporter   │
             │                                                                 └─ OTelSpanExporter→OTLP  │
             │                                                                                           │
             ├─ EventSink.emit(Event) = FanoutEventSink ── sanitised copy ──┬─ HarnessMetrics ──▶ Prometheus text
             │                                                              ├─ SLOTracker ──▶ burn-rate alerts
             │                                                              ├─ ReviewSampler ──▶ ReviewQueue (refs only)
             │                                                              ├─ JsonlEventExporter / SpanEventExporter
             │                                                              ▼
             │                                    AlertEvaluator(default_alert_rules()) / render_prometheus_rules()
             │
             ├─ StepJournal.open_run(session_id)  ── lease + heartbeat + fencing token (SQLite WAL)
             │     run.model_call(key, client, request)   replayed on resume, never re-billed
             │     run.step(key, fn, input=..., retry=RetryPolicy())
             │     IdempotentToolExecutor ── IdempotencyStore (key = sha256(session:step:call))
             │
             └─ DeploymentRegistry.pin(session_id) ── weights / drain / rollback;  FeatureFlags.snapshot(model)
```

## Quick start

```python
from harness_core.types import Event, EventType, ToolCall, ToolOutcome
from harness_ops import (AgentInstrumentation, DeploymentRegistry, FanoutEventSink, HarnessMetrics,
                         JsonlEventExporter, JsonlSpanExporter, OpsTracer, PriceTable, StepJournal,
                         prompts_tools_sha)

prices = PriceTable.load("prices.yaml")          # your prices; see metrics-and-cost.md for the schema
metrics = HarnessMetrics(prices=prices)
tracer = OpsTracer([JsonlSpanExporter("telemetry/spans.jsonl")])   # structure-only by default
sink = FanoutEventSink([metrics, JsonlEventExporter("telemetry/events.jsonl")])
inst = AgentInstrumentation(tracer, metrics=metrics, provider="anthropic")

deploys = DeploymentRegistry("state/deploy.db")
deploys.register("v1", prompts_tools_sha(SYSTEM_PROMPT, [t.to_api() for t in TOOLS]), weight=100)
journal = StepJournal("state/journal.db")

pin = deploys.pin(session_id)                    # the session stays on this version until it finishes
with journal.open_run(session_id) as run, \
     inst.invoke_agent("main", session_id=session_id, model=MODEL, version=deploys.version(pin.version_id)) as agent:
    with inst.chat(agent, MODEL, request=request) as call:
        response = run.model_call("turn-0/model", client, request)      # replayed on resume, never re-billed
        call.record(response, replayed=run.was_replayed("turn-0/model"))
    for tu in response.tool_uses:
        key = f"turn-0/tool/{tu.id}"
        with inst.execute_tool(agent, ToolCall.from_block(tu), idempotency_key=run.idempotency_key(key)) as span:
            text = run.step(key, lambda: run_tool(tu))
            span.record(ToolOutcome(text))
    agent.set_termination("success")
    run.complete()
sink.emit(Event(session_id, EventType.SESSION_END,
                {"outcome": "success", "total_cost_usd": agent.total_cost_usd, "num_turns": agent.num_turns}))
deploys.finish(session_id)

print(metrics.dashboard()["cache_hit_rate"])
open("metrics.prom", "w").write(metrics.render_prometheus())   # or serve it with PROMETHEUS_CONTENT_TYPE
```

To export to an OTel backend such as Langfuse or Phoenix, add `harness_ops.otlp_exporter()` to the tracer's exporters
(`pip install opentelemetry-sdk opentelemetry-exporter-otlp`). It reads `OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_EXPORTER_OTLP_HEADERS`.

Run the tests with `python3 -m pytest -p no:cacheprovider tests/ops` and lint with `ruff check src/harness_ops tests/ops`.
