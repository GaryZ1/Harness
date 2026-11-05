# Metrics, dashboards, cache tracking and cost accounting

## Purpose

This page covers the default ops dashboard: cache hit rate, cost per successful task, tool error rate, turns, p50/p95 latency
and TTFT, compactions, permission asks and denials, and stop reasons. All of it is exposed as Prometheus text. Cost is computed
from `Usage` against a price table that **you** supply.

## Research basis

* **Derived dashboards** (reference §10.3, checklist item 4): cache hit rate (cacheRead / total input, target >80–90%, split into expected rebuilds and unexpected misses), cost per *successful* task, tool error rate by tool, turns per task, p50/p95 wall-clock and TTFT, compaction frequency, and denial rates. "Alert on unexpected cache-miss spikes as regressions."
* **Claude Code telemetry**: `claude_code.token.usage` labelled `token.type` = input / output / cacheRead / cacheCreation, and `claude_code.cost.usage` (Prometheus name `claude_code_cost_usage_USD_total`). We mirror the shape: `harness_tokens_total{type=...}` and `harness_cost_usd_total`.
* **Cost** (§10.3 "Prompt caching as the main cost lever", §10.4). Total input = `cache_read + cache_creation + input_tokens`. Reads are priced at 0.1× base on most Claude models, 0.05× on Opus/Sonnet 5.5 and 0.025× on Fable/Mythos 5.1. 5-minute writes cost 1.25× and 1-hour writes 2×. These vary per model, so nothing is baked in.
* **Pitfalls**: compaction iterations are excluded from top-level usage ("sum `usage.iterations`"), and budget checks happen *after* each response, so spend can overshoot by one response plus subagents.

## Metrics API

```python
from harness_ops import HarnessMetrics, MetricsRegistry, PROMETHEUS_CONTENT_TYPE

metrics = HarnessMetrics(prices=prices)                 # or HarnessMetrics(MetricsRegistry(namespace="myapp"))
metrics.record_model_call("test-model", usage, latency_ms=900, ttft_ms=300, stop_reason="end_turn")
metrics.record_tool_call("bash", is_error=False, duration_ms=40)
metrics.record_permission("ask", "bash"); metrics.record_stop("end_turn"); metrics.record_compaction()
metrics.record_task("success", cost_usd=0.42, turns=12, duration_ms=95_000)
metrics.dashboard()        # dict of the default dashboard values (None = no data yet)
metrics.render_prometheus()  # text exposition format 0.0.4
```

| Metric (default namespace `harness_`) | Type | Labels |
|---|---|---|
| `tokens_total` | counter | `model`, `type` = input / output / cache_read / cache_creation |
| `cost_usd_total`, `unpriced_calls_total` | counter | `model` |
| `model_calls_total` | counter | `model`, `stop_reason` |
| `model_latency_ms`, `ttft_ms` | histogram | `model` |
| `cache_observations_total` | counter | `model`, `observation`, `reason` |
| `retries_total` | counter | `kind` |
| `tool_calls_total` | counter | `tool`, `outcome` = ok / error |
| `tool_duration_ms` | histogram | `tool` |
| `permission_decisions_total` | counter | `behavior`, `tool` |
| `stop_reasons_total` | counter | `reason` |
| `compactions_total` | counter | |
| `errors_total` | counter | `error_class` |
| `tasks_total`, `task_cost_usd_total` | counter | `outcome` |
| `task_turns`, `task_duration_ms` | histogram | `outcome` |

`dashboard()` keys: `cache_hit_rate`, `unexpected_cache_miss_rate`, `cost_usd_total`, `cost_per_successful_task_usd`
(spend of *all* tasks ÷ successful tasks), `task_success_rate`, `tool_error_rate`, `tool_error_rate_by_tool`,
`turns_per_task_p50`, `model_latency_p50_ms`/`p95_ms`, `ttft_p50_ms`/`p95_ms`, `task_duration_p50_ms`/`p95_ms`,
`compactions_per_task`, `permission_rates`, `stop_reasons`, `retries_total`.

Generic primitives: `MetricsRegistry.counter/gauge/histogram(name, help, labelnames)` are get-or-create. A name reused with a
different type or label set raises. Histograms keep cumulative buckets for `histogram_quantile` and a sliding reservoir for
exact in-process quantiles.

## Cache tracking

`CacheTracker(ttl_seconds=300, min_cacheable_tokens=1024, miss_threshold=0.5)`, one per session (`CacheMonitor` keeps them
by session id), classifies each call:

| Verdict | When |
|---|---|
| `hit` | read share ≥ `miss_threshold` |
| `cold_start` | first call of the session |
| `expected_rebuild` (reason) | declared via `expect_rebuild(reason)` (compaction, context edit, rewind...), or inferred: `model_switch`, `effort_switch`, `ttl_expired`, `fallback` |
| `unexpected_miss` | everything else. **This is the regression signal.** |
| `below_min_cacheable` | total input under the model's minimum cacheable length (such prompts silently never cache) |

`AgentInstrumentation.compaction()` declares the rebuild for you. `ChatCall.record(fallback_from=...)` declares `fallback`.

## Cost accounting

```yaml
# prices.yaml (schema: harness_ops.PRICING_SCHEMA; all numbers are yours)
version: 1
currency: USD
defaults:
  cache_read_multiplier: 0.1
  cache_write_multiplier: 1.25      # 5-minute writes
  cache_write_1h_multiplier: 2.0
models:
  test-model:
    input_per_mtok: 3.0
    output_per_mtok: 15.0
    cache_read_multiplier: 0.05     # per-model override
    tiers: [{above_input_tokens: 200000, input_per_mtok: 6.0, output_per_mtok: 22.5}]
  "test-model-*":                   # globs match dated ids; exact ids win
    input_per_mtok: 1.0
    output_per_mtok: 5.0
    cache_read_per_mtok: 0.02       # absolute price beats a multiplier
    batch_multiplier: 0.5           # only applied with cost(..., batch=True); verify per provider
```

```python
from harness_ops import PriceTable, CostLedger
prices = PriceTable.load("prices.yaml")                 # .yaml/.yml/.json; ConfigError names the bad field
c = prices.cost("test-model", usage)                    # CostBreakdown(input/output/cache_read/cache_write, total_usd)
c = prices.cost("test-model", iterations=resp_iterations, cache_write_ttl="1h")   # bill compaction passes
ledger = CostLedger(max_budget_usd=5.0); status = ledger.add(c, scope="subagent")  # status.exceeded / overshoot_usd
```

## Defaults

| Setting | Default | Anchor |
|---|---|---|
| Prices, multipliers | **none**; the table is required, and models need cache pricing directly or via `defaults` | our choice (multipliers differ per model) |
| Unknown model | `PricingError` when `strict=True` (default); `priced=False` and cost 0 otherwise. Metrics never raise and count `unpriced_calls_total` | our choice |
| Cache write TTL for pricing | `5m` | vendor default (Anthropic 5-minute TTL) |
| Hit-rate target | 80% (`CacheHitRateLow` alert) | research inference (">80–90%") |
| Cache TTL / min cacheable / miss threshold | 300 s / 1,024 tokens / 0.5 | vendor default (5 min) / our choice (set per model: 512–4,096) / our choice |
| Latency buckets | 50 ms … 300 s | our choice |
| Quantile reservoir | last 2,048 observations per label set | our choice |
| Label sets per metric | 1,000, then `__overflow__` | our choice (tool names are model-controlled) |
| `max_budget_usd` | `None` (no limit) | vendor default (Agent SDK: "No limit"; "setting a budget is a good default for production") |

## Invariants

* Counters only go up. Label names are validated (`le` and `__*` are reserved). Metric names follow the Prometheus grammar.
* Cost uses all four token types. Tiers are chosen by total input, including cache tokens.
* A price table is immutable after loading. Re-load and swap it to change prices.

## Failure modes and runbook

* **`unpriced_calls_total` rising**: a new model id (often a dated alias) is missing from the table. Add the exact id or a glob.
* **`unexpected_cache_miss_rate` spike**: diff the system prompt and tool list between the last hit and the first miss. Look for timestamps, non-deterministic JSON or tool ordering, MCP servers that connected mid-session, or a missed `expect_rebuild` declaration.
* **Cost per successful task climbing while cost per task is flat**: the success rate fell. Read the failures sampled by `ReviewSampler`.
* **High-cardinality warning**: `__overflow__` appearing in labels means more than 1,000 distinct tool names. Normalise MCP tool names before recording.
