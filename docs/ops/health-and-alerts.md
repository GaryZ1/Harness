# Health, SLOs, alert rules and review sampling

## Purpose

This component answers three questions. Is the harness healthy? Should someone be paged? Which transcripts should a human read
this week?

## Research basis

* **Alerting** (reference §10.7 checklist item 4): "Alert on unexpected cache-miss spikes as regressions". The cache-hit target is >80–90% of input tokens. Tool error rate and p95 latency are default dashboards.
* **Online trace grading** (§10.3, checklist item 20): "Trace grading is the fastest way to identify workflow-level issues". Sample production traces into graders weekly, cluster the failures, and feed them into the capability suite. The build plan adds "read transcripts".
* **Privacy**: monitor structure, not content (Anthropic). The review queue therefore stores references, never transcript text.
* **SLO method**: the research names none. We use the Google SRE Workbook's error budgets with multi-window, multi-burn-rate alerts (our choice).

## SLOs and burn rates

```python
from harness_ops import SLO, SLOTracker, tool_success_sli, FanoutEventSink

task_slo = SLOTracker(SLO("task_success", objective=0.95))                        # SESSION_END outcome == "success"
tool_slo = SLOTracker(SLO("tool_success", 0.99), classifier=tool_success_sli)     # TOOL_RESULT is_error
sink = FanoutEventSink([metrics, task_slo, tool_slo])
task_slo.burn_rate(3600); task_slo.budget_remaining(); task_slo.evaluate()        # -> [BurnAlert(...)]
```

Burn rate = error rate ÷ (1 − objective). The default policies (`DEFAULT_BURN_RATE_POLICIES`) require *both* windows to burn:

| Severity | Long window | Short window | Burn rate |
|---|---|---|---|
| page | 1 h | 5 min | 14.4 |
| page | 6 h | 30 min | 6 |
| ticket | 3 d | 6 h | 1 |

## Alert rules as data

```python
from harness_ops import AlertEvaluator, default_alert_rules, load_alert_rules, render_prometheus_rules, flatten_signals

rules = default_alert_rules() + load_alert_rules("alerts.yaml")
ev = AlertEvaluator(rules)
signals = {**flatten_signals(metrics.dashboard()), **task_slo.signals()}
for alert in ev.firing(signals): notify(alert.rule.severity, alert.rule.summary, alert.value)
open("harness-rules.yaml", "w").write(render_prometheus_rules(rules))      # rules that carry promql
```

A rule has `name`, `signal`, `op` (`> >= < <= == !=`), `threshold`, `for_seconds`, `severity` (`page`/`ticket`/`info`), and
optionally `summary`, `promql` and `runbook`. A missing or `None` signal means no data and never fires. A condition must hold
for `for_seconds` (state `pending`) before it fires, and it resets as soon as it stops holding.

| Default rule | Signal | Condition | Severity | Anchor |
|---|---|---|---|---|
| `UnexpectedCacheMissSpike` | `unexpected_cache_miss_rate` | > 5% for 10 min | page | research ("alert on unexpected cache-miss spikes"); 5% is our choice |
| `CacheHitRateLow` | `cache_hit_rate` | < 0.8 for 15 min | ticket | research target (>80–90%) |
| `ToolErrorRateHigh` | `tool_error_rate` | > 10% for 15 min | ticket | our choice |
| `ModelLatencyP95High` | `model_latency_p95_ms` | > 60 s for 15 min | ticket | our choice |
| `TaskSuccessBudgetFastBurn` | `slo.task_success.burn_rate_1h` | ≥ 14.4 | page | SRE Workbook (in-process only; no promql) |

## Health checks

`HealthRegistry.add(name, check, critical=True)` registers a check. A check returns `bool` or `(bool, detail)`, and an
exception counts as failing. `report()` → `HealthReport(status=ok|degraded|failing, checks)`: a failing critical check makes
the report `failing`, and a failing non-critical one makes it `degraded`. Typical checks are journal DB writable, lease
heartbeats alive, exporter error counters flat, and the deploy registry having an active version.

## Review sampling

```python
from harness_ops import ReviewSampler, ReviewQueue, SamplingPolicy
sampler = ReviewSampler(SamplingPolicy(base_rate=0.05, cost_outlier_usd=5.0), ReviewQueue("review/queue.jsonl"))
sink.add_exporter(sampler)            # consumes SESSION_END events
ReviewQueue("review/queue.jsonl").items()   # sorted by priority: failures/negative feedback (3), flags/cost (2), random (1)
```

Each sampled session is a `ReviewItem` with session id, trace id, reasons, priority, outcome, cost and turns, and no content.
Reviewers open the transcript through the session log's own access controls. Baseline sampling is a deterministic hash of the
session id, so every process picks the same sessions.

## Defaults

| Setting | Default | Anchor |
|---|---|---|
| SLO window / bucket | 30 days / 60 s | SRE Workbook / our choice |
| Burn-rate policies | 14.4 (1 h & 5 m), 6 (6 h & 30 m), 1 (3 d & 6 h) | SRE Workbook (our choice of method) |
| Review base rate | 5% of successful sessions | our choice |
| Always sampled | non-success outcome, negative feedback, `flags`, cost ≥ `cost_outlier_usd` (off by default) | research (failures matter most); thresholds our choice |
| Daily caps | 50 random, 500 flagged (per process) | our choice (bounds weekly reading load) |

## Invariants

* Alerts never fire on missing data. Rule names are unique. Ops, severities and thresholds are validated when rules load (`ConfigError` names the field).
* The review queue never contains event content. The `FanoutEventSink` has already sanitised the payload, and the sampler copies only scalar references.
* A crashing health check is reported as failing. It never raises out of `report()`.

## Failure modes and runbook

* **`UnexpectedCacheMissSpike`**: diff the prompt prefix (system prompt, tool definitions, their order and serialisation) between the last hit and the first miss. Check for a deploy at the same time by filtering on `harness.version`. Roll back if the regression came from a new build.
* **`TaskSuccessBudgetFastBurn`**: check `stop_reasons` and `errors_total` for the dominant failure, then the review queue's priority-3 items. If a provider is degraded (429/5xx retries rising), fall back at turn boundaries.
* **Review queue too large**: lower `base_rate` or the caps. Never drop failures from the sample; cluster them instead.
* **Caps are per process**: with many workers, put the sampler behind one aggregation process, or divide the caps by the worker count.
