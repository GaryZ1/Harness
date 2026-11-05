"""SLOs and burn rates, alert rules as data, health checks, and review sampling."""

from __future__ import annotations

import pytest
import yaml

from harness_core.types import Event, EventType
from harness_ops import (
    SLO,
    AlertEvaluator,
    AlertState,
    ConfigError,
    FanoutEventSink,
    HarnessMetrics,
    HealthRegistry,
    HealthStatus,
    ReviewQueue,
    ReviewSampler,
    SamplingPolicy,
    SLOTracker,
    default_alert_rules,
    flatten_signals,
    load_alert_rules,
    render_prometheus_rules,
)


def test_error_budget_and_multiwindow_burn_rates(clock):
    t = SLOTracker(SLO("task_success", 0.99), clock=clock)
    for _ in range(990):
        t.record(True)
    for _ in range(10):
        t.record(False)
    assert t.error_rate() == pytest.approx(0.01) and t.burn_rate() == pytest.approx(1.0)
    assert t.budget_remaining() == pytest.approx(0.0)
    clock.advance(hours=7)  # quiet period, then an incident
    for _ in range(80):
        t.record(True)
    for _ in range(20):
        t.record(False)
    assert t.burn_rate(3600) == pytest.approx(20.0)  # 20% errors / 1% budget
    alerts = t.evaluate()
    assert {(a.severity, a.policy.long_window_seconds) for a in alerts} >= {("page", 3600)}
    clock.advance(minutes=10)  # short window clears -> the fast page resets even though 1 h is still hot
    t.record(True)
    assert all(a.policy.long_window_seconds != 3600 for a in t.evaluate())
    assert t.signals()["slo.task_success.burn_rate_1h"] > 14.4


def test_slo_from_events_and_alert_evaluator_pending_then_firing(clock):
    metrics = HarnessMetrics()
    slo = SLOTracker(SLO("task_success", 0.95), clock=clock)
    sink = FanoutEventSink([metrics, slo], clock=clock)
    for i in range(10):
        sink.emit(Event(f"s{i}", EventType.SESSION_END, {"outcome": "success" if i < 5 else "error_during_execution"}))
    for i in range(20):
        sink.emit(Event("s", EventType.TOOL_RESULT, {"tool": "bash", "is_error": i % 4 == 0}))
    signals = {**flatten_signals(metrics.dashboard()), **slo.signals()}
    assert signals["tool_error_rate"] == pytest.approx(0.25) and signals["slo.task_success.burn_rate_1h"] == pytest.approx(10.0)
    ev = AlertEvaluator(default_alert_rules(), clock=clock)
    first = {a.rule.name: a.state for a in ev.evaluate(signals)}
    assert first == {"ToolErrorRateHigh": AlertState.PENDING}  # for: 15m not yet elapsed; no-data rules stay quiet
    clock.advance(minutes=16)
    assert [a.rule.name for a in ev.firing(signals)] == ["ToolErrorRateHigh"]
    assert ev.evaluate({**signals, "tool_error_rate": 0.01}) == []  # resolves and resets


def test_alert_rules_load_from_yaml_and_render_prometheus(tmp_path):
    p = tmp_path / "rules.yaml"
    p.write_text(
        "rules:\n"
        "  - {name: CostSpike, signal: cost_usd_total, op: '>', threshold: 100, for_seconds: 300, severity: page,\n"
        "     promql: 'sum(increase(harness_cost_usd_total[1h])) > 100', summary: Spend above $100/h}\n"
    )
    rules = load_alert_rules(p)
    assert rules[0].name == "CostSpike" and rules[0].holds(101) and not rules[0].holds(None)
    rendered = yaml.safe_load(render_prometheus_rules(rules + default_alert_rules()))
    items = rendered["groups"][0]["rules"]
    assert items[0] == {"alert": "CostSpike", "expr": "sum(increase(harness_cost_usd_total[1h])) > 100", "for": "5m", "labels": {"severity": "page"}, "annotations": {"summary": "Spend above $100/h"}}
    assert "TaskSuccessBudgetFastBurn" not in {i["alert"] for i in items}  # no promql: in-process only
    for bad in ({"rules": [{"name": "x", "signal": "s", "op": "~", "threshold": 1}]}, {"rules": [{"name": "x"}]}, {"nope": []}):
        with pytest.raises(ConfigError):
            load_alert_rules(bad)


def test_health_registry_statuses():
    h = HealthRegistry()
    h.add("journal_db", lambda: True)
    h.add("otlp_export", lambda: (False, "backend unreachable"), critical=False)
    assert h.report().status is HealthStatus.DEGRADED
    h.add("lease_heartbeat", lambda: 1 / 0)
    r = h.report()
    assert r.status is HealthStatus.FAILING and r.to_dict()["checks"][2]["detail"] == "check raised ZeroDivisionError"


def test_review_sampling_is_stratified_capped_and_content_free(tmp_path, clock):
    queue = ReviewQueue(tmp_path / "review.jsonl")
    sampler = ReviewSampler(SamplingPolicy(base_rate=0.05, cost_outlier_usd=5.0, max_random_per_day=10), queue, clock=clock)
    sink = FanoutEventSink([sampler], clock=clock)
    sink.emit(Event("fail-1", EventType.SESSION_END, {"outcome": "error_max_turns", "trace_id": "t" * 32, "text": "user content"}))
    sink.emit(Event("pricey", EventType.SESSION_END, {"outcome": "success", "total_cost_usd": 9.0}))
    sink.emit(Event("flagged", EventType.SESSION_END, {"outcome": "success", "flags": ["unexpected_cache_miss"]}))
    sink.emit(Event("unhappy", EventType.SESSION_END, {"outcome": "success", "user_feedback": "negative"}))
    for i in range(1000):
        sampler.consider(f"ok-{i}", outcome="success")
    items = queue.items()
    reasons = {i.session_id: i.reasons for i in items}
    assert reasons["fail-1"] == ("outcome:error_max_turns",) and reasons["pricey"] == ("cost_outlier",)
    assert reasons["flagged"] == ("flag:unexpected_cache_miss",) and reasons["unhappy"] == ("negative_feedback",)
    random_items = [i for i in items if i.reasons == ("random",)]
    assert len(random_items) == 10  # daily cap on baseline samples
    assert (items[0].session_id, items[0].priority, items[0].trace_id) == ("fail-1", 3, "t" * 32)
    assert "user content" not in (tmp_path / "review.jsonl").read_text()
    # deterministic: a fresh sampler picks the same baseline sessions
    again = ReviewSampler(SamplingPolicy(base_rate=0.05, max_random_per_day=10), clock=clock)
    assert [s for s in (f"ok-{i}" for i in range(1000)) if again.decide(s, outcome="success").sampled][:10] == [i.session_id for i in random_items]
