"""Metrics registry, Prometheus text format, dashboard values, and the fan-out EventSink."""

from __future__ import annotations

import re
import threading

import pytest

from harness_core.types import Event, EventSink, EventType, Usage
from harness_ops import (
    FanoutEventSink,
    HarnessMetrics,
    InMemoryEventExporter,
    InMemorySpanExporter,
    JsonlEventExporter,
    MetricsRegistry,
    OpsTracer,
    SpanEventExporter,
    read_events,
)
from harness_ops.metrics import OVERFLOW


def test_prometheus_text_format_counters_histograms_and_escaping():
    reg = MetricsRegistry(namespace="t")
    c = reg.counter("requests_total", "Requests\nby code", ("code", "path"))
    c.inc(code="200", path='/a"b\\c')
    c.inc(2, code="500", path="/")
    h = reg.histogram("latency_ms", "Latency", ("model",), buckets=(10, 100))
    for v in (5, 50, 500):
        h.observe(v, model="m")
    g = reg.gauge("inflight", "In flight")
    g.set(3)
    text = reg.render_prometheus()
    assert "# HELP t_requests_total Requests\\nby code" in text and "# TYPE t_requests_total counter" in text
    assert 't_requests_total{code="200",path="/a\\"b\\\\c"} 1' in text and 't_requests_total{code="500",path="/"} 2' in text
    assert 't_latency_ms_bucket{model="m",le="10"} 1' in text and 't_latency_ms_bucket{model="m",le="100"} 2' in text
    assert 't_latency_ms_bucket{model="m",le="+Inf"} 3' in text and 't_latency_ms_sum{model="m"} 555' in text
    assert 't_latency_ms_count{model="m"} 3' in text and "t_inflight 3" in text and text.endswith("\n")
    sample = re.compile(r'^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})? [0-9.e+\-Inf]+$')
    assert all(line.startswith("#") or sample.match(line) for line in text.strip().splitlines())
    with pytest.raises(ValueError):
        c.inc(-1, code="200")
    with pytest.raises(ValueError):
        reg.counter("requests_total", "x", ("other",))
    with pytest.raises(ValueError):
        MetricsRegistry().counter("bad-name", "x")


def test_histogram_quantiles_and_cardinality_cap():
    reg = MetricsRegistry(max_label_sets=3)
    h = reg.histogram("lat", "x", ("tool",))
    for v in range(1, 101):
        h.observe(v, tool="a")
    assert h.quantile(0.5) == pytest.approx(50.5) and h.quantile(0.95) == pytest.approx(95.05)
    assert h.quantile(0.5, tool="missing") is None
    c = reg.counter("calls_total", "x", ("tool",))
    for i in range(10):  # model-invented tool names must not explode cardinality
        c.inc(tool=f"tool_{i}")
    labels = {lbl["tool"] for lbl, _ in c.items()}
    assert len(labels) == 4 and OVERFLOW in labels and c.sum() == 10


def test_concurrent_counter_increments_are_exact():
    m = HarnessMetrics()

    def work():
        for _ in range(1000):
            m.record_tool_call("read", is_error=False, duration_ms=1.0)

    ts = [threading.Thread(target=work) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert m.tool_calls.sum() == 8000 and m.tool_duration.count() == 8000


def test_fanout_sink_feeds_metrics_and_sanitises_payloads(tmp_path, clock):
    metrics = HarnessMetrics()
    mem = InMemoryEventExporter()
    jsonl = JsonlEventExporter(tmp_path / "events.jsonl")
    sink = FanoutEventSink([metrics, mem, jsonl], clock=clock)
    assert isinstance(sink, EventSink)
    secret = "ghp_" + "a" * 36
    events = [
        Event("s1", EventType.USER_MESSAGE, {"text": f"please use {secret} to push"}),
        Event("s1", EventType.MODEL_RESPONSE, {"model": "test-model", "usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 900}, "stop_reason": "tool_use", "latency_ms": 800, "cost_usd": 0.01}),
        Event("s1", EventType.TOOL_RESULT, {"tool": "bash", "is_error": True, "duration_ms": 30, "content": "stack trace with " + secret}),
        Event("s1", EventType.PERMISSION, {"behavior": "deny", "tool": "bash"}),
        Event("s1", EventType.STOP, {"reason": "end_turn"}),
        Event("s1", EventType.SESSION_END, {"outcome": "success", "total_cost_usd": 0.01, "num_turns": 3, "duration_ms": 5000}),
        Event("s2", EventType.SESSION_END, {"outcome": "error_max_turns", "total_cost_usd": 0.03, "num_turns": 50}),
    ]
    returned = [sink.emit(e) for e in events]
    # caller's event keeps full fidelity (it goes to the durable log); seq/ts filled in
    assert returned[0].data["text"].startswith("please use ghp_") and [e.seq for e in returned] == [0, 1, 2, 3, 4, 5, 0]
    assert returned[0].ts == "2026-11-03T10:00:00Z"
    exported = mem.events
    assert exported[0].data["text"].keys() == {"sha256", "chars"} and exported[2].data["content"]["chars"] > 0
    on_disk = (tmp_path / "events.jsonl").read_text()
    assert secret not in on_disk and "please use" not in on_disk
    assert len(read_events(tmp_path / "events.jsonl")) == 7
    d = metrics.dashboard()
    assert d["cache_hit_rate"] == pytest.approx(0.9) and d["tool_error_rate"] == 1.0
    assert d["permission_rates"]["deny"] == 1.0 and d["stop_reasons"] == {"end_turn": 1.0}
    assert d["task_success_rate"] == 0.5 and d["cost_per_successful_task_usd"] == pytest.approx(0.04)
    assert d["model_latency_p50_ms"] == 800 and d["turns_per_task_p50"] == pytest.approx(26.5)
    assert 'harness_tokens_total{model="test-model",type="cache_read"} 900' in metrics.render_prometheus()


def test_fanout_isolates_failing_exporters_and_opt_in_capture(clock):
    from harness_ops import ContentPolicy

    class Broken:
        def export_event(self, event):
            raise OSError("disk full")

    mem = InMemoryEventExporter()
    sink = FanoutEventSink([Broken(), mem], policy=ContentPolicy(capture_content=True), clock=clock)
    sink.emit(Event("s", EventType.USER_MESSAGE, {"text": "password=hunter2hunter2 ok"}))
    assert sink.errors == {"Broken": 1} and len(mem.events) == 1
    assert mem.events[0].data["text"] == "password=[REDACTED_SECRET] ok"


def test_span_event_exporter_attaches_structure_to_current_span(clock):
    exp = InMemorySpanExporter()
    tracer = OpsTracer([exp])
    sink = FanoutEventSink([SpanEventExporter()], clock=clock)
    with tracer.span("invoke_agent main"):
        sink.emit(Event("s", EventType.TOOL_CALL, {"tool": "read", "input": {"path": "/etc/passwd"}}))
    ev = exp.spans[0].events[0]
    assert ev.name == "harness.tool_call" and ev.attributes["harness.event.tool"] == "read"
    assert "/etc/passwd" not in str(ev.attributes)


def test_record_model_call_prices_from_table_and_counts_unpriced():
    from harness_ops import PriceTable

    prices = PriceTable.from_dict({"version": 1, "defaults": {"cache_read_multiplier": 0.1, "cache_write_multiplier": 1.25, "cache_write_1h_multiplier": 2}, "models": {"test-model": {"input_per_mtok": 2, "output_per_mtok": 10}}})
    m = HarnessMetrics(prices=prices)
    assert m.record_model_call("test-model", Usage(1_000_000, 0)) == pytest.approx(2.0)
    assert m.record_model_call("unknown-model", Usage(10, 10)) is None  # never raises in telemetry
    assert m.unpriced.sum(model="unknown-model") == 1 and m.cost.sum() == pytest.approx(2.0)
