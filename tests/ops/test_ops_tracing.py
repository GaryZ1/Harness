"""Tracing: span nesting, W3C trace context, content policy, sampling, exporters and the OTel bridge."""

from __future__ import annotations

import json
import sys
import threading
from types import SimpleNamespace

import pytest

from harness_core.types import NullTracer, Tracer
from harness_ops import (
    A,
    ContentPolicy,
    InMemorySpanExporter,
    JsonlSpanExporter,
    OpsTracer,
    OTelUnavailableError,
    Sampler,
    SemConv,
    SpanContext,
    StatusCode,
    extract,
    inject,
    parse_traceparent,
    read_spans,
    span_tree,
    use_span,
)
from harness_ops.otel import OTelSpanExporter, OTelTracer, otlp_exporter


def make_tracer(**kw):
    exp = InMemorySpanExporter()
    return OpsTracer([exp], **kw), exp


def test_tracer_satisfies_shared_protocol_and_nests_spans():
    tracer, exp = make_tracer()
    assert isinstance(tracer, Tracer) and isinstance(NullTracer(), Tracer)
    with tracer.span("root", **{A.AGENT_NAME: "main"}) as root:
        with tracer.span("child") as child:
            with tracer.span("grandchild") as gc:
                pass
    spans = {s.name: s for s in exp.spans}
    assert [s.name for s in exp.spans] == ["grandchild", "child", "root"]  # exported on end
    assert len(root.trace_id) == 32 and len(root.span_id) == 16
    assert spans["child"].parent_span_id == root.span_id and spans["grandchild"].parent_span_id == child.span_id
    assert {s.trace_id for s in exp.spans} == {root.trace_id}
    assert spans["root"].parent_span_id is None and spans["root"].attributes[A.AGENT_NAME] == "main"
    assert gc.end_ns is not None and spans["root"].end_ns >= spans["child"].end_ns
    tree = span_tree(exp.spans)
    assert [s.name for s in tree[None]] == ["root"] and [s.name for s in tree[root.span_id]] == ["child"]


def test_traceparent_roundtrip_and_validation():
    ctx = SpanContext("4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7", True, "vendor=a")
    assert ctx.traceparent == "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    parsed = parse_traceparent(ctx.traceparent, "vendor=a")
    assert parsed is not None and parsed.trace_id == ctx.trace_id and parsed.sampled and parsed.remote and parsed.tracestate == "vendor=a"
    assert parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00").sampled is False
    for bad in (
        "",
        "garbage",
        "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",  # version ff invalid
        "00-00000000000000000000000000000000-00f067aa0ba902b7-01",  # zero trace id
        "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",  # zero span id
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra",  # v00 must have 4 fields
    ):
        assert parse_traceparent(bad) is None
    # future versions may append fields
    assert parse_traceparent("01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-future") is not None


def test_inject_extract_propagates_parent_across_processes_and_mcp_meta():
    tracer, exp = make_tracer()
    with tracer.span("tool call") as tool_span:
        meta = inject({})  # e.g. an MCP request's _meta (SEP-414)
    assert meta["traceparent"] == tool_span.traceparent
    # "remote" side: a different tracer continues the trace under the tool span
    remote, rexp = make_tracer()
    with remote.span("mcp server", parent=meta):
        pass
    child = rexp.spans[0]
    assert child.trace_id == tool_span.trace_id and child.parent_span_id == tool_span.span_id
    assert extract({"TraceParent": meta["traceparent"]}).span_id == tool_span.span_id  # case-insensitive carrier
    with remote.span("new root", parent=False) as fresh:
        assert fresh.parent_span_id is None and fresh.trace_id != tool_span.trace_id


def test_structure_only_by_default_content_is_hashed_and_secrets_redacted():
    tracer, exp = make_tracer()
    secret = "sk-ant-api03-" + "x" * 30
    with tracer.span("execute_tool", **{A.TOOL_NAME: "bash", A.TOOL_CALL_ARGUMENTS: {"cmd": f"curl -H 'Authorization: {secret}'"}, "note": f"key={secret}"}):
        pass
    attrs = exp.spans[0].attributes
    assert A.TOOL_CALL_ARGUMENTS not in attrs
    assert len(attrs[f"{A.TOOL_CALL_ARGUMENTS}.sha256"]) == 64 and attrs[f"{A.TOOL_CALL_ARGUMENTS}.length"] > 0
    assert secret not in json.dumps(attrs) and "[REDACTED_SECRET]" in attrs["note"]
    # identical args -> identical hash (args_hash usable for dedup / clustering)
    with tracer.span("again", **{A.TOOL_CALL_ARGUMENTS: {"cmd": f"curl -H 'Authorization: {secret}'"}}):
        pass
    assert exp.spans[1].attributes[f"{A.TOOL_CALL_ARGUMENTS}.sha256"] == attrs[f"{A.TOOL_CALL_ARGUMENTS}.sha256"]


def test_opt_in_content_capture_still_redacts_and_truncates():
    policy = ContentPolicy(capture_content=True, max_content_chars=40)
    tracer, exp = make_tracer(policy=policy)
    with tracer.span("chat", **{A.INPUT_MESSAGES: [{"role": "user", "content": "token=supersecretvalue123 " + "y" * 100}]}):
        pass
    captured = exp.spans[0].attributes[A.INPUT_MESSAGES]
    assert "supersecretvalue123" not in captured and "chars truncated" in captured
    assert ContentPolicy.from_env({"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"}).capture_content
    assert not ContentPolicy.from_env({}).capture_content
    # HMAC hashing resists dictionary attacks on short content
    keyed = ContentPolicy(hash_key=b"deploy-secret")
    assert keyed.digest("yes") != ContentPolicy().digest("yes")


def test_exception_recorded_without_message_by_default_and_reraised():
    tracer, exp = make_tracer()
    with pytest.raises(ValueError):
        with tracer.span("boom"):
            raise ValueError("user said: my password is hunter2")
    s = exp.spans[0]
    assert s.status is StatusCode.ERROR and s.attributes[A.ERROR_TYPE] == "ValueError"
    assert "hunter2" not in json.dumps(s.to_dict())


def test_semconv_rename_layer_is_one_mapping():
    sc = SemConv({A.USAGE_CACHE_READ_INPUT_TOKENS: "gen_ai.usage.cache_read_input_tokens", A.TOOL_CALL_ARGUMENTS: "gen_ai.tool.arguments", "harness.drop_me": ""})
    tracer, exp = make_tracer(semconv=sc)
    with tracer.span("x", **{A.USAGE_CACHE_READ_INPUT_TOKENS: 5, A.TOOL_CALL_ARGUMENTS: {"a": 1}, "harness.drop_me": 1}):
        pass
    attrs = exp.spans[0].attributes
    assert attrs["gen_ai.usage.cache_read_input_tokens"] == 5
    assert "gen_ai.tool.arguments.sha256" in attrs and "harness.drop_me" not in attrs


def test_head_sampling_is_trace_consistent_and_deterministic():
    tracer, exp = make_tracer(sampler=Sampler(0.0))
    with tracer.span("root") as root:
        with tracer.span("child") as child:
            child.set_attribute("k", 1)
    assert exp.spans == [] and not root.context.sampled and not child.context.sampled
    assert root.traceparent.endswith("-00")  # still propagates (unsampled) context
    half = Sampler(0.5)
    tid = "4bf92f3577b34da6a3ce929d0e0e4736"
    assert half.should_sample(tid, None) == half.should_sample(tid, None)
    # parent decision wins
    assert Sampler(0.0).should_sample(tid, SpanContext(tid, "00f067aa0ba902b7", True))


def test_contextvar_isolation_across_threads_and_use_span():
    tracer, exp = make_tracer()
    seen = {}
    with tracer.span("main") as main:

        def worker():
            with tracer.span("orphan") as s:  # new thread: no implicit parent
                seen["orphan_parent"] = s.parent_span_id
            with use_span(main):
                with tracer.span("adopted") as s2:
                    seen["adopted_parent"] = s2.parent_span_id

        t = threading.Thread(target=worker)
        t.start()
        t.join()
    assert seen["orphan_parent"] is None and seen["adopted_parent"] == main.span_id


def test_failing_exporter_never_breaks_the_loop():
    class Broken:
        def export(self, spans):
            raise RuntimeError("backend down")

        def shutdown(self):
            pass

    good = InMemorySpanExporter()
    tracer = OpsTracer([Broken(), good])
    with tracer.span("ok"):
        pass
    assert tracer.export_errors == 1 and len(good.spans) == 1


def test_jsonl_exporter_roundtrip_concurrent_writers_and_torn_tail(tmp_path):
    path = tmp_path / "spans.jsonl"
    exporters = [JsonlSpanExporter(path), JsonlSpanExporter(path)]  # two "processes" sharing one file
    tracers = [OpsTracer([e]) for e in exporters]

    def work(t, n):
        for i in range(n):
            with t.span(f"s{i}", **{"i": i}):
                pass

    threads = [threading.Thread(target=work, args=(tracers[i % 2], 25)) for i in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"name": "torn", "trace_id": ')  # crash mid-write
    spans = read_spans(path)
    assert len(spans) == 100 and all(len(s.trace_id) == 32 for s in spans)
    assert spans[0].to_dict()["duration_ms"] >= 0


def test_jsonl_rotation(tmp_path):
    path = tmp_path / "spans.jsonl"
    tracer = OpsTracer([JsonlSpanExporter(path, max_bytes=2_000, backups=2)])
    for i in range(40):
        with tracer.span(f"span-{i}"):
            pass
    assert path.exists() and (tmp_path / "spans.jsonl.1").exists()
    assert path.stat().st_size <= 2_000


def test_otel_bridge_raises_clear_import_error(monkeypatch):
    for mod in [m for m in sys.modules if m == "opentelemetry" or m.startswith("opentelemetry.")]:
        monkeypatch.setitem(sys.modules, mod, None)
    monkeypatch.setitem(sys.modules, "opentelemetry", None)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", None)
    with pytest.raises(ImportError, match="pip install opentelemetry"):
        OTelTracer()
    with pytest.raises(OTelUnavailableError):
        otlp_exporter()


def test_otel_tracer_delegates_with_sanitised_attributes():
    calls = []

    class FakeSpan:
        def __init__(self):
            self.attrs = {}

        def set_attribute(self, k, v):
            self.attrs[k] = v

        def add_event(self, name, attrs):
            pass

    class FakeOTelTracer:
        def start_as_current_span(self, name, **kwargs):
            from contextlib import contextmanager

            @contextmanager
            def cm():
                s = FakeSpan()
                calls.append((name, kwargs, s))
                yield s

            return cm()

    t = OTelTracer(FakeOTelTracer())
    with t.span("chat test-model", **{A.REQUEST_MODEL: "test-model", A.INPUT_MESSAGES: "hello"}) as s:
        s.set_attribute(A.USAGE_INPUT_TOKENS, 12)
    name, kwargs, raw = calls[0]
    assert name == "chat test-model" and kwargs["attributes"][A.REQUEST_MODEL] == "test-model"
    assert A.INPUT_MESSAGES not in kwargs["attributes"] and f"{A.INPUT_MESSAGES}.sha256" in kwargs["attributes"]
    assert raw.attrs[A.USAGE_INPUT_TOKENS] == 12


def test_otel_span_exporter_keeps_our_ids():
    class Rec(SimpleNamespace):
        pass

    def ctor(**kw):
        return Rec(**kw)

    mods = SimpleNamespace(
        ReadableSpan=ctor,
        Event=lambda name, attrs, ts: Rec(name=name, attributes=attrs, timestamp=ts),
        SpanContext=lambda trace_id, span_id, is_remote, trace_flags=None, trace_state=None: Rec(trace_id=trace_id, span_id=span_id),
        TraceFlags=lambda v: v,
        TraceState=lambda pairs=(): list(pairs),
        Link=lambda ctx, attrs: Rec(context=ctx, attributes=attrs),
        SpanKind=SimpleNamespace(INTERNAL="I", CLIENT="C", SERVER="S", PRODUCER="P", CONSUMER="K"),
        Status=lambda code, desc: Rec(code=code, desc=desc),
        StatusCode=SimpleNamespace(OK="ok", ERROR="err", UNSET="unset"),
        Resource=lambda attrs: Rec(attributes=attrs),
        InstrumentationScope=lambda name, version: Rec(name=name),
    )
    sink = []
    fake_exporter = SimpleNamespace(export=lambda spans: sink.extend(spans), shutdown=lambda: None)
    bridge = OTelSpanExporter(fake_exporter, otel_modules=mods)
    tracer = OpsTracer([bridge])
    with tracer.span("root") as root:
        with tracer.span("child"):
            pass
    child = sink[0]
    assert child.context.trace_id == int(root.trace_id, 16) and child.parent.span_id == int(root.span_id, 16)
    assert child.kind == "I" and child.status.code == "unset"


def test_critical_path_counts_parallel_fanout_once():
    from harness_ops import CriticalPath, SpanData, SpanKind, critical_path

    tid = "4bf92f3577b34da6a3ce929d0e0e4736"

    def sd(sid, parent, start, end, op=None):
        return SpanData(f"s{sid}", SpanContext(tid, f"{sid:016x}"), f"{parent:016x}" if parent else None, SpanKind.INTERNAL, start, end,
                        {A.OPERATION_NAME: op} if op else {})

    spans = [
        sd(1, None, 0, 100, "invoke_agent"),
        sd(2, 1, 0, 10, "chat"),
        sd(3, 1, 10, 50, "execute_tool"), sd(4, 1, 12, 60, "execute_tool"), sd(5, 1, 15, 40, "execute_tool"),  # parallel fan-out
        sd(6, 1, 60, 70, "chat"),
        sd(7, 1, 70, 90, "execute_tool"),
        sd(8, 7, 71, 80, "chat"), sd(9, 7, 80, 89, "chat"),  # a subagent inside the tool call: sequential steps
    ]
    cp = critical_path(spans)
    assert cp == CriticalPath(steps=6, total_steps=8, duration_ms=100 / 1e6) and cp.parallelism == pytest.approx(8 / 6)
