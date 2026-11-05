"""LayeredContextAssembler as the loop's assembler: one coherent compaction design, markers persisted durably."""

from __future__ import annotations

import logging

from integration_support import RoutingClient, approve_all_fixture, big_output_tool_turn, build, events_of, text_response  # noqa: F401

from harness import OVERFLOW_KEY, OverflowGuardClient
from harness_context.messages import META_KEY, is_compaction_marker
from harness_core.errors import ContextWindowExceededError
from harness_core.results import ResultSubtype
from harness_core.testing import ScriptedModelClient
from harness_core.types import EventType, Message, ModelRequest

logging.getLogger("harness_sandbox").setLevel(logging.ERROR)


def test_threshold_compaction_is_owned_by_the_assembler_and_persisted(tmp_path, approve_all):
    client = RoutingClient([big_output_tool_turn(i) for i in range(6)] + [text_response("final report")])
    h, _ = build(tmp_path, client, context_window_tokens=40_000, compaction_threshold_tokens=3_000, max_tokens=2_000,
                 approver=approve_all)
    result = h.run("Inspect the outputs and report.")

    assert result.subtype is ResultSubtype.SUCCESS and result.final_text == "final report"
    assert len(client.summary_requests) == 1  # one client compaction through harness_context.Compactor
    # the marker was appended to the DURABLE session (append_to_log), not only to the loop's copy
    markers = [m for m in h.session.messages() if is_compaction_marker(m)]
    assert len(markers) == 1 and markers[0].metadata[META_KEY]["trigger"] == "auto"
    # later requests start with the marker; earlier ones did not
    firsts = [is_compaction_marker(r.messages[0]) for r in client.requests]
    assert firsts[0] is False and firsts[-1] is True
    # the loop's own threshold compaction is off (assembler owns auto-compaction)
    assert h.loop.config.compaction_threshold is None and not h.session.compacted
    comp = [e.data for e in events_of(h, EventType.COMPACTION)]
    assert [c.get("kind") for c in comp] == ["client_compaction"]
    assert comp[0]["tokens_after"] < comp[0]["tokens_before"]
    # survives a process restart: the woken session projects through the marker (prefix hash verified)
    woken = h.store.wake(h.session.id)
    derived = h.assembler.derive_history(woken.messages())
    assert is_compaction_marker(derived[0]) and len(derived) < len(woken.messages())
    h.end_session()


def test_provider_overflow_uses_the_core_compactor_adapter(tmp_path, approve_all):
    script = [big_output_tool_turn(i) for i in range(4)] + [ContextWindowExceededError("prompt is too long"), text_response("done")]
    client = RoutingClient(script)
    h, _ = build(tmp_path, client, approver=approve_all)
    result = h.run("Inspect the outputs.")

    assert result.subtype is ResultSubtype.SUCCESS and result.final_text == "done"
    assert len(client.summary_requests) == 1
    comp = [e.data for e in events_of(h, EventType.COMPACTION)]
    loop_record = next(c for c in comp if c.get("delegated_to") == "harness_context")
    assert loop_record["reason"] == "overflow" and loop_record["tokens_after"] < loop_record["tokens_before"]
    assert "failed" not in loop_record  # the loop's "compaction freed nothing" / "did not shrink" checks passed
    assert h.session.compacted  # Session.record_compaction recorded the derived view ...
    view = h.session.view()
    assert is_compaction_marker(view[0])  # ... which starts with the assembler's marker
    retry = client.requests[-1]
    assert is_compaction_marker(retry.messages[0]) and len(retry.messages) < len(client.requests[-2].messages)
    # metrics count the compaction once (the delegated loop record is not double-counted)
    assert h.ops_metrics.compactions.sum() == 1
    h.end_session()


def test_local_overflow_takes_the_typed_overflow_path(tmp_path):
    client = ScriptedModelClient([text_response("unreachable")])
    # a window smaller than the tool definitions alone: prepare() cannot fit even the first request
    h, _ = build(tmp_path, client, context_window_tokens=1_500, max_tokens=500)
    result = h.run("hello")
    assert result.subtype is ResultSubtype.ERROR_CONTEXT_WINDOW  # typed, not error_during_execution
    assert client.calls == 0  # the provider was never called with a request known not to fit
    errors = [e.data for e in events_of(h, EventType.ERROR)]
    assert any(d.get("error") == "context_window_exceeded" for d in errors)
    h.end_session()


def test_overflow_guard_client_unit():
    inner = ScriptedModelClient([text_response("ok")])
    guard = OverflowGuardClient(inner)
    try:
        guard.create(ModelRequest(system="", messages=[Message.user("x")], extra={OVERFLOW_KEY: "too big"}))
    except ContextWindowExceededError as exc:
        assert "too big" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ContextWindowExceededError")
    assert inner.calls == 0
    assert guard.create(ModelRequest(system="", messages=[Message.user("x")])).text == "ok"


def test_observe_response_is_called_after_every_model_call(tmp_path, approve_all):
    client = RoutingClient([big_output_tool_turn(0), big_output_tool_turn(1), text_response("done")])
    h, _ = build(tmp_path, client, approver=approve_all)
    h.run("go")
    assert h.assembler.cache_monitor.stats.requests == client.main.calls == 3
    report = h.context_report()
    assert report is not None and report.session_id == h.session.id
    h.end_session()


def test_prompt_stack_is_populated_from_memory_session_start_context(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "AGENTS.md").write_text("# Project rules\nAlways run the linter before committing.\n")
    client = ScriptedModelClient([text_response("ok")])
    h, _ = build(tmp_path, client)
    h.run("hi")
    system = [b["text"] for b in client.requests[0].system]
    joined = "\n".join(system)
    assert "Always run the linter before committing." in joined  # instruction memory
    assert "<memory-protocol>" in joined and "memory-citation" in joined  # memory protocol + citation instructions
    assert system[0].startswith("You are a careful software engineering agent")  # base prompt first
    assert any(b.get("cache_control") for b in client.requests[0].system)  # breakpoint at the end of the stable prefix
    h.end_session()
