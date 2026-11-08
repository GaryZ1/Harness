import copy

import pytest
from context_fakes import GOOD_SUMMARY, TOOLS, RecordingSink, ScriptedClient, conversation, text_response, tool_turn

from harness_context import (
    PRE_COMPACTION_WARNING,
    AssemblerConfig,
    BudgetPolicy,
    CacheStabilityError,
    ClearingPolicy,
    Compactor,
    ConfigurationError,
    ContextError,
    ContextOverflowError,
    DynamicContext,
    LayeredContextAssembler,
    LogIntegrityError,
    PromptStack,
    ReinjectionBuilder,
    ServerContextManagement,
    TextSource,
)
from harness_context.messages import count_breakpoints, is_compaction_marker, kind_of, message_fingerprint
from harness_core.types import (
    ContextAssembler,
    EventType,
    Message,
    ModelResponse,
    OpaqueBlock,
    Role,
    RunContext,
    StopReason,
    TextBlock,
    ToolResultBlock,
    Usage,
)

CTX = RunContext("sess")
SYSTEM = "You are a careful coding agent."


def durable(window):
    return [message_fingerprint(m) for m in window if kind_of(m) != "dynamic"]


def total_breakpoints(req):
    return count_breakpoints(req.messages) + sum(1 for b in req.system if b.get("cache_control")) + sum(
        1 for t in req.tools if t.metadata.get("cache_control"))


def test_layout_breakpoints_and_dynamic_tail():
    asm = LayeredContextAssembler(stack=PromptStack().add_instructions("Use pytest."))
    assert isinstance(asm, ContextAssembler)
    log = conversation(2)
    before = copy.deepcopy(log)
    req = asm.prepare(SYSTEM, log, list(reversed(TOOLS)), CTX)
    assert log == before  # durable log untouched
    assert [t.name for t in req.tools] == ["bash", "read_file"]
    assert list(req.tools[1].input_schema) == ["properties", "required", "type"]  # canonical key order
    assert req.system[0] == {"type": "text", "text": SYSTEM}
    assert req.system[-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    tail = req.messages[-1]
    assert tail.role is Role.SYSTEM and "Context budget" in tail.text and tail.metadata["harness_context"]["turn_scoped"]
    assert req.messages[-2].content[-1].cache_control is not None  # moving breakpoint on last durable message
    assert total_breakpoints(req) <= 4
    assert [m.id for m in req.messages[:-1]] == [m.id for m in log]


def test_append_only_turns_keep_the_prefix_stable():
    asm = LayeredContextAssembler()
    log = conversation(2)
    first = asm.prepare(SYSTEM, log, TOOLS, CTX)
    log += conversation(1)
    second = asm.prepare(SYSTEM, log, TOOLS, CTX)
    assert durable(second.messages)[: len(durable(first.messages))] == durable(first.messages)
    assert asm.report("sess").cache_breaks == []
    asm2 = LayeredContextAssembler()
    asm2.prepare(SYSTEM, log, TOOLS, CTX)
    asm2.prepare(SYSTEM + " Today is 2026-11-05.", log, TOOLS, CTX)
    rep = asm2.report("sess")
    assert [b.segment for b in rep.cache_breaks] == ["system"] and not rep.cache_breaks[0].expected
    assert any(f.kind == "date" for f in rep.findings)


def test_strict_cache_rejects_timestamps_in_prefix():
    asm = LayeredContextAssembler(config=AssemblerConfig(strict_cache=True))
    with pytest.raises(CacheStabilityError):
        asm.prepare(SYSTEM + " Now: 2026-11-05T10:00:00Z", conversation(1), TOOLS, CTX)


def test_clearing_is_sticky_and_logged_as_expected_rebuild():
    sink = RecordingSink()
    cfg = AssemblerConfig(clearing=ClearingPolicy(trigger_tokens=2_500, keep_recent=2, clear_at_least_tokens=500))
    asm = LayeredContextAssembler(config=cfg, events=sink)
    log = conversation(2, result_chars=2000)
    asm.prepare(SYSTEM, log, TOOLS, CTX)
    log += conversation(4, result_chars=2000)
    req = asm.prepare(SYSTEM, log, TOOLS, CTX)
    results = [b for m in req.messages for b in m.content if isinstance(b, ToolResultBlock)]
    assert sum("<cleared-tool-result" in r.content for r in results) == 4
    assert "<cleared-tool-result" not in results[-1].content and "<cleared-tool-result" not in results[-2].content
    rep = asm.report("sess")
    assert rep.cache_breaks and all(b.expected for b in rep.cache_breaks)
    assert "tool_result_clearing" in sink.kinds()
    log += [Message.user("small follow-up"), Message.assistant([TextBlock("ok")])]
    nxt = asm.prepare(SYSTEM, log, TOOLS, CTX)
    assert durable(nxt.messages)[: len(durable(req.messages))] == durable(req.messages)  # same placeholders again
    assert asm.report("sess").cache_breaks == []
    assert all("<cleared" not in b.text for m in log for b in m.tool_results)  # never written to the log


def _compacting(threshold=6_000, window=60_000, **cfg):
    client = ScriptedClient(text_response(GOOD_SUMMARY))
    sink = RecordingSink()
    comp = Compactor(client, events=sink, reinjection=ReinjectionBuilder([TextSource("instructions", "Use pytest.")]))
    config = AssemblerConfig(budget=BudgetPolicy(window_tokens=window, compaction_threshold_tokens=threshold), clearing=None, **cfg)
    return LayeredContextAssembler(config=config, compactor=comp, events=sink), client, sink


def test_auto_compaction_appends_marker_and_projects_window():
    asm, client, sink = _compacting()
    log = conversation(8, result_chars=4000)
    original = copy.deepcopy(log)
    req = asm.prepare(SYSTEM, log, TOOLS, CTX)
    assert len(client.requests) == 1 and log[:-1] == original and is_compaction_marker(log[-1])
    first = req.messages[0]
    assert first.id == log[-1].id and first.content[len(log[-1].content) - 1].cache_control  # compaction breakpoint
    assert first.metadata["merged_ids"][0] == log[-1].id  # marker merged with the first kept user turn
    texts = [m.text for m in req.messages[1:] if m.role is Role.USER and not m.tool_results]
    assert first.content[-1].text.startswith("request 6")  # kept tail starts right after the summary
    assert texts[0].startswith("request 7") and not any(t.startswith("request 0") for t in texts)
    assert "<post-compaction-context>" in first.text
    assert any("compacted" in a for a in asm.report("sess").actions)
    assert "client_compaction" in sink.kinds()
    log += conversation(1)
    nxt = asm.prepare(SYSTEM, log, TOOLS, CTX)
    assert len(client.requests) == 1 and nxt.messages[0].id == log[8 * 4].id  # marker still first, no new compaction
    assert asm.report("sess").cache_breaks == []
    assert total_breakpoints(nxt) <= 4


def test_thrash_guard_disables_auto_compaction():
    asm, client, sink = _compacting(threshold=1_000, max_ineffective_compactions=2)
    log = conversation(4, result_chars=8000)
    for _ in range(4):
        asm.prepare(SYSTEM, log, TOOLS, CTX)
    assert len(client.requests) == 2
    assert any(e.type is EventType.ERROR and e.data["kind"] == "compaction_thrash" for e in sink.events)


def test_unknown_window_never_auto_compacts_but_manual_works():
    client = ScriptedClient(text_response())
    asm = LayeredContextAssembler(
        config=AssemblerConfig(budget=BudgetPolicy(window_tokens=None, compaction_threshold_tokens=1_000), clearing=None),
        compactor=Compactor(client),
    )
    log = conversation(4, result_chars=4000)
    asm.prepare(SYSTEM, log, TOOLS, CTX)
    assert client.requests == []
    res = asm.compact(SYSTEM, log, TOOLS, CTX, focus="the auth bug")
    assert log[-1] is res.marker and "focus on: the auth bug" in client.requests[0].messages[-1].text
    log.append(Message.assistant([TextBlock("calling a tool")]))
    a, _ = tool_turn("bash", {"command": "ls"}, "x")
    log.append(a)
    with pytest.raises(ContextError):
        asm.compact(SYSTEM, log, TOOLS, CTX)


def test_overflow_raises_with_error_event():
    sink = RecordingSink()
    asm = LayeredContextAssembler(config=AssemblerConfig(budget=BudgetPolicy(window_tokens=20_000), clearing=None, max_tokens=4_000), events=sink)
    with pytest.raises(ContextOverflowError):
        asm.prepare(SYSTEM, conversation(10, result_chars=8000), TOOLS, CTX)
    assert sink.events[-1].type is EventType.ERROR


def test_pre_compaction_warning_once_per_cycle():
    asm = LayeredContextAssembler(config=AssemblerConfig(budget=BudgetPolicy(window_tokens=100_000, compaction_threshold_tokens=5_000), clearing=None))
    log = conversation(4, result_chars=4200)
    first = asm.prepare(SYSTEM, log, TOOLS, CTX)
    assert PRE_COMPACTION_WARNING in first.messages[-1].text
    second = asm.prepare(SYSTEM, log + [Message.user("more")], TOOLS, CTX)
    assert PRE_COMPACTION_WARNING not in second.messages[-1].text


def test_server_side_params_and_server_compaction_reinjection():
    cfg = AssemblerConfig(
        clearing=None,
        server=ServerContextManagement(clear_tool_uses=True, clear_thinking=True, compact=True),
        request_extra={"betas": ["other-beta"], "output_config": {"effort": "high"}},
    )
    sink = RecordingSink()
    asm = LayeredContextAssembler(config=cfg, events=sink, reinjection=ReinjectionBuilder([TextSource("memory-index", "- prefers tabs")]))
    log = conversation(2)
    log.append(Message.assistant([OpaqueBlock({"type": "compaction", "content": "server summary"})]))
    req = asm.prepare(SYSTEM, log, TOOLS, CTX)
    edits = req.extra["context_management"]["edits"]
    assert [e["type"] for e in edits] == ["clear_thinking_20251015", "clear_tool_uses_20250919", "compact_20260112"]
    assert req.extra["betas"] == ["compact-2026-01-12", "context-management-2025-06-27", "other-beta"]
    assert req.extra["output_config"] == {"effort": "high"}
    assert kind_of(log[-1]) == "reinjection" and "prefers tabs" in log[-1].text
    comp_msg = next(m for m in req.messages if any(isinstance(b, OpaqueBlock) for b in m.content))
    assert comp_msg.content[0].raw["cache_control"]["type"] == "ephemeral"
    assert "cache_control" not in log[-2].content[0].raw  # durable block never edited
    n = len(log)
    asm.prepare(SYSTEM, log, TOOLS, CTX)
    assert len(log) == n and sink.kinds().count("server_compaction") == 1
    # a fresh assembler (resume) must not re-append the bundle either
    LayeredContextAssembler(config=cfg, reinjection=ReinjectionBuilder([TextSource("m", "x")])).prepare(SYSTEM, log, TOOLS, CTX)
    assert len(log) == n


def test_config_conflicts_rejected():
    with pytest.raises(ConfigurationError):
        AssemblerConfig(server=ServerContextManagement(clear_tool_uses=True))  # default client clearing is on
    with pytest.raises(ConfigurationError):
        LayeredContextAssembler(config=AssemblerConfig(clearing=None, server=ServerContextManagement(compact=True)),
                                compactor=Compactor(ScriptedClient(text_response())))


def test_tampering_with_compacted_history_is_detected():
    asm, _, _ = _compacting()
    log = conversation(8, result_chars=4000)
    asm.prepare(SYSTEM, log, TOOLS, CTX)
    tampered = list(log)
    tampered[0] = Message(Role.USER, [TextBlock("rewritten history")], id=log[0].id)
    with pytest.raises(LogIntegrityError):
        _compacting()[0].prepare(SYSTEM, tampered, TOOLS, CTX)


def test_mid_history_breakpoint_for_large_turns():
    asm = LayeredContextAssembler()
    log = conversation(1)
    asm.prepare(SYSTEM, log, TOOLS, CTX)
    log += conversation(1, calls_per_turn=12)
    req = asm.prepare(SYSTEM, log, TOOLS, CTX)
    bps = asm.report("sess").breakpoints
    assert any(b.startswith("mid@") for b in bps) and total_breakpoints(req) <= 4
    assert req.messages[3].content[-1].cache_control  # previous tail position keeps a breakpoint


def test_state_persists_across_instances(tmp_path):
    cfg = AssemblerConfig(clearing=ClearingPolicy(trigger_tokens=2_500, keep_recent=2, clear_at_least_tokens=500))
    log = conversation(6, result_chars=2000)
    LayeredContextAssembler(config=cfg, state_dir=tmp_path).prepare(SYSTEM, log, TOOLS, CTX)
    assert (tmp_path / "sess.context-state.json").is_file()
    resumed = LayeredContextAssembler(config=AssemblerConfig(clearing=ClearingPolicy(trigger_tokens=10**6)), state_dir=tmp_path)
    req = resumed.prepare(SYSTEM, log, TOOLS, CTX)
    assert sum("<cleared-tool-result" in b.content for m in req.messages for b in m.tool_results) == 4


def test_observe_response_calibrates_and_flags_misses():
    asm = LayeredContextAssembler()
    log = conversation(2)
    asm.prepare(SYSTEM, log, TOOLS, CTX)
    est = asm.report("sess").estimated_tokens
    resp = ModelResponse([TextBlock("ok")], StopReason.END_TURN, Usage(input_tokens=10, cache_creation_input_tokens=est * 2))
    assert asm.observe_response(CTX, resp) is False  # first request is a write, not a miss
    asm.prepare(SYSTEM, log, TOOLS, CTX)
    assert 1.8 * est <= asm.report("sess").estimated_tokens <= 2.2 * est
    assert asm.observe_response(CTX, resp) is True  # no cache read, nothing expected
    asm.declare_rebuild(CTX, "tool set changed at session boundary")
    asm.prepare(SYSTEM, log, TOOLS[:1], CTX)
    assert asm.observe_response(CTX, resp) is False
    assert all(b.expected for b in asm.report("sess").cache_breaks)


def test_user_role_dynamic_tail_merges_after_breakpoint():
    asm = LayeredContextAssembler(dynamic=DynamicContext(role=Role.USER))
    log = [Message.user("go")]
    a, r = tool_turn("read_file", {"path": "a.py"}, "print(1)")
    log += [a, r]
    req = asm.prepare(SYSTEM, log, TOOLS, CTX)
    last = req.messages[-1]
    assert last.role is Role.USER and isinstance(last.content[0], ToolResultBlock)
    assert last.content[0].cache_control and last.content[-1].cache_control is None
    assert "<system-reminder>" in last.content[-1].text and len(req.messages) == 3


def test_recitation_last_and_stuck_detection():
    dyn = DynamicContext()
    dyn.register_recitation(lambda: "PLAN: 1. [ ] write test")
    dyn.register("git", lambda ctx: "git: clean")
    asm = LayeredContextAssembler(dynamic=dyn)
    log = [Message.user("go")]
    for _ in range(4):
        log += list(tool_turn("bash", {"command": "make"}, "make: *** error"))
    tail = asm.prepare(SYSTEM, log, TOOLS, CTX).messages[-1].text
    assert tail.index("git: clean") < tail.index("Context budget") < tail.index("PLAN:")
    assert "Loop check" in tail


def test_persistent_reminders_are_deduplicated_and_safe():
    asm = LayeredContextAssembler()
    log = conversation(1)
    assert asm.append_reminder(log, "nested:src/AGENTS.md", "src rules") is not None
    assert asm.append_reminder(log, "nested:src/AGENTS.md", "src rules") is None
    assert kind_of(log[-1]) == "reminder"
    a, _ = tool_turn("bash", {}, "")
    log.append(a)
    with pytest.raises(ContextError):
        asm.append_reminder(log, "k", "v")


def test_report_renders_categories():
    asm = LayeredContextAssembler(stack=PromptStack().add_instructions("Use pytest."))
    asm.prepare(SYSTEM, conversation(2), TOOLS, CTX)
    text = asm.report("sess").render()
    for cat in ("tools", "system", "project", "history", "tool_results", "dynamic", "breakpoints"):
        assert cat in text


def test_first_response_of_each_session_is_not_a_miss():
    asm = LayeredContextAssembler()
    write = ModelResponse([TextBlock("ok")], StopReason.END_TURN, Usage(input_tokens=10, cache_creation_input_tokens=5_000))
    for sid in ("a", "b"):
        ctx = RunContext(sid)
        asm.prepare(SYSTEM, conversation(1), TOOLS, ctx)
        assert asm.observe_response(ctx, write) is False
    assert asm.cache_monitor.stats.unexpected_misses == 0
