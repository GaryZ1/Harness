import threading

import pytest
from context_fakes import GOOD_SUMMARY, TOOLS, RecordingSink, ScriptedClient, conversation, text_response, tool_turn

from harness_context import (
    CompactionConfig,
    CompactionError,
    CompactionInProgressError,
    Compactor,
    ConfigurationError,
    LogIntegrityError,
    NothingToCompactError,
    ReinjectionBuilder,
    TextSource,
    choose_cut,
    parse_summary,
)
from harness_context.messages import context_meta, is_safe_cut
from harness_core.types import Message, ModelResponse, RunContext, StopReason, TextBlock, ToolUseBlock, Usage

CTX = RunContext("s1")
SYSTEM = [{"type": "text", "text": "You are a coding agent."}]


def test_parse_summary_tagged_schema():
    s = parse_summary("preamble ignored?\n" + GOOD_SUMMARY)
    assert s.tagged and s.missing_sections == []
    assert "PKCE" in s.decisions and "small commits" in s.user_preferences and "src/auth.py" in s.files_touched
    rendered = s.render()
    assert rendered.index("## Task overview") < rendered.index("## Open tasks and next steps")


def test_parse_summary_missing_sections_and_untagged():
    s = parse_summary("<summary>\n## Task overview\nDo X\n## Next steps\nNone.\n</summary>")
    assert "open_tasks" in s.missing_sections and "user_preferences" in s.missing_sections
    assert s.render().count("None recorded.") >= 5  # every section always rendered
    bare = parse_summary("**Task overview**\nBuild it\n**Decisions**\nUse SQLite")
    assert not bare.tagged and bare.decisions == "Use SQLite"
    with pytest.raises(CompactionError):
        parse_summary("I could not summarize.")


def test_choose_cut_prefers_turn_boundaries_and_never_splits_pairs():
    log = conversation(4)
    cut = choose_cut(log, keep_turns=2)
    assert log[cut].text.startswith("request 2") and is_safe_cut(log, cut)
    # one giant turn: falls back to a safe boundary inside it, keeping >= 4 messages
    single = [Message.user("go")]
    for i in range(10):
        a, r = tool_turn("bash", {"command": f"step {i}"}, "ok")
        single += [a, r]
    cut = choose_cut(single, keep_turns=2)
    assert len(single) - cut >= 4 and is_safe_cut(single, cut) and single[cut].role.value == "assistant"
    with pytest.raises(NothingToCompactError):
        choose_cut([Message.user("hi"), Message.assistant([TextBlock("hello")])])


def test_compactor_builds_marker_with_schema_files_and_user_messages(clock):
    log = conversation(4, result_chars=3000)
    log.insert(3, Message.system("Operator: respond in British English."))
    sink = RecordingSink()
    reinj = ReinjectionBuilder([TextSource("instructions", "Use pytest."), TextSource("plan", lambda: "1. [ ] e2e test")])
    client = ScriptedClient(text_response(GOOD_SUMMARY.replace("- src/auth.py", "- src/auth.py\n- src/file_0_0.py")))
    comp = Compactor(client, reinjection=reinj, events=sink, clock=clock)
    res = comp.compact(system=SYSTEM, history=log, log=log, tools=TOOLS, ctx=CTX)
    req = client.requests[0]
    assert req.system == SYSTEM and req.tools == TOOLS  # same prefix as the session (cache sharing)
    assert "Do not call any tools" in req.messages[-1].text
    meta = context_meta(res.marker)
    assert meta["kind"] == "compaction" and meta["covers_through"] == res.covers_through
    assert meta["created"] == "2026-11-03T10:00:00Z" and len(meta["prefix_sha256"]) == 64
    body = res.marker.content[0].text
    assert "<summary>" in body and "## Files touched" in body
    assert body.count("src/file_0_0.py") == 1 and "src/file_1_0.py" in body  # harness-merged, deduped
    joined = "\n".join(b.text for b in res.marker.content)
    assert "request 0" in joined and "request 1" in joined and "request 3" not in joined  # tail kept verbatim
    assert "British English" in joined and "<post-compaction-context>" in joined and "Use pytest." in joined
    assert meta["reinjected"] == ["instructions", "plan"]
    assert sink.kinds() == ["client_compaction"] and res.tokens_after < res.tokens_before


def test_second_compaction_chains_previous_marker():
    log = conversation(4)
    comp = Compactor(ScriptedClient(text_response()))
    first = comp.compact(system=SYSTEM, history=log, log=log, tools=TOOLS, ctx=CTX)
    log.append(first.marker)
    log.extend(conversation(3))
    pos = next(i for i, m in enumerate(log) if m.id == first.covers_through)
    window = [first.marker] + [m for m in log[pos + 1:] if m is not first.marker]
    second = comp.compact(system=SYSTEM, history=window, log=log, tools=TOOLS, ctx=CTX)
    meta = context_meta(second.marker)
    assert meta["previous"] == first.marker.id
    assert any(u.startswith("request 0") for u in meta["user_messages"])  # carried from the first marker
    assert "src/file_0_0.py" in meta["files_touched"]  # files chain across compactions
    assert client_saw_previous_summary(comp)


def client_saw_previous_summary(comp):
    return "<summary>" in comp.client.requests[-1].messages[0].text


def test_compactor_retries_once_when_model_calls_a_tool_then_fails():
    tool_resp = ModelResponse([ToolUseBlock("x", "bash", {})], StopReason.TOOL_USE, Usage())
    client = ScriptedClient(tool_resp, text_response())
    log = conversation(3)
    res = Compactor(client).compact(system=SYSTEM, history=log, log=log, tools=TOOLS, ctx=CTX)
    assert len(client.requests) == 2 and "previous attempt called a tool" in client.requests[1].messages[-1].text
    assert res.summary.decisions
    with pytest.raises(CompactionError):
        Compactor(ScriptedClient(tool_resp)).compact(system=SYSTEM, history=log, log=log, tools=TOOLS, ctx=CTX)


def test_compactor_rejects_truncated_or_empty_output():
    log = conversation(3)
    for resp in (text_response(GOOD_SUMMARY, StopReason.MAX_TOKENS), text_response(""), text_response(stop=StopReason.REFUSAL)):
        with pytest.raises(CompactionError):
            Compactor(ScriptedClient(resp)).compact(system=SYSTEM, history=log, log=log, tools=TOOLS, ctx=CTX)


def test_compactor_fit_check_shrinks_tool_results_or_refuses():
    log = conversation(4, result_chars=20_000)
    client = ScriptedClient(text_response())
    Compactor(client).compact(system=SYSTEM, history=log, log=log, tools=TOOLS, ctx=CTX, max_request_tokens=15_000)
    assert "<cleared-tool-result" in str(client.requests[0].messages[2].to_api())
    with pytest.raises(CompactionError, match="too close"):
        Compactor(client).compact(system=SYSTEM, history=log, log=log, tools=TOOLS, ctx=CTX, max_request_tokens=8_050)


def test_single_flight_per_session():
    gate, entered = threading.Event(), threading.Event()

    def slow(_req):
        entered.set()
        gate.wait(5)
        return text_response()

    comp = Compactor(ScriptedClient(slow))
    log = conversation(3)
    t = threading.Thread(target=lambda: comp.compact(system=SYSTEM, history=log, log=log, tools=TOOLS, ctx=CTX))
    t.start()
    assert entered.wait(5)
    assert comp.in_progress("s1")
    with pytest.raises(CompactionInProgressError):
        comp.compact(system=SYSTEM, history=log, log=log, tools=TOOLS, ctx=CTX)
    comp.compact(system=SYSTEM, history=log, log=log, tools=TOOLS, ctx=RunContext("other"))  # other sessions unaffected
    gate.set()
    t.join(5)
    assert not comp.in_progress("s1")


def test_history_must_come_from_the_log_and_config_validation():
    log = conversation(3)
    with pytest.raises(LogIntegrityError):
        Compactor(ScriptedClient(text_response())).compact(system=SYSTEM, history=log, log=[], tools=TOOLS, ctx=CTX)
    with pytest.raises(ConfigurationError):
        CompactionConfig(keep_recent_turns=0)


def test_user_message_retention_budget():
    log = conversation(6)
    comp = Compactor(ScriptedClient(text_response()), config=CompactionConfig(retain_user_message_tokens=12))
    res = comp.compact(system=SYSTEM, history=log, log=log, tools=TOOLS, ctx=CTX)
    kept = context_meta(res.marker)["user_messages"]
    assert kept and kept[-1].startswith("request 3")  # newest summarized message kept first
    assert not any(k.startswith("request 0") for k in kept)
