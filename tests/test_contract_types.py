"""Tests for the shared cross-package contract in harness_core.types."""

from harness_core.types import (
    Event,
    EventType,
    Message,
    NullTracer,
    OpaqueBlock,
    PermissionBehavior,
    PermissionDecision,
    Role,
    StopReason,
    TextBlock,
    ToolOutcome,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
    block_from_api,
)


def test_message_roundtrip_keeps_opaque_blocks():
    api = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": "hi"},
            {"type": "tool_use", "id": "tu_1", "name": "read", "input": {"path": "a"}},
        ],
    }
    m = Message.from_api(api)
    assert isinstance(m.content[0], OpaqueBlock) and m.content[0].type == "thinking"
    assert m.text == "hi" and m.tool_uses[0].name == "read"
    assert m.to_api() == api


def test_user_and_system_messages():
    assert Message.user("x").to_api() == {"role": "user", "content": [{"type": "text", "text": "x"}]}
    assert Message.system("op").role is Role.SYSTEM
    assert Message.from_api({"role": "user", "content": "plain"}).text == "plain"


def test_tool_result_block_and_outcome():
    b = ToolOutcome("boom", is_error=True).to_block("tu_1")
    assert b.to_api() == {"type": "tool_result", "tool_use_id": "tu_1", "content": "boom", "is_error": True}
    assert ToolResultBlock("t", [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]).text == "ab"


def test_block_from_sdk_like_object():
    class Obj:
        def model_dump(self):
            return {"type": "tool_use", "id": "x", "name": "n", "input": {"a": 1}, "caller": None}

    b = block_from_api(Obj())
    assert isinstance(b, ToolUseBlock) and b.input == {"a": 1}
    assert block_from_api({"type": "text", "text": "t"}) == TextBlock("t")


def test_toolspec_to_api():
    s = ToolSpec("read", "Read a file", {"type": "object", "properties": {"path": {"type": "string"}}}, strict=True, defer_loading=True)
    d = s.to_api()
    assert d["strict"] is True and d["defer_loading"] is True and "input_schema" in d
    assert ToolSpec("memory", "", provider_type="memory_20250818").to_api() == {"type": "memory_20250818", "name": "memory"}


def test_usage_stop_reason_permission():
    u = Usage(10, 5, 100, 1) + Usage(1, 1)
    assert u.input_tokens == 11 and u.total_input == 112
    assert StopReason.parse("pause_turn") is StopReason.PAUSE_TURN
    assert StopReason.parse("something_new") is StopReason.OTHER
    assert PermissionDecision(PermissionBehavior.ALLOW).allowed


def test_event_json_roundtrip_and_null_tracer():
    e = Event("s1", EventType.TOOL_CALL, {"name": "read"}, seq=3, ts="2026-11-05T00:00:00Z")
    back = Event.from_json(e.to_json())
    assert back == e
    with NullTracer().span("x", a=1):
        pass
