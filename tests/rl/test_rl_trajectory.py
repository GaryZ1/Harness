from __future__ import annotations

import json

import pytest

from harness_core.types import Event, EventType, Message, OpaqueBlock, TextBlock, ToolResultBlock, ToolUseBlock, Usage
from harness_evals import Trajectory
from harness_rl import TrainingRecord, TrajectoryExporter, TrajectorySchemaError, from_events, from_messages, load_jsonl
from harness_rl.trajectory import tool_results_in_user_text

SECRET = "sk-ant-abcdefghijklmnopqrstuvwxyz0123"


def convo():
    return [
        Message.user("fix it"),
        Message.assistant([OpaqueBlock({"type": "thinking", "thinking": "plan", "signature": "sig-eyJabcdefghijk.eyJabcdefghijk.abcdefghijkl"}),
                           ToolUseBlock("tu_1", "bash", {"command": f"export TOKEN={SECRET}"})]),
        Message.user([ToolResultBlock("tu_1", "ok")]),
        Message.assistant([TextBlock("done")]),
    ]


def test_from_messages_pairs_tool_calls_and_validates():
    rec = from_messages(convo(), rollout_id="r1", task_id="t1", reward=1.0, seed=3).validate()
    assert rec.tool_calls[0]["name"] == "bash" and rec.tool_calls[0]["output"] == "ok"
    assert rec.turns == 2 and rec.schema == "harness_rl.trajectory/v1"


def test_validate_rejects_unpaired_and_misplaced_tool_results():
    msgs = convo()[:2]
    with pytest.raises(TrajectorySchemaError):
        from_messages(msgs, rollout_id="r", task_id="t").validate()
    from_messages(msgs, rollout_id="r", task_id="t", metadata={"truncated": True}).validate()
    bad = TrainingRecord("r", "t", [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "nope", "content": "x"}]}])
    with pytest.raises(TrajectorySchemaError):
        bad.validate()
    dup = TrainingRecord("r", "t", [{"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "x", "input": {}}] * 2}])
    with pytest.raises(TrajectorySchemaError):
        dup.validate()


def test_openai_export_uses_tool_role_never_user_text():
    chat = from_messages(convo(), rollout_id="r", task_id="t", system="sys").to_openai_chat()
    assert chat[0] == {"role": "system", "content": "sys"}
    tool_msgs = [m for m in chat if m["role"] == "tool"]
    assert tool_msgs == [{"role": "tool", "tool_call_id": "tu_1", "content": "ok"}]
    assert not any(m["role"] == "user" and "ok" == m.get("content") for m in chat)
    asst = [m for m in chat if m.get("tool_calls")][0]
    assert asst["tool_calls"][0]["function"]["name"] == "bash" and asst["reasoning_blocks"][0]["type"] == "thinking"


def test_exporter_redacts_secrets_but_keeps_opaque_signatures(tmp_path):
    p = tmp_path / "out" / "traj.jsonl"
    rec = from_messages(convo(), rollout_id="r", task_id="t")
    TrajectoryExporter(p).write([rec])
    text = p.read_text()
    assert SECRET not in text and "REDACTED" in text
    assert "sig-eyJabcdefghijk.eyJabcdefghijk.abcdefghijkl" in text  # signature must round-trip byte-exact
    back = list(load_jsonl(p))
    assert back[0].rollout_id == "r" and back[0].validate()
    TrajectoryExporter(p, fmt="openai_chat").append(rec)
    lines = p.read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[1])["messages"][0]["role"] == "user"


def test_exporter_validates_everything_before_writing(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text("previous\n")
    bad = TrainingRecord("", "t", [])
    with pytest.raises(TrajectorySchemaError):
        TrajectoryExporter(p).write([from_messages(convo(), rollout_id="r", task_id="t"), bad])
    assert p.read_text() == "previous\n"  # atomic: nothing partially written


def test_from_events_rebuilds_native_transcript():
    evs = [
        Event("s", EventType.SESSION_START, {"system": "be good"}, seq=0),
        Event("s", EventType.USER_MESSAGE, {"text": "hi"}, seq=1),
        Event("s", EventType.MODEL_RESPONSE, {"content": [{"type": "tool_use", "id": "a", "name": "read", "input": {"p": "x"}}],
                                              "usage": {"input_tokens": 10, "output_tokens": 2}, "model": "test-model"}, seq=2),
        Event("s", EventType.TOOL_RESULT, {"tool_use_id": "a", "content": "file body"}, seq=3),
        Event("s", EventType.COMPACTION, {}, seq=4),
        Event("s", EventType.MODEL_RESPONSE, {"content": [{"type": "text", "text": "done"}], "usage": {"input_tokens": 5}}, seq=5),
        Event("s", EventType.STOP, {"stop_reason": "end_turn", "cost_usd": 0.01}, seq=6),
    ]
    rec = from_events(evs, rollout_id="r", task_id="t").validate()
    assert rec.system == "be good" and rec.model == "test-model" and rec.usage["input_tokens"] == 15
    assert rec.messages[2] == {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "file body"}]}
    traj = Trajectory.from_events(evs)
    assert traj.compaction_events == 1 and traj.cost_usd == 0.01 and traj.usage == Usage(15, 2)


def test_regression_check_flags_tool_output_sent_as_user_text():
    good = convo()
    bad = [Message.user("go"), Message.assistant([ToolUseBlock("x", "bash", {})]), Message.user("Tool output: ok")]
    assert tool_results_in_user_text(good) == [] and tool_results_in_user_text(bad) == [2]
