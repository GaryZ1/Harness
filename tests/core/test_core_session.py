"""Session log: durable append-only JSONL events, replay, compaction projection, fork, lease."""

from __future__ import annotations

import json
import threading

import pytest

from harness_core import (
    CorruptLogError,
    Event,
    EventType,
    InvalidSessionIdError,
    JsonlEventLog,
    Message,
    OpaqueBlock,
    Session,
    SessionExistsError,
    SessionLockedError,
    SessionNotFoundError,
    SessionStatus,
    SessionStore,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    message_from_dict,
    message_to_dict,
)


def test_jsonl_log_assigns_contiguous_seq_and_timestamps(tmp_path, clock):
    log = JsonlEventLog(tmp_path / "s.jsonl", "s1", clock=clock)
    events = [log.emit(Event("s1", EventType.CUSTOM, {"i": i})) for i in range(5)]
    assert [e.seq for e in events] == [0, 1, 2, 3, 4]
    assert all(e.ts == clock().isoformat() for e in events)
    lines = (tmp_path / "s.jsonl").read_text().splitlines()
    assert [json.loads(line)["seq"] for line in lines] == [0, 1, 2, 3, 4]
    assert [e.data["i"] for e in log.events(1, 3)] == [1, 2]  # positional slice like getEvents()


def test_jsonl_log_rejects_foreign_session_events(tmp_path):
    log = JsonlEventLog(tmp_path / "s.jsonl", "s1")
    with pytest.raises(ValueError):
        log.emit(Event("other", EventType.CUSTOM, {}))


def test_reopened_log_continues_numbering(tmp_path):
    path = tmp_path / "s.jsonl"
    JsonlEventLog(path, "s1").emit(Event("s1", EventType.CUSTOM, {}))
    log2 = JsonlEventLog(path, "s1")
    assert log2.last_seq == 0
    assert log2.emit(Event("s1", EventType.CUSTOM, {})).seq == 1


def test_two_writers_on_one_file_interleave_without_gaps(tmp_path):
    path = tmp_path / "s.jsonl"
    a, b = JsonlEventLog(path, "s1"), JsonlEventLog(path, "s1")  # e.g. two processes
    seqs = []
    for i in range(6):
        seqs.append((a if i % 2 == 0 else b).emit(Event("s1", EventType.CUSTOM, {"i": i})).seq)
    assert seqs == list(range(6))
    assert [e.seq for e in a.events()] == list(range(6))


def test_concurrent_threads_append_safely(tmp_path):
    log = JsonlEventLog(tmp_path / "s.jsonl", "s1", fsync=False)

    def work():
        for _ in range(25):
            log.emit(Event("s1", EventType.CUSTOM, {}))

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [e.seq for e in log.events()] == list(range(100))


def test_torn_tail_is_truncated_on_open(tmp_path):
    path = tmp_path / "s.jsonl"
    log = JsonlEventLog(path, "s1")
    log.emit(Event("s1", EventType.CUSTOM, {"ok": 1}))
    with open(path, "a") as fh:
        fh.write('{"session_id": "s1", "type": "cus')  # crash mid-write
    reopened = JsonlEventLog(path, "s1")
    assert reopened.repaired_bytes > 0
    assert [e.seq for e in reopened.events()] == [0]
    assert reopened.emit(Event("s1", EventType.CUSTOM, {})).seq == 1


def test_corruption_in_the_middle_raises(tmp_path):
    path = tmp_path / "s.jsonl"
    log = JsonlEventLog(path, "s1")
    for _ in range(3):
        log.emit(Event("s1", EventType.CUSTOM, {}))
    lines = path.read_text().splitlines()
    lines[1] = "not json"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(CorruptLogError):
        JsonlEventLog(path, "s1").events()


def test_seq_gap_is_detected(tmp_path):
    path = tmp_path / "s.jsonl"
    log = JsonlEventLog(path, "s1")
    for _ in range(3):
        log.emit(Event("s1", EventType.CUSTOM, {}))
    lines = path.read_text().splitlines()
    del lines[1]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(CorruptLogError, match="expected seq 1"):
        log.events()


def test_message_serialization_is_lossless_for_opaque_blocks():
    thinking = OpaqueBlock({"type": "thinking", "thinking": "hmm", "signature": "sig=="})
    msg = Message.assistant([thinking, TextBlock("hi", cache_control={"type": "ephemeral"}), ToolUseBlock("tu1", "read", {"p": 1})],
                            stop_reason="tool_use")
    back = message_from_dict(json.loads(json.dumps(message_to_dict(msg))))
    assert back.id == msg.id and back.metadata == {"stop_reason": "tool_use"}
    assert back.to_api() == msg.to_api()


def test_store_lifecycle_replay_and_status(tmp_path, clock):
    store = SessionStore(tmp_path, clock=clock)
    s = store.create("sess-1", metadata={"user": "u1"})
    with pytest.raises(SessionExistsError):
        store.create("sess-1")
    s.append_message(Message.user("hello"))
    s.append_message(Message.assistant([OpaqueBlock({"type": "thinking", "thinking": "", "signature": "x"}), TextBlock("hi")]))
    s.emit(EventType.CUSTOM, {"note": "observability events are not transcript"})
    s.set_status(SessionStatus.IDLE)
    woken = store.wake("sess-1")
    assert [m.to_api() for m in woken.messages()] == [m.to_api() for m in s.messages()]
    assert woken.info.status is SessionStatus.IDLE and woken.info.metadata == {"user": "u1"}
    assert [i.session_id for i in store.list()] == ["sess-1"]
    with pytest.raises(SessionNotFoundError):
        store.open("missing")


@pytest.mark.parametrize("bad", ["../x", "a/b", "", ".hidden", "a b"])
def test_invalid_session_ids_are_rejected(tmp_path, bad):
    with pytest.raises(InvalidSessionIdError):
        SessionStore(tmp_path).create(bad)


def test_compaction_is_a_projection_and_canonical_history_is_kept(tmp_path):
    store = SessionStore(tmp_path)
    s = store.create("s")
    for i in range(3):
        s.append_message(Message.user(f"u{i}"))
        s.append_message(Message.assistant([TextBlock(f"a{i}")]))
    s.record_compaction([Message.user("SUMMARY")], summary="SUMMARY", reason="threshold")
    s.append_message(Message.user("after"))
    assert [m.text for m in s.view()] == ["SUMMARY", "after"]
    assert len(s.messages()) == 7
    reopened = store.open("s")
    assert [m.text for m in reopened.view()] == ["SUMMARY", "after"]
    assert reopened.compacted


def test_compaction_with_mismatched_prefix_hash_is_ignored(tmp_path):
    store = SessionStore(tmp_path)
    s = store.create("s")
    s.append_message(Message.user("u0"))
    ev = s.record_compaction([Message.user("SUMMARY")])
    path = tmp_path / "s.jsonl"
    lines = path.read_text().splitlines()
    rec = json.loads(lines[ev.seq])
    rec["data"]["prefix_hash"] = "0" * 64
    lines[ev.seq] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n")
    assert [m.text for m in store.open("s").view()] == ["u0"]


def test_fork_copies_prefix_and_refuses_to_split_a_tool_batch(tmp_path):
    store = SessionStore(tmp_path)
    s = store.create("src")
    s.append_message(Message.user("go"))
    e_tool = s.append_message(Message.assistant([ToolUseBlock("t1", "read", {})]))
    s.append_message(Message.user([ToolResultBlock("t1", "ok")]))
    cp = s.checkpoint("after-read")
    s.append_message(Message.assistant([TextBlock("done")]))
    with pytest.raises(ValueError, match="splits a tool batch"):
        store.fork("src", at_seq=e_tool.seq)
    assert s.find_checkpoint("after-read") == cp.seq
    f = store.fork("src", at_seq=cp.seq, new_session_id="fork")
    assert [m.role.value for m in f.messages()] == ["user", "assistant", "user"]
    assert f.info.parent_id == "src" and f.info.forked_at_seq == cp.seq
    assert len(store.open("src").messages()) == 4  # source untouched


def test_lease_is_exclusive_across_handles(tmp_path):
    store = SessionStore(tmp_path)
    store.create("s")
    a, b = store.open("s"), store.open("s")
    with a.lease():
        with pytest.raises(SessionLockedError):
            with b.lease():
                pass
    with b.lease():  # released
        pass
    mem = Session.in_memory()
    with mem.lease():
        with pytest.raises(SessionLockedError):
            with mem.lease():
                pass


def test_pending_tool_batch_classifies_calls(tmp_path):
    s = SessionStore(tmp_path).create("s")
    s.append_message(Message.user("go"))
    s.append_message(Message.assistant([ToolUseBlock("a", "read", {}), ToolUseBlock("b", "write", {}), ToolUseBlock("c", "read", {})]))
    s.emit(EventType.TOOL_CALL, {"phase": "executing", "tool_use_id": "a"})
    s.emit(EventType.TOOL_RESULT, {"tool_use_id": "a", "block": ToolResultBlock("a", "A").to_api()})
    s.emit(EventType.TOOL_CALL, {"phase": "executing", "tool_use_id": "b"})
    p = s.pending_tool_batch()
    assert p is not None
    assert set(p.completed) == {"a"} and p.completed["a"].text == "A"
    assert p.started == {"b"}
    assert [u.id for u in p.not_started] == ["c"]
    s.append_message(Message.user([ToolResultBlock("a", "A"), ToolResultBlock("b", "x"), ToolResultBlock("c", "C")]))
    assert s.pending_tool_batch() is None
