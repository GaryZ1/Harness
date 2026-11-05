import json

import pytest

from harness_memory import RolloutSummaryStore, SessionRegistry, TranscriptLog, ValidationError
from harness_memory.security import TaintTracker


def test_transcript_append_only_and_verify(tmp_path, clock):
    log = TranscriptLog(tmp_path / "s.jsonl", clock=clock)
    log.append("user", "fix the CI please")
    log.append("assistant", {"content": "on it"})
    evs = log.events()
    assert [e.seq for e in evs] == [0, 1] and evs[1].prev_hash == evs[0].hash
    assert log.verify()
    # Tamper: edit the first line.
    lines = (tmp_path / "s.jsonl").read_text().splitlines()
    obj = json.loads(lines[0])
    obj["payload"]["content"] = "something else"
    lines[0] = json.dumps(obj)
    (tmp_path / "s.jsonl").write_text("\n".join(lines) + "\n")
    assert not TranscriptLog(tmp_path / "s.jsonl").verify()


def test_transcript_search_and_reopen(tmp_path, clock):
    p = tmp_path / "s.jsonl"
    TranscriptLog(p, clock=clock).append("user", "deploy to staging")
    log2 = TranscriptLog(p, clock=clock)
    log2.append("tool_result", "ERROR: missing env FOO")
    assert [e.seq for e in log2.events()] == [0, 1] and log2.verify()
    assert log2.search("missing env")[0].kind == "tool_result"
    assert log2.last_activity == clock.now


def test_registry_flags_and_taint(tmp_path, clock):
    reg = SessionRegistry(tmp_path, clock=clock)
    reg.start("s1")
    reg.start("s2", subagent=True)
    with pytest.raises(ValidationError):
        reg.start("s1")
    with pytest.raises(ValidationError):
        reg.start("../evil")
    assert reg.get("s1").generation_block_reason() is None
    assert reg.get("s2").generation_block_reason() == "sub-agent session"
    t = TaintTracker()
    t.record_tool_output("web_fetch")
    reg.apply_taint("s1", t)
    assert reg.get("s1").polluted and "web_fetch" in reg.get("s1").pollution_reasons
    reg.end("s1")
    assert reg.get("s1").ended


def test_registry_transcript_redacts(tmp_path, clock):
    reg = SessionRegistry(tmp_path, clock=clock)
    reg.start("s1")
    reg.transcript("s1").append("user", "my key is ghp_abcdefghijklmnopqrstuvwxyz0123456789")
    assert "ghp_" not in (tmp_path / "sessions" / "s1.jsonl").read_text()


def test_rollout_summaries(store):
    rs = RolloutSummaryStore(store)
    path = rs.write("s1", "Fix CI!", "Fixed flaky test by pinning tz.", outcome="success")
    assert path == "rollout_summaries/s1_fix-ci.md"
    (item,) = rs.for_session("s1")
    assert item.outcome == "success" and "pinning tz" in item.body
    with pytest.raises(ValidationError):
        rs.write("s2", "x", "y", outcome="great")
    assert rs.delete_for_session("s1") == [path]
