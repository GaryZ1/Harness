from __future__ import annotations

import subprocess
import threading
import time

import pytest

from harness_core.types import Event
from harness_evals import (
    CallableAgent,
    EvalRunner,
    InfraError,
    MemoryPolicy,
    ResultStore,
    RunConditions,
    RunConfig,
    Suite,
    TrialStatus,
    parse_task,
)

COND = RunConditions(harness_version="h@1", model="test-model", effort="high")


def suite(*tasks):
    return Suite("s", "1", tuple(parse_task(t) for t in tasks))


WRITE = {"id": "write", "instruction": "write out.txt", "trials": 3, "graders": [{"kind": "file_regex", "path": "out.txt", "pattern": "ok"}]}


def runner(tmp_path, fn, **cfg):
    cfg.setdefault("concurrency", 2)
    return EvalRunner(CallableAgent(fn), store=ResultStore(tmp_path / "results"), conditions=COND,
                      config=RunConfig(workspace_root=tmp_path / "ws", **cfg))


def test_end_to_end_pass_and_records(tmp_path):
    def agent(task, ws, ctx):
        assert task.instruction == "write out.txt" and not hasattr(task, "graders")
        (ws / "out.txt").write_text("ok")
        return "finished"

    res = runner(tmp_path, agent).run(suite(WRITE), run_id="r1")
    assert len(res.records) == 3 and all(r["passed"] and r["status"] == "completed" for r in res.records)
    assert len({r["seed"] for r in res.records}) == 3
    assert res.header["conditions"]["model"] == "test-model" and res.header["suite"]["hash"]
    assert all(r["workspace_path"] is None for r in res.records)  # passing workspaces cleaned up
    assert ResultStore(tmp_path / "results").read_transcript("r1", "write.t0")["final_text"] == "finished"


def test_trials_are_isolated_no_shared_state_or_git(tmp_path):
    seen = []
    lock = threading.Lock()

    def agent(task, ws, ctx):
        with lock:
            seen.append(sorted(p.name for p in ws.iterdir()))
        (ws / "leftover.txt").write_text("from a previous trial")
        (ws / "out.txt").write_text("ok")
        (ws.parent / "memory" / "note.md").write_text("remember")
        r = subprocess.run(["git", "rev-parse", "--git-dir"], cwd=ws, env=ctx.metadata["env"], capture_output=True)
        assert r.returncode != 0, "trial must not see any enclosing git repository"
        assert ctx.metadata["memory_dir"].endswith("memory") and not any((ws.parent / "memory").glob("old*"))

    res = runner(tmp_path, agent, concurrency=1).run(suite(WRITE), run_id="iso")
    assert all(s == [] for s in seen)
    assert all(r["passed"] for r in res.records)


def test_git_init_gives_fresh_single_commit_repo(tmp_path):
    t = WRITE | {"id": "git", "trials": 2, "setup": {"files": {"a.txt": "1"}, "git_init": True}}

    def agent(task, ws, ctx):
        r = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=ws, env=ctx.metadata["env"], capture_output=True, text=True)
        assert r.stdout.strip() == "1"
        subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "agent"], cwd=ws, env=ctx.metadata["env"], check=True)
        (ws / "out.txt").write_text("ok")

    assert all(r["passed"] for r in runner(tmp_path, agent).run(suite(t), run_id="g").records)


def test_infra_errors_retried_and_recorded_agent_errors_not(tmp_path):
    calls = {"n": 0}

    def flaky(task, ws, ctx):
        calls["n"] += 1
        if ctx.metadata["attempt"] == 0:
            raise InfraError("provider 503")
        (ws / "out.txt").write_text("ok")

    res = runner(tmp_path, flaky, concurrency=1).run(suite(WRITE | {"trials": 1}), run_id="infra")
    r = res.records[0]
    assert r["passed"] and r["infra_retries"] == 1 and r["attempts"][0]["infra"] is True and calls["n"] == 2

    calls["n"] = 0

    def broken(task, ws, ctx):
        calls["n"] += 1
        raise ValueError("agent bug")

    r2 = runner(tmp_path, broken).run(suite(WRITE | {"trials": 1}), run_id="agent-err").records[0]
    assert r2["status"] == TrialStatus.AGENT_ERROR.value and not r2["passed"] and calls["n"] == 1

    def always_infra(task, ws, ctx):
        raise InfraError("sandbox lost")

    r3 = runner(tmp_path, always_infra, max_infra_retries=1).run(suite(WRITE | {"trials": 1}), run_id="infra-x").records[0]
    assert r3["status"] == "infra_error" and len(r3["attempts"]) == 2 and not r3["passed"]


def test_setup_failure_is_infra_and_retried(tmp_path):
    t = WRITE | {"id": "setup", "trials": 1, "setup": {"commands": ["exit 7"]}}
    r = runner(tmp_path, lambda *a: None, max_infra_retries=1).run(suite(t), run_id="setup").records[0]
    assert r["status"] == "infra_error" and all(a["infra"] for a in r["attempts"]) and len(r["attempts"]) == 2


def test_timeout_sets_cancel_event_and_is_not_retried(tmp_path):
    cancelled = threading.Event()

    def slow(task, ws, ctx):
        if ctx.metadata["cancel_event"].wait(5):
            cancelled.set()

    r = runner(tmp_path, slow).run(suite(WRITE | {"trials": 1, "timeout_s": 0.2}), run_id="to").records[0]
    assert r["status"] == "timeout" and not r["passed"] and len(r["attempts"]) == 1
    assert cancelled.wait(2)


def test_concurrency_cap_respected(tmp_path):
    active, peak = [0], [0]
    lock = threading.Lock()

    def agent(task, ws, ctx):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        (ws / "out.txt").write_text("ok")

    runner(tmp_path, agent, concurrency=2).run(suite(WRITE | {"trials": 6}), run_id="cc")
    assert peak[0] == 2


def test_resume_skips_completed_trials_and_rejects_changed_conditions(tmp_path):
    calls = []

    def agent(task, ws, ctx):
        calls.append(ctx.metadata["trial_id"])
        (ws / "out.txt").write_text("ok")

    r = runner(tmp_path, agent)
    r.run(suite(WRITE), run_id="res")
    r.run(suite(WRITE), run_id="res")
    assert len(calls) == 3
    other = EvalRunner(CallableAgent(agent), store=ResultStore(tmp_path / "results"), conditions=RunConditions("h@2", "test-model"),
                       config=RunConfig(workspace_root=tmp_path / "ws"))
    with pytest.raises(Exception, match="different conditions"):
        other.run(suite(WRITE), run_id="res")


def test_frozen_memory_change_is_isolation_violation_and_events_emitted(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "MEMORY.md").write_text("- fact\n")
    events: list[Event] = []

    class Sink:
        def emit(self, e):
            events.append(e)
            return e

    def agent(task, ws, ctx):
        assert (ws.parent / "memory" / "MEMORY.md").read_text() == "- fact\n"
        (ws.parent / "memory" / "MEMORY.md").write_text("- polluted\n")
        (ws / "out.txt").write_text("ok")

    rr = EvalRunner(CallableAgent(agent), store=ResultStore(tmp_path / "results"), conditions=COND, event_sink=Sink(),
                    config=RunConfig(workspace_root=tmp_path / "ws", memory=MemoryPolicy("frozen", snap)))
    rec = rr.run(suite(WRITE | {"trials": 1}), run_id="mem").records[0]
    assert rec["isolation_violations"] and not rec["passed"] and rec["workspace_path"]
    assert (snap / "MEMORY.md").read_text() == "- fact\n"
    assert len(events) == 1 and events[0].type.value == "evaluation" and events[0].data["passed"] is False


def test_limit_violation_fails_trial(tmp_path):
    from harness_evals import Trajectory

    def agent(task, ws, ctx):
        (ws / "out.txt").write_text("ok")
        return Trajectory(turns=50)

    rec = runner(tmp_path, agent).run(suite(WRITE | {"trials": 1, "limits": {"max_turns": 10}}), run_id="lim").records[0]
    assert not rec["passed"] and rec["limit_violations"]
