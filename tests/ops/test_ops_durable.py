"""Durable execution: step journal replay, crash/resume, divergence, leases/fencing, heartbeats."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from harness_core.types import Message, ModelRequest, ModelResponse, StopReason, TextBlock, Usage
from harness_ops import (
    DuplicateStepError,
    InMemorySpanExporter,
    LeaseHeldError,
    LeaseLostError,
    LeaseStore,
    NoActiveRunError,
    NonDeterminismError,
    OpsTracer,
    RetryPolicy,
    StepJournal,
    StepSerializationError,
    StepStatus,
    durable_step,
)

SRC = str(Path(__file__).resolve().parents[2] / "src")


def test_completed_steps_replay_without_rerunning(tmp_path):
    j = StepJournal(tmp_path / "j.db")
    calls = {"a": 0, "b": 0}

    def a():
        calls["a"] += 1
        return {"value": 1, "items": [1, 2]}

    def b():
        calls["b"] += 1
        return "two"

    with j.open_run("sess-1", heartbeat=False) as run:
        first = run.step("a", a)
        run.step("b", b)
    with j.open_run("sess-1", heartbeat=False) as run:
        again = run.step("a", a)
        assert run.step("b", b) == "two"
        assert run.replayed == ["a", "b"] and run.executed == []
    assert calls == {"a": 1, "b": 1} and first == again
    assert [s.status for s in j.steps("sess-1")] == [StepStatus.COMPLETED, StepStatus.COMPLETED]


def test_process_crash_mid_run_resumes_from_last_completed_step(tmp_path):
    """A worker process dies hard (os._exit) inside step 3; a new worker resumes after its lease expires."""
    db, marks = tmp_path / "j.db", tmp_path / "marks"
    marks.mkdir()
    script = textwrap.dedent(
        f"""
        import os, sys
        sys.path.insert(0, {SRC!r})
        from pathlib import Path
        from harness_ops import StepJournal
        marks = Path({str(marks)!r})
        def side(name):
            def fn():
                (marks / name).write_text(str(len(list(marks.iterdir()))))
                if name == "step3":
                    os._exit(17)  # crash after the side effect, before the journal commit
                return name.upper()
            return fn
        j = StepJournal({str(db)!r})
        run = j.open_run("sess-crash", worker_id="worker-A", lease_seconds=0.5)
        for name in ("step1", "step2", "step3", "step4"):
            run.step(name, side(name))
        """
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 17, proc.stderr
    j = StepJournal(db)
    deadline = time.time() + 10  # the dead worker's lease (0.5 s, no more heartbeats) lapses on its own
    while j.leases.holder("run:sess-crash") is not None and time.time() < deadline:
        time.sleep(0.05)
    assert j.resumable_runs() == ["sess-crash"]
    reran = []
    with j.open_run("sess-crash", worker_id="worker-B", heartbeat=False) as run:
        results = [run.step(n, lambda n=n: reran.append(n) or n.upper()) for n in ("step1", "step2", "step3", "step4")]
        run.complete(results)
    assert results == ["STEP1", "STEP2", "STEP3", "STEP4"]
    assert reran == ["step3", "step4"]  # completed steps never re-run; the interrupted one does (at-least-once)
    assert sorted(p.name for p in marks.iterdir()) == ["step1", "step2", "step3"]
    rec = j.step_record("sess-crash", "step3")
    assert rec.attempts == 2 and rec.status is StepStatus.COMPLETED
    assert j.run_info("sess-crash")["status"] == "completed" and j.resumable_runs() == []


def test_divergence_reruns_changed_step_and_every_later_step(tmp_path):
    j = StepJournal(tmp_path / "j.db")
    runs = []

    def workflow(prompts):
        with j.open_run("sess-div", heartbeat=False) as run:
            out = [run.step(f"agent-{i}", lambda p=p: runs.append(p) or f"r:{p}", input={"prompt": p}) for i, p in enumerate(prompts)]
            return out, run

    workflow(["p0", "p1", "p2"])
    runs.clear()
    out, run = workflow(["p0", "p1-edited", "p2"])
    assert runs == ["p1-edited", "p2"] and run.diverged_at == "agent-1"  # p2 unchanged but re-runs (Claude Code semantics)
    assert out == ["r:p0", "r:p1-edited", "r:p2"]
    with pytest.raises(NonDeterminismError):
        with j.open_run("sess-div", heartbeat=False, strict_determinism=True) as strict:
            strict.step("agent-0", lambda: "x", input={"prompt": "changed"})


def test_model_calls_are_never_rebilled_on_replay(tmp_path):
    class CountingClient:
        def __init__(self):
            self.calls = 0

        def create(self, request):
            self.calls += 1
            return ModelResponse([TextBlock("hi")], StopReason.END_TURN, Usage(10, 2, 3, 4), model="test-model", id="msg_x")

    j = StepJournal(tmp_path / "j.db")
    client = CountingClient()
    req = ModelRequest(system="sys", messages=[Message.user("hello")])
    for _ in range(3):
        with j.open_run("sess-m", heartbeat=False) as run:
            r = run.model_call("turn-0", client, req)
    assert client.calls == 1 and isinstance(r, ModelResponse)
    assert r.text == "hi" and r.usage == Usage(10, 2, 3, 4) and r.stop_reason is StopReason.END_TURN and r.id == "msg_x"


def test_failed_step_is_retried_on_resume_and_errors_are_redacted(tmp_path):
    j = StepJournal(tmp_path / "j.db")
    with pytest.raises(RuntimeError):
        with j.open_run("sess-f", heartbeat=False) as run:
            run.step("flaky", lambda: (_ for _ in ()).throw(RuntimeError("token=abcdefghijklmnop failed")))
    rec = j.step_record("sess-f", "flaky")
    assert rec.status is StepStatus.FAILED and "abcdefghijklmnop" not in rec.error and "RuntimeError" in rec.error
    with j.open_run("sess-f", heartbeat=False) as run:
        assert run.step("flaky", lambda: 42) == 42
    assert j.step_record("sess-f", "flaky").attempts == 2


def test_step_retry_policy_and_serialization_guard(tmp_path):
    j = StepJournal(tmp_path / "j.db")
    attempts = []

    class Overloaded(Exception):
        status_code = 529

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise Overloaded()
        return "ok"

    with j.open_run("sess-r", heartbeat=False) as run:
        assert run.step("call", flaky, retry=RetryPolicy(max_retries=4), sleep=lambda s: None) == "ok"
        with pytest.raises(StepSerializationError):
            run.step("bad", lambda: object())
        with pytest.raises(DuplicateStepError):
            run.step("call", flaky)
    assert len(attempts) == 3


def test_recorded_now_and_random_are_deterministic_on_replay(tmp_path, clock):
    j = StepJournal(tmp_path / "j.db", clock=clock)
    with j.open_run("sess-t", heartbeat=False) as run:
        t1, r1 = run.now("t"), run.random("r")
    clock.advance(hours=5)
    with j.open_run("sess-t", heartbeat=False) as run:
        assert run.now("t") == t1 and run.random("r") == r1


def test_durable_step_function_needs_an_active_run(tmp_path):
    with pytest.raises(NoActiveRunError):
        durable_step("x", lambda: 1)
    j = StepJournal(tmp_path / "j.db")
    with j.open_run("sess-ctx", heartbeat=False):
        assert durable_step("x", lambda: 1) == 1
    with j.open_run("sess-ctx", heartbeat=False):
        assert durable_step("x", lambda: 2) == 1


def test_only_one_worker_can_hold_a_run_and_zombies_are_fenced(tmp_path, clock):
    j = StepJournal(tmp_path / "j.db", clock=clock)
    zombie = j.open_run("sess-z", worker_id="A", lease_seconds=30, heartbeat=False)
    with pytest.raises(LeaseHeldError):
        j.open_run("sess-z", worker_id="B", heartbeat=False)
    zombie.step("s1", lambda: 1)
    clock.advance(seconds=31)  # A stalls (GC pause, network partition); its lease lapses
    takeover = j.open_run("sess-z", worker_id="B", heartbeat=False)
    assert takeover.fencing_token == zombie.fencing_token + 1
    with pytest.raises(LeaseLostError):
        zombie.step("s2", lambda: "stale write")  # fenced: cannot commit after takeover
    assert takeover.step("s1", lambda: 99) == 1 and takeover.step("s2", lambda: "fresh") == "fresh"
    assert j.step_record("sess-z", "s2").result == '"fresh"'
    takeover.close()


def test_threads_racing_to_open_the_same_run_get_exactly_one_owner(tmp_path):
    j = StepJournal(tmp_path / "j.db")
    winners, losers = [], []
    barrier = threading.Barrier(8)

    def contend(i):
        barrier.wait()
        try:
            run = j.open_run("sess-race", worker_id=f"w{i}", heartbeat=False)
            winners.append(run)
        except LeaseHeldError:
            losers.append(i)

    ts = [threading.Thread(target=contend, args=(i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(winners) == 1 and len(losers) == 7
    winners[0].close()


def test_heartbeat_keeps_lease_alive_and_parallel_steps_in_one_run(tmp_path):
    exp = InMemorySpanExporter()
    j = StepJournal(tmp_path / "j.db")
    with j.open_run("sess-hb", worker_id="A", lease_seconds=0.3, tracer=OpsTracer([exp])) as run:
        time.sleep(0.8)  # longer than the lease: heartbeats renew it
        assert j.leases.holder("run:sess-hb").owner == "A"
        with pytest.raises(LeaseHeldError):
            j.open_run("sess-hb", worker_id="B", heartbeat=False)
        out = {}
        ts = [threading.Thread(target=lambda i=i: out.__setitem__(i, run.step(f"read-{i}", lambda i=i: i * i))) for i in range(6)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert out == {i: i * i for i in range(6)}
    assert j.leases.holder("run:sess-hb") is None  # released on exit
    assert len([s for s in exp.spans if s.name.startswith("durable_step read-")]) == 6


def test_lease_store_tokens_release_and_expired_listing(tmp_path, clock):
    ls = LeaseStore(tmp_path / "l.db", clock=clock)
    a = ls.acquire("worker:x", "A", 10)
    assert ls.acquire("worker:x", "A", 10).token == a.token  # re-acquire = renew
    assert ls.try_acquire("worker:x", "B", 10) is None
    clock.advance(seconds=11)
    assert [lease.resource for lease in ls.expired("worker:")] == ["worker:x"]
    b = ls.acquire("worker:x", "B", 10)
    assert b.token == a.token + 1 and not ls.is_current(a)
    with pytest.raises(LeaseLostError):
        ls.renew(a)
    assert ls.release(b) and ls.holder("worker:x") is None


def test_two_processes_share_one_journal(tmp_path):
    """A second OS process sees steps committed by the first (WAL, cross-process)."""
    db = tmp_path / "nested" / "j.db"
    j = StepJournal(db)
    with j.open_run("s", heartbeat=False) as run:
        run.step("k", lambda: {"from": os.getpid()})
    script = f"import sys; sys.path.insert(0, {SRC!r}); from harness_ops import StepJournal; j = StepJournal({str(db)!r});\n" \
        "with j.open_run('s', heartbeat=False) as run: print(run.step('k', lambda: 'RERAN')['from'])"
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert int(out.stdout.strip()) == os.getpid()
