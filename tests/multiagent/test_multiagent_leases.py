from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from harness_multiagent import FileOwnership, LeaseHeldError, LeaseManager, LeaseNotHeldError, OwnershipConflict, TaskBoard, TaskStatus

SRC = str(Path(__file__).resolve().parents[2] / "src")


def test_write_lease_is_exclusive_readers_are_unlimited(tmp_path):
    lm = LeaseManager()
    w = lm.acquire_write(tmp_path, "agent-a")
    readers = [lm.acquire_read(tmp_path, f"r{i}") for i in range(5)]
    assert len(lm.readers(tmp_path)) == 5 and lm.writer_of(tmp_path).holder == "agent-a"
    with pytest.raises(LeaseHeldError, match="agent-a"):
        lm.acquire_write(tmp_path, "agent-b")
    # Overlapping paths conflict in both directions; siblings do not.
    with pytest.raises(LeaseHeldError):
        lm.acquire_write(tmp_path / "src", "agent-b")
    sibling = lm.acquire_write(tmp_path.parent / "other-workspace", "agent-b")
    w.release()
    w.release()  # idempotent
    assert lm.acquire_write(tmp_path / "src", "agent-c").held
    for r in readers:
        r.release()
    sibling.release()
    assert lm.readers(tmp_path) == []


def test_require_write_covers_children_only_for_the_holder(tmp_path):
    lm = LeaseManager()
    with lm.acquire_write(tmp_path, "writer") as lease:
        lm.require_write(tmp_path / "a" / "b.py", "writer")
        with pytest.raises(LeaseNotHeldError):
            lm.require_write(tmp_path / "a.py", "someone-else")
    with pytest.raises(LeaseNotHeldError):
        lease.assert_held()


def test_waiting_writer_gets_the_lease_after_release(tmp_path):
    lm = LeaseManager()
    first = lm.acquire_write(tmp_path, "a")
    threading.Timer(0.1, first.release).start()
    t0 = time.monotonic()
    second = lm.acquire_write(tmp_path, "b", timeout=5)
    assert second.held and time.monotonic() - t0 >= 0.05


def test_no_two_writers_ever_hold_the_same_workspace_under_contention(tmp_path):
    lm = LeaseManager(lock_dir=tmp_path / "locks")
    lock = threading.Lock()
    state = {"holders": 0, "peak": 0, "wins": 0}

    def worker(i: int) -> None:
        for _ in range(20):
            try:
                lease = lm.acquire_write(tmp_path / "ws", f"w{i}", timeout=0.5)
            except LeaseHeldError:
                continue
            with lock:
                state["holders"] += 1
                state["peak"] = max(state["peak"], state["holders"])
                state["wins"] += 1
            time.sleep(0.001)
            with lock:
                state["holders"] -= 1
            lease.release()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert state["peak"] == 1 and state["wins"] > 0


def test_write_lease_is_exclusive_across_processes(tmp_path):
    lock_dir, ws = tmp_path / "locks", tmp_path / "ws"
    ws.mkdir()
    lm = LeaseManager(lock_dir=lock_dir)
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {SRC!r})
        from harness_multiagent import LeaseManager, LeaseHeldError
        try:
            LeaseManager(lock_dir={str(lock_dir)!r}).acquire_write({str(ws)!r}, "other-process")
            print("ACQUIRED")
        except LeaseHeldError as exc:
            print("REFUSED", exc)
    """)
    with lm.acquire_write(ws, "this-process"):
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60).stdout
        assert out.startswith("REFUSED") and "this-process" in out
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60).stdout
    assert out.startswith("ACQUIRED")


def test_file_ownership_claims_are_disjoint_and_all_or_nothing():
    own = FileOwnership()
    own.claim("a", ["src/auth", "docs/auth.md"])
    with pytest.raises(OwnershipConflict, match="owned by 'a'"):
        own.claim("b", ["src/billing", "src/auth/jwt.py"])
    assert own.owner_of("src/billing/x.py") is None  # nothing from the failed claim was recorded
    own.claim("b", ["src/billing"])
    own.check("a", "src/auth/jwt.py")
    with pytest.raises(OwnershipConflict):
        own.check("a", "src/billing/x.py")
    with pytest.raises(OwnershipConflict, match="relative"):
        own.claim("c", ["../etc"])
    own.release("a")
    assert own.snapshot() == {"b": ["src/billing"]}


# ----------------------------------------------------------------------------- task board


def test_task_board_dependencies_auto_unblock_and_file_conflicts(tmp_path, clock):
    board = TaskBoard(tmp_path / "tasks.json", clock=clock)
    board.add("schema", "Design schema", files=["db/schema.sql"])
    board.add("api", "Build API", depends_on=["schema"], files=["src/api"])
    board.add("migr", "Write migration", files=["db"])
    first = board.claim("a1")
    assert first.id == "schema"
    # "api" is blocked on schema; "migr" overlaps schema's files -> nothing claimable.
    assert board.claim("a2") is None
    board.complete("schema", "a1", "done")
    assert board.claim("a2").id == "api"
    assert board.claim("a3").id == "migr"
    with pytest.raises(OwnershipConflict):
        board.complete("api", "a3")
    assert board.summary()["in_progress"] == 2


def test_task_board_concurrent_claims_never_double_claim(tmp_path):
    board = TaskBoard(tmp_path / "tasks.json")
    for i in range(30):
        board.add(f"t{i}", f"task {i}")
    claimed: list[str] = []
    lock = threading.Lock()

    def worker(name: str) -> None:
        while (t := board.claim(name)) is not None:
            with lock:
                claimed.append(t.id)
            board.complete(t.id, name)

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(claimed) == sorted(f"t{i}" for i in range(30))
    assert board.summary()["completed"] == 30


def test_task_board_expired_lease_returns_task_and_retries_are_bounded(tmp_path, clock):
    board = TaskBoard(tmp_path / "tasks.json", lease_seconds=60, max_attempts=2, clock=clock)
    board.add("t", "flaky")
    assert board.claim("crashed-agent").id == "t"
    clock.advance(seconds=61)
    t = board.claim("agent-2")
    assert t.owner == "agent-2" and t.attempts == 2
    final = board.fail("t", "agent-2", "boom")
    assert final.status is TaskStatus.FAILED and board.claim("agent-3") is None
