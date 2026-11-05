"""Workspace snapshots: capture shell side effects, restore, diff, secret exclusion, symlink safety, expiry."""

from __future__ import annotations

import os
import signal
from datetime import timedelta

import pytest

from harness_sandbox import LocalRunner, SandboxPolicy, SnapshotError, WorkspaceSnapshotter


@pytest.fixture(autouse=True)
def _deadline():
    signal.signal(signal.SIGALRM, lambda *_: pytest.fail("test exceeded 60s deadline"))
    signal.alarm(60)
    yield
    signal.alarm(0)


@pytest.fixture
def ws(tmp_path):
    w = tmp_path / "ws"
    (w / "src").mkdir(parents=True)
    (w / "src" / "a.py").write_text("A = 1\n")
    (w / "README.md").write_text("readme\n")
    (w / ".env").write_text("TOKEN=secret\n")
    (w / ".git").mkdir()
    (w / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    return w


@pytest.fixture
def snap(tmp_path, ws, clock):
    return WorkspaceSnapshotter.for_policy(SandboxPolicy(), ws, tmp_path / "store", clock=clock)


def test_snapshot_restore_undoes_shell_side_effects(ws, snap):
    s1 = snap.snapshot("turn-1")
    with LocalRunner(SandboxPolicy(), ws, network_namespace=False) as r:
        res = r.run("rm README.md && echo 'A = 2' > src/a.py && mkdir -p build/out && echo bin > build/out/x.o", timeout=20)
    assert res.ok
    d = snap.diff(s1.id)
    assert d.added == ("build/out/x.o",) and d.modified == ("src/a.py",) and d.deleted == ("README.md",)
    report = snap.restore(s1.id)
    assert report.deleted == ("build/out/x.o",) and report.added == ("README.md",)
    assert (ws / "README.md").read_text() == "readme\n" and (ws / "src" / "a.py").read_text() == "A = 1\n"
    assert not (ws / "build").exists()
    assert snap.diff(s1.id).empty


def test_secrets_and_git_are_never_captured_or_touched(ws, snap, tmp_path):
    s1 = snap.snapshot()
    blobs = b"".join(p.read_bytes() for p in (tmp_path / "store" / "objects").rglob("*") if p.is_file())
    assert b"secret" not in blobs and b"refs/heads" not in blobs
    (ws / ".env").write_text("TOKEN=rotated\n")
    (ws / ".env.local").write_text("X=1\n")
    snap.restore(s1.id)
    assert (ws / ".env").read_text() == "TOKEN=rotated\n" and (ws / ".env.local").exists()
    assert (ws / ".git" / "HEAD").exists()


def test_symlinks_are_recorded_not_followed(ws, snap, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "id_rsa_like").write_text("PRIVATE KEY MATERIAL")
    (ws / "escape").symlink_to(outside)
    s1 = snap.snapshot()
    blobs = b"".join(p.read_bytes() for p in (tmp_path / "store" / "objects").rglob("*") if p.is_file())
    assert b"PRIVATE KEY MATERIAL" not in blobs
    (ws / "escape").unlink()
    (ws / "escape").mkdir()  # replaced by a real dir; restore must turn it back into the link
    (ws / "escape" / "junk").write_text("j")
    snap.restore(s1.id)
    assert os.path.islink(ws / "escape") and os.readlink(ws / "escape") == str(outside)
    assert (outside / "id_rsa_like").read_text() == "PRIVATE KEY MATERIAL"  # restore never wrote through it


def test_restore_replaces_planted_symlink_instead_of_writing_through(ws, snap, tmp_path):
    s1 = snap.snapshot()
    target = tmp_path / "victim.txt"
    target.write_text("do not overwrite")
    (ws / "src" / "a.py").unlink()
    (ws / "src" / "a.py").symlink_to(target)
    snap.restore(s1.id)
    assert target.read_text() == "do not overwrite"
    assert not os.path.islink(ws / "src" / "a.py") and (ws / "src" / "a.py").read_text() == "A = 1\n"


def test_dedup_list_diff_between_snapshots_and_prune(ws, snap, clock, tmp_path):
    s1 = snap.snapshot("one")
    n_objects = sum(1 for p in (tmp_path / "store" / "objects").rglob("*") if p.is_file())
    clock.advance(hours=1)
    s2 = snap.snapshot("two")  # unchanged tree: no new objects
    assert sum(1 for p in (tmp_path / "store" / "objects").rglob("*") if p.is_file()) == n_objects
    (ws / "new.txt").write_text("n")
    clock.advance(hours=1)
    s3 = snap.snapshot("three")
    assert [s.label for s in snap.list()] == ["one", "two", "three"] and s3.seq == 3
    assert snap.diff(s2.id, s3.id).added == ("new.txt",)
    assert snap.prune(max_age=timedelta(minutes=90)) == [s1.id]
    assert snap.prune(keep_last=1) == [s2.id]
    assert [s.id for s in snap.list()] == [s3.id]
    with pytest.raises(SnapshotError):
        snap.restore(s1.id)


def test_store_must_be_outside_workspace_and_destroy(ws, tmp_path):
    with pytest.raises(SnapshotError):
        WorkspaceSnapshotter(ws, ws / ".snapshots")
    s = WorkspaceSnapshotter(ws, tmp_path / "st")
    s.snapshot()
    s.destroy()
    assert not (tmp_path / "st").exists()


def test_oversized_files_are_skipped_and_preserved(ws, tmp_path):
    (ws / "big.bin").write_bytes(b"x" * 5000)
    s = WorkspaceSnapshotter(ws, tmp_path / "st", max_file_bytes=1000)
    snap = s.snapshot()
    assert snap.skipped == ("big.bin",)
    (ws / "big.bin").write_bytes(b"y" * 6000)
    s.restore(snap.id)
    assert (ws / "big.bin").read_bytes() == b"y" * 6000  # not captured, so not touched
