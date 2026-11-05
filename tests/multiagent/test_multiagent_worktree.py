from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from test_multiagent_support import ScriptedModel, ctx, defn, make_manager, report, text

from harness_core.types import RunContext, ToolCall, ToolOutcome, ToolSpec
from harness_multiagent import (
    DelegationBrief,
    Integrator,
    LeaseManager,
    MergeConflictError,
    WorktreeManager,
    WriteAccess,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    git(r, "config", "user.email", "t@example.com")
    git(r, "config", "user.name", "t")
    (r / "app.py").write_text("x = 1\n")
    (r / "src").mkdir()
    (r / "src" / "a.py").write_text("a = 1\n")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "init")
    return r


def test_worktree_writer_is_isolated_and_merged_by_the_single_integrator(repo):
    wtm = WorktreeManager(repo)
    leases = LeaseManager()
    wt = wtm.create("fixer", owned_files=["src"])
    assert wt.path.is_dir() and wt.branch.startswith("harness/fixer-")
    (wt.path / "src" / "a.py").write_text("a = 2\n")
    assert (repo / "src" / "a.py").read_text() == "a = 1\n"  # main checkout untouched while the writer works
    assert git(repo, "status", "--porcelain") == ""          # worktrees are excluded from main's status
    seen = []
    integ = Integrator(wtm, leases, verify=lambda path: seen.append(path) or (path / "src" / "a.py").read_text() == "a = 2\n")
    res = integ.merge(wt)
    assert res.merged and res.verified and res.changed_files == ["src/a.py"] and seen == [wt.path]
    assert (repo / "src" / "a.py").read_text() == "a = 2\n"
    assert leases.writer_of(repo) is None
    wtm.remove(wt)
    assert not wt.path.exists() and wt.branch not in git(repo, "branch")


def test_integrator_refuses_unverified_or_out_of_ownership_changes(repo):
    wtm, leases = WorktreeManager(repo), LeaseManager()
    wt = wtm.create("fixer", owned_files=["src"])
    (wt.path / "app.py").write_text("x = 99\n")  # outside its ownership
    res = Integrator(wtm, leases, verify=lambda p: True).merge(wt)
    assert not res.merged and "does not own" in res.reason
    wt2 = wtm.create("fixer2")
    (wt2.path / "app.py").write_text("x = 3\n")
    assert "no verifier" in Integrator(wtm, leases).merge(wt2).reason
    rejected = Integrator(wtm, leases, verify=lambda p: False).merge(wt2)
    assert not rejected.merged and rejected.verified is False
    assert (repo / "app.py").read_text() == "x = 1\n"


def test_parallel_worktree_writers_conflict_is_detected_and_aborted(repo):
    wtm, leases = WorktreeManager(repo), LeaseManager()
    a, b = wtm.create("a"), wtm.create("b")
    (a.path / "app.py").write_text("x = 'a'\n")
    (b.path / "app.py").write_text("x = 'b'\n")
    integ = Integrator(wtm, leases, verify=lambda p: True)
    assert integ.merge(a).merged
    with pytest.raises(MergeConflictError, match="app.py"):
        integ.merge(b)
    assert (repo / "app.py").read_text() == "x = 'a'\n" and git(repo, "status", "--porcelain") == ""


def test_integrator_waits_out_a_workspace_writer(repo):
    wtm, leases = WorktreeManager(repo), LeaseManager()
    wt = wtm.create("w")
    (wt.path / "app.py").write_text("x = 5\n")
    with leases.acquire_write(repo, "some-workspace-writer"):
        res = Integrator(wtm, leases, verify=lambda p: True, lease_timeout=0.05).merge(wt)
    assert not res.merged and "lease" in res.reason


def test_manager_runs_own_worktree_writer_in_its_worktree(repo):
    class FsTools:
        def specs(self):
            return [ToolSpec("write_file", "write", {"type": "object"}, read_only=False)]

        def execute(self, call: ToolCall, ctx: RunContext) -> ToolOutcome:
            target = Path(ctx.cwd) / call.input["path"]
            target.write_text(call.input["content"])
            return ToolOutcome("written")

    from test_multiagent_support import tool_use

    model = ScriptedModel(lambda req, n: tool_use("write_file", {"path": "app.py", "content": "x = 7\n"}) if n == 0 else text(report("edited", files=["app.py"])))
    wtm = WorktreeManager(repo)
    mgr = make_manager(model, defn("wt-writer", write_access=WriteAccess.OWN_WORKTREE), tools=FsTools(), worktrees=wtm)
    res = mgr.spawn("wt-writer", DelegationBrief("Bump x"), ctx(str(repo)))
    assert res.ok and res.workspace and Path(res.workspace) != repo
    assert (Path(res.workspace) / "app.py").read_text() == "x = 7\n" and (repo / "app.py").read_text() == "x = 1\n"
    wt = mgr.worktree(res.run_id)
    assert Integrator(wtm, mgr.leases, verify=lambda p: True).merge(wt).merged
    assert (repo / "app.py").read_text() == "x = 7\n"
    # Without a WorktreeManager the spawn is refused instead of writing to the shared checkout.
    bare = make_manager(model, defn("wt-writer", write_access=WriteAccess.OWN_WORKTREE), tools=FsTools())
    assert bare.spawn("wt-writer", DelegationBrief("Bump x"), ctx(str(repo))).refusal_code == "no_worktree"
