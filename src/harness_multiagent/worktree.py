"""Git-worktree isolation for parallel writers, merged back through one integrator.

Research basis (reference 8.7 item 11; 8.4 inferences): parallel writers are allowed only with
(a) one git worktree (or container) per writer, (b) a file-ownership map or task locks, (c) a shared
test oracle plus a CI gate before merge, and (d) a **single integrator that owns the main branch**.
Claude Code's ``isolation: worktree`` runs a subagent in a temporary worktree; Anthropic's compiler
swarm arbitrated through git with lock files; when the oracle was weak, parallel agents "solved the
wrong problem".

* :class:`WorktreeManager` creates ``<repo>/.harness/worktrees/<agent>-<id>`` on a fresh branch
  ``harness/<agent>-<id>`` from ``base`` and removes it afterwards.
* :class:`Integrator` is the only code path that writes to the target branch. It holds the
  workspace write lease (so no ``workspace`` writer can run concurrently), commits the writer's
  worktree, checks changed files against the writer's ownership, runs the verifier (test oracle)
  **inside the worktree**, and only then merges with ``--no-ff``. Conflicts abort the merge and
  leave the target branch untouched.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from harness_core.types import new_id

from .errors import LeaseHeldError, MergeConflictError, OwnershipConflict, WorktreeError
from .leases import LeaseManager, _norm_rel, _rel_covers

_SAFE = re.compile(r"[^A-Za-z0-9_.\-]")

#: Used only when the environment has no git identity configured (throwaway commits in worktrees).
_FALLBACK_IDENTITY = {
    "GIT_AUTHOR_NAME": "harness-multiagent",
    "GIT_AUTHOR_EMAIL": "harness-multiagent@localhost",
    "GIT_COMMITTER_NAME": "harness-multiagent",
    "GIT_COMMITTER_EMAIL": "harness-multiagent@localhost",
}


def _git(args: Sequence[str], cwd: Path, *, check: bool = True, git: str = "git") -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    for k, v in _FALLBACK_IDENTITY.items():
        env.setdefault(k, v)
    proc = subprocess.run([git, *args], cwd=cwd, capture_output=True, text=True, env=env)
    if check and proc.returncode != 0:
        raise WorktreeError(f"git {' '.join(args)} failed in {cwd}: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc


@dataclass(frozen=True)
class Worktree:
    agent: str
    path: Path
    branch: str
    base: str
    owned_files: tuple[str, ...] = ()


@dataclass
class MergeResult:
    merged: bool
    branch: str
    commit: str | None = None
    changed_files: list[str] = field(default_factory=list)
    verified: bool | None = None
    conflicts: list[str] = field(default_factory=list)
    reason: str = ""


class WorktreeManager:
    """Create and remove per-writer git worktrees. Thread-safe (git worktree ops are serialised)."""

    def __init__(self, repo: str | Path, *, root: str | Path | None = None, git: str = "git") -> None:
        self.repo = Path(repo).expanduser().resolve()
        self.git = git
        top = _git(["rev-parse", "--show-toplevel"], self.repo, git=git).stdout.strip()
        self.repo = Path(top).resolve()
        self.root = Path(root).expanduser().resolve() if root else self.repo / ".harness" / "worktrees"
        self._lock = threading.Lock()
        self._active: dict[str, Worktree] = {}

    def create(self, agent: str, *, base: str = "HEAD", owned_files: Sequence[str] = ()) -> Worktree:
        slug = f"{_SAFE.sub('-', agent)}-{new_id('wt')[3:15]}"
        path = self.root / slug
        branch = f"harness/{slug}"
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            self._ensure_excluded()
            base_sha = _git(["rev-parse", "--verify", base], self.repo, git=self.git).stdout.strip()
            _git(["worktree", "add", "-b", branch, str(path), base_sha], self.repo, git=self.git)
            wt = Worktree(agent=agent, path=path, branch=branch, base=base_sha, owned_files=tuple(_norm_rel(f) for f in owned_files))
            self._active[str(path)] = wt
        return wt

    def _ensure_excluded(self) -> None:
        """Keep worktrees out of the main checkout's ``git status`` (uses .git/info/exclude, not .gitignore)."""
        try:
            rel = self.root.relative_to(self.repo).as_posix()
        except ValueError:
            return
        common = Path(_git(["rev-parse", "--git-common-dir"], self.repo, git=self.git).stdout.strip())
        if not common.is_absolute():
            common = self.repo / common
        exclude = common / "info" / "exclude"
        exclude.parent.mkdir(parents=True, exist_ok=True)
        existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
        line = f"/{rel}/"
        if line not in existing.splitlines():
            with open(exclude, "a", encoding="utf-8") as fh:
                fh.write(("" if existing.endswith("\n") or not existing else "\n") + line + "\n")

    def active(self) -> list[Worktree]:
        with self._lock:
            return list(self._active.values())

    def changed_files(self, wt: Worktree) -> list[str]:
        """Files changed in the worktree relative to its base (committed and uncommitted)."""
        committed = _git(["diff", "--name-only", f"{wt.base}..HEAD"], wt.path, git=self.git).stdout.split()
        dirty = _git(["status", "--porcelain", "--untracked-files=all"], wt.path, git=self.git).stdout.splitlines()
        names = set(committed)
        for line in dirty:
            name = line[3:].strip()
            if " -> " in name:
                name = name.split(" -> ", 1)[1]
            names.add(name.strip('"'))
        return sorted(n for n in names if n)

    def commit_all(self, wt: Worktree, message: str) -> str | None:
        """Commit every change in the worktree; returns the new HEAD sha or ``None`` if clean."""
        _git(["add", "-A"], wt.path, git=self.git)
        if _git(["diff", "--cached", "--quiet"], wt.path, check=False, git=self.git).returncode == 0:
            return None
        _git(["commit", "-q", "-m", message], wt.path, git=self.git)
        return _git(["rev-parse", "HEAD"], wt.path, git=self.git).stdout.strip()

    def remove(self, wt: Worktree, *, delete_branch: bool = True) -> None:
        with self._lock:
            _git(["worktree", "remove", "--force", str(wt.path)], self.repo, check=False, git=self.git)
            _git(["worktree", "prune"], self.repo, check=False, git=self.git)
            if delete_branch:
                _git(["branch", "-D", wt.branch], self.repo, check=False, git=self.git)
            self._active.pop(str(wt.path), None)


Verifier = Callable[[Path], bool]


class Integrator:
    """The single writer of the target branch. Merges verified worktree branches one at a time."""

    def __init__(
        self,
        worktrees: WorktreeManager,
        leases: LeaseManager,
        *,
        verify: Verifier | None = None,
        name: str = "integrator",
        lease_timeout: float = 30.0,
    ) -> None:
        self.worktrees = worktrees
        self.leases = leases
        self.verify = verify
        self.name = name
        self.lease_timeout = lease_timeout
        self._lock = threading.Lock()

    def merge(self, wt: Worktree, *, message: str | None = None, require_verifier: bool = True) -> MergeResult:
        """Commit, ownership-check, verify, then merge ``wt.branch`` into the main checkout's branch.

        Raises :class:`MergeConflictError` on conflicts (after ``git merge --abort``). Returns
        ``merged=False`` with a reason when ownership or verification fails.
        """
        repo = self.worktrees.repo
        with self._lock:
            try:
                lease = self.leases.acquire_write(repo, self.name, timeout=self.lease_timeout)
            except LeaseHeldError as exc:
                return MergeResult(False, wt.branch, reason=f"integrator could not take the workspace lease: {exc}")
            with lease:
                self.worktrees.commit_all(wt, message or f"{wt.agent}: subagent changes")
                changed = self.worktrees.changed_files(wt)
                if not changed:
                    return MergeResult(False, wt.branch, reason="no changes", changed_files=[])
                if wt.owned_files:
                    outside = [f for f in changed if not any(_rel_covers(o, f) for o in wt.owned_files)]
                    if outside:
                        return MergeResult(False, wt.branch, changed_files=changed,
                                           reason=str(OwnershipConflict(f"{wt.agent} changed files it does not own: {outside}")))
                if self.verify is None and require_verifier:
                    return MergeResult(False, wt.branch, changed_files=changed,
                                       reason="no verifier configured; refusing to merge without a test oracle")
                verified = None
                if self.verify is not None:
                    try:
                        verified = bool(self.verify(wt.path))
                    except Exception as exc:  # a crashing oracle is a failed gate
                        return MergeResult(False, wt.branch, changed_files=changed, verified=False, reason=f"verifier raised: {exc}")
                    if not verified:
                        return MergeResult(False, wt.branch, changed_files=changed, verified=False, reason="verifier rejected the change")
                if _git(["status", "--porcelain", "--untracked-files=no"], repo, git=self.worktrees.git).stdout.strip():
                    return MergeResult(False, wt.branch, changed_files=changed, verified=verified,
                                       reason="main checkout has uncommitted changes; refusing to merge")
                proc = _git(["merge", "--no-ff", "-m", f"Merge {wt.branch} ({wt.agent})", wt.branch], repo, check=False, git=self.worktrees.git)
                if proc.returncode != 0:
                    conflicts = _git(["diff", "--name-only", "--diff-filter=U"], repo, check=False, git=self.worktrees.git).stdout.split()
                    _git(["merge", "--abort"], repo, check=False, git=self.worktrees.git)
                    raise MergeConflictError(f"merging {wt.branch} conflicted on {conflicts or 'unknown files'}; merge aborted")
                head = _git(["rev-parse", "HEAD"], repo, git=self.worktrees.git).stdout.strip()
                return MergeResult(True, wt.branch, commit=head, changed_files=changed, verified=verified)


__all__ = ["Worktree", "WorktreeManager", "Integrator", "MergeResult", "Verifier"]
