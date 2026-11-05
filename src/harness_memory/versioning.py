"""Git-backed versioning for memory repositories (Codex git baselines / Letta MemFS semantics).

Every writable scope is its own git repository. Consolidation commits with typed messages
(``feat|fix|chore(memory): ...``), diffs against a *baseline ref* to find new inputs, and makes
**no commit when nothing changed**.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .errors import MemoryError_

BASELINE_REF = "refs/harness/memory-baseline"


class GitUnavailableError(MemoryError_):
    pass


class GitVersioner:
    def __init__(self, root: str | Path, *, author_name: str = "harness-memory", author_email: str = "harness-memory@localhost") -> None:
        self.root = Path(root)
        self._env_args = ["-c", f"user.name={author_name}", "-c", f"user.email={author_email}", "-c", "commit.gpgsign=false"]
        if shutil.which("git") is None:
            raise GitUnavailableError("git executable not found on PATH")

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run(["git", *self._env_args, *args], cwd=self.root, capture_output=True, text=True, check=False)
        if check and proc.returncode != 0:
            raise MemoryError_(f"git {' '.join(args)} failed: {proc.stderr.strip() or proc.stdout.strip()}")
        return proc

    def ensure_repo(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not (self.root / ".git").exists():
            self._git("init", "-q")
            gitignore = self.root / ".gitignore"
            if not gitignore.exists():
                gitignore.write_text(".lock\n*.tmp-*\n.consolidation.lock\n", encoding="utf-8")

    def head(self) -> str | None:
        proc = self._git("rev-parse", "--verify", "-q", "HEAD", check=False)
        return proc.stdout.strip() or None

    def has_changes(self) -> bool:
        return bool(self._git("status", "--porcelain").stdout.strip())

    def commit_all(self, message: str) -> str | None:
        """Stage everything and commit. Returns the new commit sha, or ``None`` if nothing changed."""
        self.ensure_repo()
        self._git("add", "-A")
        if not self._git("diff", "--cached", "--name-only").stdout.strip():
            return None
        self._git("commit", "-q", "-m", message)
        return self.head()

    def baseline(self) -> str | None:
        proc = self._git("rev-parse", "--verify", "-q", BASELINE_REF, check=False)
        return proc.stdout.strip() or None

    def set_baseline(self, sha: str | None = None) -> None:
        sha = sha or self.head()
        if sha:
            self._git("update-ref", BASELINE_REF, sha)

    def diff_since_baseline(self, *paths: str) -> str:
        """Unified diff of the working tree (incl. untracked files) against the baseline ref."""
        self.ensure_repo()
        self._git("add", "-A", "--intent-to-add", ".", check=False)
        base = self.baseline()
        args = ["diff", "--no-color"]
        if base:
            args.append(base)
        elif self.head() is None:
            # No commits at all: everything is new.
            args = ["diff", "--no-color", "--cached"]
            self._git("add", "-A")
        args += ["--", *paths] if paths else []
        return self._git(*args, check=False).stdout

    def log(self, n: int = 20) -> list[str]:
        proc = self._git("log", f"-n{n}", "--pretty=format:%h %s", check=False)
        return [ln for ln in proc.stdout.splitlines() if ln]
