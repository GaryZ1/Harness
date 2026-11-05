"""Per-trial isolation: a fresh directory tree per trial, isolated env, hidden-test overlay.

Why (reference §9.7 step 4, §9.8 "Non-isolated trials"): Claude once read git history left by
earlier trials, and shared memory or caches make trials non-independent and inflate pass rates.
Each :class:`TrialSandbox` therefore gets:

* ``workspace/`` - materialised from the task's setup only. ``git_init`` creates a *new* repo per
  trial; otherwise there is no ``.git`` at all, and ``GIT_CEILING_DIRECTORIES`` stops git from
  discovering any enclosing repository (e.g. the harness repo itself).
* ``home/`` - ``HOME`` / ``XDG_*`` / ``TMPDIR`` point here, so tool caches do not leak.
* ``memory/`` - the only memory root (``HARNESS_MEMORY_DIR``). ``reset`` = empty; ``frozen`` =
  a copy of a snapshot, verified unchanged after the trial (a changed frozen memory is recorded
  as an isolation violation).
* ``grade/`` - created only after the agent finishes: a copy of the workspace with hidden tests
  overlaid (replacing any same-named files the agent planted). Graders run here, *outside* the
  agent's sandbox, so tests are never visible or writable to the agent.

Process isolation (containers, resource limits) belongs to the sandboxing package; this module is
the filesystem/env contract the runner needs and works on any POSIX host.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .errors import IsolationError, SetupError
from .task import Task

_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


@dataclass(frozen=True)
class MemoryPolicy:
    """``reset`` (default): empty memory per trial. ``frozen``: copy of ``snapshot``, must not change."""

    mode: str = "reset"
    snapshot: Path | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("reset", "frozen"):
            raise ValueError("memory mode must be 'reset' or 'frozen'")
        if self.mode == "frozen" and (self.snapshot is None or not Path(self.snapshot).is_dir()):
            raise ValueError("frozen memory needs an existing snapshot directory")


@dataclass(frozen=True)
class CommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float

    @property
    def output(self) -> str:
        return self.stdout + ("\n" + self.stderr if self.stderr else "")


def run_command(
    command: str | Sequence[str], *, cwd: Path, env: Mapping[str, str] | None = None, timeout_s: float = 600.0, max_output_chars: int = 200_000
) -> CommandResult:
    """Run a shell command (string) or argv (list) with a hard timeout that kills the process group."""
    shell = isinstance(command, str)
    start = time.monotonic()
    proc = subprocess.Popen(
        command, shell=shell, cwd=str(cwd), env=dict(env) if env is not None else None,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace", start_new_session=True,
    )
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):  # pragma: no cover - already gone
            proc.kill()
        out, err = proc.communicate()
    shown = command if shell else " ".join(command)
    return CommandResult(
        str(shown), proc.returncode if not timed_out else -9, (out or "")[-max_output_chars:], (err or "")[-max_output_chars:],
        timed_out, time.monotonic() - start,
    )


def file_manifest(root: Path) -> dict[str, str]:
    """``{relative posix path: sha256}`` for every regular file under ``root`` (``.git`` and caches skipped)."""
    out: dict[str, str] = {}
    if not root.exists():
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if p.is_symlink() or not p.is_file():
                continue
            out[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


@dataclass(frozen=True)
class ManifestDiff:
    added: tuple[str, ...]
    modified: tuple[str, ...]
    deleted: tuple[str, ...]

    @property
    def changed(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.added) | set(self.modified)))

    @property
    def touched(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.added) | set(self.modified) | set(self.deleted)))


def diff_manifests(before: Mapping[str, str], after: Mapping[str, str]) -> ManifestDiff:
    added = tuple(sorted(set(after) - set(before)))
    deleted = tuple(sorted(set(before) - set(after)))
    modified = tuple(sorted(p for p in set(before) & set(after) if before[p] != after[p]))
    return ManifestDiff(added, modified, deleted)


class TrialSandbox:
    """Directory tree for one trial attempt. Use :class:`SandboxFactory` to create one."""

    def __init__(self, root: Path, task: Task, *, memory: MemoryPolicy, seed: int) -> None:
        self.root = root
        self.task = task
        self.memory = memory
        self.seed = seed
        self.workspace = root / "workspace"
        self.home = root / "home"
        self.memory_dir = root / "memory"
        self.grade_dir = root / "grade"
        self.manifest_before: dict[str, str] = {}
        self._memory_manifest: dict[str, str] = {}

    # ------------------------------------------------------------------ env
    def env(self) -> dict[str, str]:
        """Environment for agent tools, setup and graders: isolated HOME, caches, memory and git."""
        keep = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "LC_ALL", "PYTHONPATH", "SYSTEMROOT", "VIRTUAL_ENV")}
        tmp = self.home / "tmp"
        return keep | {
            "HOME": str(self.home),
            "XDG_CACHE_HOME": str(self.home / ".cache"),
            "XDG_CONFIG_HOME": str(self.home / ".config"),
            "TMPDIR": str(tmp),
            "HARNESS_MEMORY_DIR": str(self.memory_dir),
            "HARNESS_TRIAL_SEED": str(self.seed),
            "GIT_CEILING_DIRECTORIES": str(self.root.parent),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": str(self.home / ".gitconfig"),
            "GIT_AUTHOR_NAME": "eval", "GIT_AUTHOR_EMAIL": "eval@localhost",
            "GIT_COMMITTER_NAME": "eval", "GIT_COMMITTER_EMAIL": "eval@localhost",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

    # ------------------------------------------------------------------ lifecycle
    def materialise(self, *, setup_timeout_s: float = 600.0) -> None:
        """Create the workspace from the task's setup. Any failure is a :class:`SetupError` (infra)."""
        try:
            for d in (self.workspace, self.home / "tmp", self.memory_dir):
                d.mkdir(parents=True, exist_ok=False)
            src = self.task.source_dir
            for cd in self.task.setup.copy_dirs:
                if src is None:
                    raise SetupError(f"copy_dirs needs a task directory: {cd}")
                shutil.copytree(src / cd, self.workspace, dirs_exist_ok=True, symlinks=True)
            for rel, content in self.task.setup.files.items():
                target = self.workspace / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            if self.memory.mode == "frozen" and self.memory.snapshot is not None:
                shutil.copytree(self.memory.snapshot, self.memory_dir, dirs_exist_ok=True)
                self._memory_manifest = file_manifest(self.memory_dir)
            env = self.env()
            if self.task.setup.git_init:
                for cmd in (["git", "init", "-q"], ["git", "add", "-A"], ["git", "commit", "-q", "--allow-empty", "-m", "task setup"]):
                    r = run_command(cmd, cwd=self.workspace, env=env, timeout_s=setup_timeout_s)
                    if r.exit_code != 0:
                        raise SetupError(f"git setup failed: {r.output[-500:]}")
            for cmd in self.task.setup.commands:
                r = run_command(cmd, cwd=self.workspace, env=env, timeout_s=setup_timeout_s)
                if r.exit_code != 0 or r.timed_out:
                    raise SetupError(f"setup command failed ({r.exit_code}): {cmd}: {r.output[-500:]}")
        except SetupError:
            raise
        except OSError as exc:
            raise SetupError(f"workspace materialisation failed: {exc}") from exc
        self.verify_isolation()
        self.manifest_before = file_manifest(self.workspace)

    def verify_isolation(self) -> None:
        """Fail closed when the workspace can see git history it should not have."""
        git_dir = self.workspace / ".git"
        if git_dir.exists() and not self.task.setup.git_init:
            raise IsolationError("workspace contains a .git directory but the task did not request git_init")
        if self.task.setup.git_init:
            r = run_command(["git", "rev-list", "--count", "HEAD"], cwd=self.workspace, env=self.env(), timeout_s=30)
            if r.exit_code == 0 and r.stdout.strip() not in ("1",):
                raise IsolationError(f"fresh repo has unexpected history ({r.stdout.strip()} commits)")

    def memory_violations(self) -> list[str]:
        if self.memory.mode != "frozen":
            return []
        after = file_manifest(self.memory_dir)
        d = diff_manifests(self._memory_manifest, after)
        return [f"frozen memory changed: {p}" for p in d.touched]

    def prepare_grading(self) -> Path:
        """Copy the end-state workspace to ``grade/`` and overlay hidden tests (agent copies replaced)."""
        if self.grade_dir.exists():
            shutil.rmtree(self.grade_dir)
        shutil.copytree(self.workspace, self.grade_dir, symlinks=True)
        hidden_dir = self.task.hidden_dir
        if hidden_dir is not None and hidden_dir.is_dir():
            dest = self.grade_dir / hidden_dir.name
            if dest.is_symlink() or dest.is_file():
                dest.unlink()
            elif dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(hidden_dir, dest)
        for rel, content in self.task.hidden_files.items():
            target = self.grade_dir / rel
            if target.is_symlink() or target.exists():
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return self.grade_dir

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


class SandboxFactory:
    """Creates uniquely named sandboxes under ``base_dir`` (a private temp dir by default)."""

    def __init__(self, base_dir: str | Path | None = None, *, memory: MemoryPolicy | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir is not None else Path(tempfile.mkdtemp(prefix="harness-evals-"))
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.memory = memory or MemoryPolicy()
        self._lock = threading.Lock()

    def create(self, task: Task, *, trial_id: str, attempt: int, seed: int) -> TrialSandbox:
        with self._lock:
            root = self.base_dir / f"{trial_id}.a{attempt}.{uuid.uuid4().hex[:6]}"
            root.mkdir(parents=True, exist_ok=False)
        return TrialSandbox(root, task, memory=self.memory, seed=seed)


__all__ = [
    "CommandResult",
    "ManifestDiff",
    "MemoryPolicy",
    "SandboxFactory",
    "TrialSandbox",
    "diff_manifests",
    "file_manifest",
    "run_command",
]
