"""Scopes, mounts and the on-disk layout.

::

    ~/.harness/
      org/                                  # READ-ONLY mount (org standards, shared skills)
      users/<user_id>/memory/               # git repo, read_write
      projects/<repo_id>/memory/            # git repo, read_write; shared across worktrees
      agents/<agent_name>/memory/           # subagent memory
      sessions/<session_id>.jsonl           # immutable transcripts

Rules: org memory is always read-only; at most 8 stores per session, attached at session start
only (Managed Agents semantics); subagents never see main-conversation memory unless forked, and
isolation is by layout, not by agent name.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from ._time import Clock, utcnow
from .errors import AccessDeniedError, LimitExceededError, ValidationError
from .frontmatter import Scope
from .security import SecretRedactor
from .store import Access, FileMemoryStore, StoreLimits
from .versioning import GitVersioner

MAX_MOUNTS_PER_SESSION = 8
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$")


def _check_id(kind: str, value: str) -> str:
    if not _ID.match(value or ""):
        raise ValidationError(f"invalid {kind}: {value!r}")
    return value


def repo_id_for(path: str | Path) -> str:
    """Stable project id shared by all worktrees of one repository (hash of ``git --git-common-dir``)."""
    p = Path(path).resolve()
    proc = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=p, capture_output=True, text=True)
    anchor = proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else str(p)
    digest = hashlib.sha256(anchor.encode("utf-8")).hexdigest()[:12]
    name = re.sub(r"[^A-Za-z0-9_.\-]", "-", Path(anchor).parent.name if anchor.endswith(".git") else p.name) or "repo"
    return f"{name}-{digest}"


@dataclass(frozen=True)
class MemoryLayout:
    base: Path = field(default_factory=lambda: Path("~/.harness").expanduser())

    def __post_init__(self) -> None:
        object.__setattr__(self, "base", Path(self.base).expanduser())

    @property
    def org_dir(self) -> Path:
        return self.base / "org"

    def user_dir(self, user_id: str) -> Path:
        return self.base / "users" / _check_id("user id", user_id) / "memory"

    def project_dir(self, repo_id: str) -> Path:
        return self.base / "projects" / _check_id("repo id", repo_id) / "memory"

    def agent_dir(self, agent_name: str, *, scope: Scope = Scope.PROJECT, repo_root: Path | None = None) -> Path:
        """Subagent memory: ``user`` -> under base, ``project`` -> ``<repo>/.harness/agent-memory``, ``local`` -> gitignored."""
        _check_id("agent name", agent_name)
        if scope is Scope.USER:
            return self.base / "agents" / agent_name / "memory"
        if repo_root is None:
            raise ValidationError("project/local agent memory needs repo_root")
        sub = "agent-memory" if scope is Scope.PROJECT else "agent-memory-local"
        return Path(repo_root) / ".harness" / sub / agent_name

    @property
    def sessions_base(self) -> Path:
        return self.base


@dataclass
class Mount:
    name: str
    scope: Scope
    store: FileMemoryStore
    instructions: str = ""

    @property
    def read_only(self) -> bool:
        return not self.store.writable


class MountSet:
    """The stores attached to one session. Frozen once the session starts."""

    def __init__(self) -> None:
        self._mounts: dict[str, Mount] = {}
        self._frozen = False

    def attach(self, mount: Mount) -> Mount:
        if self._frozen:
            raise AccessDeniedError("memory stores can only be attached at session start")
        if len(self._mounts) >= MAX_MOUNTS_PER_SESSION:
            raise LimitExceededError(f"at most {MAX_MOUNTS_PER_SESSION} memory stores per session")
        if mount.name in self._mounts:
            raise ValidationError(f"mount {mount.name!r} already attached")
        if len(mount.instructions) > 4096:
            raise LimitExceededError("mount instructions are limited to 4,096 characters")
        if mount.scope is Scope.ORG and mount.store.writable:
            raise AccessDeniedError("org memory must be mounted read-only")
        self._mounts[mount.name] = mount
        return mount

    def freeze(self) -> None:
        self._frozen = True

    def __iter__(self) -> Iterator[Mount]:
        return iter(self._mounts.values())

    def __getitem__(self, name: str) -> Mount:
        return self._mounts[name]

    def __contains__(self, name: str) -> bool:
        return name in self._mounts

    def writable(self) -> list[Mount]:
        return [m for m in self if not m.read_only]

    def describe(self) -> str:
        """System-prompt note describing each mount (as Managed Agents adds automatically)."""
        lines = []
        for m in self:
            mode = "read-only" if m.read_only else "read-write"
            line = f"- {m.name} ({m.scope.value}, {mode}): {m.store.root}"
            if m.instructions:
                line += f" — {m.instructions}"
            lines.append(line)
        return "Memory stores attached to this session:\n" + "\n".join(lines) if lines else ""


def open_store(
    path: Path,
    scope: Scope,
    *,
    access: Access = Access.READ_WRITE,
    git: bool = True,
    clock: Clock = utcnow,
    redactor: SecretRedactor | None = None,
    limits: StoreLimits | None = None,
    session_tainted: bool = False,
) -> FileMemoryStore:
    """Open (and for writable scopes, create + git-init) a store."""
    if scope is Scope.ORG:
        access = Access.READ_ONLY
    versioner = None
    if access is Access.READ_WRITE and git:
        versioner = GitVersioner(path)
        versioner.ensure_repo()
    if access is Access.READ_ONLY and not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return FileMemoryStore(
        path,
        scope=scope,
        access=access,
        versioner=versioner,
        clock=clock,
        redactor=redactor,
        limits=limits,
        session_tainted=session_tainted,
    )
