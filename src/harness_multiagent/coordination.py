"""Durable coordination substrate: a shared task board with locked claiming.

Research basis (reference 8.7 item 12, 8.2 agent teams / C compiler): Claude Code agent teams use a
shared task list (pending / in-progress / completed, dependencies, auto-unblocking, **file-locked
claiming**); Anthropic's 16-agent compiler swarm claimed tasks by writing lock files and let git
arbitrate. :class:`TaskBoard` is the file-backed equivalent:

* one JSON document, rewritten atomically (temp + fsync + ``os.replace``) under a cross-process
  ``fcntl`` lock, so concurrent claimers in threads *or* processes never double-claim;
* ``claim`` returns the first pending task (insertion order, deterministic) whose dependencies are
  all completed - dependents unblock automatically;
* claims are **leases** with an expiry: a crashed agent's task returns to the pool after
  ``lease_seconds`` instead of blocking dependents forever;
* tasks may declare ``files``; a claim is refused while another in-progress task owns an overlapping
  path (the ownership-map rule for parallel writers).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from harness_core.types import Clock, utcnow
from harness_memory.locking import atomic_write_text, file_lock

from .errors import MultiAgentError, OwnershipConflict
from .leases import _norm_rel, _rel_overlaps


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BoardTask:
    id: str
    title: str
    status: TaskStatus = TaskStatus.PENDING
    depends_on: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    owner: str | None = None
    lease_expires: str | None = None
    attempts: int = 0
    result: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BoardTask":
        return cls(**{**d, "status": TaskStatus(d.get("status", "pending"))})


class TaskBoard:
    """File-backed task list shared by agents (threads or processes)."""

    def __init__(self, path: str | Path, *, lease_seconds: int = 1800, max_attempts: int = 3, clock: Clock = utcnow) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.path.with_name(self.path.name + ".lock")
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts
        self.clock = clock

    # ------------------------------------------------------------------ persistence
    def _read(self) -> list[BoardTask]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8") or "[]")
        return [BoardTask.from_dict(d) for d in data]

    def _write(self, tasks: list[BoardTask]) -> None:
        atomic_write_text(self.path, json.dumps([t.to_dict() for t in tasks], indent=1, ensure_ascii=False))

    def tasks(self) -> list[BoardTask]:
        with file_lock(self._lock_path):
            return self._read()

    # ------------------------------------------------------------------ mutation
    def add(self, task_id: str, title: str, *, depends_on: Iterable[str] = (), files: Iterable[str] = ()) -> BoardTask:
        with file_lock(self._lock_path):
            tasks = self._read()
            ids = {t.id for t in tasks}
            if task_id in ids:
                raise MultiAgentError(f"task {task_id!r} already exists")
            deps = list(depends_on)
            missing = [d for d in deps if d not in ids]
            if missing:
                raise MultiAgentError(f"task {task_id!r} depends on unknown tasks {missing} (add dependencies first)")
            task = BoardTask(task_id, title, depends_on=deps, files=[_norm_rel(f) for f in files])
            tasks.append(task)
            self._write(tasks)
            return task

    def _expire(self, tasks: list[BoardTask], now: datetime) -> None:
        for t in tasks:
            if t.status is TaskStatus.IN_PROGRESS and t.lease_expires and datetime.fromisoformat(t.lease_expires) <= now:
                t.status, t.owner, t.lease_expires = TaskStatus.PENDING, None, None

    def claim(self, agent: str) -> BoardTask | None:
        """Atomically claim the first ready task, or ``None`` if nothing is claimable now."""
        with file_lock(self._lock_path):
            tasks = self._read()
            now = self.clock()
            self._expire(tasks, now)
            done = {t.id for t in tasks if t.status is TaskStatus.COMPLETED}
            busy_files = [f for t in tasks if t.status is TaskStatus.IN_PROGRESS for f in t.files]
            for t in tasks:
                if t.status is not TaskStatus.PENDING or t.attempts >= self.max_attempts:
                    continue
                if not all(d in done for d in t.depends_on):
                    continue
                if any(_rel_overlaps(f, b) for f in t.files for b in busy_files):
                    continue
                t.status, t.owner, t.attempts = TaskStatus.IN_PROGRESS, agent, t.attempts + 1
                t.lease_expires = (now + timedelta(seconds=self.lease_seconds)).isoformat()
                self._write(tasks)
                return t
            self._write(tasks)
            return None

    def _finish(self, task_id: str, agent: str, status: TaskStatus, result: str) -> BoardTask:
        with file_lock(self._lock_path):
            tasks = self._read()
            self._expire(tasks, self.clock())
            for t in tasks:
                if t.id == task_id:
                    if t.status is not TaskStatus.IN_PROGRESS or t.owner != agent:
                        raise OwnershipConflict(f"{agent!r} does not hold task {task_id!r} (status={t.status.value}, owner={t.owner})")
                    if status is TaskStatus.PENDING and t.attempts >= self.max_attempts:
                        status = TaskStatus.FAILED  # retries exhausted
                    t.status, t.result, t.lease_expires = status, result[:2_000], None
                    if status is TaskStatus.PENDING:
                        t.owner = None
                    self._write(tasks)
                    return t
            raise MultiAgentError(f"unknown task {task_id!r}")

    def complete(self, task_id: str, agent: str, result: str = "") -> BoardTask:
        return self._finish(task_id, agent, TaskStatus.COMPLETED, result)

    def fail(self, task_id: str, agent: str, error: str = "", *, retry: bool = True) -> BoardTask:
        """Mark failed; with ``retry`` the task returns to pending until ``max_attempts``."""
        return self._finish(task_id, agent, TaskStatus.PENDING if retry else TaskStatus.FAILED, error)

    def renew(self, task_id: str, agent: str) -> None:
        with file_lock(self._lock_path):
            tasks = self._read()
            for t in tasks:
                if t.id == task_id and t.owner == agent and t.status is TaskStatus.IN_PROGRESS:
                    t.lease_expires = (self.clock() + timedelta(seconds=self.lease_seconds)).isoformat()
                    self._write(tasks)
                    return
            raise OwnershipConflict(f"{agent!r} does not hold task {task_id!r}")

    def summary(self) -> dict[str, int]:
        out = {s.value: 0 for s in TaskStatus}
        for t in self.tasks():
            out[t.status.value] += 1
        return out


__all__ = ["TaskBoard", "BoardTask", "TaskStatus"]
