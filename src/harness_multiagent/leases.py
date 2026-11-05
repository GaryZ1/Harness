""""Many readers, one writer", enforced in code.

Research basis (reference 8.7 item 11, 8.8 "Parallel writers on shared files"): the rule shared by
Anthropic, Cognition and the controlled studies is to fan out reads, reviews and hypotheses but keep
writes single-threaded. "Two teammates editing the same file leads to overwrites"; sixteen agents
overwrote each other's fixes on one kernel bug. Parallel writers are acceptable only with strict
partitioning (one worktree per writer, a file-ownership map or task locks) and a single integrator.

This module provides the two partitioning primitives:

* :class:`LeaseManager` hands out **exclusive write leases** per resource (a workspace path or any
  string key) and **unlimited read leases**. Readers never block the writer and never block each
  other; a second writer on the same or an *overlapping* path (``/repo`` vs ``/repo/src``) is refused
  or waits. With ``lock_dir`` set, exact-resource leases are also exclusive **across processes**
  (``fcntl.flock`` on a per-resource lock file, released by the kernel if the process dies, so a
  crashed writer can never wedge the workspace). A JSON holder file next to it aids diagnostics.
* :class:`FileOwnership` is the C-compiler / agent-teams style ownership map: writers claim disjoint
  path prefixes, all-or-nothing, and every write is checked against the claimant.

The tool-level enforcement (no write tool executes without a held lease and owned path) lives in
:class:`harness_multiagent.tools.ScopedToolExecutor`.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from harness_core.types import Clock, new_id, utcnow
from harness_memory.locking import atomic_write_text

from .cancel import CancellationToken
from .errors import LeaseHeldError, LeaseNotHeldError, OwnershipConflict

try:  # pragma: no cover - platform dependent
    import fcntl

    _HAS_FCNTL = True
except ImportError:  # pragma: no cover
    _HAS_FCNTL = False

READ = "read"
WRITE = "write"


def normalize_resource(resource: str | os.PathLike[str]) -> str:
    """Paths become absolute, resolved POSIX strings; other keys (``"db:prod"``) pass through."""
    if isinstance(resource, os.PathLike) or (isinstance(resource, str) and (os.sep in resource or resource.startswith("."))):
        return Path(resource).expanduser().resolve().as_posix()
    if not isinstance(resource, str) or not resource:
        raise ValueError("resource must be a non-empty string or path")
    return resource


def _overlaps(a: str, b: str) -> bool:
    if a == b:
        return True
    if a.startswith("/") and b.startswith("/"):
        pa, pb = PurePosixPath(a), PurePosixPath(b)
        return pa in pb.parents or pb in pa.parents
    return False


def _covers(lease_resource: str, target: str) -> bool:
    if lease_resource == target:
        return True
    if lease_resource.startswith("/") and target.startswith("/"):
        return PurePosixPath(lease_resource) in PurePosixPath(target).parents
    return False


@dataclass(frozen=True)
class LeaseInfo:
    resource: str
    holder: str
    lease_id: str
    mode: str
    acquired_at: str


class Lease:
    """A held lease. Idempotent :meth:`release`; usable as a context manager."""

    def __init__(self, manager: "LeaseManager", info: LeaseInfo) -> None:
        self._manager = manager
        self.info = info
        self._released = False

    @property
    def resource(self) -> str:
        return self.info.resource

    @property
    def holder(self) -> str:
        return self.info.holder

    @property
    def held(self) -> bool:
        return not self._released

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._manager._release(self)

    def __enter__(self) -> "Lease":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.info.resource!r}, holder={self.info.holder!r}, held={self.held})"


class WriteLease(Lease):
    def assert_held(self) -> None:
        if not self.held:
            raise LeaseNotHeldError(f"write lease on {self.resource} (holder {self.holder}) was released")


class ReadLease(Lease):
    pass


class LeaseManager:
    """Exclusive write leases and unlimited read leases per resource. Thread-safe."""

    def __init__(self, lock_dir: str | Path | None = None, *, clock: Clock = utcnow) -> None:
        self.lock_dir = Path(lock_dir).expanduser() if lock_dir is not None else None
        if self.lock_dir is not None:
            self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self._cond = threading.Condition()
        self._writers: dict[str, WriteLease] = {}
        self._readers: dict[str, dict[str, ReadLease]] = {}
        self._fds: dict[str, int] = {}

    # ------------------------------------------------------------------ write leases
    def acquire_write(
        self,
        resource: str | os.PathLike[str],
        holder: str,
        *,
        timeout: float = 0.0,
        token: CancellationToken | None = None,
    ) -> WriteLease:
        """Acquire the single write lease on ``resource``.

        ``timeout=0`` (default) refuses immediately with :class:`LeaseHeldError` when any overlapping
        writer exists; a positive timeout waits (cancellable through ``token``).
        """
        key = normalize_resource(resource)
        deadline = time.monotonic() + max(0.0, timeout)
        with self._cond:
            while True:
                conflict = self._conflict_locked(key)
                if conflict is None:
                    # Reserve the in-process slot first, then take the OS lock without holding the
                    # condition (a cross-process wait must not stall unrelated in-process leases).
                    info = LeaseInfo(key, holder, new_id("lease"), WRITE, self.clock().isoformat())
                    lease = WriteLease(self, info)
                    self._writers[key] = lease
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0 or (token is not None and token.cancelled):
                    raise LeaseHeldError(
                        f"write lease on {key} is held by {conflict.holder!r} (overlapping {conflict.resource}); "
                        "one writer per workspace - use a read-only agent, an own_worktree writer, or wait"
                    )
                self._cond.wait(min(remaining, 0.05))
        try:
            fd = self._os_lock(key, deadline, token)
        except BaseException:
            with self._cond:
                if self._writers.get(key) is lease:
                    del self._writers[key]
                self._cond.notify_all()
            raise
        if fd is not None:
            with self._cond:
                self._fds[key] = fd
            self._write_holder(key, info)
        return lease

    def _conflict_locked(self, key: str) -> LeaseInfo | None:
        for res, lease in self._writers.items():
            if _overlaps(res, key):
                return lease.info
        return None

    def _lock_path(self, key: str) -> Path:
        assert self.lock_dir is not None
        return self.lock_dir / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()[:24]}.lock"

    def _os_lock(self, key: str, deadline: float, token: CancellationToken | None) -> int | None:
        if self.lock_dir is None or not _HAS_FCNTL:
            return None
        fd = os.open(self._lock_path(key), os.O_RDWR | os.O_CREAT, 0o600)
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except BlockingIOError:
                if time.monotonic() >= deadline or (token is not None and token.cancelled):
                    os.close(fd)
                    other = self._read_holder(key)
                    raise LeaseHeldError(f"write lease on {key} is held by another process ({other or 'unknown holder'})") from None
                time.sleep(0.05)

    def _holder_path(self, key: str) -> Path:
        return self._lock_path(key).with_suffix(".holder.json")

    def _write_holder(self, key: str, info: LeaseInfo) -> None:
        payload = {"resource": key, "holder": info.holder, "lease_id": info.lease_id, "pid": os.getpid(), "acquired_at": info.acquired_at}
        atomic_write_text(self._holder_path(key), json.dumps(payload))

    def _read_holder(self, key: str) -> str:
        try:
            d = json.loads(self._holder_path(key).read_text(encoding="utf-8"))
            return f"{d.get('holder')} pid={d.get('pid')}"
        except (OSError, ValueError):
            return ""

    # ------------------------------------------------------------------ read leases
    def acquire_read(self, resource: str | os.PathLike[str], holder: str) -> ReadLease:
        """Readers are unlimited and never conflict with the writer (they see its committed state)."""
        key = normalize_resource(resource)
        info = LeaseInfo(key, holder, new_id("lease"), READ, self.clock().isoformat())
        lease = ReadLease(self, info)
        with self._cond:
            self._readers.setdefault(key, {})[info.lease_id] = lease
        return lease

    # ------------------------------------------------------------------ release / queries
    def _release(self, lease: Lease) -> None:
        with self._cond:
            key = lease.info.resource
            if lease.info.mode == WRITE:
                if self._writers.get(key) is lease:
                    del self._writers[key]
                    fd = self._fds.pop(key, None)
                    if fd is not None:
                        try:
                            self._holder_path(key).unlink(missing_ok=True)
                        finally:
                            try:
                                fcntl.flock(fd, fcntl.LOCK_UN)
                            finally:
                                os.close(fd)
            else:
                self._readers.get(key, {}).pop(lease.info.lease_id, None)
                if not self._readers.get(key):
                    self._readers.pop(key, None)
            self._cond.notify_all()

    def writer_of(self, resource: str | os.PathLike[str]) -> LeaseInfo | None:
        key = normalize_resource(resource)
        with self._cond:
            return self._conflict_locked(key)

    def readers(self, resource: str | os.PathLike[str]) -> list[LeaseInfo]:
        key = normalize_resource(resource)
        with self._cond:
            return [r.info for r in self._readers.get(key, {}).values()]

    def holds_write(self, resource: str | os.PathLike[str], holder: str) -> bool:
        key = normalize_resource(resource)
        with self._cond:
            return any(_covers(res, key) and lease.holder == holder for res, lease in self._writers.items())

    def require_write(self, resource: str | os.PathLike[str], holder: str) -> None:
        """Raise :class:`LeaseNotHeldError` unless ``holder`` holds a write lease covering ``resource``."""
        if not self.holds_write(resource, holder):
            raise LeaseNotHeldError(f"{holder!r} holds no write lease covering {normalize_resource(resource)}")


# ----------------------------------------------------------------------------- file ownership


def _norm_rel(path: str) -> str:
    p = PurePosixPath(str(path).replace("\\", "/").strip())
    parts = [x for x in p.parts if x not in ("", ".")]
    if any(x == ".." for x in parts) or p.is_absolute():
        raise OwnershipConflict(f"ownership paths must be relative and inside the workspace: {path!r}")
    return "/".join(parts)


def _rel_overlaps(a: str, b: str) -> bool:
    return a == b or a == "" or b == "" or a.startswith(b + "/") or b.startswith(a + "/")


def _rel_covers(owned: str, path: str) -> bool:
    return owned == "" or path == owned or path.startswith(owned + "/")


class FileOwnership:
    """Disjoint path-prefix claims per writer (workspace-relative). Thread-safe, all-or-nothing."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._claims: dict[str, set[str]] = {}

    def claim(self, owner: str, paths: Iterable[str]) -> tuple[str, ...]:
        wanted = tuple(dict.fromkeys(_norm_rel(p) for p in paths))
        if not wanted:
            raise OwnershipConflict("claim requires at least one path")
        with self._lock:
            for other, owned in self._claims.items():
                if other == owner:
                    continue
                for w in wanted:
                    for o in owned:
                        if _rel_overlaps(w, o):
                            raise OwnershipConflict(f"{owner!r} cannot claim {w!r}: overlaps {o!r} owned by {other!r}")
            self._claims.setdefault(owner, set()).update(wanted)
        return wanted

    def release(self, owner: str) -> None:
        with self._lock:
            self._claims.pop(owner, None)

    def owner_of(self, path: str) -> str | None:
        rel = _norm_rel(path)
        with self._lock:
            for owner, owned in self._claims.items():
                if any(_rel_covers(o, rel) for o in owned):
                    return owner
        return None

    def check(self, owner: str, path: str) -> None:
        rel = _norm_rel(path)
        with self._lock:
            owned = self._claims.get(owner, set())
            if not any(_rel_covers(o, rel) for o in owned):
                raise OwnershipConflict(f"{owner!r} does not own {rel!r} (owns: {sorted(owned) or 'nothing'})")

    def snapshot(self) -> dict[str, list[str]]:
        with self._lock:
            return {k: sorted(v) for k, v in sorted(self._claims.items())}


__all__ = [
    "LeaseManager", "Lease", "WriteLease", "ReadLease", "LeaseInfo", "FileOwnership", "normalize_resource", "READ", "WRITE",
]
