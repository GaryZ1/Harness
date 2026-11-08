"""Small file-locking and atomic-write helpers (POSIX ``fcntl``; thread-only fallback elsewhere).

Kept private and dependency-free so :mod:`harness_core` never imports a sibling package.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:  # pragma: no cover - platform dependent
    import fcntl

    HAS_FCNTL = True
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]
    HAS_FCNTL = False

_THREAD_LOCKS: dict[str, threading.Lock] = {}
_GUARD = threading.Lock()


def _thread_lock(key: str) -> threading.Lock:
    with _GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = _THREAD_LOCKS[key] = threading.Lock()
        return lock


@contextmanager
def exclusive_lock(path: str | Path) -> Iterator[None]:
    """Blocking exclusive lock across threads and processes (not re-entrant)."""
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    tlock = _thread_lock(str(lock_path.resolve()))
    with tlock:
        if not HAS_FCNTL:  # pragma: no cover
            yield
            return
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


class Lease:
    """A non-blocking exclusive lease on a file, held until :meth:`release`.

    Backed by ``flock`` on its own open file description, so a second lease on the same path
    fails even inside the same process, and the OS drops the lease if the holder crashes, which
    is what makes ``wake(session_id)`` after a crash safe.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fd: int | None = None
        self._tlock: threading.Lock | None = None

    def try_acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tlock = _thread_lock("lease:" + str(self.path.resolve()))
        if not tlock.acquire(blocking=False):
            return False
        if not HAS_FCNTL:  # pragma: no cover
            self._tlock = tlock
            return True
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            tlock.release()
            return False
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        self._fd, self._tlock = fd, tlock
        return True

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None
        if self._tlock is not None:
            self._tlock.release()
            self._tlock = None

    @property
    def held(self) -> bool:
        return self._tlock is not None


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (temp file + fsync + ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
