"""Cross-thread and cross-process file locks (POSIX ``fcntl``; thread-only fallback elsewhere)."""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .errors import LockUnavailableError

try:  # pragma: no cover - platform dependent
    import fcntl

    _HAS_FCNTL = True
except ImportError:  # pragma: no cover
    _HAS_FCNTL = False

_THREAD_LOCKS: dict[str, threading.RLock] = {}
_REGISTRY_GUARD = threading.Lock()
_HELD = threading.local()  # per-thread {path: (depth, fd)} so nested acquisitions are re-entrant


def _thread_lock(key: str) -> threading.RLock:
    with _REGISTRY_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = _THREAD_LOCKS[key] = threading.RLock()
        return lock


@contextmanager
def file_lock(path: str | Path, *, blocking: bool = True, timeout: float | None = None) -> Iterator[None]:
    """Hold an exclusive, re-entrant (per thread) lock on ``path`` (created if missing).

    ``blocking=False`` raises :class:`LockUnavailableError` immediately when another thread or
    process holds the lock; ``timeout`` bounds the wait in blocking mode.
    """
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    key = str(lock_path.resolve())
    held: dict[str, list] = getattr(_HELD, "locks", None) or {}
    _HELD.locks = held
    if key in held:  # re-entrant acquisition in the same thread
        held[key][0] += 1
        try:
            yield
        finally:
            held[key][0] -= 1
        return

    tlock = _thread_lock(key)
    acquired = tlock.acquire(blocking=blocking, timeout=-1 if (timeout is None or not blocking) else timeout)
    if not acquired:
        raise LockUnavailableError(f"lock busy: {lock_path}")
    fd = None
    try:
        if _HAS_FCNTL:
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            deadline = None if timeout is None else time.monotonic() + timeout
            while True:
                try:
                    flags = fcntl.LOCK_EX | (0 if (blocking and timeout is None) else fcntl.LOCK_NB)
                    fcntl.flock(fd, flags)
                    break
                except BlockingIOError as exc:
                    if not blocking or (deadline is not None and time.monotonic() >= deadline):
                        os.close(fd)
                        fd = None
                        raise LockUnavailableError(f"lock busy: {lock_path}") from exc
                    time.sleep(0.05)
        held[key] = [1, fd]
        try:
            yield
        finally:
            del held[key]
    finally:
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        tlock.release()


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (temp file + fsync + ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
