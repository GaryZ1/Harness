"""Small shared helpers: canonical hashing, deterministic bucketing, time and SQLite plumbing."""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def canonical_json(value: Any) -> str:
    """Deterministic JSON (sorted keys, no whitespace). Non-JSON values fall back to ``str``.

    Deterministic serialisation matters twice in this package: hashes must be stable across
    processes, and (Manus, Codex) non-deterministic serialisation silently breaks prefix caching.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_hex(text: str | bytes, key: bytes | None = None) -> str:
    data = text.encode("utf-8") if isinstance(text, str) else text
    if key:
        return hmac.new(key, data, hashlib.sha256).hexdigest()
    return hashlib.sha256(data).hexdigest()


def stable_bucket(key: str, salt: str = "", buckets: int = 10_000) -> int:
    """Map ``key`` to ``[0, buckets)`` deterministically (same answer in every process and run)."""
    digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def from_ts(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


class SQLiteDB:
    """Per-thread SQLite connections (WAL, 30 s busy timeout) with ``BEGIN IMMEDIATE`` transactions.

    Same pattern as :class:`harness_memory.JobStore`: WAL lets readers proceed while one writer
    commits, and ``BEGIN IMMEDIATE`` takes the write lock up front so read-check-write sequences
    are atomic across threads *and* processes.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._all: list[sqlite3.Connection] = []
        self._guard = threading.Lock()

    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA busy_timeout=30000")
            self._local.conn = c
            with self._guard:
                self._all.append(c)
        return c

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        c = self.conn()
        c.execute("BEGIN IMMEDIATE")
        try:
            yield c
        except BaseException:
            c.execute("ROLLBACK")
            raise
        else:
            c.execute("COMMIT")

    def execute(self, sql: str, params: tuple = ()) -> list[tuple]:
        return self.conn().execute(sql, params).fetchall()

    def close(self) -> None:
        with self._guard:
            for c in self._all:
                try:
                    c.close()
                except sqlite3.Error:  # pragma: no cover - best effort
                    pass
            self._all.clear()
        self._local = threading.local()
