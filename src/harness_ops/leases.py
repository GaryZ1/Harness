"""Worker leases with fencing tokens and heartbeats (SQLite; safe across threads and processes).

Research basis (reference §10.3 "Durable execution", "Append-only session log"; checklist item 7):

* A crashed harness must be replaceable (Managed Agents ``wake(sessionId)``) without two workers
  ever driving the same session. A lease names one owner until ``expires_at``; a heartbeat renews
  it while the owner is alive; a crashed owner's lease simply expires and another worker takes over.
* Every acquisition by a *new* owner increments a **fencing token**. Writers guard their commits
  with ``owner + token`` (see :class:`~harness_ops.durable.StepJournal`), so a paused "zombie"
  worker whose lease was taken over can never commit stale results - expiry alone is not enough.
* Same lease-in-SQLite design as Codex's memory pipeline and :class:`harness_memory.JobStore`.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from harness_core.types import Clock, utcnow

from ._util import SQLiteDB
from .errors import LeaseHeldError, LeaseLostError

log = logging.getLogger("harness_ops.leases")

DEFAULT_LEASE_SECONDS = 30.0  # our choice: short enough for fast takeover, renewed every 10 s


@dataclass(frozen=True)
class Lease:
    resource: str
    owner: str
    token: int
    expires_at: float


_SCHEMA = """
CREATE TABLE IF NOT EXISTS leases (
    resource TEXT PRIMARY KEY,
    owner TEXT,
    token INTEGER NOT NULL DEFAULT 0,
    expires_at REAL NOT NULL DEFAULT 0,
    acquired_at REAL
)
"""


class LeaseStore:
    """``acquire`` / ``renew`` / ``release`` leases on named resources."""

    def __init__(self, db: SQLiteDB | str | Path, *, clock: Clock = utcnow) -> None:
        self.db = db if isinstance(db, SQLiteDB) else SQLiteDB(db)
        self.clock = clock
        self.db.conn().execute(_SCHEMA)

    def _now(self) -> float:
        return self.clock().timestamp()

    def acquire(self, resource: str, owner: str, ttl_s: float = DEFAULT_LEASE_SECONDS) -> Lease:
        """Take the lease or raise :class:`LeaseHeldError`. Re-acquiring your own live lease renews it (same token)."""
        if ttl_s <= 0:
            raise ValueError("ttl_s must be > 0")
        now = self._now()
        with self.db.tx() as c:
            row = c.execute("SELECT owner, token, expires_at FROM leases WHERE resource=?", (resource,)).fetchone()
            if row is None:
                token = 1
                c.execute("INSERT INTO leases(resource, owner, token, expires_at, acquired_at) VALUES (?,?,?,?,?)", (resource, owner, token, now + ttl_s, now))
            else:
                cur_owner, cur_token, expires = row
                if cur_owner is not None and cur_owner != owner and expires > now:
                    raise LeaseHeldError(resource, cur_owner, expires)
                token = cur_token if cur_owner == owner and expires > now else cur_token + 1
                c.execute("UPDATE leases SET owner=?, token=?, expires_at=?, acquired_at=? WHERE resource=?", (owner, token, now + ttl_s, now, resource))
        return Lease(resource, owner, token, now + ttl_s)

    def try_acquire(self, resource: str, owner: str, ttl_s: float = DEFAULT_LEASE_SECONDS) -> Lease | None:
        try:
            return self.acquire(resource, owner, ttl_s)
        except LeaseHeldError:
            return None

    def renew(self, lease: Lease, ttl_s: float = DEFAULT_LEASE_SECONDS) -> Lease:
        """Extend a lease we still hold (owner and token match), even if it lapsed but nobody took it."""
        now = self._now()
        with self.db.tx() as c:
            cur = c.execute("UPDATE leases SET expires_at=? WHERE resource=? AND owner=? AND token=?", (now + ttl_s, lease.resource, lease.owner, lease.token))
            if cur.rowcount != 1:
                raise LeaseLostError(f"lease on {lease.resource!r} lost by {lease.owner!r} (token {lease.token})")
        return Lease(lease.resource, lease.owner, lease.token, now + ttl_s)

    def release(self, lease: Lease) -> bool:
        with self.db.tx() as c:
            cur = c.execute(
                "UPDATE leases SET owner=NULL, expires_at=0 WHERE resource=? AND owner=? AND token=?", (lease.resource, lease.owner, lease.token)
            )
            return cur.rowcount == 1

    def holder(self, resource: str) -> Lease | None:
        row = self.db.conn().execute("SELECT owner, token, expires_at FROM leases WHERE resource=?", (resource,)).fetchone()
        if row is None or row[0] is None or row[2] <= self._now():
            return None
        return Lease(resource, row[0], row[1], row[2])

    def is_current(self, lease: Lease) -> bool:
        row = self.db.conn().execute("SELECT owner, token FROM leases WHERE resource=?", (lease.resource,)).fetchone()
        return row is not None and row[0] == lease.owner and row[1] == lease.token

    def expired(self, prefix: str = "") -> list[Lease]:
        """Leases whose owner stopped heart-beating (crashed workers) - candidates for ``wake``."""
        rows = self.db.conn().execute(
            r"SELECT resource, owner, token, expires_at FROM leases WHERE owner IS NOT NULL AND expires_at <= ? AND resource LIKE ? ESCAPE '\'",
            (self._now(), prefix.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_") + "%"),
        ).fetchall()
        return [Lease(*r) for r in rows]


class Heartbeat:
    """Background renewal: calls ``renew()`` every ``interval_s`` until stopped.

    A :class:`LeaseLostError` from ``renew`` sets :attr:`lost`; the owner must check it (and
    :meth:`raise_if_lost`) before committing work. Other exceptions are logged and retried.
    """

    def __init__(self, renew: Callable[[], object], interval_s: float, *, name: str = "harness-ops-heartbeat") -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be > 0")
        self._renew = renew
        self.interval_s = interval_s
        self.lost = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.beats = 0

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self._renew()
                self.beats += 1
            except LeaseLostError:
                self.lost.set()
                return
            except Exception as exc:  # noqa: BLE001 - transient DB errors: keep trying until the lease lapses
                log.warning("heartbeat renew failed: %s", type(exc).__name__)

    def start(self) -> "Heartbeat":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=5)

    def raise_if_lost(self) -> None:
        if self.lost.is_set():
            raise LeaseLostError("lease lost (heartbeat could not renew)")

    def __enter__(self) -> "Heartbeat":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
