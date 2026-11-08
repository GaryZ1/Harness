"""Rainbow deployments: versioned harness builds, per-session pinning, deterministic traffic splits.

Research basis (reference §10.3 "Rainbow deployments and versioned workflows", checklist item 17,
pitfall "Hot-swapping prompts or tools under in-flight agents"):

* Anthropic runs old and new versions side by side and shifts traffic gradually, so agents in
  flight are not broken. Prompts, tools and workflow code are versioned **together**
  (``prompts_tools_sha``) and stamped on every root span.
* **Invariant:** a session finishes on the version it started on. :meth:`DeploymentRegistry.pin`
  is idempotent per session and never re-routes a pinned session - not on a weight change, not on
  drain, not on rollback (unless the operator explicitly aborts in-flight sessions).
* New sessions are routed by a deterministic hash of a routing key (session or user id), so the
  assignment is reproducible across processes and a ramp from 10% to 50% only *adds* sessions to
  the new version (the newest version owns the low end of the hash range).
* Drain stops new traffic; retire is refused while sessions are in flight; rollback sends all new
  traffic back to a known-good version.

State lives in SQLite so every harness process sees the same weights and pins.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from harness_core.types import Clock, utcnow

from ._util import SQLiteDB, canonical_json, from_ts, sha256_hex, stable_bucket
from .errors import DeployError, SessionAbortedError
from .semconv import A

BUCKETS = 10_000


class VersionState(str, Enum):
    ACTIVE = "active"  # eligible for new sessions (weight > 0)
    DRAINING = "draining"  # no new sessions; in-flight sessions finish here
    RETIRED = "retired"  # no sessions at all
    ROLLED_BACK = "rolled_back"  # bad build; no new sessions


@dataclass(frozen=True)
class HarnessVersion:
    version_id: str
    prompts_tools_sha: str
    created_at: datetime
    state: VersionState = VersionState.ACTIVE
    weight: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def span_attributes(self) -> dict[str, Any]:
        """Stamp these on the ``invoke_agent`` root span so regressions can be attributed."""
        return {A.VERSION: self.version_id, A.PROMPTS_TOOLS_SHA: self.prompts_tools_sha}


def prompts_tools_sha(system_prompt: str, tools: list[dict[str, Any]], extra: Mapping[str, Any] | None = None) -> str:
    """Content hash of everything the model sees as scaffold (system prompt + tool definitions + extras)."""
    return sha256_hex(canonical_json({"system": system_prompt, "tools": tools, "extra": dict(extra or {})}))


@dataclass(frozen=True)
class SessionPin:
    session_id: str
    version_id: str
    pinned_at: datetime
    status: str  # active | finished | aborted
    snapshot: Mapping[str, Any] = field(default_factory=dict)  # frozen per-session config (e.g. feature flags)


_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS versions (
        version_id TEXT PRIMARY KEY,
        prompts_tools_sha TEXT NOT NULL,
        state TEXT NOT NULL,
        weight REAL NOT NULL DEFAULT 0,
        ordinal INTEGER NOT NULL,
        metadata TEXT,
        created_at REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS pins (
        session_id TEXT PRIMARY KEY,
        version_id TEXT NOT NULL REFERENCES versions(version_id),
        status TEXT NOT NULL,
        snapshot TEXT,
        pinned_at REAL NOT NULL,
        finished_at REAL
    )""",
    "CREATE INDEX IF NOT EXISTS pins_by_version ON pins(version_id, status)",
)


class DeploymentRegistry:
    """Version registry + traffic router + session pins."""

    def __init__(self, path: str | Path, *, clock: Clock = utcnow, salt: str = "harness-rainbow") -> None:
        self.db = SQLiteDB(path)
        self.clock = clock
        self.salt = salt
        with self.db.tx() as c:
            for stmt in _SCHEMA:
                c.execute(stmt)

    def _now(self) -> float:
        return self.clock().timestamp()

    # ------------------------------------------------------------------ versions

    def register(self, version_id: str, prompts_tools_sha: str, *, metadata: Mapping[str, Any] | None = None, weight: float = 0.0) -> HarnessVersion:
        """Register a build. Re-registering the same id with a different SHA is refused (versions are immutable)."""
        if not version_id or not prompts_tools_sha:
            raise DeployError("version_id and prompts_tools_sha are required")
        with self.db.tx() as c:
            row = c.execute("SELECT prompts_tools_sha FROM versions WHERE version_id=?", (version_id,)).fetchone()
            if row is not None:
                if row[0] != prompts_tools_sha:
                    raise DeployError(f"version {version_id!r} already registered with a different prompts_tools_sha; versions are immutable")
            else:
                ordinal = c.execute("SELECT COALESCE(MAX(ordinal), 0) + 1 FROM versions").fetchone()[0]
                c.execute(
                    "INSERT INTO versions(version_id, prompts_tools_sha, state, weight, ordinal, metadata, created_at) VALUES (?,?,?,?,?,?,?)",
                    (version_id, prompts_tools_sha, VersionState.ACTIVE.value, 0.0, ordinal, canonical_json(dict(metadata or {})), self._now()),
                )
        if weight:
            self.set_weights({**self.weights(), version_id: weight})
        return self.version(version_id)

    def version(self, version_id: str) -> HarnessVersion:
        row = self.db.conn().execute(
            "SELECT version_id, prompts_tools_sha, created_at, state, weight, metadata FROM versions WHERE version_id=?", (version_id,)
        ).fetchone()
        if row is None:
            raise DeployError(f"unknown version {version_id!r}")
        return HarnessVersion(row[0], row[1], from_ts(row[2]), VersionState(row[3]), row[4], json.loads(row[5] or "{}"))

    def versions(self) -> list[HarnessVersion]:
        ids = [r[0] for r in self.db.execute("SELECT version_id FROM versions ORDER BY ordinal")]
        return [self.version(v) for v in ids]

    def weights(self) -> dict[str, float]:
        return {r[0]: r[1] for r in self.db.execute("SELECT version_id, weight FROM versions WHERE weight > 0 ORDER BY ordinal")}

    def set_weights(self, weights: Mapping[str, float]) -> dict[str, float]:
        """Replace the traffic split. Weights are relative (normalised); unnamed versions get 0.

        Only ``active`` (or ``draining`` - which re-activates it) versions may receive weight; the
        total must be positive.
        """
        if any(w < 0 for w in weights.values()):
            raise DeployError("weights must be >= 0")
        if sum(weights.values()) <= 0:
            raise DeployError("at least one version needs a positive weight")
        with self.db.tx() as c:
            states = {r[0]: r[1] for r in c.execute("SELECT version_id, state FROM versions").fetchall()}
            for vid, w in weights.items():
                if vid not in states:
                    raise DeployError(f"unknown version {vid!r}")
                if w > 0 and states[vid] in (VersionState.RETIRED.value, VersionState.ROLLED_BACK.value):
                    raise DeployError(f"version {vid!r} is {states[vid]} and cannot take traffic")
            c.execute("UPDATE versions SET weight=0")
            for vid, w in weights.items():
                new_state = VersionState.ACTIVE.value if w > 0 else (VersionState.DRAINING.value if states[vid] == VersionState.ACTIVE.value else states[vid])
                c.execute("UPDATE versions SET weight=?, state=? WHERE version_id=?", (float(w), new_state, vid))
            for vid, st in states.items():
                if vid not in weights and st == VersionState.ACTIVE.value:
                    c.execute("UPDATE versions SET state=? WHERE version_id=?", (VersionState.DRAINING.value, vid))
        return self.weights()

    def shift(self, to_version: str, percent: float) -> dict[str, float]:
        """Gradual ramp: give ``to_version`` ``percent`` of new traffic, scaling the others proportionally."""
        if not 0 <= percent <= 100:
            raise DeployError("percent must be within [0, 100]")
        current = {k: v for k, v in self.weights().items() if k != to_version}
        rest = sum(current.values())
        if percent < 100 and rest <= 0:
            raise DeployError("no other version carries traffic to shift from")
        new = {k: (100 - percent) * v / rest for k, v in current.items()} if rest else {}
        new[to_version] = percent
        return self.set_weights({k: v for k, v in new.items() if v > 0} or {to_version: 100.0})

    # ------------------------------------------------------------------ routing & pins

    def route(self, routing_key: str) -> str:
        """Pure function of (routing_key, current weights): which version a *new* session would get."""
        rows = self.db.execute("SELECT version_id, weight FROM versions WHERE weight > 0 AND state=? ORDER BY ordinal DESC", (VersionState.ACTIVE.value,))
        if not rows:
            raise DeployError("no version is taking traffic")
        total = sum(w for _, w in rows)
        bucket = stable_bucket(routing_key, self.salt, BUCKETS)
        edge = 0.0
        for vid, w in rows:  # newest first: ramping the newest version only moves sessions *to* it
            edge += w / total * BUCKETS
            if bucket < edge:
                return vid
        return rows[-1][0]

    def pin(self, session_id: str, *, routing_key: str | None = None, snapshot: Mapping[str, Any] | None = None) -> SessionPin:
        """Return the session's pinned version, pinning it now if it is new. Idempotent and race-free."""
        with self.db.tx() as c:
            row = c.execute("SELECT session_id, version_id, pinned_at, status, snapshot FROM pins WHERE session_id=?", (session_id,)).fetchone()
            if row is None:
                vid = self.route(routing_key or session_id)
                c.execute(
                    "INSERT INTO pins(session_id, version_id, status, snapshot, pinned_at) VALUES (?,?,?,?,?)",
                    (session_id, vid, "active", canonical_json(dict(snapshot or {})), self._now()),
                )
                row = (session_id, vid, self._now(), "active", canonical_json(dict(snapshot or {})))
        pin = SessionPin(row[0], row[1], from_ts(row[2]), row[3], json.loads(row[4] or "{}"))
        if pin.status == "aborted":
            raise SessionAbortedError(f"session {session_id!r} was aborted by rollback of {pin.version_id!r}")
        return pin

    def pinned_version(self, session_id: str) -> HarnessVersion | None:
        row = self.db.conn().execute("SELECT version_id FROM pins WHERE session_id=?", (session_id,)).fetchone()
        return self.version(row[0]) if row else None

    def finish(self, session_id: str) -> None:
        with self.db.tx() as c:
            c.execute("UPDATE pins SET status='finished', finished_at=? WHERE session_id=? AND status='active'", (self._now(), session_id))

    def inflight(self, version_id: str) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM pins WHERE version_id=? AND status='active'", (version_id,))[0][0])

    def inflight_sessions(self, version_id: str) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT session_id FROM pins WHERE version_id=? AND status='active' ORDER BY pinned_at", (version_id,))]

    # ------------------------------------------------------------------ drain / retire / rollback

    def drain(self, version_id: str) -> int:
        """Stop new sessions on ``version_id``; returns how many sessions are still in flight there."""
        v = self.version(version_id)
        remaining = {k: w for k, w in self.weights().items() if k != version_id}
        if v.weight > 0 and not remaining:
            raise DeployError(f"cannot drain {version_id!r}: it is the only version taking traffic")
        with self.db.tx() as c:
            c.execute("UPDATE versions SET weight=0, state=? WHERE version_id=? AND state=?", (VersionState.DRAINING.value, version_id, VersionState.ACTIVE.value))
        return self.inflight(version_id)

    def is_drained(self, version_id: str) -> bool:
        return self.version(version_id).weight == 0 and self.inflight(version_id) == 0

    def retire(self, version_id: str) -> None:
        """Remove a drained version. Refused while any session is still pinned to it and active."""
        if self.version(version_id).weight > 0:
            raise DeployError(f"drain {version_id!r} before retiring it")
        n = self.inflight(version_id)
        if n:
            raise DeployError(f"{n} session(s) still in flight on {version_id!r}; sessions finish on the version they started on")
        with self.db.tx() as c:
            c.execute("UPDATE versions SET state=? WHERE version_id=?", (VersionState.RETIRED.value, version_id))

    def rollback(self, bad_version: str, to_version: str, *, abort_inflight: bool = False) -> list[str]:
        """Send all new traffic to ``to_version`` and mark ``bad_version`` rolled back.

        In-flight sessions on the bad version keep running on it by default (the rainbow
        invariant); with ``abort_inflight=True`` they are marked aborted and their next
        :meth:`pin` raises :class:`SessionAbortedError`. Returns the affected session ids.
        """
        if bad_version == to_version:
            raise DeployError("rollback target must differ from the bad version")
        target = self.version(to_version)
        if target.state in (VersionState.RETIRED, VersionState.ROLLED_BACK):
            raise DeployError(f"rollback target {to_version!r} is {target.state.value}")
        self.version(bad_version)
        affected = self.inflight_sessions(bad_version)
        with self.db.tx() as c:
            c.execute("UPDATE versions SET weight=0, state=CASE WHEN state=? THEN ? ELSE state END", (VersionState.ACTIVE.value, VersionState.DRAINING.value))
            c.execute("UPDATE versions SET weight=100, state=? WHERE version_id=?", (VersionState.ACTIVE.value, to_version))
            c.execute("UPDATE versions SET state=? WHERE version_id=?", (VersionState.ROLLED_BACK.value, bad_version))
            if abort_inflight:
                c.execute("UPDATE pins SET status='aborted', finished_at=? WHERE version_id=? AND status='active'", (self._now(), bad_version))
        return affected

    def close(self) -> None:
        self.db.close()
