"""Idempotency keys for side-effecting tool calls: a retry or replay returns the recorded outcome.

Research basis (reference §10.3 "Idempotency keys for side-effecting tools", checklist item 7,
pitfall "retrying non-idempotent tools without keys"):

* Key = ``sha256(session_id:step_idx:tool_call_id)`` - stable across retries and replays of the
  same step, different across steps (2026 durable-execution guidance: ``(workflow_id, step_id)``).
* Read-only tools are retry-safe and bypass the store. MCP ``idempotentHint`` is **not** trusted
  (default false, and only a hint from possibly untrusted servers): unknown tools are treated as
  side-effecting.
* The key is passed to the tool (``ctx.metadata["idempotency_key"]``) so it can forward it to
  downstream APIs that accept one, and recorded on the ``execute_tool`` span.
* Reusing a key with a *different* request is an error (:class:`IdempotencyConflictError`), as in
  payment APIs: it means two different side effects collided on one key.

Concurrency: claims use ``BEGIN IMMEDIATE`` plus an in-progress lease, so exactly one thread or
process executes the side effect; concurrent duplicates get :class:`IdempotencyInProgressError`
(retry later) or, once it completed, the recorded result.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

from harness_core.types import Clock, RunContext, ToolCall, ToolExecutor, ToolOutcome, ToolSpec, utcnow

from ._util import SQLiteDB, canonical_json, sha256_hex
from .errors import IdempotencyConflictError, IdempotencyInProgressError, LeaseLostError

T = TypeVar("T")

DEFAULT_IN_PROGRESS_LEASE_S = 300.0  # our choice: longest expected side effect before a crashed claim is taken over


def idempotency_key(session_id: str, step_idx: int, tool_call_id: str) -> str:
    """The research reference key: ``sha256(f"{session_id}:{step_idx}:{tool_call_id}")``."""
    return sha256_hex(f"{session_id}:{step_idx}:{tool_call_id}")


def request_fingerprint(name: str, args: Any) -> str:
    return sha256_hex(canonical_json({"name": name, "input": args}))


@dataclass(frozen=True)
class IdempotentResult(Generic[T]):
    key: str
    value: T
    replayed: bool


_SCHEMA = """
CREATE TABLE IF NOT EXISTS idempotency (
    key TEXT PRIMARY KEY,
    fingerprint TEXT,
    status TEXT NOT NULL,          -- in_progress | completed
    result TEXT,
    owner TEXT,
    lease_expires REAL,
    created_at REAL NOT NULL,
    completed_at REAL
)
"""


class IdempotencyStore:
    def __init__(self, path: str | Path, *, clock: Clock = utcnow, worker_id: str | None = None, lease_seconds: float = DEFAULT_IN_PROGRESS_LEASE_S) -> None:
        self.db = SQLiteDB(path)
        self.clock = clock
        self.worker_id = worker_id or f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.lease_seconds = lease_seconds
        self.db.conn().execute(_SCHEMA)

    def _now(self) -> float:
        return self.clock().timestamp()

    def get(self, key: str) -> Any | None:
        row = self.db.conn().execute("SELECT status, result FROM idempotency WHERE key=?", (key,)).fetchone()
        if row is None or row[0] != "completed":
            return None
        return json.loads(row[1])

    def execute(
        self,
        key: str,
        fn: Callable[[], T],
        *,
        fingerprint: str | None = None,
        dumps: Callable[[Any], str] = lambda v: json.dumps(v, sort_keys=True, ensure_ascii=False, default=str),
        loads: Callable[[str], Any] = json.loads,
    ) -> IdempotentResult[T]:
        """Run ``fn`` at most once per ``key`` (until :meth:`purge`). Exceptions release the claim so a retry can run."""
        owner = f"{self.worker_id}:{uuid.uuid4().hex[:6]}"  # per call: two threads of one worker never share a claim
        now = self._now()
        with self.db.tx() as c:
            row = c.execute("SELECT fingerprint, status, result, owner, lease_expires FROM idempotency WHERE key=?", (key,)).fetchone()
            if row is not None:
                fp, status, result, cur_owner, expires = row
                if fingerprint is not None and fp is not None and fp != fingerprint:
                    raise IdempotencyConflictError(f"idempotency key {key[:16]}... reused with a different request")
                if status == "completed":
                    return IdempotentResult(key, loads(result), True)
                if expires is not None and expires > now:
                    raise IdempotencyInProgressError(f"side effect for key {key[:16]}... is in progress by {cur_owner}")
                c.execute(
                    "UPDATE idempotency SET owner=?, lease_expires=?, fingerprint=COALESCE(fingerprint, ?) WHERE key=?",
                    (owner, now + self.lease_seconds, fingerprint, key),
                )
            else:
                c.execute(
                    "INSERT INTO idempotency(key, fingerprint, status, owner, lease_expires, created_at) VALUES (?,?,?,?,?,?)",
                    (key, fingerprint, "in_progress", owner, now + self.lease_seconds, now),
                )
        try:
            value = fn()
            text = dumps(value)
        except BaseException:
            self._release(key, owner)
            raise
        with self.db.tx() as c:
            cur = c.execute(
                "UPDATE idempotency SET status='completed', result=?, completed_at=?, lease_expires=NULL WHERE key=? AND owner=? AND status='in_progress'",
                (text, self._now(), key, owner),
            )
            if cur.rowcount != 1:
                raise LeaseLostError(f"idempotency claim for {key[:16]}... was taken over before completion (side effect may have run twice)")
        return IdempotentResult(key, loads(text), False)

    def _release(self, key: str, owner: str) -> None:
        with self.db.tx() as c:
            c.execute("UPDATE idempotency SET owner=NULL, lease_expires=NULL WHERE key=? AND owner=? AND status='in_progress'", (key, owner))

    def purge(self, older_than_seconds: float) -> int:
        """Drop completed records older than the retention window (our choice: caller decides; e.g. 7 days)."""
        with self.db.tx() as c:
            cur = c.execute("DELETE FROM idempotency WHERE status='completed' AND completed_at < ?", (self._now() - older_than_seconds,))
            return cur.rowcount

    def close(self) -> None:
        self.db.close()


class _Uncached(Exception):
    def __init__(self, outcome: ToolOutcome) -> None:
        super().__init__("uncached tool error")
        self.outcome = outcome


def _outcome_dumps(o: ToolOutcome) -> str:
    return json.dumps({"content": o.content, "is_error": o.is_error, "metadata": o.metadata}, sort_keys=True, ensure_ascii=False, default=str)


def _outcome_loads(s: str) -> ToolOutcome:
    return ToolOutcome(**json.loads(s))


class IdempotentToolExecutor:
    """``ToolExecutor`` wrapper: side-effecting calls execute at most once per idempotency key.

    ``step_index_key`` names the ``ctx.metadata`` entry holding the loop step index (default
    ``"step_idx"``; ``0`` when absent). Error outcomes are not cached by default (``cache_errors``),
    so a retry can re-attempt a failed side effect; raised exceptions are never cached.
    """

    def __init__(
        self,
        inner: ToolExecutor,
        store: IdempotencyStore,
        *,
        step_index_key: str = "step_idx",
        cache_errors: bool = False,
        is_retry_safe: Callable[[ToolSpec | None], bool] = lambda spec: spec is not None and spec.read_only,
    ) -> None:
        self.inner = inner
        self.store = store
        self.step_index_key = step_index_key
        self.cache_errors = cache_errors
        self.is_retry_safe = is_retry_safe

    def specs(self) -> list[ToolSpec]:
        return self.inner.specs()

    def _spec(self, name: str) -> ToolSpec | None:
        return next((s for s in self.inner.specs() if s.name == name), None)

    def key_for(self, call: ToolCall, ctx: RunContext) -> str:
        return idempotency_key(ctx.session_id, int(ctx.metadata.get(self.step_index_key, 0) or 0), call.id)

    def execute(self, call: ToolCall, ctx: RunContext) -> ToolOutcome:
        spec = self._spec(call.name)
        if self.is_retry_safe(spec):
            return self.inner.execute(call, ctx)
        key = self.key_for(call, ctx)
        ctx2 = replace(ctx, metadata={**ctx.metadata, "idempotency_key": key})

        def run() -> ToolOutcome:
            outcome = self.inner.execute(call, ctx2)
            if outcome.is_error and not self.cache_errors:
                raise _Uncached(outcome)
            return outcome

        try:
            res = self.store.execute(key, run, fingerprint=request_fingerprint(call.name, call.input), dumps=_outcome_dumps, loads=_outcome_loads)
        except _Uncached as u:
            out = u.outcome
            out.metadata = {**out.metadata, "idempotency_key": key, "idempotent_replay": False}
            return out
        out = res.value
        out.metadata = {**out.metadata, "idempotency_key": key, "idempotent_replay": res.replayed}
        return out
