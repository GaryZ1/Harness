"""Durable execution: a SQLite step journal so a crashed run resumes without re-running completed steps.

Research basis (reference §10.3 "Durable execution: workflow plus activities", "Deterministic replay
for orchestration scripts"; checklist items 7-8; build plan "Operations"):

* Each agent session is a durable workflow keyed by ``session_id``; every model call and every
  side-effecting tool call is a **step** whose result is persisted, so replay never re-bills a
  model call or re-executes a completed side effect (Temporal activities, Restate/DBOS journals,
  Anthropic "resume from where the agent was when the errors occurred").
* Replay follows Claude Code workflow semantics: completed steps return their recorded result *up
  to the first step whose input differs*; that step **and every later one** re-run
  (``strict_determinism=True`` raises :class:`NonDeterminismError` instead).
* Orchestration code must be deterministic: use :meth:`DurableRun.now` / :meth:`DurableRun.random`
  (recorded values) instead of wall-clock or randomness inside a run.
* One worker drives a run at a time: the run is leased (:mod:`harness_ops.leases`) with a
  heartbeat, and every journal write is fenced by ``owner + token``, so a zombie worker that lost
  its lease cannot commit.

Guarantee: a step whose result was committed never runs again for that run. A step interrupted
between finishing ``fn`` and committing **does** run again (at-least-once) - give side-effecting
steps an idempotency key (:meth:`DurableRun.idempotency_key`, :mod:`harness_ops.idempotency`).
"""

from __future__ import annotations

import contextvars
import json
import os
import random
import threading
import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

from harness_core.types import (
    Clock,
    ModelClient,
    ModelRequest,
    ModelResponse,
    StopReason,
    ToolOutcome,
    Tracer,
    Usage,
    block_from_api,
    messages_to_api,
    utcnow,
)
from harness_memory import SecretRedactor

from ._util import SQLiteDB, canonical_json, from_ts, sha256_hex
from .errors import DuplicateStepError, LeaseLostError, NoActiveRunError, NonDeterminismError, OpsError, StepSerializationError
from .leases import DEFAULT_LEASE_SECONDS, Heartbeat, Lease, LeaseStore
from .retry import RetryPolicy, retry_call
from .semconv import A, HarnessOp

T = TypeVar("T")


@dataclass(frozen=True)
class Codec:
    """How a step result is stored. ``loads(dumps(x))`` is what *both* first execution and replay return."""

    dumps: Callable[[Any], str]
    loads: Callable[[str], Any]


JSON_CODEC = Codec(lambda v: json.dumps(v, sort_keys=True, ensure_ascii=False, allow_nan=False), json.loads)


def response_to_dict(r: ModelResponse) -> dict[str, Any]:
    return {"content": [b.to_api() for b in r.content], "stop_reason": r.stop_reason.value, "usage": asdict(r.usage), "model": r.model, "id": r.id}


def response_from_dict(d: dict[str, Any]) -> ModelResponse:
    return ModelResponse(
        content=[block_from_api(b) for b in d.get("content", [])],
        stop_reason=StopReason.parse(d.get("stop_reason")),
        usage=Usage(**d.get("usage", {})),
        model=d.get("model", ""),
        id=d.get("id", ""),
    )


MODEL_RESPONSE_CODEC = Codec(lambda r: json.dumps(response_to_dict(r), sort_keys=True, ensure_ascii=False), lambda s: response_from_dict(json.loads(s)))
TOOL_OUTCOME_CODEC = Codec(
    lambda o: json.dumps({"content": o.content, "is_error": o.is_error, "metadata": o.metadata}, sort_keys=True, ensure_ascii=False, default=str),
    lambda s: ToolOutcome(**json.loads(s)),
)


def request_fingerprint(request: ModelRequest) -> dict[str, Any]:
    """The replay key of a model call: everything that changes what the model sees."""
    return {
        "system": request.system,
        "messages": messages_to_api(request.messages),
        "tools": [t.to_api() for t in request.tools],
        "max_tokens": request.max_tokens,
        "extra": request.extra,
    }


class StepStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class StepRecord:
    run_id: str
    key: str
    status: StepStatus
    input_hash: str | None
    result: str | None
    error: str | None
    attempts: int
    seq: int | None
    started_at: datetime | None
    finished_at: datetime | None
    token: int


_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        result TEXT,
        error TEXT,
        metadata TEXT,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS steps (
        run_id TEXT NOT NULL,
        step_key TEXT NOT NULL,
        status TEXT NOT NULL,
        input_hash TEXT,
        result TEXT,
        error TEXT,
        attempts INTEGER NOT NULL DEFAULT 0,
        seq INTEGER,
        token INTEGER NOT NULL DEFAULT 0,
        started_at REAL,
        finished_at REAL,
        PRIMARY KEY (run_id, step_key)
    )""",
)

_ACTIVE: contextvars.ContextVar["DurableRun | None"] = contextvars.ContextVar("harness_ops_active_run", default=None)


def current_run() -> "DurableRun | None":
    return _ACTIVE.get()


def durable_step(key: str, fn: Callable[[], T], **kwargs: Any) -> T:
    """Run ``fn`` as step ``key`` of the run activated with ``with run:`` (see :meth:`DurableRun.step`)."""
    run = _ACTIVE.get()
    if run is None:
        raise NoActiveRunError("durable_step() needs an active run: use `with journal.open_run(run_id) as run:`")
    return run.step(key, fn, **kwargs)


class StepJournal:
    """The SQLite journal shared by every worker (WAL; one file per deployment or per tenant)."""

    def __init__(self, path: str | Path, *, clock: Clock = utcnow, redactor: SecretRedactor | None = None) -> None:
        self.db = SQLiteDB(path)
        self.clock = clock
        self.redactor = redactor or SecretRedactor()
        with self.db.tx() as c:
            for stmt in _SCHEMA:
                c.execute(stmt)
        self.leases = LeaseStore(self.db, clock=clock)

    def _now(self) -> float:
        return self.clock().timestamp()

    @staticmethod
    def lease_resource(run_id: str) -> str:
        return f"run:{run_id}"

    # ------------------------------------------------------------------ opening runs

    def open_run(
        self,
        run_id: str,
        *,
        worker_id: str | None = None,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
        heartbeat: bool = True,
        strict_determinism: bool = False,
        metadata: dict[str, Any] | None = None,
        tracer: Tracer | None = None,
    ) -> "DurableRun":
        """Lease ``run_id`` for this worker (raises :class:`~harness_ops.errors.LeaseHeldError` if a live worker has it)."""
        owner = worker_id or f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        lease = self.leases.acquire(self.lease_resource(run_id), owner, lease_seconds)
        now = self._now()
        with self.db.tx() as c:
            c.execute(
                "INSERT INTO runs(run_id, status, metadata, created_at, updated_at) VALUES (?,?,?,?,?) "
                "ON CONFLICT(run_id) DO UPDATE SET updated_at=excluded.updated_at",
                (run_id, RunStatus.RUNNING.value, canonical_json(metadata or {}), now, now),
            )
        return DurableRun(self, run_id, lease, lease_seconds=lease_seconds, heartbeat=heartbeat, strict_determinism=strict_determinism, tracer=tracer)

    def run(self, run_id: str, workflow: Callable[["DurableRun"], T], **open_kwargs: Any) -> T:
        """Open (or resume) ``run_id``, execute ``workflow(run)``, record completion, release the lease.

        If ``workflow`` raises, the run stays ``running`` (resumable) and the lease is released.
        """
        with self.open_run(run_id, **open_kwargs) as run:
            result = workflow(run)
            run.complete(result)
            return result

    # ------------------------------------------------------------------ queries

    def step_record(self, run_id: str, key: str) -> StepRecord | None:
        row = self.db.conn().execute(
            "SELECT run_id, step_key, status, input_hash, result, error, attempts, seq, started_at, finished_at, token FROM steps WHERE run_id=? AND step_key=?",
            (run_id, key),
        ).fetchone()
        return self._record(row) if row else None

    def steps(self, run_id: str) -> list[StepRecord]:
        rows = self.db.conn().execute(
            "SELECT run_id, step_key, status, input_hash, result, error, attempts, seq, started_at, finished_at, token FROM steps WHERE run_id=? ORDER BY seq, step_key",
            (run_id,),
        ).fetchall()
        return [self._record(r) for r in rows]

    @staticmethod
    def _record(row: tuple) -> StepRecord:
        return StepRecord(
            row[0], row[1], StepStatus(row[2]), row[3], row[4], row[5], row[6], row[7],
            from_ts(row[8]) if row[8] is not None else None, from_ts(row[9]) if row[9] is not None else None, row[10],
        )

    def run_info(self, run_id: str) -> dict[str, Any] | None:
        row = self.db.conn().execute("SELECT run_id, status, result, error, metadata, created_at, updated_at FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            return None
        return {
            "run_id": row[0], "status": row[1], "result": row[2], "error": row[3], "metadata": json.loads(row[4] or "{}"),
            "created_at": from_ts(row[5]), "updated_at": from_ts(row[6]),
        }

    def resumable_runs(self) -> list[str]:
        """Runs still ``running`` whose lease is free or expired: crashed workers to ``wake``."""
        rows = self.db.conn().execute(
            "SELECT r.run_id FROM runs r LEFT JOIN leases l ON l.resource = 'run:' || r.run_id "
            "WHERE r.status = ? AND (l.owner IS NULL OR l.expires_at <= ?) ORDER BY r.updated_at",
            (RunStatus.RUNNING.value, self._now()),
        ).fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        self.db.close()

    # ------------------------------------------------------------------ fenced writes

    def _fence(self, c: Any, lease: Lease) -> None:
        row = c.execute("SELECT owner, token FROM leases WHERE resource=?", (lease.resource,)).fetchone()
        if row is None or row[0] != lease.owner or row[1] != lease.token:
            raise LeaseLostError(f"{lease.owner!r} no longer holds {lease.resource!r} (token {lease.token}); refusing to write")

    def _begin_step(self, lease: Lease, run_id: str, key: str, input_hash: str | None, seq: int) -> None:
        now = self._now()
        with self.db.tx() as c:
            self._fence(c, lease)
            c.execute(
                "INSERT INTO steps(run_id, step_key, status, input_hash, attempts, seq, token, started_at) VALUES (?,?,?,?,1,?,?,?) "
                "ON CONFLICT(run_id, step_key) DO UPDATE SET status=excluded.status, input_hash=excluded.input_hash, "
                "attempts=steps.attempts+1, seq=excluded.seq, token=excluded.token, started_at=excluded.started_at, "
                "result=NULL, error=NULL, finished_at=NULL",
                (run_id, key, StepStatus.RUNNING.value, input_hash, seq, lease.token, now),
            )

    def _end_step(self, lease: Lease, run_id: str, key: str, status: StepStatus, result: str | None, error: str | None) -> None:
        now = self._now()
        with self.db.tx() as c:
            self._fence(c, lease)
            c.execute(
                "UPDATE steps SET status=?, result=?, error=?, finished_at=? WHERE run_id=? AND step_key=? AND token=?",
                (status.value, result, error, now, run_id, key, lease.token),
            )
            c.execute("UPDATE runs SET updated_at=? WHERE run_id=?", (now, run_id))

    def _finish_run(self, lease: Lease, run_id: str, status: RunStatus, result: str | None, error: str | None) -> None:
        with self.db.tx() as c:
            self._fence(c, lease)
            c.execute("UPDATE runs SET status=?, result=?, error=?, updated_at=? WHERE run_id=?", (status.value, result, error, self._now(), run_id))

    def _error_text(self, exc: BaseException) -> str:
        return f"{type(exc).__qualname__}: {self.redactor.redact(str(exc))[:500]}"


class DurableRun:
    """One worker's handle on a leased run. Use as a context manager (activates :func:`durable_step`).

    Steps may be called from several threads of the same run (parallel read-only fan-out); step
    keys must be unique within one execution.
    """

    def __init__(
        self,
        journal: StepJournal,
        run_id: str,
        lease: Lease,
        *,
        lease_seconds: float,
        heartbeat: bool,
        strict_determinism: bool,
        tracer: Tracer | None,
    ) -> None:
        self.journal = journal
        self.run_id = run_id
        self._lease = lease
        self.lease_seconds = lease_seconds
        self.strict_determinism = strict_determinism
        self.tracer = tracer
        self._lock = threading.Lock()
        self._seen: set[str] = set()
        self._seq = 0
        self.diverged_at: str | None = None
        self.executed: list[str] = []
        self.replayed: list[str] = []
        self._closed = False
        self._tokens: list[contextvars.Token] = []
        self._heartbeat: Heartbeat | None = None
        if heartbeat:
            self._heartbeat = Heartbeat(self._renew, max(lease_seconds / 3.0, 0.05), name=f"heartbeat-{run_id}").start()

    # ------------------------------------------------------------------ lease

    @property
    def worker_id(self) -> str:
        return self._lease.owner

    @property
    def fencing_token(self) -> int:
        return self._lease.token

    def _renew(self) -> None:
        self._lease = self.journal.leases.renew(self._lease, self.lease_seconds)

    def _check_lease(self) -> None:
        if self._closed:
            raise OpsError(f"run {self.run_id!r} is closed")
        if self._heartbeat is not None:
            self._heartbeat.raise_if_lost()

    # ------------------------------------------------------------------ steps

    def idempotency_key(self, step_key: str) -> str:
        """Stable across retries and replays of the same step; differs across steps (``(workflow_id, step_id)``)."""
        return sha256_hex(f"{self.run_id}:{step_key}")

    def step(
        self,
        key: str,
        fn: Callable[[], T],
        *,
        input: Any = None,
        codec: Codec = JSON_CODEC,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> T:
        """Return ``key``'s recorded result, or run ``fn`` (optionally under ``retry``) and record it.

        ``input`` is any JSON-able description of what ``fn`` depends on (a prompt, tool args); a
        changed input marks the run as diverged from that step on.
        """
        self._check_lease()
        input_hash = sha256_hex(canonical_json(input)) if input is not None else None
        with self._lock:
            if key in self._seen:
                raise DuplicateStepError(f"step key {key!r} used twice in run {self.run_id!r}")
            self._seen.add(key)
            seq = self._seq
            self._seq += 1
            diverged = self.diverged_at is not None
        rec = self.journal.step_record(self.run_id, key)
        if rec is not None and rec.status is StepStatus.COMPLETED and rec.result is not None:
            if rec.input_hash != input_hash:
                if self.strict_determinism:
                    raise NonDeterminismError(f"step {key!r} of run {self.run_id!r}: recorded input differs from current input")
                with self._lock:
                    self.diverged_at = self.diverged_at or key
                    diverged = True
            if not diverged:
                with self._lock:
                    self.replayed.append(key)
                self._trace(key, replayed=True)
                return codec.loads(rec.result)
        return self._execute(key, fn, input_hash, seq, codec, retry, sleep)

    durable_step = step

    def was_replayed(self, key: str) -> bool:
        """``True`` if ``key`` returned a recorded result in this execution (telemetry: do not re-count it)."""
        with self._lock:
            return key in self.replayed

    def _trace(self, key: str, *, replayed: bool) -> None:
        if self.tracer is None or not replayed:
            return
        with self.tracer.span(f"durable_step {key}", **{A.OPERATION: HarnessOp.DURABLE_STEP.value, A.STEP_KEY: key, A.STEP_REPLAYED: True}):
            pass

    def _execute(self, key: str, fn: Callable[[], T], input_hash: str | None, seq: int, codec: Codec, retry: RetryPolicy | None, sleep: Callable[[float], None]) -> T:
        self.journal._begin_step(self._lease, self.run_id, key, input_hash, seq)
        span = (
            self.tracer.span(f"durable_step {key}", **{A.OPERATION: HarnessOp.DURABLE_STEP.value, A.STEP_KEY: key, A.STEP_REPLAYED: False})
            if self.tracer is not None
            else nullcontext()
        )
        with span:
            try:
                value = retry_call(fn, retry, sleep=sleep) if retry is not None else fn()
            except BaseException as exc:
                self._safe_end(key, StepStatus.FAILED, None, self.journal._error_text(exc))
                raise
            try:
                text = codec.dumps(value)
                out = codec.loads(text)
            except Exception as exc:
                self._safe_end(key, StepStatus.FAILED, None, f"serialization: {type(exc).__qualname__}")
                raise StepSerializationError(f"step {key!r} returned a value the codec cannot store: {type(exc).__name__}") from exc
            self._check_lease()
            self.journal._end_step(self._lease, self.run_id, key, StepStatus.COMPLETED, text, None)
        with self._lock:
            self.executed.append(key)
        return out

    def _safe_end(self, key: str, status: StepStatus, result: str | None, error: str | None) -> None:
        try:
            self.journal._end_step(self._lease, self.run_id, key, status, result, error)
        except LeaseLostError:
            pass  # the new owner will redo the step; surface the original error

    def model_call(self, key: str, client: ModelClient, request: ModelRequest, *, retry: RetryPolicy | None = None, sleep: Callable[[float], None] = time.sleep) -> ModelResponse:
        """A model invocation as a durable step: replay returns the recorded response, never re-bills."""
        return self.step(key, lambda: client.create(request), input=request_fingerprint(request), codec=MODEL_RESPONSE_CODEC, retry=retry, sleep=sleep)

    def now(self, key: str) -> datetime:
        """Recorded wall-clock value (deterministic on replay)."""
        return datetime.fromisoformat(self.step(key, lambda: self.journal.clock().isoformat()))

    def random(self, key: str) -> float:
        """Recorded random value (deterministic on replay)."""
        return float(self.step(key, random.random))

    # ------------------------------------------------------------------ lifecycle

    def complete(self, result: Any = None) -> None:
        self._check_lease()
        self.journal._finish_run(self._lease, self.run_id, RunStatus.COMPLETED, canonical_json(result) if result is not None else None, None)

    def fail(self, error: BaseException | str) -> None:
        """Mark the run terminally failed (it will not show up in :meth:`StepJournal.resumable_runs`)."""
        self._check_lease()
        text = self.journal._error_text(error) if isinstance(error, BaseException) else self.journal.redactor.redact(error)[:500]
        self.journal._finish_run(self._lease, self.run_id, RunStatus.FAILED, None, text)

    def close(self) -> None:
        """Stop heart-beating and release the lease so another worker can resume immediately."""
        if self._closed:
            return
        self._closed = True
        if self._heartbeat is not None:
            self._heartbeat.stop()
        self.journal.leases.release(self._lease)

    def __enter__(self) -> "DurableRun":
        self._tokens.append(_ACTIVE.set(self))
        return self

    def __exit__(self, *exc: object) -> None:
        if self._tokens:
            _ACTIVE.reset(self._tokens.pop())
        self.close()

    def steps(self) -> Iterator[StepRecord]:
        return iter(self.journal.steps(self.run_id))
