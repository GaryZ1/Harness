"""Durable, append-only session log: the source of truth the harness projects context from.

Research basis (reference report section 1.7 step 1, build plan "Architecture"):

* Anthropic Managed Agents splits an agent into a **session** ("an append-only event log"), a
  **harness** and a **sandbox**. Events persist via ``emitEvent(id, event)``, ``getEvents()``
  returns positional slices, and a replacement harness calls ``wake(sessionId)`` to resume "from
  the last event" after a crash. :class:`SessionStore.wake` and :meth:`EventLog.events` mirror that.
* Compaction, memory writes and trimming are irreversible, so the model-visible context must be a
  *projection* of a log that is never destroyed. Compaction is recorded as a new ``compaction``
  event (OpenHands tombstone-style ``Condensation``) carrying the replacement view plus a hash of
  the canonical prefix it summarises (Cline's hash-validated ``compaction.json``).
* Codex persists sessions as append-only JSONL rollouts; Claude Code keeps JSONL transcripts with
  ``--resume`` and ``--fork-session``.

Invariants enforced here:

* events are only ever appended; ``seq`` is contiguous from 0 and assigned under a cross-process
  lock; every append is flushed and ``fsync``-ed before :meth:`EventLog.emit` returns;
* a torn final line (crash mid-write) is truncated on open; any other malformed line or ``seq``
  gap raises :class:`~harness_core.errors.CorruptLogError` instead of being silently skipped;
* the canonical message history is rebuilt only from events whose ``data`` carries a
  ``"message"`` key, so observability events never leak into the transcript;
* a session is driven by at most one harness at a time (:meth:`Session.lease`), and the OS drops
  the lease when that process dies.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol, Sequence, runtime_checkable

from ._locking import Lease, atomic_write_text, exclusive_lock
from .errors import (
    CorruptLogError,
    InvalidSessionIdError,
    SessionExistsError,
    SessionLockedError,
    SessionNotFoundError,
)
from .types import (
    Clock,
    ContentBlock,
    Event,
    EventType,
    Message,
    Role,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
    block_from_api,
    messages_to_api,
    new_id,
    utcnow,
)

_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$")


def validate_session_id(session_id: str) -> str:
    if not isinstance(session_id, str) or not _SESSION_ID.match(session_id) or ".." in session_id:
        raise InvalidSessionIdError(f"invalid session id: {session_id!r}")
    return session_id


def _iso(clock: Clock) -> str:
    return clock().isoformat()


# ----------------------------------------------------------------------------- serialization


def block_to_dict(block: ContentBlock) -> dict[str, Any]:
    return block.to_api()


def message_to_dict(message: Message) -> dict[str, Any]:
    """Lossless JSON form of a :class:`Message` (keeps ``id``, ``metadata`` and opaque blocks verbatim)."""
    return {
        "id": message.id,
        "role": message.role.value,
        "content": [block_to_dict(b) for b in message.content],
        "metadata": dict(message.metadata),
    }


def message_from_dict(d: dict[str, Any]) -> Message:
    content = d.get("content", "")
    blocks: list[ContentBlock] = [block_from_api({"type": "text", "text": content})] if isinstance(content, str) else [
        block_from_api(b) for b in content
    ]
    return Message(Role(d["role"]), blocks, id=d.get("id") or new_id("msg"), metadata=dict(d.get("metadata") or {}))


def history_hash(messages: Sequence[Message]) -> str:
    """SHA-256 of the canonical API form of ``messages`` (used to validate compaction projections)."""
    canonical = json.dumps(messages_to_api(messages), sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- event logs


@runtime_checkable
class EventLog(Protocol):
    """An append-only event store for one session (an :class:`~harness_core.types.EventSink`)."""

    session_id: str

    def emit(self, event: Event) -> Event: ...

    def events(self, start: int = 0, end: int | None = None) -> list[Event]: ...

    @property
    def last_seq(self) -> int: ...


class InMemoryEventLog:
    """Thread-safe, non-durable :class:`EventLog` for tests and ephemeral sub-agents."""

    def __init__(self, session_id: str, *, clock: Clock = utcnow) -> None:
        self.session_id = session_id
        self.clock = clock
        self._events: list[Event] = []
        self._lock = threading.Lock()

    def emit(self, event: Event) -> Event:
        if event.session_id != self.session_id:
            raise ValueError(f"event for session {event.session_id!r} emitted to log {self.session_id!r}")
        with self._lock:
            event.seq = len(self._events)
            event.ts = event.ts or _iso(self.clock)
            self._events.append(event)
            return event

    def events(self, start: int = 0, end: int | None = None) -> list[Event]:
        with self._lock:
            return list(self._events[start:end])

    @property
    def last_seq(self) -> int:
        with self._lock:
            return len(self._events) - 1


class JsonlEventLog:
    """Durable JSONL :class:`EventLog`: one :meth:`Event.to_json` line per event, fsync on append.

    Safe for concurrent writers across threads and processes: each append takes an exclusive
    ``flock`` on ``<file>.lock``, re-reads the tail if another writer appended meanwhile, assigns
    the next ``seq``, writes one line with a single ``write`` call and ``fsync``-s it.
    """

    def __init__(self, path: str | Path, session_id: str, *, clock: Clock = utcnow, fsync: bool = True) -> None:
        self.path = Path(path)
        self.session_id = validate_session_id(session_id)
        self.clock = clock
        self.fsync = fsync
        self.repaired_bytes = 0
        self._lock_path = self.path.with_name(f".{self.path.name}.lock")
        self._tlock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._tlock, exclusive_lock(self._lock_path):
            self.repaired_bytes = self._repair_torn_tail()
            self._known_size, self._last_seq = self._scan_tail()

    # ------------------------------------------------------------------ recovery
    def _repair_torn_tail(self) -> int:
        """Truncate a final line that was cut off by a crash (no newline, or not valid JSON)."""
        if not self.path.exists():
            return 0
        with open(self.path, "rb+") as fh:
            data = fh.read()
            if not data:
                return 0
            cut = len(data)
            if not data.endswith(b"\n"):
                cut = data.rfind(b"\n") + 1
            else:
                last_start = data.rfind(b"\n", 0, len(data) - 1) + 1
                try:
                    json.loads(data[last_start:].decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    cut = last_start
            if cut == len(data):
                return 0
            fh.truncate(cut)
            fh.flush()
            os.fsync(fh.fileno())
            return len(data) - cut

    def _scan_tail(self) -> tuple[int, int]:
        if not self.path.exists():
            return 0, -1
        size = self.path.stat().st_size
        if size == 0:
            return 0, -1
        with open(self.path, "rb") as fh:
            back = min(size, 1 << 16)
            while True:
                fh.seek(size - back)
                chunk = fh.read(back)
                body = chunk[:-1] if chunk.endswith(b"\n") else chunk
                nl = body.rfind(b"\n")
                if nl >= 0 or back == size:
                    line = body[nl + 1:]
                    break
                back = min(size, back * 2)
        try:
            return size, int(json.loads(line.decode("utf-8"))["seq"])
        except (ValueError, KeyError, UnicodeDecodeError) as exc:
            raise CorruptLogError(f"{self.path}: unreadable last event") from exc

    # ------------------------------------------------------------------ EventSink
    def emit(self, event: Event) -> Event:
        if event.session_id != self.session_id:
            raise ValueError(f"event for session {event.session_id!r} emitted to log {self.session_id!r}")
        with self._tlock, exclusive_lock(self._lock_path):
            size = self.path.stat().st_size if self.path.exists() else 0
            if size != self._known_size:  # another process appended since our last write
                self._known_size, self._last_seq = self._scan_tail()
            event.seq = self._last_seq + 1
            event.ts = event.ts or _iso(self.clock)
            line = (event.to_json() + "\n").encode("utf-8")
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(fd, line)
                if self.fsync:
                    os.fsync(fd)
            finally:
                os.close(fd)
            self._last_seq = event.seq
            self._known_size = size + len(line)
            return event

    # ------------------------------------------------------------------ reads
    def iter_events(self) -> Iterator[Event]:
        if not self.path.exists():
            return
        expected = 0
        with open(self.path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                if not line.endswith("\n"):
                    return  # concurrent writer mid-append; the line is not committed yet
                try:
                    event = Event.from_json(line)
                except (ValueError, KeyError, TypeError) as exc:
                    raise CorruptLogError(f"{self.path}:{lineno}: malformed event") from exc
                if event.seq != expected:
                    raise CorruptLogError(f"{self.path}:{lineno}: expected seq {expected}, found {event.seq}")
                expected += 1
                yield event

    def events(self, start: int = 0, end: int | None = None) -> list[Event]:
        """Positional slice by ``seq`` (``start`` inclusive, ``end`` exclusive), like ``getEvents()``."""
        out = []
        for e in self.iter_events():
            if end is not None and e.seq >= end:
                break
            if e.seq >= start:
                out.append(e)
        return out

    @property
    def last_seq(self) -> int:
        with self._tlock:
            return self._last_seq


class FanoutSink:
    """Forward every event to several sinks (e.g. the session log plus a UI stream)."""

    def __init__(self, *sinks: Any) -> None:
        self.sinks = list(sinks)

    def emit(self, event: Event) -> Event:
        for s in self.sinks:
            s.emit(event)
        return event


# ----------------------------------------------------------------------------- sessions


class SessionStatus(str, Enum):
    ACTIVE = "active"            # created, never run
    RUNNING = "running"          # a harness holds the lease
    IDLE = "idle"                # last run ended successfully; can take more input
    PAUSED = "paused"            # waiting for an approval decision
    INTERRUPTED = "interrupted"  # hard-cancelled
    FAILED = "failed"            # last run ended with an error subtype
    COMPLETED = "completed"      # explicitly ended (session_end)


@dataclass
class SessionInfo:
    session_id: str
    created_at: str
    updated_at: str
    status: SessionStatus = SessionStatus.ACTIVE
    parent_id: str | None = None
    forked_at_seq: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SessionInfo":
        d = dict(d)
        d["status"] = SessionStatus(d.get("status", "active"))
        return cls(**d)


@dataclass
class PendingToolBatch:
    """Tool calls of the last assistant turn that have no recorded results yet (resume state)."""

    assistant_message: Message
    assistant_seq: int
    completed: dict[str, ToolResultBlock]  # tool_use_id -> result recorded before the crash/pause
    started: set[str]                      # execution began but no result was recorded
    not_started: list[ToolUseBlock]

    @property
    def tool_uses(self) -> list[ToolUseBlock]:
        return self.assistant_message.tool_uses


class Session:
    """One conversation: an :class:`EventLog` plus metadata, canonical history and projected view.

    The canonical history (:meth:`messages`) is every message ever appended. The *view*
    (:meth:`view`) is what the model sees: the latest compaction's replacement messages plus all
    messages appended after it.
    """

    def __init__(self, log: EventLog, info: SessionInfo, *, store: "SessionStore | None" = None, clock: Clock = utcnow) -> None:
        self.log = log
        self.info = info
        self.store = store
        self.clock = clock
        self._lock = threading.RLock()
        self._lease: Lease | None = None
        self._mem_lease = threading.Lock()
        self._messages: list[tuple[int, Message]] = []
        self._compaction: tuple[int, int, list[Message]] | None = None  # (event seq, through_seq, view)
        self._rebuild()

    # ------------------------------------------------------------------ construction
    @classmethod
    def in_memory(cls, session_id: str | None = None, *, clock: Clock = utcnow, metadata: dict[str, Any] | None = None) -> "Session":
        sid = validate_session_id(new_id("sess") if session_id is None else session_id)
        now = _iso(clock)
        return cls(InMemoryEventLog(sid, clock=clock), SessionInfo(sid, now, now, metadata=dict(metadata or {})), clock=clock)

    @property
    def id(self) -> str:
        return self.info.session_id

    # ------------------------------------------------------------------ replay
    def _rebuild(self) -> None:
        msgs: list[tuple[int, Message]] = []
        compaction = None
        for e in self.log.events():
            if "message" in e.data:
                msgs.append((e.seq, message_from_dict(e.data["message"])))
            elif e.type is EventType.COMPACTION and "view" in e.data:
                through = int(e.data["through_seq"])
                prefix = [m for s, m in msgs if s <= through]
                if e.data.get("prefix_hash") == history_hash(prefix):
                    compaction = (e.seq, through, [message_from_dict(m) for m in e.data["view"]])
        with self._lock:
            self._messages = msgs
            self._compaction = compaction

    def reload(self) -> None:
        """Re-read the log (e.g. after another process appended)."""
        self._rebuild()

    def messages(self) -> list[Message]:
        """Canonical, full-fidelity history (never affected by compaction)."""
        with self._lock:
            return [m for _, m in self._messages]

    def view(self) -> list[Message]:
        """Model-visible projection: latest compaction view + messages appended after it."""
        with self._lock:
            if self._compaction is None:
                return [m for _, m in self._messages]
            _, through, view = self._compaction
            return list(view) + [m for s, m in self._messages if s > through]

    @property
    def compacted(self) -> bool:
        return self._compaction is not None

    def events(self, start: int = 0, end: int | None = None) -> list[Event]:
        return self.log.events(start, end)

    # ------------------------------------------------------------------ writes
    def emit(self, type: EventType, data: dict[str, Any] | None = None) -> Event:
        return self.log.emit(Event(self.id, type, dict(data or {})))

    def append_message(self, message: Message, *, kind: str = "", event_type: EventType | None = None, **data: Any) -> Event:
        """Append ``message`` to the canonical history (one durable event)."""
        if event_type is None:
            event_type = EventType.MODEL_RESPONSE if message.role is Role.ASSISTANT else EventType.USER_MESSAGE
        payload = {"message": message_to_dict(message), **data}
        if kind:
            payload["kind"] = kind
        with self._lock:
            event = self.emit(event_type, payload)
            self._messages.append((event.seq, message))
            return event

    def record_compaction(self, view: Sequence[Message], *, summary: str = "", reason: str = "", **data: Any) -> Event:
        """Record a compaction as a new event; the canonical history is untouched.

        The event stores the replacement view and the hash of the canonical prefix it replaces,
        so a projection whose prefix no longer matches is ignored on replay.
        """
        with self._lock:
            through = self._messages[-1][0] if self._messages else -1
            prefix = [m for _, m in self._messages]
            view = list(view)
            event = self.emit(EventType.COMPACTION, {
                "through_seq": through,
                "prefix_hash": history_hash(prefix),
                "view": [message_to_dict(m) for m in view],
                "summary": summary,
                "reason": reason,
                **data,
            })
            self._compaction = (event.seq, through, view)
            return event

    def checkpoint(self, label: str, **data: Any) -> Event:
        """Record a named checkpoint (a fork/rewind target)."""
        return self.emit(EventType.CUSTOM, {"checkpoint": label, **data})

    def find_checkpoint(self, label: str) -> int | None:
        seq = None
        for e in self.log.events():
            if e.type is EventType.CUSTOM and e.data.get("checkpoint") == label:
                seq = e.seq
        return seq

    def set_status(self, status: SessionStatus) -> None:
        self.info.status = status
        self.info.updated_at = _iso(self.clock)
        if self.store is not None:
            self.store._write_info(self.info)

    # ------------------------------------------------------------------ resume helpers
    def has_event(self, type: EventType) -> bool:
        return any(e.type is type for e in self.log.events())

    def total_usage(self) -> tuple[Usage, float, int]:
        """(usage, estimated cost USD, model calls) summed over every recorded model response."""
        usage, cost, calls = Usage(), 0.0, 0
        for e in self.log.events():
            if e.type is EventType.MODEL_RESPONSE and "usage" in e.data:
                usage = usage + Usage(**e.data["usage"])
                cost += float(e.data.get("cost_usd") or 0.0)
                calls += 1
        return usage, cost, calls

    def pending_tool_batch(self) -> PendingToolBatch | None:
        """Tool calls in the last assistant message that still lack a result message.

        Uses per-call ``tool_call``/``tool_result`` events written after that message to tell
        which calls completed, which started (crash mid-execution: side effects unknown) and
        which never started (crash before execution, or a durable approval pause).
        """
        with self._lock:
            if not self._messages:
                return None
            seq, last = self._messages[-1]
        if last.role is not Role.ASSISTANT or not last.tool_uses:
            return None
        completed: dict[str, ToolResultBlock] = {}
        started: set[str] = set()
        for e in self.log.events(seq + 1):
            tid = e.data.get("tool_use_id")
            if not tid:
                continue
            if e.type is EventType.TOOL_RESULT and "block" in e.data:
                b = block_from_api(e.data["block"])
                if isinstance(b, ToolResultBlock):
                    completed[tid] = b
            elif e.type is EventType.TOOL_CALL and e.data.get("phase") == "executing":
                started.add(tid)
        started -= set(completed)
        not_started = [u for u in last.tool_uses if u.id not in completed and u.id not in started]
        return PendingToolBatch(last, seq, completed, started, not_started)

    # ------------------------------------------------------------------ lease
    @contextmanager
    def lease(self) -> Iterator[None]:
        """Exclusive right to drive this session; raises :class:`SessionLockedError` if held elsewhere."""
        if self.store is None:
            if not self._mem_lease.acquire(blocking=False):
                raise SessionLockedError(f"session {self.id} is already running")
            try:
                yield
            finally:
                self._mem_lease.release()
            return
        lease = Lease(self.store.root / f"{self.id}.lease")
        if not lease.try_acquire():
            raise SessionLockedError(f"session {self.id} is held by another harness")
        try:
            yield
        finally:
            lease.release()


class SessionStore:
    """Directory of sessions: ``<id>.jsonl`` (events), ``<id>.meta.json`` (status), ``<id>.lease``."""

    def __init__(self, root: str | Path, *, clock: Clock = utcnow, fsync: bool = True) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.fsync = fsync

    def _log_path(self, sid: str) -> Path:
        return self.root / f"{validate_session_id(sid)}.jsonl"

    def _meta_path(self, sid: str) -> Path:
        return self.root / f"{validate_session_id(sid)}.meta.json"

    def _write_info(self, info: SessionInfo) -> None:
        atomic_write_text(self._meta_path(info.session_id), json.dumps(info.to_dict(), indent=2, sort_keys=True) + "\n")

    def exists(self, session_id: str) -> bool:
        return self._meta_path(session_id).exists() or self._log_path(session_id).exists()

    def create(self, session_id: str | None = None, *, metadata: dict[str, Any] | None = None) -> Session:
        sid = validate_session_id(new_id("sess") if session_id is None else session_id)
        with exclusive_lock(self.root / ".store.lock"):
            if self.exists(sid):
                raise SessionExistsError(f"session {sid} already exists")
            now = _iso(self.clock)
            info = SessionInfo(sid, now, now, metadata=dict(metadata or {}))
            self._write_info(info)
            self._log_path(sid).touch()
        return self.open(sid)

    def open(self, session_id: str) -> Session:
        if not self.exists(session_id):
            raise SessionNotFoundError(session_id)
        meta = self._meta_path(session_id)
        if meta.exists():
            info = SessionInfo.from_dict(json.loads(meta.read_text(encoding="utf-8")))
        else:  # log without metadata (e.g. metadata write lost): reconstruct
            now = _iso(self.clock)
            info = SessionInfo(session_id, now, now)
        log = JsonlEventLog(self._log_path(session_id), session_id, clock=self.clock, fsync=self.fsync)
        return Session(log, info, store=self, clock=self.clock)

    def wake(self, session_id: str) -> Session:
        """Managed Agents ``wake(sessionId)``: reopen a session and rebuild its state from the log."""
        return self.open(session_id)

    def list(self) -> list[SessionInfo]:
        out = []
        for p in sorted(self.root.glob("*.meta.json")):
            try:
                out.append(SessionInfo.from_dict(json.loads(p.read_text(encoding="utf-8"))))
            except (ValueError, TypeError):
                continue
        return sorted(out, key=lambda i: i.updated_at, reverse=True)

    def fork(self, session_id: str, *, at_seq: int | None = None, new_session_id: str | None = None,
             metadata: dict[str, Any] | None = None) -> Session:
        """Copy events ``0..at_seq`` into a new session (fork / conversation-only rewind).

        Refuses a fork point that would leave an assistant tool batch without its results, because
        resuming would silently re-execute those tools.
        """
        src = self.open(session_id)
        events = src.log.events(0, None if at_seq is None else at_seq + 1)
        last_msg = None
        for e in events:
            if "message" in e.data:
                last_msg = message_from_dict(e.data["message"])
        if last_msg is not None and last_msg.role is Role.ASSISTANT and last_msg.tool_uses:
            raise ValueError(f"fork point seq={at_seq} splits a tool batch; fork before or after it")
        dst = self.create(new_session_id, metadata=metadata)
        for e in events:
            dst.log.emit(Event(dst.id, e.type, e.data, ts=e.ts, id=e.id))
        fork_seq = events[-1].seq if events else -1
        dst.emit(EventType.CUSTOM, {"forked_from": src.id, "at_seq": fork_seq})
        dst.info.parent_id = src.id
        dst.info.forked_at_seq = fork_seq
        dst.set_status(SessionStatus.IDLE)
        dst.reload()
        return dst


def iter_messages(events: Iterable[Event]) -> Iterator[Message]:
    """Messages carried by ``events`` in order (the replay rule used by :class:`Session`)."""
    for e in events:
        if "message" in e.data:
            yield message_from_dict(e.data["message"])
