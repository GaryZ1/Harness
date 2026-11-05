"""Episodic memory: what happened, when, in which session.

Two tiers, both *evidence* rather than always-loaded context:

* **Raw transcripts** (``sessions/<session_id>.jsonl``) - append-only, hash-chained, never
  rewritten ("Raw rollouts are immutable evidence"). Searched only for exact evidence.
* **Rollout summaries** (``rollout_summaries/<session_id>_<slug>.md``) - one distilled page per
  session written by Phase-1 extraction, cited from ``MEMORY.md`` task groups.

The :class:`SessionRegistry` records each session's flags (ephemeral, sub-agent, memory enabled,
``memory_mode`` clean/polluted) that gate whether memory may be generated from it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from ._time import Clock, parse_iso, to_iso, utcnow
from .errors import NotFoundError, ValidationError
from .frontmatter import join_frontmatter, split_frontmatter
from .locking import atomic_write_text, file_lock
from .security import PathSandbox, SecretRedactor, TaintTracker
from .store import FileMemoryStore

_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$")
GENESIS_HASH = "0" * 64
OUTCOMES = ("success", "partial", "uncertain", "fail")


def validate_session_id(session_id: str) -> str:
    if not _SESSION_ID.match(session_id or ""):
        raise ValidationError(f"invalid session id: {session_id!r}")
    return session_id


@dataclass(frozen=True)
class TranscriptEvent:
    seq: int
    ts: str
    kind: str  # user | assistant | tool_call | tool_result | system | compaction | note
    payload: dict[str, Any]
    prev_hash: str
    hash: str

    @property
    def text(self) -> str:
        content = self.payload.get("content", self.payload)
        return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def _event_hash(seq: int, ts: str, kind: str, payload: dict[str, Any], prev_hash: str) -> str:
    canonical = json.dumps(
        {"seq": seq, "ts": ts, "kind": kind, "payload": payload, "prev": prev_hash},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class TranscriptLog:
    """Append-only, tamper-evident JSONL transcript for one session."""

    def __init__(self, path: str | Path, *, clock: Clock = utcnow, redactor: SecretRedactor | None = None) -> None:
        self.path = Path(path)
        self.clock = clock
        self.redactor = redactor
        self._lock_path = self.path.with_name(f".{self.path.name}.lock")
        self._tail: tuple[int, str] | None = None
        self._tlock = threading.Lock()

    def _load_tail(self) -> tuple[int, str]:
        if self._tail is not None:
            return self._tail
        seq, last = -1, GENESIS_HASH
        if self.path.exists():
            with open(self.path, "rb") as fh:
                for raw in fh:
                    if raw.strip():
                        obj = json.loads(raw)
                        seq, last = obj["seq"], obj["hash"]
        self._tail = (seq, last)
        return self._tail

    def append(self, kind: str, payload: dict[str, Any] | str) -> TranscriptEvent:
        if isinstance(payload, str):
            payload = {"content": payload}
        if self.redactor is not None:
            payload = json.loads(self.redactor.redact(json.dumps(payload, ensure_ascii=False)))
        with self._tlock, file_lock(self._lock_path):
            self._tail = None  # another process may have appended
            seq, prev = self._load_tail()
            seq += 1
            ts = to_iso(self.clock()) or ""
            h = _event_hash(seq, ts, kind, payload, prev)
            event = TranscriptEvent(seq, ts, kind, payload, prev, h)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(asdict(event), ensure_ascii=False, separators=(",", ":")) + "\n"
            # O_APPEND guarantees we never overwrite existing evidence.
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(fd, line.encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)
            self._tail = (seq, h)
            return event

    def __iter__(self) -> Iterator[TranscriptEvent]:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as fh:
            for raw in fh:
                if raw.strip():
                    obj = json.loads(raw)
                    yield TranscriptEvent(**obj)

    def events(self) -> list[TranscriptEvent]:
        return list(self)

    def verify(self) -> bool:
        """Recompute the hash chain; ``False`` means the file was edited or truncated mid-chain."""
        prev, expected_seq = GENESIS_HASH, 0
        for ev in self:
            if ev.seq != expected_seq or ev.prev_hash != prev:
                return False
            if _event_hash(ev.seq, ev.ts, ev.kind, ev.payload, ev.prev_hash) != ev.hash:
                return False
            prev, expected_seq = ev.hash, expected_seq + 1
        return True

    def search(self, pattern: str, *, kinds: set[str] | None = None, max_hits: int = 20) -> list[TranscriptEvent]:
        """Exact-evidence search (the last step of the quick memory pass)."""
        rx = re.compile(pattern, re.I)
        hits = []
        for ev in self:
            if kinds and ev.kind not in kinds:
                continue
            if rx.search(ev.text):
                hits.append(ev)
                if len(hits) >= max_hits:
                    break
        return hits

    @property
    def last_activity(self) -> datetime | None:
        last = None
        for ev in self:
            last = ev.ts
        return parse_iso(last)


@dataclass
class SessionMeta:
    session_id: str
    started: str
    ended: str | None = None
    ephemeral: bool = False
    subagent: bool = False
    memory_enabled: bool = True
    memory_mode: str = "clean"  # clean | polluted
    pollution_reasons: list[str] = field(default_factory=list)
    parent_session: str | None = None
    user_id: str | None = None
    project_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def polluted(self) -> bool:
        return self.memory_mode == "polluted"

    def generation_block_reason(self) -> str | None:
        """Why memory must *not* be generated from this session (``None`` = eligible)."""
        if self.subagent:
            return "sub-agent session"
        if self.ephemeral:
            return "ephemeral session"
        if not self.memory_enabled:
            return "memory disabled"
        if self.polluted:
            return "memory_mode=polluted (ingested untrusted content)"
        return None


class SessionRegistry:
    """Session metadata + transcripts under ``<base>/sessions/``."""

    def __init__(self, base: str | Path, *, clock: Clock = utcnow, redactor: SecretRedactor | None = None) -> None:
        self.dir = Path(base).expanduser() / "sessions"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sandbox = PathSandbox(self.dir)
        self.clock = clock
        self.redactor = redactor or SecretRedactor()

    def _meta_path(self, session_id: str) -> Path:
        return self.sandbox.resolve(f"{validate_session_id(session_id)}.meta.json")

    def transcript(self, session_id: str) -> TranscriptLog:
        return TranscriptLog(self.sandbox.resolve(f"{validate_session_id(session_id)}.jsonl"), clock=self.clock, redactor=self.redactor)

    def start(self, session_id: str, **flags: Any) -> SessionMeta:
        path = self._meta_path(session_id)
        if path.exists():
            raise ValidationError(f"session {session_id} already exists")
        known = {f for f in SessionMeta.__dataclass_fields__}
        meta = SessionMeta(
            session_id=session_id,
            started=to_iso(self.clock()) or "",
            **{k: v for k, v in flags.items() if k in known},
            extra={k: v for k, v in flags.items() if k not in known},
        )
        self._save(meta)
        return meta

    def get(self, session_id: str) -> SessionMeta:
        path = self._meta_path(session_id)
        if not path.exists():
            raise NotFoundError(f"unknown session {session_id}")
        return SessionMeta(**json.loads(path.read_text(encoding="utf-8")))

    def _save(self, meta: SessionMeta) -> None:
        with file_lock(self.dir / ".registry.lock"):
            atomic_write_text(self._meta_path(meta.session_id), json.dumps(asdict(meta), indent=2, sort_keys=True))

    def mark_polluted(self, session_id: str, reason: str) -> SessionMeta:
        meta = self.get(session_id)
        meta.memory_mode = "polluted"
        meta.pollution_reasons.append(reason)
        self._save(meta)
        return meta

    def apply_taint(self, session_id: str, tracker: TaintTracker) -> SessionMeta:
        if tracker.polluted:
            meta = self.get(session_id)
            meta.memory_mode = "polluted"
            meta.pollution_reasons.extend(r for r in tracker.reasons if r not in meta.pollution_reasons)
            self._save(meta)
            return meta
        return self.get(session_id)

    def end(self, session_id: str) -> SessionMeta:
        meta = self.get(session_id)
        meta.ended = to_iso(self.clock())
        self._save(meta)
        return meta

    def all(self) -> list[SessionMeta]:
        out = []
        for p in sorted(self.dir.glob("*.meta.json")):
            out.append(SessionMeta(**json.loads(p.read_text(encoding="utf-8"))))
        return out

    def last_activity(self, session_id: str) -> datetime | None:
        meta = self.get(session_id)
        return self.transcript(session_id).last_activity or parse_iso(meta.ended) or parse_iso(meta.started)


@dataclass(frozen=True)
class RolloutSummary:
    session_id: str
    slug: str
    path: str
    outcome: str
    body: str
    created: str | None


class RolloutSummaryStore:
    """Per-session distilled summaries in ``rollout_summaries/`` of a learned-memory store."""

    PREFIX = "rollout_summaries"

    def __init__(self, store: FileMemoryStore) -> None:
        self.store = store

    def write(self, session_id: str, slug: str, summary: str, *, outcome: str = "uncertain", tainted: bool = False) -> str:
        validate_session_id(session_id)
        if outcome not in OUTCOMES:
            raise ValidationError(f"outcome must be one of {OUTCOMES}")
        slug = re.sub(r"[^a-z0-9-]+", "-", slug.lower()).strip("-")[:48] or "session"
        rel = f"{self.PREFIX}/{session_id}_{slug}.md"
        front = {
            "session_id": session_id,
            "slug": slug,
            "outcome": outcome,
            "created": to_iso(self.store.clock()),
            "tainted": tainted,
        }
        self.store.write(rel, join_frontmatter(front, summary.strip() + "\n"))
        return rel

    def list(self) -> list[RolloutSummary]:
        out = []
        for rel in self.store.iter_files(self.PREFIX, suffix=".md"):
            data, body = split_frontmatter(self.store.read(rel).content)
            data = data or {}
            out.append(
                RolloutSummary(
                    session_id=str(data.get("session_id", "")),
                    slug=str(data.get("slug", "")),
                    path=rel,
                    outcome=str(data.get("outcome", "uncertain")),
                    body=body,
                    created=data.get("created"),
                )
            )
        return out

    def for_session(self, session_id: str) -> list[RolloutSummary]:
        return [r for r in self.list() if r.session_id == session_id]

    def delete_for_session(self, session_id: str) -> list[str]:
        removed = []
        for r in self.for_session(session_id):
            self.store.delete(r.path)
            removed.append(r.path)
        return removed
