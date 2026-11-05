"""Decay and forgetting.

Policies (build checklist step 9):

* **Phase-2 input selection**: drop raw memories whose last citation is older than
  ``max_unused_days`` (never-used ones age from ``generated_at``), rank by ``usage_count`` then
  recency, cap at ``max_raw_memories`` (256, the Agents SDK default).
* **Topic-file expiry**: active memories not used within ``expire_after_days`` are *archived*
  (moved to ``archive/`` and removed from the index - git history keeps them), not hard-deleted.
* **Explicit deletion**: :func:`forget_derived` cascades a user's "forget that session" request to
  memories supported *solely* by that session; shared evidence is kept with the session removed
  from provenance ("delete only memory supported by deleted inputs").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, Sequence

from ._time import Clock, parse_iso, utcnow
from .errors import ValidationError
from .frontmatter import MemoryDocument, Status
from .index import MemoryIndex
from .store import FileMemoryStore
from .usage import UsageStats

ARCHIVE_PREFIX = "archive"


@dataclass
class ForgettingPolicy:
    max_unused_days: int = 90
    max_raw_memories: int = 256
    expire_after_days: int | None = 180
    max_rollout_age_days: int | None = 30


@dataclass
class RawMemoryRef:
    session_id: str
    generated_at: datetime
    path: str
    usage_count: int = 0
    last_usage: datetime | None = None


def select_raw_memories(refs: Sequence[RawMemoryRef], policy: ForgettingPolicy, *, now: datetime | None = None) -> list[RawMemoryRef]:
    now = now or utcnow()
    cutoff = now - timedelta(days=policy.max_unused_days)
    eligible = [r for r in refs if (r.last_usage or r.generated_at) >= cutoff]
    eligible.sort(key=lambda r: (-r.usage_count, -(r.last_usage or r.generated_at).timestamp()))
    return eligible[: policy.max_raw_memories]


@dataclass
class ForgettingReport:
    archived: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    trimmed_provenance: list[str] = field(default_factory=list)


def _last_touch(doc: MemoryDocument, stats: dict[str, UsageStats] | None) -> datetime | None:
    candidates = [doc.meta.last_used, doc.meta.created]
    if stats and doc.path in stats:
        candidates.append(stats[doc.path].last_used)
    return max(filter(None, candidates), default=None)


def expire_unused(
    store: FileMemoryStore,
    index: MemoryIndex,
    policy: ForgettingPolicy,
    *,
    usage: dict[str, UsageStats] | None = None,
    clock: Clock = utcnow,
    dry_run: bool = False,
) -> ForgettingReport:
    """Archive root-level topic files that have not been used within ``policy.expire_after_days``."""
    report = ForgettingReport()
    if policy.expire_after_days is None:
        return report
    cutoff = clock() - timedelta(days=policy.expire_after_days)
    for doc in list(store.iter_documents()):
        if "/" in doc.path:
            continue
        if doc.meta.status is not Status.ACTIVE:
            continue
        touched = _last_touch(doc, usage)
        if touched is not None and touched < cutoff:
            report.archived.append(doc.path)
            if not dry_run:
                archive(store, index, doc.path)
    return report


def archive(store: FileMemoryStore, index: MemoryIndex, path: str) -> str:
    dest = f"{ARCHIVE_PREFIX}/{path}"
    n = 1
    while store.exists(dest):
        n += 1
        dest = f"{ARCHIVE_PREFIX}/{n}-{path}"
    with store.lock():
        store.rename(path, dest)
        index.remove(path)
    return dest


def forget_derived(
    store: FileMemoryStore,
    index: MemoryIndex,
    session_ids: Iterable[str],
    *,
    dry_run: bool = False,
) -> ForgettingReport:
    """Delete memories supported only by ``session_ids``; trim those sessions from shared provenance."""
    doomed = set(session_ids)
    if not doomed:
        raise ValidationError("no session ids given")
    report = ForgettingReport()
    for doc in list(store.iter_documents()):
        sessions = set(doc.meta.source.sessions)
        if not sessions & doomed:
            continue
        remaining = sessions - doomed
        if not remaining and not [e for e in doc.meta.source.evidence if not any(s in e for s in doomed)]:
            report.deleted.append(doc.path)
            if not dry_run:
                with store.lock():
                    store.delete(doc.path)
                    index.remove(doc.path)
        else:
            report.trimmed_provenance.append(doc.path)
            if not dry_run:
                doc.meta.source.sessions = [s for s in doc.meta.source.sessions if s not in doomed]
                doc.meta.source.evidence = [e for e in doc.meta.source.evidence if not any(s in e for s in doomed)]
                store.write(doc.path, doc.to_markdown())
    return report


def resolve_age(value: str | datetime | None) -> datetime | None:
    return parse_iso(value)
