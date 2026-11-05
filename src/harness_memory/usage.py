"""Citation-based usage telemetry (Codex ``<oai-mem-citation>`` pattern).

Every answer that used memory ends with a structured trailer::

    <memory-citation>
    feedback_testing.md:3-7|note=[CI runs make test-fast first]
    rollout_summaries/s_8f2c_fix-ci.md:12-30
    <session_ids>s_8f2c, s_91aa</session_ids>
    </memory-citation>

:func:`parse_citations` extracts it, :func:`strip_citations` removes it before display, and
:class:`UsageLog` appends to ``.usage.jsonl``. Consolidation folds the log into each memory's
``use_count`` / ``last_used`` frontmatter, which drives forgetting. Usage is *not* written into
memory files mid-session, so the cached prompt prefix is never invalidated.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ._time import Clock, parse_iso, to_iso, utcnow
from .errors import ValidationError
from .frontmatter import MemoryDocument
from .locking import file_lock
from .store import FileMemoryStore

CITATION_INSTRUCTIONS = (
    "If you used memory to answer, end your reply with a <memory-citation> block: one line per memory "
    "file as `path:start-end|note=[why it was used]`, then `<session_ids>id, ...</session_ids>` for any "
    "rollout summaries you relied on. Omit the block if you did not use memory."
)

_BLOCK = re.compile(r"<memory-citation>\s*(?P<body>.*?)\s*</memory-citation>", re.S)
_LINE = re.compile(r"^(?P<path>[^\s:|]+):(?P<start>\d+)(?:-(?P<end>\d+))?(?:\|note=\[(?P<note>.*)\])?$")
_SESSIONS = re.compile(r"<session_ids>(?P<ids>.*?)</session_ids>", re.S)


@dataclass(frozen=True)
class Citation:
    path: str
    start: int
    end: int
    note: str = ""


@dataclass(frozen=True)
class CitationBlock:
    citations: tuple[Citation, ...]
    session_ids: tuple[str, ...]


def parse_citations(text: str, *, strict: bool = False) -> CitationBlock | None:
    m = _BLOCK.search(text)
    if not m:
        return None
    body = m.group("body")
    sessions: tuple[str, ...] = ()
    sm = _SESSIONS.search(body)
    if sm:
        sessions = tuple(s.strip() for s in sm.group("ids").split(",") if s.strip())
        body = _SESSIONS.sub("", body)
    cites = []
    for raw in body.splitlines():
        line = raw.strip().lstrip("-").strip()
        if not line:
            continue
        lm = _LINE.match(line)
        if not lm:
            if strict:
                raise ValidationError(f"malformed citation line: {raw!r}")
            continue
        start = int(lm.group("start"))
        end = int(lm.group("end") or start)
        cites.append(Citation(lm.group("path"), start, max(start, end), lm.group("note") or ""))
    return CitationBlock(tuple(cites), sessions)


def strip_citations(text: str) -> str:
    return _BLOCK.sub("", text).rstrip()


@dataclass(frozen=True)
class UsageStats:
    count: int
    last_used: datetime | None


class UsageLog:
    """Append-only ``.usage.jsonl`` inside a memory store."""

    FILENAME = ".usage.jsonl"

    def __init__(self, store: FileMemoryStore, *, clock: Clock = utcnow) -> None:
        self.store = store
        self.clock = clock
        self.path: Path = store.root / self.FILENAME

    def record(self, block: CitationBlock | None, *, session_id: str) -> int:
        if block is None or (not block.citations and not block.session_ids):
            return 0
        ts = to_iso(self.clock())
        rows = [
            {"ts": ts, "session": session_id, "path": c.path, "start": c.start, "end": c.end, "note": c.note} for c in block.citations
        ] + [{"ts": ts, "session": session_id, "rollout": sid} for sid in block.session_ids]
        with file_lock(self.store.root / ".usage.lock"):
            with open(self.path, "a", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        return len(rows)

    def record_reply(self, reply: str, *, session_id: str) -> int:
        return self.record(parse_citations(reply), session_id=session_id)

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def stats(self) -> dict[str, UsageStats]:
        """Aggregate per memory path (and ``rollout:<session_id>`` for cited rollouts)."""
        agg: dict[str, tuple[int, datetime | None]] = {}
        for r in self.rows():
            key = r.get("path") or (f"rollout:{r['rollout']}" if r.get("rollout") else None)
            if not key:
                continue
            ts = parse_iso(r.get("ts"))
            count, last = agg.get(key, (0, None))
            agg[key] = (count + 1, max(filter(None, [last, ts]), default=None))
        return {k: UsageStats(c, t) for k, (c, t) in agg.items()}

    def apply_to_store(self) -> list[str]:
        """Fold aggregated usage into frontmatter (run during consolidation). Returns updated paths."""
        updated = []
        stats = self.stats()
        for path, st in stats.items():
            if path.startswith("rollout:") or not self.store.exists(path):
                continue
            try:
                f = self.store.read(path)
                doc = MemoryDocument.from_markdown(f.content, path=f.path)
            except ValidationError:
                continue
            if doc.meta.use_count == st.count and doc.meta.last_used == st.last_used:
                continue
            doc.meta.use_count = max(doc.meta.use_count, st.count)
            if st.last_used and (doc.meta.last_used is None or st.last_used > doc.meta.last_used):
                doc.meta.last_used = st.last_used
            # store.write (not write_document) so `modified` is preserved: usage is telemetry, not content.
            self.store.write(path, doc.to_markdown(), expected_sha256=f.sha256)
            updated.append(path)
        return updated
