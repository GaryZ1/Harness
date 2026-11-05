"""The always-loaded layer of learned memory: the bounded ``MEMORY.md`` index and ``memory_summary.md``.

``MEMORY.md`` holds **one line per memory** (Claude Code auto memory). Only the first 200 lines or
25 KB load at session start. Near the cap the harness reminds the model to compact; past the cap
the write is *accepted* but an error tells the model to rewrite the index.

``memory_summary.md`` is the Codex / Agents SDK prompt-loaded summary. Its first line must be
exactly ``v1`` (schema version) or it is regenerated, and it carries four sections.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from .errors import NotFoundError
from .store import FileMemoryStore

INDEX_FILE = "MEMORY.md"
SUMMARY_FILE = "memory_summary.md"
MAX_INDEX_LINES = 200
MAX_INDEX_BYTES = 25_000
NEAR_CAP_RATIO = 0.9

NEAR_CAP_REMINDER = (
    "MEMORY.md is close to its limit: keep one line per entry, move detail into topic files, and merge or drop stale entries."
)
OVER_CAP_ERROR = (
    "Error: MEMORY.md exceeds {lines} lines / {bytes} bytes; only the first {max_lines} lines or "
    "{max_bytes} bytes are loaded. Rewrite the index now: keep one line per entry, move detail into "
    "topic files, and merge or drop stale entries."
)

_ENTRY = re.compile(r"^\s*[-*]\s+\[(?P<title>[^\]]+)\]\((?P<target>[^)\s]+)\)(?P<rest>.*)$")

SUMMARY_VERSION = "v1"
SUMMARY_SECTIONS = ("User Profile", "User preferences", "General Tips", "What's in Memory")


class IndexHealth(str, Enum):
    OK = "ok"
    NEAR_CAP = "near_cap"
    OVER_CAP = "over_cap"


@dataclass(frozen=True)
class IndexStatus:
    health: IndexHealth
    lines: int
    bytes: int

    @property
    def message(self) -> str | None:
        if self.health is IndexHealth.NEAR_CAP:
            return NEAR_CAP_REMINDER
        if self.health is IndexHealth.OVER_CAP:
            return OVER_CAP_ERROR.format(lines=self.lines, bytes=self.bytes, max_lines=MAX_INDEX_LINES, max_bytes=MAX_INDEX_BYTES)
        return None


def index_status(text: str, *, max_lines: int = MAX_INDEX_LINES, max_bytes: int = MAX_INDEX_BYTES) -> IndexStatus:
    n_lines = len(text.splitlines())
    n_bytes = len(text.encode("utf-8"))
    if n_lines > max_lines or n_bytes > max_bytes:
        health = IndexHealth.OVER_CAP
    elif n_lines >= max_lines * NEAR_CAP_RATIO or n_bytes >= max_bytes * NEAR_CAP_RATIO:
        health = IndexHealth.NEAR_CAP
    else:
        health = IndexHealth.OK
    return IndexStatus(health, n_lines, n_bytes)


def truncate_for_prompt(text: str, *, max_lines: int = MAX_INDEX_LINES, max_bytes: int = MAX_INDEX_BYTES) -> str:
    """Return the prefix of ``text`` that is loaded at session start (first N lines, then byte cap)."""
    lines = text.splitlines(keepends=True)[:max_lines]
    out, size = [], 0
    for line in lines:
        b = len(line.encode("utf-8"))
        if size + b > max_bytes:
            break
        out.append(line)
        size += b
    return "".join(out)


@dataclass
class IndexEntry:
    title: str
    target: str
    note: str = ""

    def render(self) -> str:
        note = f" — {self.note}" if self.note else ""
        return f"- [{self.title}]({self.target}){note}"


class MemoryIndex:
    """Structured editor for ``MEMORY.md`` that preserves free-form lines (headings, comments)."""

    HEADER = "# Memory index\n\nOne line per memory. Open the linked topic file for detail.\n\n"

    def __init__(
        self, store: FileMemoryStore, filename: str = INDEX_FILE, *, max_lines: int = MAX_INDEX_LINES, max_bytes: int = MAX_INDEX_BYTES
    ) -> None:
        self.store = store
        self.filename = filename
        self.max_lines = max_lines
        self.max_bytes = max_bytes

    def text(self) -> str:
        try:
            return self.store.read(self.filename).content
        except NotFoundError:
            return ""

    def entries(self) -> list[IndexEntry]:
        out = []
        for line in self.text().splitlines():
            m = _ENTRY.match(line)
            if m:
                rest = m.group("rest").strip()
                rest = re.sub(r"^[—–-]\s*", "", rest)
                out.append(IndexEntry(m.group("title").strip(), m.group("target"), rest))
        return out

    def get(self, target: str) -> IndexEntry | None:
        return next((e for e in self.entries() if e.target == target), None)

    def _rewrite(self, transform) -> IndexStatus:
        with self.store.lock():
            current = self.text() or self.HEADER
            lines = current.splitlines()
            new_lines = transform(lines)
            text = "\n".join(new_lines).rstrip("\n") + "\n"
            self.store.write(self.filename, text)
        return self.status(text)

    def upsert(self, entry: IndexEntry) -> IndexStatus:
        """Insert or replace the line pointing at ``entry.target``. Always writes; check the status."""
        if "\n" in entry.title or "\n" in entry.note:
            raise ValueError("index entries must be single-line")

        def transform(lines: list[str]) -> list[str]:
            replaced = False
            out = []
            for line in lines:
                m = _ENTRY.match(line)
                if m and m.group("target") == entry.target:
                    if not replaced:
                        out.append(entry.render())
                        replaced = True
                    continue
                out.append(line)
            if not replaced:
                while out and out[-1] == "":
                    out.pop()
                out.append(entry.render())
            return out

        return self._rewrite(transform)

    def remove(self, target: str) -> IndexStatus:
        def transform(lines: list[str]) -> list[str]:
            return [ln for ln in lines if not ((m := _ENTRY.match(ln)) and m.group("target") == target)]

        return self._rewrite(transform)

    def status(self, text: str | None = None) -> IndexStatus:
        return index_status(self.text() if text is None else text, max_lines=self.max_lines, max_bytes=self.max_bytes)

    def for_prompt(self) -> str:
        return truncate_for_prompt(self.text(), max_lines=self.max_lines, max_bytes=self.max_bytes)

    def dangling(self) -> list[IndexEntry]:
        """Entries whose target file no longer exists."""
        return [e for e in self.entries() if not self.store.exists(e.target)]


@dataclass
class MemorySummary:
    """Parsed ``memory_summary.md``: first line ``v1`` then four fixed ``##`` sections."""

    sections: dict[str, str] = field(default_factory=lambda: {s: "" for s in SUMMARY_SECTIONS})

    def render(self) -> str:
        parts = [SUMMARY_VERSION, ""]
        for name in SUMMARY_SECTIONS:
            parts.append(f"## {name}")
            parts.append((self.sections.get(name) or "").strip() or "_(none yet)_")
            parts.append("")
        return "\n".join(parts).rstrip() + "\n"

    @classmethod
    def parse(cls, text: str) -> "MemorySummary | None":
        """Return ``None`` when the schema version is wrong (caller must regenerate)."""
        lines = text.splitlines()
        if not lines or lines[0].strip() != SUMMARY_VERSION:
            return None
        sections: dict[str, list[str]] = {s: [] for s in SUMMARY_SECTIONS}
        current: str | None = None
        for line in lines[1:]:
            if line.startswith("## "):
                name = line[3:].strip()
                current = name if name in sections else None
                continue
            if current:
                sections[current].append(line)
        out = {}
        for k, v in sections.items():
            body = "\n".join(v).strip()
            out[k] = "" if body == "_(none yet)_" else body
        return cls(out)

    @classmethod
    def load(cls, store: FileMemoryStore) -> "MemorySummary | None":
        try:
            return cls.parse(store.read(SUMMARY_FILE).content)
        except NotFoundError:
            return None

    def save(self, store: FileMemoryStore) -> None:
        store.write(SUMMARY_FILE, self.render())
