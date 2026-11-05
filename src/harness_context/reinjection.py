"""Post-compaction re-injection of durable state (compaction ladder rung L4).

"After any summary, re-load from disk the things a summary must not be trusted to carry"
(reference report 3.3 technique 20). The Claude Code list this module implements:

* root instruction files and unscoped rules, the memory index, the plan / todo list, a fresh git
  status snapshot (all via :class:`TextSource` callables supplied by the integrator - e.g.
  ``harness_memory.MemorySystem.post_compaction_context``);
* active skill bodies, **5,000 tokens each / 25,000 total**, oldest dropped first, truncated from
  the end (:class:`SkillBodiesSource`);
* up to **5** most recently read or edited files, re-read from disk; files over **5,000 tokens**
  come back as a path-only "referenced file" (:class:`RecentFilesSource`);
* a reminder that background jobs / sub-agents are still running, so they are not duplicated.

The bundle is rendered *once* at compaction time and stored inside the compaction marker message,
so every later window carries identical bytes (cache-stable) and a resumed session sees exactly
what the model saw.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence, runtime_checkable

from harness_core.types import Message, RunContext, ToolCall, ToolUseBlock

from .references import extract_paths
from .tokens import HeuristicTokenCounter, TokenCounter

RECENT_FILES_MAX = 5
RECENT_FILE_MAX_TOKENS = 5_000
SKILL_MAX_TOKENS = 5_000
SKILLS_TOTAL_MAX_TOKENS = 25_000

#: Tools whose path arguments count as "touched files" (read or edit).
DEFAULT_FILE_TOOLS = frozenset(
    {"read", "read_file", "Read", "view", "edit", "edit_file", "Edit", "write", "write_file", "Write",
     "str_replace_based_edit_tool", "MultiEdit", "apply_patch", "create_file", "NotebookEdit"}
)


@runtime_checkable
class ReinjectionSource(Protocol):
    name: str

    def render(self, ctx: RunContext) -> str | None: ...


@dataclass
class TextSource:
    """A static string or a zero/one-argument callable evaluated at compaction time."""

    name: str
    content: str | Callable[[], str | None] | Callable[[RunContext], str | None]
    pass_context: bool = False

    def render(self, ctx: RunContext) -> str | None:
        if isinstance(self.content, str):
            return self.content
        return self.content(ctx) if self.pass_context else self.content()  # type: ignore[call-arg]


class FileActivityTracker:
    """Most-recently touched files (thread-safe), fed from tool calls or replayed from history."""

    def __init__(self, file_tools: Iterable[str] = DEFAULT_FILE_TOOLS, *, capacity: int = 50) -> None:
        self.file_tools = frozenset(file_tools)
        self.capacity = capacity
        self._recent: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()

    def touch(self, path: str) -> None:
        with self._lock:
            self._recent.pop(path, None)
            self._recent[path] = None
            while len(self._recent) > self.capacity:
                self._recent.popitem(last=False)

    def observe_call(self, call: ToolCall | ToolUseBlock) -> None:
        if call.name in self.file_tools:
            for p in extract_paths(call.input):
                self.touch(p)

    def observe_messages(self, messages: Sequence[Message]) -> None:
        for m in messages:
            for u in m.tool_uses:
                self.observe_call(u)

    def recent(self, n: int = RECENT_FILES_MAX) -> list[str]:
        """Newest first."""
        with self._lock:
            return list(reversed(self._recent))[:n]


class RecentFilesSource:
    """Re-read the most recently touched files from disk (sandboxed to ``root``)."""

    name = "recent-files"

    def __init__(
        self,
        tracker: FileActivityTracker,
        root: str | Path,
        *,
        max_files: int = RECENT_FILES_MAX,
        max_tokens_per_file: int = RECENT_FILE_MAX_TOKENS,
        counter: TokenCounter | None = None,
    ) -> None:
        self.tracker = tracker
        self.root = Path(root).resolve()
        self.max_files = max_files
        self.max_tokens_per_file = max_tokens_per_file
        self.counter = counter or HeuristicTokenCounter()

    def _resolve(self, rel: str) -> Path | None:
        p = Path(rel)
        target = (p if p.is_absolute() else self.root / p).resolve()
        if target != self.root and self.root not in target.parents:
            return None
        return target

    def render(self, ctx: RunContext) -> str | None:
        parts = []
        for rel in self.tracker.recent(self.max_files):
            target = self._resolve(rel)
            if target is None or not target.is_file():
                parts.append(f'<referenced-file path="{rel}" status="unavailable"/>')
                continue
            try:
                text = target.read_text(encoding="utf-8", errors="replace")
            except OSError:
                parts.append(f'<referenced-file path="{rel}" status="unreadable"/>')
                continue
            tokens = self.counter.count_text(text)
            if tokens > self.max_tokens_per_file:
                parts.append(f'<referenced-file path="{rel}" tokens="~{tokens}">Too large to re-inject; read it with a range if needed.</referenced-file>')
            else:
                parts.append(f'<file path="{rel}">\n{text.rstrip()}\n</file>')
        return "\n".join(parts) if parts else None


class SkillBodiesSource:
    """Active skill bodies: ``provider()`` returns ``[(name, body)]`` in invocation order (oldest first)."""

    name = "active-skills"

    def __init__(
        self,
        provider: Callable[[], Sequence[tuple[str, str]]],
        *,
        per_skill_tokens: int = SKILL_MAX_TOKENS,
        total_tokens: int = SKILLS_TOTAL_MAX_TOKENS,
        counter: TokenCounter | None = None,
    ) -> None:
        self.provider = provider
        self.per_skill_tokens = per_skill_tokens
        self.total_tokens = total_tokens
        self.counter = counter or HeuristicTokenCounter()

    def _truncate(self, body: str) -> str:
        if self.counter.count_text(body) <= self.per_skill_tokens:
            return body
        lo, hi = 0, len(body)
        while lo < hi:  # longest prefix within the per-skill budget (truncated from the end)
            mid = (lo + hi + 1) // 2
            if self.counter.count_text(body[:mid]) <= self.per_skill_tokens:
                lo = mid
            else:
                hi = mid - 1
        return body[:lo].rstrip() + "\n[... skill body truncated; re-open the skill file for the rest]"

    def render(self, ctx: RunContext) -> str | None:
        skills = list(self.provider())
        rendered: list[str] = []
        used = 0
        for name, body in reversed(skills):  # newest first so the oldest are dropped
            text = f'<skill name="{name}">\n{self._truncate(body).rstrip()}\n</skill>'
            cost = self.counter.count_text(text)
            if used + cost > self.total_tokens:
                break  # this and every older skill are dropped
            rendered.append(text)
            used += cost
        rendered.reverse()  # back to invocation order
        return "\n".join(rendered) if rendered else None


@dataclass
class ReinjectionBundle:
    text: str
    sections: list[str]


class ReinjectionBuilder:
    """Render configured sources (in configured order) into one ``<post-compaction-context>`` block."""

    def __init__(self, sources: Sequence[ReinjectionSource] = (), *, background_jobs: Callable[[], Sequence[str]] | None = None) -> None:
        names = [s.name for s in sources]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate re-injection source names: {names}")
        self.sources = list(sources)
        self.background_jobs = background_jobs

    def build(self, ctx: RunContext) -> ReinjectionBundle:
        parts: list[str] = []
        names: list[str] = []
        for src in self.sources:
            try:
                text = src.render(ctx)
            except Exception as exc:  # a failing source must not abort compaction
                text = f"(could not re-load {src.name}: {type(exc).__name__}: {exc})"
            if text and text.strip():
                parts.append(f'<section name="{src.name}">\n{text.strip()}\n</section>')
                names.append(src.name)
        jobs = list(self.background_jobs()) if self.background_jobs else []
        if jobs:
            listing = "\n".join(f"- {j}" for j in jobs)
            parts.append(
                '<section name="background-jobs">\nThese are still running; do not start them again.\n'
                f"{listing}\n</section>"
            )
            names.append("background-jobs")
        if not parts:
            return ReinjectionBundle("", [])
        body = "\n".join(parts)
        text = (
            "<post-compaction-context>\nRe-loaded from durable state after compaction; this is current and "
            f"takes precedence over the summary where they differ.\n{body}\n</post-compaction-context>"
        )
        return ReinjectionBundle(text, names)


__all__ = [
    "DEFAULT_FILE_TOOLS",
    "FileActivityTracker",
    "RecentFilesSource",
    "ReinjectionBuilder",
    "ReinjectionBundle",
    "ReinjectionSource",
    "SkillBodiesSource",
    "TextSource",
]
