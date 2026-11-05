"""Working memory: the in-context scratchpad for the current task.

Function axis: *working* memory. Loading policy: lives in the context window, but is mirrored to
disk so it survives compaction and context resets ("ASSUME INTERRUPTION").

Responsibilities

* a todo/plan list with at most one ``in_progress`` item (Claude Code todo semantics);
* bounded scratch notes (Anthropic "structured note-taking" / ``NOTES.md``);
* **recitation**: a compact block rendered at the *end* of the context so the global plan stays
  in attention (Manus ``todo.md`` rewrite) - appended as a message, never written into the cached
  prompt prefix;
* context-budget tracking with a **pre-compaction warning** ("save important results to memory
  before they are cleared") and a post-compaction re-injection bundle.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ._time import Clock, to_iso, utcnow
from .errors import LimitExceededError, NotFoundError, ValidationError
from .locking import atomic_write_text

PRE_COMPACTION_WARNING = (
    "Context is close to the compaction threshold. Before older tool results are cleared, save any "
    "important findings, decisions and progress to memory or the task progress log. Anything not "
    "recorded may be lost."
)


class TodoStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


@dataclass
class TodoItem:
    id: str
    content: str
    status: TodoStatus = TodoStatus.PENDING
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class ContextBudget:
    window_tokens: int
    compaction_threshold: float = 0.8
    warn_margin: float = 0.05
    used_tokens: int = 0

    @property
    def fraction(self) -> float:
        return self.used_tokens / self.window_tokens if self.window_tokens else 1.0

    @property
    def should_warn(self) -> bool:
        return self.fraction >= self.compaction_threshold - self.warn_margin

    @property
    def should_compact(self) -> bool:
        return self.fraction >= self.compaction_threshold


@dataclass
class WorkingMemory:
    """Per-session working memory with JSON persistence."""

    goal: str = ""
    todos: list[TodoItem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    max_notes: int = 50
    max_note_chars: int = 2_000
    budget: ContextBudget | None = None
    clock: Clock = field(default=utcnow, repr=False, compare=False)
    _warned: bool = field(default=False, repr=False, compare=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    # ------------------------------------------------------------------ todos
    def set_plan(self, items: list[str | dict[str, Any]]) -> None:
        """Replace the plan. Accepts strings or ``{"content", "status"?, "id"?}`` dicts."""
        new: list[TodoItem] = []
        for i, raw in enumerate(items, start=1):
            if isinstance(raw, str):
                new.append(TodoItem(id=str(i), content=raw))
            else:
                new.append(
                    TodoItem(
                        id=str(raw.get("id") or i),
                        content=str(raw["content"]),
                        status=TodoStatus(raw.get("status", "pending")),
                        note=str(raw.get("note", "")),
                    )
                )
        self._check_single_in_progress(new)
        with self._lock:
            self.todos = new

    def add_todo(self, content: str) -> TodoItem:
        with self._lock:
            existing = {t.id for t in self.todos}
            n = len(self.todos) + 1
            while str(n) in existing:
                n += 1
            item = TodoItem(id=str(n), content=content)
            self.todos.append(item)
            return item

    def _get(self, todo_id: str) -> TodoItem:
        for t in self.todos:
            if t.id == todo_id:
                return t
        raise NotFoundError(f"no todo {todo_id}")

    def update(self, todo_id: str, status: TodoStatus | str, note: str = "") -> TodoItem:
        status = TodoStatus(status)
        with self._lock:
            item = self._get(todo_id)
            if status is TodoStatus.IN_PROGRESS:
                for t in self.todos:
                    if t.status is TodoStatus.IN_PROGRESS and t.id != todo_id:
                        raise ValidationError(f"todo {t.id} is already in progress; finish or block it first")
            item.status = status
            if note:
                item.note = note
            return item

    def start_next(self) -> TodoItem | None:
        with self._lock:
            if any(t.status is TodoStatus.IN_PROGRESS for t in self.todos):
                return next(t for t in self.todos if t.status is TodoStatus.IN_PROGRESS)
            for t in self.todos:
                if t.status is TodoStatus.PENDING:
                    t.status = TodoStatus.IN_PROGRESS
                    return t
        return None

    @staticmethod
    def _check_single_in_progress(items: list[TodoItem]) -> None:
        if sum(1 for t in items if t.status is TodoStatus.IN_PROGRESS) > 1:
            raise ValidationError("at most one todo may be in_progress")

    @property
    def done(self) -> bool:
        return bool(self.todos) and all(t.status is TodoStatus.COMPLETED for t in self.todos)

    # ------------------------------------------------------------------ notes
    def note(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if len(text) > self.max_note_chars:
            raise LimitExceededError(f"note is {len(text)} chars; limit is {self.max_note_chars}. Save detail to a file instead.")
        with self._lock:
            self.notes.append(f"[{to_iso(self.clock())}] {text}")
            # Keep the newest notes; working memory is bounded by design.
            if len(self.notes) > self.max_notes:
                del self.notes[: len(self.notes) - self.max_notes]

    # ------------------------------------------------------------------ recitation & budget
    def recitation(self, *, max_notes: int = 5) -> str:
        """Compact plan block to append at the end of the context (never into the cached prefix)."""
        marks = {TodoStatus.PENDING: "[ ]", TodoStatus.IN_PROGRESS: "[>]", TodoStatus.COMPLETED: "[x]", TodoStatus.BLOCKED: "[!]"}
        lines = ["<working-memory>"]
        if self.goal:
            lines.append(f"Goal: {self.goal}")
        if self.todos:
            done = sum(1 for t in self.todos if t.status is TodoStatus.COMPLETED)
            lines.append(f"Plan ({done}/{len(self.todos)} done):")
            for t in self.todos:
                suffix = f" — {t.note}" if t.note else ""
                lines.append(f"  {marks[t.status]} {t.id}. {t.content}{suffix}")
        if self.notes and max_notes:
            lines.append("Recent notes:")
            lines.extend(f"  - {n}" for n in self.notes[-max_notes:])
        lines.append("</working-memory>")
        return "\n".join(lines)

    def observe_usage(self, used_tokens: int) -> str | None:
        """Update context usage; returns the pre-compaction warning once per approach to the threshold."""
        if self.budget is None:
            return None
        self.budget.used_tokens = used_tokens
        if self.budget.should_warn and not self._warned:
            self._warned = True
            return PRE_COMPACTION_WARNING
        if not self.budget.should_warn:
            self._warned = False
        return None

    def after_compaction(self) -> None:
        """Reset the warning latch; the caller re-injects :meth:`recitation` with the other bundles."""
        self._warned = False

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "todos": [t.to_dict() for t in self.todos],
            "notes": list(self.notes),
            "saved_at": to_iso(self.clock()),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs: Any) -> "WorkingMemory":
        wm = cls(goal=data.get("goal", ""), notes=list(data.get("notes", [])), **kwargs)
        wm.set_plan(data.get("todos", []))
        return wm

    def save(self, path: str | Path) -> None:
        with self._lock:
            atomic_write_text(Path(path), json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "WorkingMemory":
        p = Path(path)
        if not p.exists():
            return cls(**kwargs)
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")), **kwargs)

    def to_markdown(self) -> str:
        """Human-readable ``NOTES.md`` rendering."""
        out = ["# Working notes", ""]
        if self.goal:
            out += [f"**Goal:** {self.goal}", ""]
        if self.todos:
            out.append("## Plan")
            for t in self.todos:
                box = "x" if t.status is TodoStatus.COMPLETED else " "
                tag = "" if t.status in (TodoStatus.PENDING, TodoStatus.COMPLETED) else f" _({t.status.value})_"
                out.append(f"- [{box}] {t.content}{tag}")
            out.append("")
        if self.notes:
            out.append("## Notes")
            out.extend(f"- {n}" for n in self.notes)
        return "\n".join(out).rstrip() + "\n"
