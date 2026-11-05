"""The single "dynamic context" channel (checklist step 4).

Everything time-varying - wall-clock time, git status, budget notes, todo/plan *recitation*,
stuck-loop warnings - goes through one appender and is placed at the **end** of the window, after
the moving cache breakpoint, never in the cached system prefix:

* **turn-scoped** items render only in the window being built (they are never stored in the
  durable log), mirroring Claude's turn-scoped system messages
  (``clear_at: "next_user_message"``): they cost nothing on later turns and never pile up;
* **persistent** items (a newly loaded nested instruction file, a file-changed notice) must stay,
  so :meth:`DynamicContext.persistent_message` returns a message for the integrator to append to
  the log, de-duplicated by key + content hash so reminders do not accumulate.

Recitation goes last (Manus ``todo.md``: the global plan sits at the end of context, against
lost-in-the-middle and goal drift).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from harness_core.types import Message, Role, RunContext, TextBlock

from ._io import sha256_text
from .messages import KIND_DYNAMIC, KIND_REMINDER, META_KEY

RECITATION_PRIORITY = 1_000  # rendered last


@dataclass(frozen=True)
class DynamicItem:
    key: str
    text: str
    priority: int = 100


Provider = Callable[[RunContext], "str | None"]


def safe_to_append(log: "list[Message]") -> bool:
    """True when appending a user message now cannot separate a ``tool_use`` from its ``tool_result``."""
    return not log or not (log[-1].role is Role.ASSISTANT and log[-1].tool_uses)


class DynamicContext:
    """Collects turn-scoped items and renders them as one tail message.

    :param role: ``Role.SYSTEM`` (mid-conversation system message, supported models) or
        ``Role.USER`` (``<system-reminder>``-wrapped text for other providers; the assembler merges
        it into the final user message).
    """

    def __init__(self, *, role: Role = Role.SYSTEM, wrap_tag: str = "system-reminder") -> None:
        if role not in (Role.SYSTEM, Role.USER):
            raise ValueError("dynamic context role must be SYSTEM or USER")
        self.role = role
        self.wrap_tag = wrap_tag
        self._providers: dict[str, tuple[int, Provider]] = {}
        self._pending: dict[str, DynamicItem] = {}
        self._persisted: dict[str, str] = {}
        self._lock = threading.Lock()

    def register(self, key: str, provider: Provider, *, priority: int = 100) -> None:
        """Recurring turn-scoped provider (time, git status, recitation...), evaluated every window."""
        with self._lock:
            self._providers[key] = (priority, provider)

    def register_recitation(self, provider: Callable[[], "str | None"], *, key: str = "recitation") -> None:
        """Todo / plan recitation, always rendered last."""
        self.register(key, lambda ctx: provider(), priority=RECITATION_PRIORITY)

    def push(self, key: str, text: str, *, priority: int = 100) -> None:
        """One-shot turn-scoped item for the next window only (same key replaces)."""
        with self._lock:
            self._pending[key] = DynamicItem(key, text, priority)

    def collect(self, ctx: RunContext, extra: list[DynamicItem] | None = None) -> list[DynamicItem]:
        with self._lock:
            providers = dict(self._providers)
            pending = dict(self._pending)
            self._pending.clear()
        items: dict[str, DynamicItem] = {}
        for key, (priority, fn) in providers.items():
            try:
                text = fn(ctx)
            except Exception as exc:  # a broken provider must not break the request
                text = f"({key} unavailable: {type(exc).__name__})"
            if text and text.strip():
                items[key] = DynamicItem(key, text.strip(), priority)
        items.update(pending)
        for it in extra or []:
            items[it.key] = it
        return sorted(items.values(), key=lambda i: (i.priority, i.key))

    def render(self, items: list[DynamicItem]) -> Message | None:
        if not items:
            return None
        body = "\n\n".join(i.text for i in items)
        text = f"<{self.wrap_tag}>\n{body}\n</{self.wrap_tag}>" if self.wrap_tag else body
        return Message(
            self.role,
            [TextBlock(text)],
            metadata={META_KEY: {"kind": KIND_DYNAMIC, "turn_scoped": True, "clear_at": "next_user_message",
                                 "keys": [i.key for i in items]}},
        )

    def persistent_message(self, key: str, text: str) -> Message | None:
        """A reminder to append to the durable log, or ``None`` if identical content was already appended."""
        digest = sha256_text(text)
        with self._lock:
            if self._persisted.get(key) == digest:
                return None
            self._persisted[key] = digest
        wrapped = f"<{self.wrap_tag}>\n{text.strip()}\n</{self.wrap_tag}>" if self.wrap_tag else text.strip()
        return Message(Role.USER, [TextBlock(wrapped)], metadata={META_KEY: {"kind": KIND_REMINDER, "key": key}})


__all__ = ["DynamicContext", "DynamicItem", "RECITATION_PRIORITY", "safe_to_append"]
