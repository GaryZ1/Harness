"""Default :class:`~harness_core.types.ContextAssembler` and compaction contract for the loop.

The loop never decides *what* the model sees; it hands the session's projected view to a
``ContextAssembler`` (reference report section 1.7 step 1: "Build each prompt by projecting the
log"). Cache-stable layering and the compaction ladder live in a dedicated context package; this
module provides the trivial pass-through assembler and the protocol the loop uses to ask for a
durable compaction.

Compaction rules the loop enforces in code (section 2.7 step 7, section 2.8 "Compaction failures"):

* compaction is *recorded* (``Session.record_compaction``), never a rewrite of the log;
* the replacement view must keep every ``tool_result`` paired with a ``tool_use`` that precedes it
  (:func:`validate_tool_pairing`), otherwise :class:`~harness_core.errors.CompactionError`;
* a compaction that does not shrink the estimated window counts as a failed attempt, and the loop
  stops after a few consecutive failures (thrash guard) instead of looping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

from .errors import CompactionError
from .types import Message, ModelRequest, Role, RunContext, ToolSpec, messages_to_api


class PassthroughAssembler:
    """Send the view unchanged: ``system``, ``messages`` and ``tools`` as given."""

    def __init__(self, *, max_tokens: int = 16_000, extra: dict[str, Any] | None = None) -> None:
        self.max_tokens = max_tokens
        self.extra = dict(extra or {})

    def prepare(self, system: str, messages: list[Message], tools: list[ToolSpec], ctx: RunContext) -> ModelRequest:
        return ModelRequest(system=system, messages=list(messages), tools=list(tools), max_tokens=self.max_tokens,
                            extra=dict(self.extra))


@dataclass
class CompactionResult:
    """Replacement view produced by a :class:`Compactor`."""

    messages: list[Message]
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Compactor(Protocol):
    """Produce a smaller model-visible view; return ``None`` when nothing can be freed.

    ``reason`` is ``"threshold"`` (proactive, before overflow) or ``"overflow"`` (after a
    context-window-exceeded signal). Implementations must not call tools.
    """

    def compact(self, messages: list[Message], ctx: RunContext, *, reason: str) -> CompactionResult | None: ...


def estimate_tokens(messages: Sequence[Message], system: str = "") -> int:
    """Rough token estimate (UTF-8 JSON bytes / 4). Used only to check that compaction freed space."""
    payload = json.dumps(messages_to_api(messages), ensure_ascii=False, default=str)
    return (len(payload.encode("utf-8")) + len(system.encode("utf-8"))) // 4


def validate_tool_pairing(messages: Sequence[Message]) -> None:
    """Raise :class:`CompactionError` if a ``tool_result`` has no earlier matching ``tool_use``,
    or if an assistant tool batch that is followed by another message is missing results."""
    seen: set[str] = set()
    for i, m in enumerate(messages):
        if m.role is Role.ASSISTANT:
            ids = {u.id for u in m.tool_uses}
            seen |= ids
            if ids and i + 1 < len(messages):
                nxt = messages[i + 1]
                got = {r.tool_use_id for r in nxt.tool_results}
                if not ids <= got:
                    raise CompactionError(f"tool_use ids {sorted(ids - got)} lack results in the following message")
        for r in m.tool_results:
            if r.tool_use_id not in seen:
                raise CompactionError(f"tool_result {r.tool_use_id} has no preceding tool_use")
