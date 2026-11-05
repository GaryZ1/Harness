"""Server-side context management: emit Anthropic ``context_management`` params into ``ModelRequest.extra``.

The research prefers provider mechanisms where available, because the server edits the window
*without* the client rewriting history:

* ``clear_tool_uses_20250919`` (beta ``context-management-2025-06-27``): trigger **100,000**
  input tokens, keep the **last 3** tool uses, ``clear_at_least``, ``exclude_tools``,
  ``clear_tool_inputs`` (default ``false``);
* ``clear_thinking_20251015``: ``keep`` = ``{"type": "thinking_turns", "value": N}`` or ``"all"``;
  must be listed **first** when combined;
* ``compact_20260112`` (beta ``compact-2026-01-12``): ``trigger`` default
  ``{"type": "input_tokens", "value": 150000}``, value **>= 50,000**; ``pause_after_compaction``;
  ``instructions`` fully replaces the default prompt (we pass the structured summary prompt,
  which includes "do not call any tools"). The returned ``compaction`` block must be passed back
  unchanged; the assembler keeps it verbatim and puts a cache breakpoint on it.

The adapter is expected to forward ``extra["betas"]`` as beta headers and
``extra["context_management"]`` as the request parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .budget import DEFAULT_COMPACTION_THRESHOLD, MIN_SERVER_COMPACTION_TRIGGER
from .clearing import DEFAULT_EXCLUDED_TOOLS
from .compaction import SUMMARY_PROMPT
from .errors import ConfigurationError

CONTEXT_MANAGEMENT_BETA = "context-management-2025-06-27"
COMPACTION_BETA = "compact-2026-01-12"
CLEAR_TOOL_USES = "clear_tool_uses_20250919"
CLEAR_THINKING = "clear_thinking_20251015"
COMPACT = "compact_20260112"


@dataclass(frozen=True)
class ServerContextManagement:
    clear_tool_uses: bool = False
    clear_trigger_tokens: int = 100_000
    keep_tool_uses: int = 3
    clear_at_least_tokens: int | None = 5_000
    exclude_tools: tuple[str, ...] = field(default_factory=lambda: tuple(sorted(DEFAULT_EXCLUDED_TOOLS)))
    clear_tool_inputs: bool = False
    clear_thinking: bool = False
    keep_thinking_turns: int | None = None   # None = keep "all"
    compact: bool = False
    compact_trigger_tokens: int = DEFAULT_COMPACTION_THRESHOLD
    pause_after_compaction: bool = False
    compact_instructions: str | None = SUMMARY_PROMPT

    def __post_init__(self) -> None:
        if self.compact and self.compact_trigger_tokens < MIN_SERVER_COMPACTION_TRIGGER:
            raise ConfigurationError(f"compact trigger must be >= {MIN_SERVER_COMPACTION_TRIGGER} input tokens")
        if self.keep_tool_uses < 0 or self.clear_trigger_tokens <= 0:
            raise ConfigurationError("invalid clear_tool_uses settings")
        if self.keep_thinking_turns is not None and self.keep_thinking_turns < 1:
            raise ConfigurationError("keep_thinking_turns must be >= 1 (or None for all)")

    @property
    def enabled(self) -> bool:
        return self.clear_tool_uses or self.clear_thinking or self.compact

    def edits(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if self.clear_thinking:  # must come first
            keep: Any = "all" if self.keep_thinking_turns is None else {"type": "thinking_turns", "value": self.keep_thinking_turns}
            out.append({"type": CLEAR_THINKING, "keep": keep})
        if self.clear_tool_uses:
            edit: dict[str, Any] = {
                "type": CLEAR_TOOL_USES,
                "trigger": {"type": "input_tokens", "value": self.clear_trigger_tokens},
                "keep": {"type": "tool_uses", "value": self.keep_tool_uses},
            }
            if self.clear_at_least_tokens:
                edit["clear_at_least"] = {"type": "input_tokens", "value": self.clear_at_least_tokens}
            if self.exclude_tools:
                edit["exclude_tools"] = sorted(self.exclude_tools)
            if self.clear_tool_inputs:
                edit["clear_tool_inputs"] = True
            out.append(edit)
        if self.compact:
            edit = {"type": COMPACT, "trigger": {"type": "input_tokens", "value": self.compact_trigger_tokens}}
            if self.pause_after_compaction:
                edit["pause_after_compaction"] = True
            if self.compact_instructions:
                edit["instructions"] = self.compact_instructions
            out.append(edit)
        return out

    def betas(self) -> list[str]:
        out = []
        if self.clear_tool_uses or self.clear_thinking:
            out.append(CONTEXT_MANAGEMENT_BETA)
        if self.compact:
            out.append(COMPACTION_BETA)
        return out

    def apply(self, extra: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of ``extra`` with ``context_management`` and merged, sorted ``betas``."""
        out = dict(extra)
        if not self.enabled:
            return out
        out["context_management"] = {"edits": self.edits()}
        out["betas"] = sorted(set(out.get("betas", [])) | set(self.betas()))
        return out


__all__ = [
    "CLEAR_THINKING",
    "CLEAR_TOOL_USES",
    "COMPACT",
    "COMPACTION_BETA",
    "CONTEXT_MANAGEMENT_BETA",
    "ServerContextManagement",
]
