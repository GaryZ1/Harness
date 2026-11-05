"""Output budgets: cap every tool result, coach the model to narrow, offload oversize payloads.

Research basis (``reports/SOTA agentic harness reference.md`` §5.7 step 5, §5.8 "Unbounded or
silently truncated outputs"):

* A **global per-result cap of 25,000 tokens** (Claude Code's default) applies to every tool;
  individual tools may set a lower cap (e.g. ``bash``).
* Truncation is **never silent**: the result says how much was cut and how to narrow the query
  ("use offset/limit", "narrow the pattern"), because truncation messages that steer toward
  small targeted calls are what Anthropic found to work.
* **Offload, don't drop.** When a workspace directory is available, the full payload is written
  to ``<offload_dir>/<call_id>.txt`` and the model receives a head+tail preview, the absolute path
  and the size, so it can page through with ``read``/``grep`` (deepagents ``_blob_offload``
  pattern; "return path + preview + size").

Token counts are estimated at ``chars_per_token=4`` characters per token, the usual rule of
thumb for English and code; this is deliberately provider-neutral (no tokenizer dependency).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness_memory.locking import atomic_write_text

#: Claude Code's default per-tool-result cap.
DEFAULT_MAX_RESULT_TOKENS = 25_000
DEFAULT_CHARS_PER_TOKEN = 4
DEFAULT_PREVIEW_TOKENS = 2_000

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def estimate_tokens(text: str | Any, chars_per_token: int = DEFAULT_CHARS_PER_TOKEN) -> int:
    """Rough token estimate (ceil(chars / chars_per_token)); non-strings are JSON-encoded first."""
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False, default=str)
    return -(-len(text) // max(chars_per_token, 1))


@dataclass(frozen=True)
class OutputPolicy:
    """Per-result output budget.

    :param max_tokens: hard cap per tool result (default 25,000 = Claude Code).
    :param preview_tokens: size of the head+tail preview returned when a result is offloaded.
    :param offload_dir: absolute directory for full payloads; ``None`` disables offloading
        (results are then truncated in place, head+tail).
    :param head_fraction: share of the preview kept from the start (rest from the end, where
        errors and summaries usually are).
    """

    max_tokens: int = DEFAULT_MAX_RESULT_TOKENS
    chars_per_token: int = DEFAULT_CHARS_PER_TOKEN
    preview_tokens: int = DEFAULT_PREVIEW_TOKENS
    offload_dir: Path | None = None
    head_fraction: float = 0.6

    @property
    def max_chars(self) -> int:
        return self.max_tokens * self.chars_per_token

    def with_max_tokens(self, max_tokens: int | None) -> "OutputPolicy":
        if max_tokens is None or max_tokens >= self.max_tokens:
            return self
        return OutputPolicy(max_tokens, self.chars_per_token, min(self.preview_tokens, max_tokens), self.offload_dir, self.head_fraction)


@dataclass(frozen=True)
class BudgetedText:
    text: str
    truncated: bool
    original_chars: int
    offload_path: str | None = None


def head_tail(text: str, budget_chars: int, head_fraction: float = 0.6) -> tuple[str, str, int]:
    """Return (head, tail, omitted_chars) keeping ``budget_chars`` in total, cut at line breaks when possible."""
    if len(text) <= budget_chars:
        return text, "", 0
    head_n = max(int(budget_chars * head_fraction), 0)
    tail_n = max(budget_chars - head_n, 0)
    head = text[:head_n]
    nl = head.rfind("\n")
    if nl > head_n * 0.5:
        head = head[: nl + 1]
    tail = text[len(text) - tail_n :] if tail_n else ""
    nl = tail.find("\n")
    if 0 <= nl < tail_n * 0.5:
        tail = tail[nl + 1 :]
    return head, tail, len(text) - len(head) - len(tail)


def apply_budget(
    text: str,
    policy: OutputPolicy,
    *,
    call_id: str = "result",
    hint: str = "Narrow the request (smaller range, more specific pattern, or a filter) to see less at once.",
) -> BudgetedText:
    """Fit ``text`` into ``policy``. Offloads the full text when ``policy.offload_dir`` is set."""
    if len(text) <= policy.max_chars:
        return BudgetedText(text, False, len(text))
    total_tokens = estimate_tokens(text, policy.chars_per_token)
    if policy.offload_dir is not None:
        path = Path(policy.offload_dir) / f"{_SAFE_NAME.sub('_', call_id)[:120] or 'result'}.txt"
        atomic_write_text(path, text)
        head, tail, omitted = head_tail(text, policy.preview_tokens * policy.chars_per_token, policy.head_fraction)
        note = (
            f"\n\n[... {omitted:,} characters omitted. The full output ({len(text):,} characters, ~{total_tokens:,} tokens) "
            f"exceeded the {policy.max_tokens:,}-token result limit and was saved to {path} . "
            f"Use read with offset/limit or grep on that file to inspect it. {hint}]\n\n"
        )
        return BudgetedText(head + note + tail, True, len(text), str(path))
    note_template = (
        "\n\n[... {omitted:,} characters omitted: output was ~{tokens:,} tokens, over the {limit:,}-token result limit. {hint}]\n\n"
    )
    reserve = len(note_template) + 64 + len(hint)
    head, tail, omitted = head_tail(text, max(policy.max_chars - reserve, 0), policy.head_fraction)
    note = note_template.format(omitted=omitted, tokens=total_tokens, limit=policy.max_tokens, hint=hint)
    return BudgetedText(head + note + tail, True, len(text))


__all__ = [
    "BudgetedText",
    "DEFAULT_CHARS_PER_TOKEN",
    "DEFAULT_MAX_RESULT_TOKENS",
    "DEFAULT_PREVIEW_TOKENS",
    "OutputPolicy",
    "apply_budget",
    "estimate_tokens",
    "head_tail",
]
