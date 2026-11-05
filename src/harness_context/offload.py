"""Ingest-time tool-output hygiene: the 25k-token cap (ladder rung L0) and file offload (rung L1).

Research basis (reference report 3.3 technique 10 and 21, checklist step 6, compaction ladder):

* "Claude Code restricts tool responses to 25,000 tokens by default"; truncation must *say what
  was cut and how to get more* (narrower query, ranges, pagination) - never truncate silently.
* Outputs above an offload threshold are written to the workspace and returned as
  **path + preview + summary stats** (Manus "the file system is the ultimate context";
  deepagents ``_blob_offload.py``). This is *restorable* compression: the content can be
  re-read with ranges, grep, ``head``/``tail``.

Both happen when a tool result is *created*, before it is appended to the durable log, so the log
stays append-only. :class:`OffloadingToolExecutor` wraps any
:class:`~harness_core.types.ToolExecutor` to apply this transparently.

Files are written atomically under ``<root>/<session_id>/`` with sanitized names; restore reads
are sandboxed to ``root``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from harness_core.types import RunContext, ToolCall, ToolExecutor, ToolOutcome, ToolSpec

from ._io import atomic_write_text, sha256_text
from .errors import ConfigurationError, OffloadError
from .tokens import HeuristicTokenCounter, TokenCounter

DEFAULT_MAX_RESULT_TOKENS = 25_000      # vendor default (Claude Code)
DEFAULT_OFFLOAD_THRESHOLD_TOKENS = 8_000

_SAFE = re.compile(r"[^A-Za-z0-9_-]+")


def _safe_name(text: str, limit: int = 80) -> str:
    cleaned = _SAFE.sub("_", text).strip("_") or "x"
    return cleaned[:limit]


@dataclass(frozen=True)
class OffloadPolicy:
    """Defaults: 25k-token ingest cap *(vendor default)*; offload above 8k tokens *(our choice)*."""

    max_result_tokens: int = DEFAULT_MAX_RESULT_TOKENS
    offload_threshold_tokens: int = DEFAULT_OFFLOAD_THRESHOLD_TOKENS
    preview_head_chars: int = 1_500
    preview_tail_chars: int = 500
    exclude_tools: frozenset[str] = field(default_factory=lambda: frozenset({"memory"}))

    def __post_init__(self) -> None:
        if self.max_result_tokens <= 0 or self.offload_threshold_tokens <= 0:
            raise ConfigurationError("token limits must be positive")
        if self.offload_threshold_tokens > self.max_result_tokens:
            raise ConfigurationError("offload_threshold_tokens must not exceed max_result_tokens")


@dataclass(frozen=True)
class OffloadedOutput:
    path: Path
    display_path: str
    bytes: int
    lines: int
    tokens: int
    sha256: str


def truncation_note(shown_tokens: int, total_tokens: int, *, tool: str = "") -> str:
    who = f" from {tool}" if tool else ""
    return (
        f"\n[Output{who} truncated: showing the first ~{shown_tokens:,} of ~{total_tokens:,} tokens. "
        "Narrow the request (filters, line ranges, pagination, a more specific query) to see the rest.]"
    )


def cap_text(text: str, max_tokens: int, counter: TokenCounter, *, tool: str = "") -> tuple[str, bool]:
    """Deterministically cut ``text`` to about ``max_tokens`` tokens with a coaching note."""
    total = counter.count_text(text)
    if total <= max_tokens:
        return text, False
    ratio = max_tokens / total
    keep = max(0, int(len(text) * ratio) - 200)
    cut = text[:keep]
    if "\n" in cut[-500:]:
        cut = cut[: cut.rfind("\n")]
    return cut + truncation_note(counter.count_text(cut), total, tool=tool), True


class ToolOutputOffloader:
    """Apply the ingest cap and offload large tool outputs to files.

    :param root: offload directory; ``None`` disables offloading (the cap still applies).
    :param display_path: maps the absolute file path to what the model sees (e.g. relative to cwd).
    """

    def __init__(
        self,
        root: str | Path | None,
        *,
        policy: OffloadPolicy | None = None,
        counter: TokenCounter | None = None,
        display_path: Callable[[Path], str] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve() if root is not None else None
        self.policy = policy or OffloadPolicy()
        self.counter = counter or HeuristicTokenCounter()
        self.display_path = display_path or (lambda p: p.as_posix())

    # ------------------------------------------------------------------ main entry point
    def process(self, call: ToolCall, outcome: ToolOutcome, ctx: RunContext) -> ToolOutcome:
        """Return ``outcome`` unchanged, capped, or replaced by an offload reference."""
        if call.name in self.policy.exclude_tools:
            return outcome
        text = self._text_of(outcome)
        if text is None:
            return outcome
        tokens = self.counter.count_text(text)
        meta = dict(outcome.metadata)
        if self.root is not None and tokens > self.policy.offload_threshold_tokens:
            saved = self.offload(text, call=call, ctx=ctx, tokens=tokens)
            meta.update({"offloaded_path": saved.display_path, "original_tokens": tokens, "bytes": saved.bytes})
            return ToolOutcome(self.reference_text(text, saved, tool=call.name), outcome.is_error, meta)
        if tokens > self.policy.max_result_tokens:
            capped, _ = cap_text(text, self.policy.max_result_tokens, self.counter, tool=call.name)
            meta.update({"truncated": True, "original_tokens": tokens})
            return ToolOutcome(capped, outcome.is_error, meta)
        return outcome

    @staticmethod
    def _text_of(outcome: ToolOutcome) -> str | None:
        if isinstance(outcome.content, str):
            return outcome.content
        items = outcome.content
        if all(isinstance(i, dict) and i.get("type") == "text" for i in items):
            return "".join(str(i.get("text", "")) for i in items)
        return None  # images/documents: leave untouched

    def offload(self, text: str, *, call: ToolCall, ctx: RunContext, tokens: int | None = None) -> OffloadedOutput:
        if self.root is None:
            raise OffloadError("offloading is disabled (no root directory configured)")
        directory = self.root / _safe_name(ctx.session_id)
        path = directory / f"{_safe_name(call.name, 40)}-{_safe_name(call.id)}.txt"
        try:
            atomic_write_text(path, text)
        except OSError as exc:
            raise OffloadError(f"could not offload output of {call.name} to {path}: {exc}") from exc
        return OffloadedOutput(
            path=path,
            display_path=self.display_path(path),
            bytes=len(text.encode("utf-8")),
            lines=text.count("\n") + (1 if text and not text.endswith("\n") else 0),
            tokens=self.counter.count_text(text) if tokens is None else tokens,
            sha256=sha256_text(text),
        )

    def reference_text(self, text: str, saved: OffloadedOutput, *, tool: str = "") -> str:
        p = self.policy
        head = text[: p.preview_head_chars]
        tail = text[-p.preview_tail_chars:] if len(text) > p.preview_head_chars + p.preview_tail_chars else ""
        omitted = len(text) - len(head) - len(tail)
        parts = [
            f'<offloaded-tool-output path="{saved.display_path}" bytes="{saved.bytes}" lines="{saved.lines}" '
            f'tokens="~{saved.tokens}" sha256="{saved.sha256[:16]}">',
            f"The full output{' of ' + tool if tool else ''} (~{saved.tokens:,} tokens) was saved to the file above "
            "instead of being placed in context.",
            f"Preview (first {len(head):,} chars):",
            head,
        ]
        if tail:
            parts += [f"[... {omitted:,} chars omitted ...]", f"Last {len(tail):,} chars:", tail]
        parts += [
            "Load only what you need: read the file with an offset/limit or line range, grep it, or head/tail it.",
            "</offloaded-tool-output>",
        ]
        return "\n".join(parts)

    def restore(self, path: str | Path) -> str:
        """Read an offloaded file back (sandboxed to ``root``)."""
        if self.root is None:
            raise OffloadError("offloading is disabled")
        target = Path(path)
        target = (target if target.is_absolute() else self.root / target).resolve()
        if target != self.root and self.root not in target.parents:
            raise OffloadError(f"path outside the offload directory: {path}")
        try:
            return target.read_text(encoding="utf-8")
        except OSError as exc:
            raise OffloadError(f"cannot restore {path}: {exc}") from exc


class OffloadingToolExecutor:
    """:class:`~harness_core.types.ToolExecutor` decorator applying :class:`ToolOutputOffloader`."""

    def __init__(self, inner: ToolExecutor, offloader: ToolOutputOffloader) -> None:
        self.inner = inner
        self.offloader = offloader

    def specs(self) -> list[ToolSpec]:
        return self.inner.specs()

    def execute(self, call: ToolCall, ctx: RunContext) -> ToolOutcome:
        return self.offloader.process(call, self.inner.execute(call, ctx), ctx)


__all__ = [
    "DEFAULT_MAX_RESULT_TOKENS",
    "DEFAULT_OFFLOAD_THRESHOLD_TOKENS",
    "OffloadPolicy",
    "OffloadedOutput",
    "OffloadingToolExecutor",
    "ToolOutputOffloader",
    "cap_text",
    "truncation_note",
]
