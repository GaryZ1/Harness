"""Token budget policy: quality-oriented thresholds, headroom and calibration.

Research basis (reference report 3.3 technique 4 and 25, checklist step 11 "Guards", 3.5 table):

* Trigger compaction for **quality, not overflow**: Claude's threshold default is **150,000**
  (minimum 50,000); Manus' "pre-rot" threshold is 128K-200K. The hard ceiling is
  ``min(configured, 90% of the window)`` with a **95%** effective window (Codex).
* Estimate the next request from the last ``usage`` (``input + cache_read + cache_creation`` +
  output) and compact *before* the window - the summarization call itself must fit.
* Skip auto-compaction when the model's window is unknown (Cline).
* Give the model a budget signal: a short turn-scoped "tokens used / remaining / next compaction
  at N" note.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum

from harness_core.types import Usage

from .errors import ConfigurationError

DEFAULT_COMPACTION_THRESHOLD = 150_000   # Anthropic compact_20260112 default trigger
MIN_SERVER_COMPACTION_TRIGGER = 50_000   # Anthropic minimum
MAX_THRESHOLD_FRACTION = 0.90            # Codex: min(configured, 90% of window)
EFFECTIVE_WINDOW_FRACTION = 0.95         # Codex effective_context_window_percent


class BudgetLevel(str, Enum):
    OK = "ok"
    WARN = "warn"            # approaching compaction: pre-compaction warning
    COMPACT = "compact"      # at/over the compaction trigger
    OVERFLOW = "overflow"    # over the effective window


@dataclass(frozen=True)
class BudgetPolicy:
    """:param window_tokens: the model's context window (``None`` = unknown: no auto-compaction)."""

    window_tokens: int | None = None
    compaction_threshold_tokens: int = DEFAULT_COMPACTION_THRESHOLD
    max_threshold_fraction: float = MAX_THRESHOLD_FRACTION
    effective_window_fraction: float = EFFECTIVE_WINDOW_FRACTION
    warn_fraction: float = 0.9           # warn at 90% of the compaction trigger (our choice)

    def __post_init__(self) -> None:
        if self.window_tokens is not None and self.window_tokens <= 0:
            raise ConfigurationError("window_tokens must be positive")
        if self.compaction_threshold_tokens <= 0:
            raise ConfigurationError("compaction_threshold_tokens must be positive")
        for name in ("max_threshold_fraction", "effective_window_fraction", "warn_fraction"):
            v = getattr(self, name)
            if not 0 < v <= 1:
                raise ConfigurationError(f"{name} must be in (0, 1]")

    @property
    def window_known(self) -> bool:
        return self.window_tokens is not None

    @property
    def effective_window(self) -> int | None:
        return int(self.window_tokens * self.effective_window_fraction) if self.window_tokens else None

    @property
    def compaction_trigger(self) -> int:
        if self.window_tokens is None:
            return self.compaction_threshold_tokens
        return min(self.compaction_threshold_tokens, int(self.window_tokens * self.max_threshold_fraction))

    @property
    def warn_at(self) -> int:
        return int(self.compaction_trigger * self.warn_fraction)

    def classify(self, tokens: int, *, max_tokens: int = 0) -> BudgetLevel:
        eff = self.effective_window
        if eff is not None and tokens + max_tokens > eff:
            return BudgetLevel.OVERFLOW
        if tokens >= self.compaction_trigger:
            return BudgetLevel.COMPACT
        if tokens >= self.warn_at:
            return BudgetLevel.WARN
        return BudgetLevel.OK

    def note(self, tokens: int) -> str:
        """Turn-scoped budget note for the dynamic channel."""
        parts = [f"Context budget: ~{tokens:,} tokens in use"]
        if self.window_tokens:
            parts.append(f"of a {self.window_tokens:,}-token window (~{max(0, self.window_tokens - tokens):,} remaining)")
        parts.append(f"next compaction at ~{self.compaction_trigger:,}")
        return "; ".join(parts) + "."


PRE_COMPACTION_WARNING = (
    "Context is approaching the compaction threshold. Before older tool results are cleared and history is "
    "summarized, save important findings, decisions and progress to memory or the progress log, and finish or "
    "checkpoint the current sub-task. Anything not recorded may be lost."
)


class UsageCalibration:
    """Scale heuristic estimates by the observed ``actual / estimated`` ratio of the last response.

    ``actual`` is the provider's total input (``input + cache_read + cache_creation``). The ratio
    is clamped to ``[0.5, 3.0]`` so one odd response cannot swing decisions wildly.
    """

    def __init__(self, ratio: float = 1.0) -> None:
        self._lock = threading.Lock()
        self.ratio = ratio
        self.last_actual: int | None = None

    def observe(self, estimated: int, usage: Usage) -> float:
        actual = usage.total_input
        with self._lock:
            if estimated > 0 and actual > 0:
                self.ratio = min(3.0, max(0.5, actual / estimated))
                self.last_actual = actual
            return self.ratio

    def apply(self, estimate: int) -> int:
        with self._lock:
            return int(round(estimate * self.ratio))


__all__ = [
    "BudgetLevel",
    "BudgetPolicy",
    "DEFAULT_COMPACTION_THRESHOLD",
    "EFFECTIVE_WINDOW_FRACTION",
    "MAX_THRESHOLD_FRACTION",
    "MIN_SERVER_COMPACTION_TRIGGER",
    "PRE_COMPACTION_WARNING",
    "UsageCalibration",
]
