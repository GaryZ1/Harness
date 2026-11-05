"""Prompt-cache hit tracking with expected rebuilds separated from unexpected misses.

Research basis (reference §10.3 "Cache-hit rate as a first-class KPI", checklist item 4):

* Manus calls KV-cache hit rate "the single most important metric for a production-stage AI
  agent"; hit rate = cache_read / total input, total input = read + write + uncached.
* Log cache-write tokens and **expected rebuilds** (compaction, tool-result clearing, model or
  effort switch, TTL expiry) separately from **unexpected misses**, and alert on the latter as
  regressions (a timestamp in the system prompt, non-deterministic tool serialisation...).
* Prompts below the model's minimum cacheable length (512-4,096 tokens on Anthropic, 1,024 on
  OpenAI) silently never cache; they are classified ``below_min_cacheable``, not as misses.

One :class:`CacheTracker` per session (thread-safe); :class:`CacheMonitor` keeps them by session id.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from harness_core.types import Clock, Usage, utcnow


class CacheObservation(str, Enum):
    HIT = "hit"
    COLD_START = "cold_start"
    EXPECTED_REBUILD = "expected_rebuild"
    UNEXPECTED_MISS = "unexpected_miss"
    BELOW_MIN_CACHEABLE = "below_min_cacheable"
    NO_INPUT = "no_input"


#: Reasons the harness can declare (``expect_rebuild``) or that the tracker infers itself.
EXPECTED_REASONS = frozenset(
    {"compaction", "context_edit", "tool_result_clearing", "model_switch", "effort_switch", "ttl_expired", "fallback", "rewind", "tools_changed"}
)


@dataclass(frozen=True)
class CacheVerdict:
    observation: CacheObservation
    hit_ratio: float | None
    reason: str = ""

    @property
    def is_regression(self) -> bool:
        return self.observation is CacheObservation.UNEXPECTED_MISS


def hit_ratio(usage: Usage) -> float | None:
    total = usage.total_input
    return usage.cache_read_input_tokens / total if total else None


class CacheTracker:
    """Classifies each model call of one session.

    Defaults (our choices, tunable): ``ttl_seconds=300`` (Anthropic default 5-minute TTL),
    ``min_cacheable_tokens=1024`` (set it per model from your config), ``miss_threshold=0.5``
    (a call whose read share is below half of its input is a miss).
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = 300.0,
        min_cacheable_tokens: int = 1024,
        miss_threshold: float = 0.5,
        clock: Clock = utcnow,
    ) -> None:
        if not 0.0 < miss_threshold <= 1.0:
            raise ValueError("miss_threshold must be within (0, 1]")
        self.ttl_seconds = ttl_seconds
        self.min_cacheable_tokens = min_cacheable_tokens
        self.miss_threshold = miss_threshold
        self.clock = clock
        self._lock = threading.Lock()
        self._last_at: datetime | None = None
        self._last_model: str | None = None
        self._last_effort: str | None = None
        self._pending: str | None = None
        self.counts: dict[CacheObservation, int] = {o: 0 for o in CacheObservation}

    def expect_rebuild(self, reason: str) -> None:
        """Declare that the *next* call will rebuild the cache (compaction, context edit, rewind...)."""
        with self._lock:
            self._pending = reason

    def observe(self, usage: Usage, *, model: str | None = None, effort: str | None = None, at: datetime | None = None) -> CacheVerdict:
        now = at or self.clock()
        ratio = hit_ratio(usage)
        with self._lock:
            pending, self._pending = self._pending, None
            first = self._last_at is None
            idle = None if first else (now - self._last_at).total_seconds()  # type: ignore[operator]
            switched_model = not first and model is not None and self._last_model is not None and model != self._last_model
            switched_effort = not first and effort is not None and self._last_effort is not None and effort != self._last_effort
            self._last_at = now
            self._last_model = model if model is not None else self._last_model
            self._last_effort = effort if effort is not None else self._last_effort

            if ratio is None:
                verdict = CacheVerdict(CacheObservation.NO_INPUT, None)
            elif usage.total_input < self.min_cacheable_tokens:
                verdict = CacheVerdict(CacheObservation.BELOW_MIN_CACHEABLE, ratio)
            elif ratio >= self.miss_threshold:
                verdict = CacheVerdict(CacheObservation.HIT, ratio)
            elif first:
                verdict = CacheVerdict(CacheObservation.COLD_START, ratio, "first_call")
            elif pending:
                verdict = CacheVerdict(CacheObservation.EXPECTED_REBUILD, ratio, pending)
            elif switched_model:
                verdict = CacheVerdict(CacheObservation.EXPECTED_REBUILD, ratio, "model_switch")
            elif switched_effort:
                verdict = CacheVerdict(CacheObservation.EXPECTED_REBUILD, ratio, "effort_switch")
            elif idle is not None and idle > self.ttl_seconds:
                verdict = CacheVerdict(CacheObservation.EXPECTED_REBUILD, ratio, "ttl_expired")
            else:
                verdict = CacheVerdict(CacheObservation.UNEXPECTED_MISS, ratio)
            self.counts[verdict.observation] += 1
        return verdict


class CacheMonitor:
    """Session id -> :class:`CacheTracker` (created on first use with shared settings)."""

    def __init__(self, **tracker_kwargs: object) -> None:
        self._kwargs = tracker_kwargs
        self._trackers: dict[str, CacheTracker] = {}
        self._lock = threading.Lock()

    def tracker(self, session_id: str) -> CacheTracker:
        with self._lock:
            t = self._trackers.get(session_id)
            if t is None:
                t = self._trackers[session_id] = CacheTracker(**self._kwargs)  # type: ignore[arg-type]
            return t

    def forget(self, session_id: str) -> None:
        with self._lock:
            self._trackers.pop(session_id, None)
