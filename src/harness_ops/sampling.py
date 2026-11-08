"""Transcript/trace sampling for human review and online trace grading.

Research basis (reference §10.3 "Online trace grading and failure clustering", checklist item 20;
build plan "read transcripts"):

* "Trace grading is the fastest way to identify workflow-level issues" (OpenAI). Sample production
  traces into graders and human review **weekly**, cluster the failures, and feed them into the
  capability suite.
* Failures are worth far more review time than successes, so the sampler is stratified: every
  failed or flagged session (error subtype, negative feedback, cost outlier, unexpected cache
  misses, permission denials) is queued with a priority; successes are sampled at a small base
  rate. The random choice is a deterministic hash of the session id, so every process agrees.
* The queue stores **references** (session id, trace id, reasons, numbers) - never transcript
  content - so sampling never widens who can read user data; reviewers open the session log
  through its own access controls.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from harness_core.types import Clock, Event, EventType, utcnow
from harness_memory.locking import file_lock

from ._util import stable_bucket, to_iso


@dataclass(frozen=True)
class SamplingPolicy:
    base_rate: float = 0.05  # our choice: 5% of successful sessions
    sample_failures: bool = True
    cost_outlier_usd: float | None = None  # sample any session that cost at least this much
    max_random_per_day: int = 50  # our choice: bounds weekly reading load (~350 transcripts)
    max_flagged_per_day: int = 500
    salt: str = "harness-review"

    def __post_init__(self) -> None:
        if not 0.0 <= self.base_rate <= 1.0:
            raise ValueError("base_rate must be within [0, 1]")


@dataclass(frozen=True)
class ReviewItem:
    session_id: str
    reasons: tuple[str, ...]
    priority: int
    outcome: str
    sampled_at: str
    trace_id: str | None = None
    total_cost_usd: float | None = None
    num_turns: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        d["reasons"] = list(self.reasons)
        return json.dumps(d, sort_keys=True, ensure_ascii=False, default=str)

    @classmethod
    def from_json(cls, line: str) -> "ReviewItem":
        d = json.loads(line)
        d["reasons"] = tuple(d.get("reasons", ()))
        return cls(**d)


@dataclass(frozen=True)
class ReviewDecision:
    sampled: bool
    reasons: tuple[str, ...] = ()
    priority: int = 0


class ReviewQueue:
    """Append-only JSONL queue of :class:`ReviewItem` (cross-process safe)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.path.with_name(f".{self.path.name}.lock")

    def put(self, item: ReviewItem) -> None:
        data = (item.to_json() + "\n").encode("utf-8")
        with file_lock(self._lock_path):
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view):]
            finally:
                os.close(fd)

    def items(self) -> list[ReviewItem]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(ReviewItem.from_json(line))
            except (ValueError, TypeError):
                continue
        return sorted(out, key=lambda i: (-i.priority, i.sampled_at))


class ReviewSampler:
    """Decides which finished sessions go to review. ``export_event`` consumes ``SESSION_END`` events.

    ``SESSION_END`` data keys used: ``outcome`` (or ``termination``), ``total_cost_usd``,
    ``num_turns``, ``trace_id``, ``user_feedback`` (``"negative"`` flags), ``flags`` (list of
    strings, e.g. ``["unexpected_cache_miss", "permission_denied"]``).
    """

    def __init__(self, policy: SamplingPolicy | None = None, queue: ReviewQueue | None = None, *, clock: Clock = utcnow) -> None:
        self.policy = policy or SamplingPolicy()
        self.queue = queue
        self.clock = clock
        self._daily: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()
        self.items: list[ReviewItem] = []

    def decide(self, session_id: str, *, outcome: str, cost_usd: float | None = None, flags: Iterable[str] = (), user_feedback: str | None = None) -> ReviewDecision:
        p = self.policy
        reasons: list[str] = []
        priority = 0
        if p.sample_failures and outcome != "success":
            reasons.append(f"outcome:{outcome}")
            priority = max(priority, 3)
        if user_feedback == "negative":
            reasons.append("negative_feedback")
            priority = max(priority, 3)
        if p.cost_outlier_usd is not None and cost_usd is not None and cost_usd >= p.cost_outlier_usd:
            reasons.append("cost_outlier")
            priority = max(priority, 2)
        for f in flags:
            reasons.append(f"flag:{f}")
            priority = max(priority, 2)
        day = self.clock().date().isoformat()
        with self._lock:
            if reasons:
                key = (day, "flagged")
                if self._daily.get(key, 0) >= p.max_flagged_per_day:
                    return ReviewDecision(False, tuple(reasons), priority)
                self._daily[key] = self._daily.get(key, 0) + 1
                return ReviewDecision(True, tuple(reasons), priority)
            if stable_bucket(session_id, p.salt, 1_000_000) < p.base_rate * 1_000_000:
                key = (day, "random")
                if self._daily.get(key, 0) >= p.max_random_per_day:
                    return ReviewDecision(False, ("random",), 1)
                self._daily[key] = self._daily.get(key, 0) + 1
                return ReviewDecision(True, ("random",), 1)
        return ReviewDecision(False)

    def consider(self, session_id: str, *, outcome: str, trace_id: str | None = None, cost_usd: float | None = None, num_turns: int | None = None, flags: Iterable[str] = (), user_feedback: str | None = None) -> ReviewItem | None:
        flags = tuple(flags)
        d = self.decide(session_id, outcome=outcome, cost_usd=cost_usd, flags=flags, user_feedback=user_feedback)
        if not d.sampled:
            return None
        item = ReviewItem(session_id, d.reasons, d.priority, outcome, to_iso(self.clock()), trace_id, cost_usd, num_turns)
        with self._lock:
            self.items.append(item)
        if self.queue is not None:
            self.queue.put(item)
        return item

    def export_event(self, event: Event) -> None:
        if event.type is not EventType.SESSION_END:
            return
        d = event.data or {}
        flags = d.get("flags") or ()
        self.consider(
            event.session_id,
            outcome=str(d.get("outcome") or d.get("termination") or "unknown"),
            trace_id=d.get("trace_id") if isinstance(d.get("trace_id"), str) else None,
            cost_usd=d.get("total_cost_usd"),
            num_turns=d.get("num_turns"),
            flags=[str(f) for f in flags] if isinstance(flags, (list, tuple)) else (),
            user_feedback=d.get("user_feedback") if isinstance(d.get("user_feedback"), str) else None,
        )
