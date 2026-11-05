"""Denial escalation counters and approval-fatigue monitoring.

* **Denials** (research §7.3 technique 9, checklist §7.7 step 8): every denial - rule, hook,
  classifier, ``dont-ask`` conversion or sandbox violation - counts into the *same* counters to
  prevent "grinding". At **3 consecutive or 20 total** denials auto mode escalates to the human;
  headless runs terminate. An allowed action resets the consecutive counter; a human decision
  resets it too.
* **Approval fatigue** (§7.1, §7.8): users "accept 93% of them anyway", so a high approval rate is
  a *warning sign* that prompts are noise. :class:`FatigueMonitor` tracks the ask rate over a
  sliding window of decisions, the human approval rate and approval latency, and reports
  fatigue plus concrete narrow-rule suggestions for patterns the human approved repeatedly.
  Thresholds are our choice (the research names the 93% observation, not a threshold).
"""

from __future__ import annotations

import statistics
import threading
from collections import Counter, deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DenialState:
    consecutive: int
    total: int
    limit_reached: bool
    reason: str = ""


class DenialTracker:
    """Thread-safe per-session denial counters."""

    def __init__(self, escalate_consecutive: int = 3, escalate_total: int = 20) -> None:
        if escalate_consecutive < 1 or escalate_total < 1:
            raise ValueError("escalation thresholds must be >= 1")
        self.escalate_consecutive = escalate_consecutive
        self.escalate_total = escalate_total
        self.consecutive = 0
        self.total = 0
        self.by_source: Counter[str] = Counter()
        self._lock = threading.Lock()

    def _state(self) -> DenialState:
        why = ""
        if self.consecutive >= self.escalate_consecutive:
            why = f"{self.consecutive} consecutive denials (limit {self.escalate_consecutive})"
        elif self.total >= self.escalate_total:
            why = f"{self.total} total denials (limit {self.escalate_total})"
        return DenialState(self.consecutive, self.total, bool(why), why)

    def record_denial(self, source: str = "policy") -> DenialState:
        with self._lock:
            self.consecutive += 1
            self.total += 1
            self.by_source[source] += 1
            return self._state()

    def record_allow(self) -> None:
        with self._lock:
            self.consecutive = 0

    def record_human_resolution(self) -> None:
        """A human looked at the situation: the consecutive streak is broken (total stays)."""
        with self._lock:
            self.consecutive = 0

    @property
    def state(self) -> DenialState:
        with self._lock:
            return self._state()


@dataclass(frozen=True)
class FatigueReport:
    decisions: int
    asks: int
    ask_rate: float
    human_decisions: int
    approval_rate: float | None
    median_latency_s: float | None
    fatigued: bool
    reasons: tuple[str, ...] = ()
    suggested_rules: tuple[str, ...] = ()


@dataclass
class FatigueMonitor:
    window: int = 50
    max_ask_rate: float = 0.25
    rubber_stamp_rate: float = 0.93
    min_human_decisions: int = 20
    fast_approval_s: float = 2.0
    suggest_after: int = 3
    _checks: deque = field(default_factory=deque, repr=False)
    _human: deque = field(default_factory=deque, repr=False)
    _approved_patterns: Counter = field(default_factory=Counter, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_check(self, behavior: str) -> None:
        with self._lock:
            self._checks.append(behavior)
            while len(self._checks) > self.window:
                self._checks.popleft()

    def record_human(self, approved: bool, latency_s: float, suggestion: str | None = None) -> None:
        with self._lock:
            self._human.append((approved, max(0.0, latency_s)))
            while len(self._human) > max(self.window, self.min_human_decisions):
                self._human.popleft()
            if approved and suggestion:
                self._approved_patterns[suggestion] += 1

    def report(self) -> FatigueReport:
        with self._lock:
            checks = list(self._checks)
            human = list(self._human)
            patterns = dict(self._approved_patterns)
        asks = sum(1 for b in checks if b == "ask")
        ask_rate = asks / len(checks) if checks else 0.0
        approvals = [a for a, _ in human]
        approval_rate = (sum(approvals) / len(approvals)) if approvals else None
        median = statistics.median(lat for _, lat in human) if human else None
        reasons: list[str] = []
        if len(checks) >= min(self.window, 10) and ask_rate > self.max_ask_rate:
            reasons.append(f"ask rate {ask_rate:.0%} over the last {len(checks)} decisions exceeds {self.max_ask_rate:.0%}")
        if approval_rate is not None and len(human) >= self.min_human_decisions and approval_rate >= self.rubber_stamp_rate:
            reasons.append(f"human approved {approval_rate:.0%} of {len(human)} prompts (rubber-stamping threshold {self.rubber_stamp_rate:.0%})")
        if median is not None and len(human) >= self.min_human_decisions and median < self.fast_approval_s:
            reasons.append(f"median approval latency {median:.1f}s is below {self.fast_approval_s:.1f}s")
        suggestions = tuple(sorted(p for p, n in patterns.items() if n >= self.suggest_after))
        return FatigueReport(
            decisions=len(checks), asks=asks, ask_rate=ask_rate, human_decisions=len(human), approval_rate=approval_rate,
            median_latency_s=median, fatigued=bool(reasons), reasons=tuple(reasons), suggested_rules=suggestions,
        )
