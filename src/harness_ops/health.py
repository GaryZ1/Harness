"""Health checks, SLOs, error budgets and multi-window burn rates.

Research basis: the reference (§10.7) asks for dashboards and alerts on cost, cache, tool errors
and latency but names no SLO method; we use the Google SRE Workbook's error-budget /
multi-window, multi-burn-rate method (our choice), which turns "is the agent healthy?" into a
small number of page/ticket conditions:

=========  ==========  ============  ===========================================
Severity   Long window Short window  Burn rate (budget consumed)
=========  ==========  ============  ===========================================
page       1 h         5 min         14.4 (2% of a 30-day budget in 1 h)
page       6 h         30 min        6 (5% in 6 h)
ticket     3 d         6 h           1 (10% in 3 d)
=========  ==========  ============  ===========================================

Burn rate = observed error rate / (1 - objective). A short window must also be burning so an
alert resets quickly after the problem is fixed.

SLIs are fed from session events (``export_event``) with a classifier; defaults cover task success
(``SESSION_END.outcome == "success"``) and tool success (``TOOL_RESULT.is_error``).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

from harness_core.types import Clock, Event, EventType, utcnow


@dataclass(frozen=True)
class SLO:
    name: str
    objective: float  # e.g. 0.99 = 99% of events good
    window_seconds: float = 30 * 86_400
    description: str = ""

    def __post_init__(self) -> None:
        if not 0.0 < self.objective < 1.0:
            raise ValueError("objective must be within (0, 1)")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")

    @property
    def error_budget(self) -> float:
        return 1.0 - self.objective


@dataclass(frozen=True)
class BurnRatePolicy:
    severity: str
    long_window_seconds: float
    short_window_seconds: float
    burn_rate: float


DEFAULT_BURN_RATE_POLICIES: tuple[BurnRatePolicy, ...] = (
    BurnRatePolicy("page", 3600, 300, 14.4),
    BurnRatePolicy("page", 6 * 3600, 1800, 6.0),
    BurnRatePolicy("ticket", 3 * 86_400, 6 * 3600, 1.0),
)


@dataclass(frozen=True)
class BurnAlert:
    slo: str
    severity: str
    long_burn: float
    short_burn: float
    policy: BurnRatePolicy


def task_success_sli(event: Event) -> bool | None:
    if event.type is EventType.SESSION_END:
        return str((event.data or {}).get("outcome") or (event.data or {}).get("termination") or "") == "success"
    return None


def tool_success_sli(event: Event) -> bool | None:
    if event.type is EventType.TOOL_RESULT:
        return not bool((event.data or {}).get("is_error"))
    return None


class SLOTracker:
    """Good/total counts in fixed time buckets (default 60 s) over the SLO window. Thread-safe."""

    def __init__(
        self,
        slo: SLO,
        *,
        clock: Clock = utcnow,
        bucket_seconds: int = 60,
        classifier: Callable[[Event], bool | None] = task_success_sli,
    ) -> None:
        if bucket_seconds <= 0:
            raise ValueError("bucket_seconds must be > 0")
        self.slo = slo
        self.clock = clock
        self.bucket_seconds = bucket_seconds
        self.classifier = classifier
        self._buckets: dict[int, list[int]] = {}  # bucket -> [good, total]
        self._lock = threading.Lock()

    def _bucket(self, at: datetime) -> int:
        return int(at.timestamp()) // self.bucket_seconds

    def record(self, good: bool, *, at: datetime | None = None, count: int = 1) -> None:
        b = self._bucket(at or self.clock())
        with self._lock:
            slot = self._buckets.setdefault(b, [0, 0])
            slot[0] += count if good else 0
            slot[1] += count
            self._prune(b)

    def _prune(self, newest: int) -> None:
        oldest = newest - int(self.slo.window_seconds // self.bucket_seconds) - 1
        if len(self._buckets) > 4 and min(self._buckets) < oldest:
            for k in [k for k in self._buckets if k < oldest]:
                del self._buckets[k]

    def export_event(self, event: Event) -> None:
        verdict = self.classifier(event)
        if verdict is not None:
            self.record(verdict)

    def counts(self, window_seconds: float | None = None) -> tuple[int, int]:
        window = self.slo.window_seconds if window_seconds is None else window_seconds
        now_b = self._bucket(self.clock())
        first = now_b - int(window // self.bucket_seconds) + 1
        with self._lock:
            good = total = 0
            for b, (g, t) in self._buckets.items():
                if first <= b <= now_b:
                    good += g
                    total += t
        return good, total

    def error_rate(self, window_seconds: float | None = None) -> float | None:
        good, total = self.counts(window_seconds)
        return (total - good) / total if total else None

    def burn_rate(self, window_seconds: float | None = None) -> float | None:
        rate = self.error_rate(window_seconds)
        return None if rate is None else rate / self.slo.error_budget

    def budget_remaining(self) -> float | None:
        """Fraction of the window's error budget left (1.0 = untouched, <0 = SLO violated)."""
        rate = self.error_rate()
        return None if rate is None else 1.0 - rate / self.slo.error_budget

    def evaluate(self, policies: Sequence[BurnRatePolicy] = DEFAULT_BURN_RATE_POLICIES) -> list[BurnAlert]:
        out = []
        for p in policies:
            long_b = self.burn_rate(p.long_window_seconds)
            short_b = self.burn_rate(p.short_window_seconds)
            if long_b is not None and short_b is not None and long_b >= p.burn_rate and short_b >= p.burn_rate:
                out.append(BurnAlert(self.slo.name, p.severity, long_b, short_b, p))
        return out

    def signals(self) -> dict[str, float | None]:
        """Values for :class:`~harness_ops.alerts.AlertEvaluator` snapshots (``slo.<name>.*``)."""
        n = self.slo.name
        return {
            f"slo.{n}.burn_rate_1h": self.burn_rate(3600),
            f"slo.{n}.burn_rate_5m": self.burn_rate(300),
            f"slo.{n}.burn_rate_6h": self.burn_rate(6 * 3600),
            f"slo.{n}.budget_remaining": self.budget_remaining(),
        }


class HealthStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    FAILING = "failing"


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    critical: bool
    detail: str = ""


@dataclass(frozen=True)
class HealthReport:
    status: HealthStatus
    checks: tuple[CheckResult, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "checks": [{"name": c.name, "ok": c.ok, "critical": c.critical, "detail": c.detail} for c in self.checks]}


class HealthRegistry:
    """Named liveness/readiness checks. A check returns ``bool`` or ``(bool, detail)``; raising counts as failing."""

    def __init__(self) -> None:
        self._checks: dict[str, tuple[Callable[[], Any], bool]] = {}
        self._lock = threading.Lock()

    def add(self, name: str, check: Callable[[], Any], *, critical: bool = True) -> None:
        with self._lock:
            self._checks[name] = (check, critical)

    def report(self) -> HealthReport:
        with self._lock:
            checks = list(self._checks.items())
        results = []
        for name, (fn, critical) in checks:
            try:
                out = fn()
                ok, detail = (bool(out[0]), str(out[1])) if isinstance(out, tuple) else (bool(out), "")
            except Exception as exc:  # noqa: BLE001 - a crashing check is a failing check
                ok, detail = False, f"check raised {type(exc).__name__}"
            results.append(CheckResult(name, ok, critical, detail))
        if any(not r.ok and r.critical for r in results):
            status = HealthStatus.FAILING
        elif any(not r.ok for r in results):
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.OK
        return HealthReport(status, tuple(results))


def flatten_signals(snapshot: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten ``HarnessMetrics.dashboard()``-style nested dicts into ``a.b`` -> number (``None`` dropped)."""
    out: dict[str, float] = {}
    for k, v in snapshot.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(flatten_signals(v, f"{key}."))
        elif isinstance(v, bool):
            out[key] = float(v)
        elif isinstance(v, (int, float)):
            out[key] = float(v)
    return out
