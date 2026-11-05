"""Reliability metrics and statistics (stdlib only; bootstrap implemented here).

* :func:`pass_at_k` - unbiased estimator of P(at least one of k trials succeeds) from n >= k
  trials with c successes (Chen et al. 2021): ``1 - C(n-c, k) / C(n, k)``. Use only where a
  verifier or human picks among attempts, and label it so (reference §9.7 step 8).
* :func:`pass_hat_k` - unbiased estimator of pass^k = P(all k trials succeed) (tau-bench):
  ``C(c, k) / C(n, k)``. The reliability metric for user-facing agents (k = 3-8; Sierra).
* :func:`mean_se` / :func:`bootstrap_ci` / :func:`cluster_bootstrap_ci` - mean pass@1 with SE
  *across tasks* (we always state the unit; §9.8 "Unreported ± semantics").
* :func:`paired_compare` - harness A/B on the same tasks with the model fixed: paired bootstrap
  over per-task deltas plus an exact two-sided sign test.
* :func:`percentile` and :class:`EfficiencySummary` - cost per success and p50/p95 latency are
  gates alongside pass rate (§9.7 step 9).

All resampling uses a seeded ``random.Random`` so reports are reproducible.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

#: Default bootstrap resamples. 10,000 keeps percentile-CI Monte Carlo error well below 1 point.
DEFAULT_RESAMPLES = 10_000
#: Deltas below this many points (as a fraction) are treated as noise for single-run decisions
#: (2.2-6.0 point run-to-run band; "~5 points on an ~89-task suite", reference §9.7 step 7).
NOISE_BAND = 0.05


def pass_at_k(n: int, c: int, k: int) -> float:
    if not 0 <= c <= n:
        raise ValueError("need 0 <= c <= n")
    if k < 1 or k > n:
        raise ValueError(f"pass@k needs 1 <= k <= n (n={n}, k={k})")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    if not 0 <= c <= n:
        raise ValueError("need 0 <= c <= n")
    if k < 1 or k > n:
        raise ValueError(f"pass^k needs 1 <= k <= n (n={n}, k={k})")
    return math.comb(c, k) / math.comb(n, k)


def suite_metric(per_task: Mapping[str, tuple[int, int]], k: int, fn: Callable[[int, int, int], float]) -> float:
    """Mean of a per-task estimator over tasks with at least ``k`` trials (raises if none qualify)."""
    vals = [fn(n, c, k) for n, c in per_task.values() if n >= k]
    if not vals:
        raise ValueError(f"no task has >= {k} trials")
    return sum(vals) / len(vals)


def mean_se(values: Sequence[float]) -> tuple[float, float]:
    """Mean and standard error of the mean (SE = sample stdev / sqrt(n); 0.0 when n < 2)."""
    if not values:
        raise ValueError("mean_se of empty sequence")
    m = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return m, se


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, ``q`` in [0, 100]."""
    if not values:
        raise ValueError("percentile of empty sequence")
    if not 0 <= q <= 100:
        raise ValueError("q must be within [0, 100]")
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def bootstrap_ci(
    values: Sequence[float], *, stat: Callable[[Sequence[float]], float] = statistics.fmean, n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95, seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for ``stat`` over i.i.d. ``values``."""
    if not values:
        raise ValueError("bootstrap of empty sequence")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    rng = random.Random(seed)
    n = len(values)
    vals = list(values)
    stats_ = sorted(stat([vals[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples))
    alpha = (1 - confidence) / 2
    return percentile(stats_, 100 * alpha), percentile(stats_, 100 * (1 - alpha))


def cluster_bootstrap_ci(
    per_task: Mapping[str, Sequence[float]], *, n_resamples: int = DEFAULT_RESAMPLES, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """CI for the mean of per-task means, resampling *tasks* (trials of one task are correlated)."""
    means = [statistics.fmean(v) for v in per_task.values() if v]
    return bootstrap_ci(means, n_resamples=n_resamples, confidence=confidence, seed=seed)


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign test p-value (ties excluded by the caller)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


@dataclass(frozen=True)
class PairedComparison:
    """Candidate (b) minus baseline (a) over the same tasks."""

    n_tasks: int
    mean_a: float
    mean_b: float
    delta: float
    ci_low: float
    ci_high: float
    confidence: float
    wins: int
    losses: int
    ties: int
    p_sign: float
    min_effect: float
    alpha: float

    @property
    def significant(self) -> bool:
        """CI excludes zero in the candidate's favour AND the effect clears the noise band."""
        return self.ci_low > 0 and self.delta >= self.min_effect

    @property
    def regressed(self) -> bool:
        return self.ci_high < 0

    @property
    def verdict(self) -> str:
        if self.significant:
            return "improved"
        if self.regressed:
            return "regressed"
        if abs(self.delta) < self.min_effect:
            return "within_noise"
        return "inconclusive"

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("n_tasks", "mean_a", "mean_b", "delta", "ci_low", "ci_high", "confidence", "wins", "losses",
                                              "ties", "p_sign", "min_effect", "alpha")} | {"significant": self.significant, "verdict": self.verdict}


def paired_compare(
    a: Mapping[str, float], b: Mapping[str, float], *, n_resamples: int = DEFAULT_RESAMPLES, confidence: float = 0.95, seed: int = 0,
    min_effect: float = NOISE_BAND, alpha: float | None = None, allow_partial_overlap: bool = False,
) -> PairedComparison:
    """Paired bootstrap of per-task score deltas (b - a) plus a sign test.

    ``a``/``b`` map task id -> per-task score (e.g. mean pass over trials). Both variants must have
    run the *same* tasks; a mismatch raises unless ``allow_partial_overlap`` (then the intersection
    is used and the count is reported).
    """
    keys = sorted(set(a) & set(b))
    if not allow_partial_overlap and set(a) != set(b):
        raise ValueError(f"paired comparison needs the same task set (only in a: {sorted(set(a) - set(b))[:5]}, only in b: {sorted(set(b) - set(a))[:5]})")
    if not keys:
        raise ValueError("no common tasks to compare")
    deltas = [b[k] - a[k] for k in keys]
    lo, hi = bootstrap_ci(deltas, n_resamples=n_resamples, confidence=confidence, seed=seed)
    wins = sum(d > 1e-12 for d in deltas)
    losses = sum(d < -1e-12 for d in deltas)
    return PairedComparison(
        len(keys), statistics.fmean(a[k] for k in keys), statistics.fmean(b[k] for k in keys), statistics.fmean(deltas), lo, hi, confidence,
        wins, losses, len(keys) - wins - losses, sign_test(wins, losses), min_effect, alpha if alpha is not None else 1 - confidence,
    )


@dataclass(frozen=True)
class EfficiencySummary:
    n_trials: int
    n_success: int
    total_cost_usd: float | None
    cost_per_success_usd: float | None
    wall_p50_s: float | None
    wall_p95_s: float | None
    mean_turns: float | None
    mean_tool_calls: float | None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_share: float | None = None
    tool_errors_by_tool: Mapping[str, int] = field(default_factory=dict)
    compaction_events: int = 0

    @classmethod
    def from_trials(cls, trials: Iterable[Mapping[str, Any]]) -> "EfficiencySummary":
        """``trials``: dicts with ``passed``, ``cost_usd``, ``wall_clock_s``, ``turns``, ``tool_calls``, ``usage``..."""
        ts = list(trials)
        succ = sum(1 for t in ts if t.get("passed"))
        costs = [float(t["cost_usd"]) for t in ts if t.get("cost_usd") is not None]
        total_cost = sum(costs) if costs else None
        walls = [float(t["wall_clock_s"]) for t in ts if t.get("wall_clock_s") is not None]
        turns = [t["turns"] for t in ts if t.get("turns") is not None]
        calls = [t["tool_calls"] for t in ts if t.get("tool_calls") is not None]
        u = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
        errs: dict[str, int] = {}
        comp = 0
        for t in ts:
            for k in u:
                u[k] += int((t.get("usage") or {}).get(k, 0) or 0)
            for tool, n in (t.get("tool_errors_by_tool") or {}).items():
                errs[tool] = errs.get(tool, 0) + int(n)
            comp += int(t.get("compaction_events", 0) or 0)
        total_in = u["input_tokens"] + u["cache_read_input_tokens"] + u["cache_creation_input_tokens"]
        return cls(
            len(ts), succ, total_cost,
            (total_cost / succ) if (total_cost is not None and succ) else None,
            percentile(walls, 50) if walls else None, percentile(walls, 95) if walls else None,
            statistics.fmean(turns) if turns else None, statistics.fmean(calls) if calls else None,
            u["input_tokens"], u["output_tokens"], u["cache_read_input_tokens"], u["cache_creation_input_tokens"],
            (u["cache_read_input_tokens"] / total_in) if total_in else None, dict(sorted(errs.items())), comp,
        )

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__} | {"tool_errors_by_tool": dict(self.tool_errors_by_tool)}


__all__ = [
    "DEFAULT_RESAMPLES",
    "EfficiencySummary",
    "NOISE_BAND",
    "PairedComparison",
    "bootstrap_ci",
    "cluster_bootstrap_ci",
    "mean_se",
    "paired_compare",
    "pass_at_k",
    "pass_hat_k",
    "percentile",
    "sign_test",
    "suite_metric",
]
