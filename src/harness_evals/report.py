"""Run summaries, markdown/JSON reports and harness A/B comparisons.

Reporting rules from reference §9.7 steps 7-9, 11, 17 and §9.8:

* Report **mean pass@1 ± SE, stating the unit** (here always *across tasks*: per-task means, then
  mean and SE over tasks) plus a task-cluster bootstrap CI.
* Report **pass^k** (k = 3-8) next to pass@1; report pass@k only labelled as "any-of-k
  (verifier/human selects)", never next to someone else's pass@1.
* Report **strict and partial** together; never headline partial alone.
* **Never merge tiers**: capability, regression, etc. are summarised separately.
* Pin conditions in the header (harness, model, effort, thinking, suite hash, runs, error policy,
  judge model + prompt version, user simulator, safeguards).
* Efficiency is part of the result: cost per success, p50/p95 wall clock, tokens, cache-read share.
* A/B comparisons refuse confounded inputs (different model, effort, thinking, suite, judge or
  simulator) unless explicitly overridden, and use paired statistics over the same tasks.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .errors import ConfoundedComparisonError
from .guards import SaturationReport, noise_warnings, saturation_report
from .metrics import (
    DEFAULT_RESAMPLES,
    NOISE_BAND,
    EfficiencySummary,
    PairedComparison,
    cluster_bootstrap_ci,
    mean_se,
    paired_compare,
    pass_at_k,
    pass_hat_k,
)

ERROR_STATUSES = ("infra_error", "grader_error")
#: Conditions that must be identical for a harness A/B (the variables under test may differ).
HELD_FIXED = ("model", "effort", "thinking", "judge_model", "user_simulator")


def included(records: Sequence[Mapping[str, Any]], error_policy: str) -> list[Mapping[str, Any]]:
    if error_policy == "exclude":
        return [r for r in records if r["status"] not in ERROR_STATUSES]
    return list(records)


def per_task_counts(records: Sequence[Mapping[str, Any]], *, key: str = "passed") -> dict[str, tuple[int, int]]:
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in records:
        out[r["task_id"]][0] += 1
        out[r["task_id"]][1] += 1 if r.get(key) else 0
    return {t: (n, c) for t, (n, c) in sorted(out.items())}


@dataclass
class TierSummary:
    tier: str
    n_tasks: int
    n_trials: int
    trials_min: int
    trials_max: int
    pass1: float
    pass1_se: float
    pass1_ci: tuple[float, float]
    strict: float
    partial: float
    pass_hat_k: dict[int, float]
    pass_at_k: dict[int, float]
    per_task: dict[str, dict[str, Any]]
    saturation: SaturationReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier, "n_tasks": self.n_tasks, "n_trials": self.n_trials, "trials_per_task": [self.trials_min, self.trials_max],
            "pass@1": self.pass1, "pass@1_se_across_tasks": self.pass1_se, "pass@1_ci": list(self.pass1_ci), "strict": self.strict,
            "partial": self.partial, "pass^k": {str(k): v for k, v in self.pass_hat_k.items()},
            "pass@k_any_of_k": {str(k): v for k, v in self.pass_at_k.items()}, "per_task": self.per_task, "saturation": self.saturation.to_dict(),
        }


@dataclass
class RunSummary:
    run_id: str
    header: dict[str, Any]
    error_policy: str
    tiers: dict[str, TierSummary]
    efficiency: EfficiencySummary
    status_counts: dict[str, int]
    excluded: int
    infra_retries: int
    hack_signal_trials: int
    isolation_violation_trials: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "suite": self.header.get("suite"), "conditions": self.header.get("conditions"), "judge": self.header.get("judge"),
            "error_policy": self.error_policy, "se_unit": "across tasks", "tiers": {k: v.to_dict() for k, v in self.tiers.items()},
            "efficiency": self.efficiency.to_dict(), "status_counts": self.status_counts, "excluded_trials": self.excluded,
            "infra_retries": self.infra_retries, "hack_signal_trials": self.hack_signal_trials,
            "isolation_violation_trials": self.isolation_violation_trials, "warnings": self.warnings,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)

    def to_markdown(self) -> str:
        c = self.header.get("conditions") or {}
        s = self.header.get("suite") or {}
        j = self.header.get("judge") or {}
        e = self.efficiency
        lines = [f"# Eval report `{self.run_id}`", "", "```yaml",
                 f"suite: {s.get('name')}@{s.get('version')} (hash {s.get('hash')}, {s.get('n_tasks')} tasks)",
                 f"harness: {c.get('harness_version')} (prompts sha {c.get('prompts_sha') or '-'}, tools sha {c.get('tools_sha') or '-'}, profile {c.get('profile') or '-'})",
                 f"model: {c.get('model')}   effort: {c.get('effort') or 'unpinned'}   thinking: {c.get('thinking') or 'unpinned'}",
                 f"sandbox_image: {c.get('sandbox_image') or '-'}   user_simulator: {c.get('user_simulator') or '-'}",
                 f"judge: {(j.get('model') or '-')}@{j.get('prompt_version') or '-'}" if j else "judge: none",
                 f"safeguards: {json.dumps(c.get('safeguards') or {}, sort_keys=True)}",
                 f"errors_counted_as: {'failure' if self.error_policy == 'failure' else 'excluded'}   excluded_trials: {self.excluded}   infra_retries: {self.infra_retries}",
                 "se_unit: across tasks", "```", ""]
        lines += ["| tier | tasks | trials/task | pass@1 ± SE | 95% CI | strict | partial | pass^k | pass@k (any-of-k) |", "|---|---|---|---|---|---|---|---|---|"]
        for t in self.tiers.values():
            phk = ", ".join(f"^{k}={v:.3f}" for k, v in t.pass_hat_k.items()) or "-"
            pak = ", ".join(f"@{k}={v:.3f}" for k, v in t.pass_at_k.items()) or "-"
            tpt = str(t.trials_min) if t.trials_min == t.trials_max else f"{t.trials_min}-{t.trials_max}"
            lines.append(f"| {t.tier} | {t.n_tasks} | {tpt} | {t.pass1:.3f} ± {t.pass1_se:.3f} | [{t.pass1_ci[0]:.3f}, {t.pass1_ci[1]:.3f}] | {t.strict:.3f} | {t.partial:.3f} | {phk} | {pak} |")
        lines += ["", "## Efficiency", "", "| metric | value |", "|---|---|",
                  f"| cost per success (USD) | {_fmt(e.cost_per_success_usd)} |", f"| total cost (USD) | {_fmt(e.total_cost_usd)} |",
                  f"| wall clock p50 / p95 (s) | {_fmt(e.wall_p50_s)} / {_fmt(e.wall_p95_s)} |", f"| mean turns / tool calls | {_fmt(e.mean_turns)} / {_fmt(e.mean_tool_calls)} |",
                  f"| tokens in / out / cache-read / cache-write | {e.input_tokens} / {e.output_tokens} / {e.cache_read_tokens} / {e.cache_write_tokens} |",
                  f"| cache-read share | {_fmt(e.cache_read_share)} |", f"| tool errors by tool | {json.dumps(dict(e.tool_errors_by_tool)) if e.tool_errors_by_tool else '-'} |",
                  f"| compaction events | {e.compaction_events} |", "",
                  "## Trial status", "", ", ".join(f"{k}: {v}" for k, v in sorted(self.status_counts.items())) or "-",
                  f"\nTrials with tamper/suspicious signals: {self.hack_signal_trials}; isolation violations: {self.isolation_violation_trials}", ""]
        if self.warnings:
            lines += ["## Warnings", ""] + [f"- {w}" for w in self.warnings] + [""]
        for t in self.tiers.values():
            lines += [f"## Tasks ({t.tier})", "", "| task | trials | passed | rate |", "|---|---|---|---|"]
            lines += [f"| {tid} | {d['n']} | {d['c']} | {d['rate']:.2f} |" for tid, d in sorted(t.per_task.items(), key=lambda kv: (kv[1]["rate"], kv[0]))]
            lines.append("")
        return "\n".join(lines)


def _fmt(v: Any) -> str:
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def summarize(
    header: Mapping[str, Any], records: Sequence[Mapping[str, Any]], *, k_values: Sequence[int] = (3, 5, 8), n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95, seed: int = 0,
) -> RunSummary:
    policy = (header.get("config") or {}).get("error_policy", "failure")
    recs = included(records, policy)
    tiers: dict[str, TierSummary] = {}
    by_tier: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in recs:
        by_tier[r.get("suite_tier", "capability")].append(r)
    warnings: list[str] = []
    for tier, rs in sorted(by_tier.items()):
        counts = per_task_counts(rs)
        strict_counts = per_task_counts(rs, key="strict_pass")
        partial_by_task: dict[str, list[float]] = defaultdict(list)
        pass_by_task: dict[str, list[float]] = defaultdict(list)
        for r in rs:
            partial_by_task[r["task_id"]].append(float(r.get("partial") or 0.0))
            pass_by_task[r["task_id"]].append(1.0 if r.get("passed") else 0.0)
        rates = [c / n for n, c in counts.values()]
        m, se = mean_se(rates)
        ci = cluster_bootstrap_ci(pass_by_task, n_resamples=n_resamples, confidence=confidence, seed=seed)
        tmin = min(n for n, _ in counts.values())
        tmax = max(n for n, _ in counts.values())
        phk = {k: sum(pass_hat_k(n, c, k) for n, c in counts.values()) / len(counts) for k in k_values if k <= tmin}
        pak = {k: sum(pass_at_k(n, c, k) for n, c in counts.values()) / len(counts) for k in k_values if k <= tmin}
        sat = saturation_report(counts)
        tiers[tier] = TierSummary(
            tier, len(counts), len(rs), tmin, tmax, m, se, ci,
            sum(c / n for n, c in strict_counts.values()) / len(strict_counts),
            sum(sum(v) / len(v) for v in partial_by_task.values()) / len(partial_by_task),
            phk, pak, {t: {"n": n, "c": c, "rate": c / n} for t, (n, c) in counts.items()}, sat,
        )
        warnings += [f"[{tier}] {w}" for w in noise_warnings(trials_per_task=tmin, n_tasks=len(counts) if tier == "capability" else None)]
        if sat.broken_suspects:
            warnings.append(f"[{tier}] 0% over >= 5 trials (audit as broken task or grader first): {', '.join(sat.broken_suspects)}")
        if tier == "capability" and sat.graduate:
            warnings.append(f"[{tier}] at ~100%, graduate to regression: {', '.join(sat.graduate)}")
        if tier == "capability" and sat.suite_saturated:
            warnings.append(f"[{tier}] suite mean {sat.suite_mean:.2f} is saturated; add harder tasks from fresh failures")
    cond = header.get("conditions") or {}
    if not cond.get("effort"):
        warnings.append("effort is not pinned; vendor defaults differ by model and change scores")
    if header.get("judge") and not (cond.get("extra") or {}).get("judge_human_agreement"):
        warnings.append("LLM judge used without a recorded judge-human agreement; calibrate on a human-labelled sample")
    status_counts = dict(Counter(r["status"] for r in records))
    return RunSummary(
        str(header.get("run_id", "")), dict(header), policy, tiers, EfficiencySummary.from_trials(recs), status_counts,
        len(records) - len(recs), sum(int(r.get("infra_retries", 0)) for r in records),
        sum(1 for r in records if r.get("hack_signals")), sum(1 for r in records if r.get("isolation_violations")), warnings,
    )


@dataclass
class ComparisonReport:
    baseline_run: str
    candidate_run: str
    differs_in: dict[str, tuple[Any, Any]]
    tiers: dict[str, PairedComparison]
    baseline_efficiency: EfficiencySummary
    candidate_efficiency: EfficiencySummary
    warnings: list[str]

    @property
    def cost_per_success_ratio(self) -> float | None:
        a, b = self.baseline_efficiency.cost_per_success_usd, self.candidate_efficiency.cost_per_success_usd
        return (b / a) if (a and b is not None) else None

    def to_dict(self) -> dict[str, Any]:
        return {"baseline": self.baseline_run, "candidate": self.candidate_run, "differs_in": {k: list(v) for k, v in self.differs_in.items()},
                "tiers": {k: v.to_dict() for k, v in self.tiers.items()}, "baseline_efficiency": self.baseline_efficiency.to_dict(),
                "candidate_efficiency": self.candidate_efficiency.to_dict(), "cost_per_success_ratio": self.cost_per_success_ratio, "warnings": self.warnings}

    def to_markdown(self) -> str:
        lines = [f"# Harness A/B: `{self.candidate_run}` vs baseline `{self.baseline_run}`", "",
                 "Variables under test: " + (", ".join(f"{k}: {a!r} -> {b!r}" for k, (a, b) in self.differs_in.items()) or "none"), "",
                 "| tier | tasks | baseline | candidate | delta | CI | wins/losses/ties | sign p | verdict |", "|---|---|---|---|---|---|---|---|---|"]
        for tier, pc in self.tiers.items():
            lines.append(f"| {tier} | {pc.n_tasks} | {pc.mean_a:.3f} | {pc.mean_b:.3f} | {pc.delta:+.3f} | [{pc.ci_low:+.3f}, {pc.ci_high:+.3f}] | "
                         f"{pc.wins}/{pc.losses}/{pc.ties} | {pc.p_sign:.3g} | {pc.verdict} |")
        a, b = self.baseline_efficiency, self.candidate_efficiency
        lines += ["", f"Cost per success: {_fmt(a.cost_per_success_usd)} -> {_fmt(b.cost_per_success_usd)}; p95 wall: {_fmt(a.wall_p95_s)} -> {_fmt(b.wall_p95_s)} s"]
        if self.warnings:
            lines += ["", "## Warnings", ""] + [f"- {w}" for w in self.warnings]
        return "\n".join(lines) + "\n"


def compare_runs(
    baseline_header: Mapping[str, Any], baseline_records: Sequence[Mapping[str, Any]], candidate_header: Mapping[str, Any],
    candidate_records: Sequence[Mapping[str, Any]], *, allow_confounded: bool = False, min_effect: float = NOISE_BAND,
    n_resamples: int = DEFAULT_RESAMPLES, confidence: float = 0.95, seed: int = 0,
) -> ComparisonReport:
    """Paired harness A/B over the shared tasks of each tier, with the model held fixed."""
    ca, cb = baseline_header.get("conditions") or {}, candidate_header.get("conditions") or {}
    confounds = {k: (ca.get(k), cb.get(k)) for k in HELD_FIXED if ca.get(k) != cb.get(k)}
    sa, sb = (baseline_header.get("suite") or {}), (candidate_header.get("suite") or {})
    if sa.get("hash") != sb.get("hash"):
        confounds["suite_hash"] = (sa.get("hash"), sb.get("hash"))
    if confounds and not allow_confounded:
        raise ConfoundedComparisonError(f"runs differ in held-fixed conditions: {confounds}")
    differs = {k: (ca.get(k), cb.get(k)) for k in sorted(set(ca) | set(cb)) if ca.get(k) != cb.get(k) and k not in HELD_FIXED}
    pa = (baseline_header.get("config") or {}).get("error_policy", "failure")
    pb = (candidate_header.get("config") or {}).get("error_policy", "failure")
    ra, rb = included(baseline_records, pa), included(candidate_records, pb)
    warnings = [f"confounded comparison: {confounds}"] if confounds else []
    tiers: dict[str, PairedComparison] = {}
    for tier in sorted({r.get("suite_tier", "capability") for r in ra} & {r.get("suite_tier", "capability") for r in rb}):
        cnt_a = per_task_counts([r for r in ra if r.get("suite_tier", "capability") == tier])
        cnt_b = per_task_counts([r for r in rb if r.get("suite_tier", "capability") == tier])
        if set(cnt_a) != set(cnt_b):
            warnings.append(f"[{tier}] task sets differ; comparing the {len(set(cnt_a) & set(cnt_b))} shared tasks")
        pc = paired_compare({t: c / n for t, (n, c) in cnt_a.items()}, {t: c / n for t, (n, c) in cnt_b.items()}, n_resamples=n_resamples,
                            confidence=confidence, seed=seed, min_effect=min_effect, allow_partial_overlap=True)
        tiers[tier] = pc
        tmin = min(min(n for n, _ in cnt_a.values()), min(n for n, _ in cnt_b.values()))
        warnings += [f"[{tier}] {w}" for w in noise_warnings(trials_per_task=tmin, delta=pc.delta)]
        if tier == "regression" and pc.delta < 0:
            warnings.append(f"[regression] candidate is lower by {-pc.delta:.3f}; a harness change must not regress the regression suite")
    return ComparisonReport(str(baseline_header.get("run_id")), str(candidate_header.get("run_id")), differs, tiers,
                            EfficiencySummary.from_trials(ra), EfficiencySummary.from_trials(rb), warnings)


__all__ = ["ComparisonReport", "HELD_FIXED", "RunSummary", "TierSummary", "compare_runs", "included", "per_task_counts", "summarize"]
