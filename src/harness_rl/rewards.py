"""Reward functions built on :mod:`harness_evals` graders, with reward-hacking penalties.

Ranking of mitigations from Anthropic + Redwood's production-RL study (reference §11.4, §11.8):
*preventing* hacks is best. So rewards here are computed from the same hardened grading path as
evals (hidden, read-only tests in a separate grading copy; end state, not claims), and on top:

* :class:`HackPenalty` **overrides** the reward (it is not additive) when a tamper signal fired:
  deleting or editing tests, editing grader files, planting files at protected paths. A trajectory
  that hacked can never score above ``penalty`` no matter what else it did. Suspicious signals
  (new ``sys.exit``, skip markers, new ``conftest.py``) carry a configurable smaller penalty.
* :func:`rl_readiness` refuses tasks that have no deterministic end-state grader, or whose tests
  are not hidden/protected - "move to RL only when a stable, high-volume task has an automatic,
  hack-resistant grader" (§11.7 step 12).
* LLM-judge components are capped (``max_judge_share``, default 0.3): judges agree with experts
  66% vs 71% human-human on GDPval, so they never dominate a reward.
* :func:`group_advantages` gives GRPO-style group-relative advantages and flags zero-variance
  groups (no learning signal).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from harness_evals import GradeResult, HackSignal, Task, TaskScore, Trajectory, grader_kinds

from .errors import RewardConfigError

#: Grader kinds that verify end state deterministically (verifiable rewards).
VERIFIABLE_KINDS = frozenset({"unit_tests", "command", "state_check", "file_exists", "file_regex", "file_equals", "json_assert", "answer"})
JUDGE_KINDS = frozenset({"llm_rubric", "llm_judge"})


@dataclass
class EpisodeOutcome:
    """Everything a reward function may look at for one finished rollout."""

    task: Task
    trajectory: Trajectory
    grades: Sequence[GradeResult]
    score: TaskScore | None
    hack_signals: Sequence[HackSignal] = ()
    status: str = "completed"


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    components: Mapping[str, float] = field(default_factory=dict)
    hack_detected: bool = False
    flags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"total": self.total, "components": dict(self.components), "hack_detected": self.hack_detected, "flags": list(self.flags)}


class RewardFn(Protocol):
    def __call__(self, ep: EpisodeOutcome) -> float: ...


def outcome_reward(ep: EpisodeOutcome, *, strict: bool = True) -> float:
    """1.0 when the task passed (strict: every grader passed), else 0.0. Errors and timeouts score 0."""
    if ep.score is None or ep.status != "completed":
        return 0.0
    return 1.0 if (ep.score.strict_pass if strict else ep.score.passed) else 0.0


def verifiable_reward(ep: EpisodeOutcome) -> float:
    """Weighted mean score over deterministic end-state graders only (partial credit for shaping)."""
    rs = [g for g in ep.grades if g.kind in VERIFIABLE_KINDS and g.error is None and g.score is not None]
    if not rs or ep.status != "completed":
        return 0.0
    w = sum(g.weight for g in rs)
    return sum(float(g.score) * g.weight for g in rs) / w if w > 0 else 0.0


def rubric_reward(ep: EpisodeOutcome) -> float:
    """Mean LLM-judge score; unknown verdicts contribute 0 (no reward for unverifiable work)."""
    rs = [g for g in ep.grades if g.kind in JUDGE_KINDS and g.error is None]
    if not rs:
        return 0.0
    return sum(float(g.score) if (g.score is not None and g.passed is not None) else 0.0 for g in rs) / len(rs)


@dataclass(frozen=True)
class HackPenalty:
    tamper_penalty: float = -1.0
    suspicious_penalty: float = 0.0

    def apply(self, total: float, signals: Sequence[HackSignal]) -> tuple[float, bool, tuple[str, ...]]:
        tamper = [s for s in signals if s.severity == "tamper"]
        suspicious = [s for s in signals if s.severity != "tamper"]
        flags = tuple(f"{s.severity}:{s.kind}" for s in signals)
        if tamper:
            return min(total, self.tamper_penalty), True, flags
        if suspicious and self.suspicious_penalty:
            return total + self.suspicious_penalty, False, flags
        return total, False, flags


@dataclass(frozen=True)
class EfficiencyPenalty:
    """Small shaping term: fewer turns (agent RFT reduced turns). Off by default."""

    per_turn: float = 0.0
    per_1k_output_tokens: float = 0.0
    max_penalty: float = 0.2

    def __call__(self, ep: EpisodeOutcome) -> float:
        p = self.per_turn * (ep.trajectory.turns or 0) + self.per_1k_output_tokens * ep.trajectory.usage.output_tokens / 1000
        return -min(self.max_penalty, max(0.0, p))


class CompositeReward:
    """``total = clip(sum(weight_i * component_i)) then hack override``.

    Default: ``outcome`` (weight 1.0) only. ``max_judge_share`` caps the fraction of positive
    weight that LLM-judge components may hold.
    """

    def __init__(
        self,
        components: Sequence[tuple[str, Callable[[EpisodeOutcome], float], float]] | None = None,
        *,
        hack: HackPenalty | None = None,
        clip: tuple[float, float] = (-1.0, 1.0),
        judge_components: Sequence[str] = ("rubric",),
        max_judge_share: float = 0.3,
    ) -> None:
        self.components = list(components) if components is not None else [("outcome", outcome_reward, 1.0)]
        if not self.components:
            raise RewardConfigError("at least one reward component is required")
        names = [n for n, _, _ in self.components]
        if len(set(names)) != len(names):
            raise RewardConfigError("reward component names must be unique")
        pos = sum(w for _, _, w in self.components if w > 0)
        judge = sum(w for n, _, w in self.components if n in judge_components and w > 0)
        if pos > 0 and judge / pos > max_judge_share + 1e-12:
            raise RewardConfigError(f"LLM-judge share {judge / pos:.2f} exceeds max_judge_share {max_judge_share}")
        self.hack = hack or HackPenalty()
        self.clip = clip

    def __call__(self, ep: EpisodeOutcome) -> RewardBreakdown:
        comps = {name: float(fn(ep)) for name, fn, _ in self.components}
        raw = sum(comps[name] * w for name, _, w in self.components)
        lo, hi = self.clip
        total = max(lo, min(hi, raw))
        total, hacked, flags = self.hack.apply(total, list(ep.hack_signals))
        if math.isnan(total):
            raise RewardConfigError("reward is NaN")
        return RewardBreakdown(round(total, 6), comps, hacked, flags)


def default_reward() -> CompositeReward:
    return CompositeReward()


def rl_readiness(task: Task) -> list[str]:
    """Problems that make ``task`` unsafe to optimise against (empty list = ready)."""
    kinds = [str(g.get("kind")) for g in task.graders]
    known = grader_kinds()
    problems = [f"unknown grader kind {k}" for k in kinds if k not in known]
    if not any(k in VERIFIABLE_KINDS for k in kinds):
        problems.append("no deterministic end-state grader (judge-only or transcript-only rewards are hackable)")
    if any(k in ("unit_tests",) for k in kinds) and not task.hidden_paths():
        problems.append("unit_tests grader but no hidden tests: tests would be visible/writable to the policy")
    if any(k == "command" for k in kinds) and not task.all_protected():
        problems.append("command grader without protected or hidden paths: the policy can edit what the command checks")
    return problems


def group_advantages(rewards: Sequence[float], *, eps: float = 1e-6) -> tuple[list[float], bool]:
    """GRPO group-relative advantages ``(r - mean) / (std + eps)``; returns (advantages, has_signal)."""
    if not rewards:
        return [], False
    m = statistics.fmean(rewards)
    sd = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    if sd < eps:
        return [0.0] * len(rewards), False
    return [(r - m) / (sd + eps) for r in rewards], True


__all__ = [
    "CompositeReward",
    "EfficiencyPenalty",
    "EpisodeOutcome",
    "HackPenalty",
    "JUDGE_KINDS",
    "RewardBreakdown",
    "VERIFIABLE_KINDS",
    "default_reward",
    "group_advantages",
    "outcome_reward",
    "rl_readiness",
    "rubric_reward",
    "verifiable_reward",
]
