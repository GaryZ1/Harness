"""Grade one finished trial: the single code path shared by the eval runner and RL environments.

Order of operations (reference §9.7 step 5): snapshot the agent's end state, build a *separate*
grading copy with hidden tests overlaid, run every grader there (exceptions become grader errors),
compute tamper signals, then score. With ``fail_on_tamper`` (default) a trial that modified,
deleted or planted protected files cannot pass regardless of what the graders said - the
hackable-grader pitfall is closed in code, not left to each task author.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from .agent import Trajectory
from .graders.base import GradeContext, Grader, GradeResult
from .graders.composite import TaskScore, safe_grade, score_task
from .graders.integrity import HackSignal, detect_hack_signals, hack_pattern_counts, has_tamper
from .isolation import TrialSandbox, file_manifest


@dataclass
class GradingOutcome:
    grades: list[GradeResult]
    score: TaskScore
    hack_signals: list[HackSignal]
    manifest_after: dict[str, str]

    @property
    def grader_error(self) -> bool:
        return self.score.grader_error


def grade_trial(
    sandbox: TrialSandbox,
    trajectory: Trajectory,
    graders: Sequence[Grader],
    *,
    patterns_before: dict[str, dict[str, int]] | None = None,
    fail_on_tamper: bool = True,
    timeout_s: float = 600.0,
) -> GradingOutcome:
    task = sandbox.task
    manifest_after = file_manifest(sandbox.workspace)
    patterns_after = hack_pattern_counts(sandbox.workspace)
    grade_dir = sandbox.prepare_grading()
    ctx = GradeContext(
        task=task, trajectory=trajectory, workspace=grade_dir, agent_workspace=sandbox.workspace,
        manifest_before=sandbox.manifest_before, manifest_after=manifest_after, patterns_before=patterns_before or {},
        env=sandbox.env(), timeout_s=timeout_s,
    )
    grades = [safe_grade(g, ctx) for g in graders]
    signals = detect_hack_signals(
        task, sandbox.manifest_before, manifest_after, patterns_before=patterns_before or {}, patterns_after=patterns_after, trajectory=trajectory,
    )
    score = score_task(grades, task.scoring, task.threshold)
    if fail_on_tamper and has_tamper(signals):
        score = replace(score, passed=False, strict_pass=False, failing=tuple(score.failing) + ("<tamper>",))
    return GradingOutcome(grades, score, signals, manifest_after)


__all__ = ["GradingOutcome", "grade_trial"]
