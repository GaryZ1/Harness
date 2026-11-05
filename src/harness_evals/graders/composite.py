"""Combine grader results into a task score: binary, weighted or hybrid; strict always reported.

Anthropic defines three scoring modes (reference §9.4): **binary** requires every grader to pass
(the release gate), **weighted** compares a weighted score with a threshold, **hybrid** mixes
them (here: ``required: true`` graders gate, the rest are weighted). Partial credit helps
hill-climbing but must never be the headline alone - OSWorld 2.0 is 81.8% with partial credit and
48.7% strict - so :class:`TaskScore` carries both ``strict_pass`` and ``partial``.

Unknown verdicts count as *not passed* for strict and binary scoring and are excluded from the
weighted mean (they are reported, never silently converted to a pass). Grader errors make the
task ``grader_error`` so grader QA audits it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..errors import GraderConfigError
from ..task import ScoringMode
from .base import GradeContext, Grader, GraderDeps, GradeResult, build_grader, common_kwargs, register_grader


@dataclass(frozen=True)
class TaskScore:
    passed: bool
    strict_pass: bool
    partial: float
    mode: str
    n_unknown: int = 0
    grader_error: bool = False
    failing: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "strict_pass": self.strict_pass, "partial": self.partial, "mode": self.mode,
                "n_unknown": self.n_unknown, "grader_error": self.grader_error, "failing": list(self.failing)}


def _weighted(results: Sequence[GradeResult]) -> float:
    scored = [(r.score, r.weight) for r in results if r.error is None and r.score is not None]
    total_w = sum(w for _, w in scored)
    if not scored or total_w <= 0:
        return 0.0
    return sum(float(s) * w for s, w in scored) / total_w


def score_task(results: Sequence[GradeResult], mode: ScoringMode | str = ScoringMode.BINARY, threshold: float = 1.0) -> TaskScore:
    mode = ScoringMode(mode)
    if not results:
        raise GraderConfigError("cannot score a task with no grader results")
    errors = any(r.error is not None for r in results)
    unknown = sum(1 for r in results if r.error is None and r.passed is None)
    failing = tuple(r.grader for r in results if r.passed is not True)
    strict = not failing
    partial = _weighted(results)
    if mode is ScoringMode.BINARY:
        passed = strict
    elif mode is ScoringMode.WEIGHTED:
        passed = (not errors) and partial >= threshold - 1e-12
    else:
        gates = [r for r in results if r.required]
        rest = [r for r in results if not r.required] or list(results)
        passed = (not errors) and all(r.passed is True for r in gates) and _weighted(rest) >= threshold - 1e-12
    return TaskScore(bool(passed) and not errors, strict and not errors, round(partial, 6), mode.value, unknown, errors, failing)


@register_grader("composite")
class CompositeGrader(Grader):
    """Nested group of graders scored with its own mode, e.g. ``any`` style checks via weighted + threshold."""

    kind = "composite"

    def __init__(self, graders: Sequence[Grader], *, mode: ScoringMode | str = ScoringMode.BINARY, threshold: float = 1.0, **kw: Any) -> None:
        super().__init__(**kw)
        if not graders:
            raise GraderConfigError("composite grader needs children")
        self.children, self.mode, self.threshold = list(graders), ScoringMode(mode), float(threshold)
        self.deterministic = all(getattr(g, "deterministic", True) for g in self.children)

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any], deps: GraderDeps) -> "CompositeGrader":
        children = [build_grader(s, deps) for s in spec.get("graders") or []]
        return cls(children, mode=spec.get("mode", "binary"), threshold=float(spec.get("threshold", 1.0)), **common_kwargs(spec))

    def grade(self, ctx: GradeContext) -> GradeResult:
        results = [safe_grade(g, ctx) for g in self.children]
        ts = score_task(results, self.mode, self.threshold)
        err = "child grader error" if ts.grader_error else None
        return self.result(ts.passed if not err else None, ts.partial, details={"children": [r.to_dict() for r in results], "score": ts.to_dict()}, error=err)


def safe_grade(grader: Grader, ctx: GradeContext) -> GradeResult:
    """Run a grader; any exception becomes an error result (never an agent failure, never a crash)."""
    try:
        return grader.grade(ctx)
    except Exception as exc:  # noqa: BLE001 - graders are untrusted plugins
        return GradeResult(grader.name, grader.kind, None, None, grader.weight, grader.required, error=f"{type(exc).__name__}: {exc}", version=grader.version)


__all__ = ["CompositeGrader", "TaskScore", "safe_grade", "score_task"]
