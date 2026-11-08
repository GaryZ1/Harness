"""Grader protocol, results and the kind registry.

A grader turns an end state (and, for transcript graders, the trajectory) into a
:class:`GradeResult`. Three verdicts exist, not two: ``passed=None`` means *unknown* (Anthropic
2026: give judges an "Unknown" option) and ``error`` means the grader itself failed, which grader
QA must audit rather than count against the agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Mapping

from harness_core.types import ModelClient

from ..agent import Trajectory
from ..errors import GraderConfigError
from ..isolation import ManifestDiff, diff_manifests
from ..task import Task


@dataclass
class GradeContext:
    task: Task
    trajectory: Trajectory
    workspace: Path                    # grading copy (hidden tests overlaid) - graders may run code here
    agent_workspace: Path              # the agent's end state, read-only evidence
    manifest_before: Mapping[str, str] = field(default_factory=dict)
    manifest_after: Mapping[str, str] = field(default_factory=dict)
    patterns_before: Mapping[str, Mapping[str, int]] = field(default_factory=dict)   # hack-pattern counts pre-agent
    env: Mapping[str, str] | None = None
    timeout_s: float = 600.0

    @property
    def diff(self) -> ManifestDiff:
        return diff_manifests(self.manifest_before, self.manifest_after)


@dataclass
class GradeResult:
    grader: str
    kind: str
    passed: bool | None                # None = unknown
    score: float | None = None         # 0..1; None = not scoreable
    weight: float = 1.0
    required: bool = False             # hybrid scoring gate
    details: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    error: str | None = None
    version: str = "1"

    def __post_init__(self) -> None:
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError(f"grader {self.grader}: score {self.score} outside [0, 1]")
        if self.score is None and self.passed is not None and self.error is None:
            self.score = 1.0 if self.passed else 0.0

    @property
    def status(self) -> str:
        if self.error is not None:
            return "error"
        return "unknown" if self.passed is None else ("pass" if self.passed else "fail")

    def to_dict(self) -> dict[str, Any]:
        return {
            "grader": self.grader, "kind": self.kind, "passed": self.passed, "score": self.score, "weight": self.weight,
            "required": self.required, "details": self.details, "evidence": self.evidence, "error": self.error,
            "version": self.version, "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GradeResult":
        return cls(
            str(d["grader"]), str(d["kind"]), d.get("passed"), d.get("score"), float(d.get("weight", 1.0)),
            bool(d.get("required", False)), dict(d.get("details") or {}), list(d.get("evidence") or []), d.get("error"),
            str(d.get("version", "1")),
        )


@dataclass(frozen=True)
class GraderDeps:
    """Dependencies some graders need. Judge model ids are always passed in, never defaulted."""

    judge_client: ModelClient | None = None
    judge_model: str | None = None


class Grader:
    """Base class. Subclasses set ``kind``/``version`` and implement :meth:`grade`."""

    kind: ClassVar[str] = ""
    version: ClassVar[str] = "1"
    deterministic: ClassVar[bool] = True     # False for model-based graders (RL readiness, reward shares)

    def __init__(self, name: str | None = None, *, weight: float = 1.0, required: bool = False) -> None:
        if weight < 0:
            raise GraderConfigError("grader weight must be >= 0")
        self.name = name or self.kind
        self.weight = float(weight)
        self.required = bool(required)

    def grade(self, ctx: GradeContext) -> GradeResult:  # pragma: no cover - interface
        raise NotImplementedError

    def result(self, passed: bool | None, score: float | None = None, **kw: Any) -> GradeResult:
        return GradeResult(self.name, self.kind, passed, score, self.weight, self.required, version=self.version, **kw)

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any], deps: GraderDeps) -> "Grader":  # pragma: no cover - interface
        raise NotImplementedError


_REGISTRY: dict[str, type[Grader]] = {}


def register_grader(*kinds: str) -> Callable[[type[Grader]], type[Grader]]:
    def deco(cls: type[Grader]) -> type[Grader]:
        for k in kinds:
            if k in _REGISTRY and _REGISTRY[k] is not cls:
                raise GraderConfigError(f"grader kind {k!r} already registered")
            _REGISTRY[k] = cls
        return cls

    return deco


def grader_kinds() -> dict[str, type[Grader]]:
    return dict(_REGISTRY)


def common_kwargs(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {"name": spec.get("name") or spec.get("dimension"), "weight": float(spec.get("weight", 1.0)), "required": bool(spec.get("required", False))}


def build_grader(spec: Mapping[str, Any], deps: GraderDeps | None = None) -> Grader:
    kind = str(spec.get("kind") or "")
    cls = _REGISTRY.get(kind)
    if cls is None:
        raise GraderConfigError(f"unknown grader kind {kind!r}; known: {sorted(_REGISTRY)}")
    try:
        return cls.from_spec(spec, deps or GraderDeps())
    except GraderConfigError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise GraderConfigError(f"invalid {kind} grader spec: {exc}") from exc


def build_graders(task: Task, deps: GraderDeps | None = None) -> list[Grader]:
    """Build every grader of ``task``; unnamed duplicates get ``#2``, ``#3``... so results stay keyed."""
    graders = [build_grader(s, deps) for s in task.graders]
    seen: dict[str, int] = {}
    for g in graders:
        seen[g.name] = seen.get(g.name, 0) + 1
        if seen[g.name] > 1:
            g.name = f"{g.name}#{seen[g.name]}"
    return graders


__all__ = [
    "GradeContext",
    "GradeResult",
    "Grader",
    "GraderDeps",
    "build_grader",
    "build_graders",
    "common_kwargs",
    "grader_kinds",
    "register_grader",
]
