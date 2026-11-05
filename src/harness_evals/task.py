"""Task and suite format (YAML / JSON / JSONL, or a Harbor-style task directory).

One artifact serves CI evals, public-benchmark comparisons and later RL (reference §9.7 step 2,
§11.7 step 5). A task directory follows the Harbor shape (exact Harbor file names were not
captured in the research, so this layout is our convention)::

    tasks/fix-timezone-parse-001/
      task.yaml          # metadata + graders
      instruction.md     # what the agent sees (used when task.yaml has no `instruction`)
      tests/             # HIDDEN from the agent; overlaid only into the grading copy
      solution/          # reference solution (proves solvability: "two experts would agree")
      files/             # optional seed files copied into the workspace

Invariants enforced at load time: ids are unique and filename-safe; every path is relative and
traversal-free (:class:`harness_memory.PathSandbox` rules); every task has at least one grader;
hidden test paths are automatically *protected* (tamper detection) and never part of
:meth:`Task.agent_view`.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from harness_memory import PathSandbox, PathSecurityError

from .agent import AgentTask, Limits
from .errors import TaskFormatError

TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class SuiteTier(str, Enum):
    """Suite tiers from the evals-notes stack (§9.7 step 10). Never merge tiers into one pass rate."""

    TOOL = "tool"              # tier 0: per-tool unit evals
    CAPABILITY = "capability"  # tier 1: hill to climb (starts low)
    REGRESSION = "regression"  # tier 2: graduated tasks (~100%)
    PUBLIC = "public"          # tier 3: external calibration
    ONLINE = "online"          # tier 4: sampled production traces


class Polarity(str, Enum):
    POSITIVE = "positive"   # the behaviour should happen
    NEGATIVE = "negative"   # the behaviour should NOT happen (balance both, Anthropic 2026)


class ScoringMode(str, Enum):
    BINARY = "binary"       # every grader must pass (release gate)
    WEIGHTED = "weighted"   # weighted score >= threshold
    HYBRID = "hybrid"       # required graders must pass AND weighted score of the rest >= threshold


#: Default trials per task when neither the task nor the run config says (§9.7 step 7).
DEFAULT_TRIALS: dict[SuiteTier, int] = {
    SuiteTier.TOOL: 3,
    SuiteTier.CAPABILITY: 5,
    SuiteTier.REGRESSION: 3,
    SuiteTier.PUBLIC: 5,
    SuiteTier.ONLINE: 1,
}

_SANDBOX = PathSandbox("/")  # only used for its pure normalise() rules


def safe_relpath(rel: str, *, what: str) -> str:
    """Validate a task-relative path (no traversal, not absolute, no NUL)."""
    try:
        clean = _SANDBOX.normalise(str(rel))
    except PathSecurityError as exc:
        raise TaskFormatError(f"{what}: {exc}") from exc
    if not clean:
        raise TaskFormatError(f"{what}: empty path")
    return clean


@dataclass(frozen=True)
class Setup:
    """How to materialise the starting workspace. Commands run with the trial's isolated env."""

    files: Mapping[str, str] = field(default_factory=dict)
    commands: tuple[str, ...] = ()
    copy_dirs: tuple[str, ...] = ()       # directories (relative to the task dir) copied into the workspace
    git_init: bool = False                # fresh repo per trial: never shared history

    def to_dict(self) -> dict[str, Any]:
        return {"files": dict(self.files), "commands": list(self.commands), "copy_dirs": list(self.copy_dirs), "git_init": self.git_init}


@dataclass(frozen=True)
class Task:
    id: str
    instruction: str
    graders: tuple[Mapping[str, Any], ...]
    suite: SuiteTier = SuiteTier.CAPABILITY
    polarity: Polarity = Polarity.POSITIVE
    origin: str = ""
    tags: tuple[str, ...] = ()
    setup: Setup = field(default_factory=Setup)
    hidden_files: Mapping[str, str] = field(default_factory=dict)   # overlaid at grade time only
    hidden_dir: Path | None = None                                  # e.g. <task>/tests, overlaid at grade time
    protected_paths: tuple[str, ...] = ()                           # globs the agent must not modify/delete
    scoring: ScoringMode = ScoringMode.BINARY
    threshold: float = 1.0
    trials: int | None = None
    limits: Limits = field(default_factory=Limits)
    reference_solution: Setup | None = None
    solution_dir: Path | None = None
    canary: str | None = None
    version: str = "1"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_dir: Path | None = None

    # ------------------------------------------------------------------ views
    def agent_view(self) -> AgentTask:
        """The agent-visible projection: no graders, hidden tests, solution or canary."""
        return AgentTask(self.id, self.instruction, self.limits, self.tags, dict(self.metadata.get("agent", {}) or {}))

    def hidden_paths(self) -> list[str]:
        paths = sorted(self.hidden_files)
        if self.hidden_dir is not None and self.hidden_dir.is_dir():
            base = self.hidden_dir
            prefix = base.name
            for p in sorted(base.rglob("*")):
                if p.is_file():
                    paths.append(f"{prefix}/{p.relative_to(base).as_posix()}")
        return sorted(set(paths))

    def all_protected(self) -> tuple[str, ...]:
        """Explicit protected globs plus every hidden path (agents must never ship their own copies)."""
        return tuple(sorted(set(self.protected_paths) | set(self.hidden_paths())))

    def trials_for(self, override: int | None = None) -> int:
        return override or self.trials or DEFAULT_TRIALS[self.suite]

    def content_hash(self) -> str:
        """Stable hash of everything that defines the task (inputs, graders, hidden files)."""
        h = hashlib.sha256()
        payload = {
            "id": self.id, "instruction": self.instruction, "graders": [dict(g) for g in self.graders],
            "suite": self.suite.value, "polarity": self.polarity.value, "setup": self.setup.to_dict(),
            "hidden_files": dict(self.hidden_files), "protected": list(self.protected_paths),
            "scoring": self.scoring.value, "threshold": self.threshold, "limits": self.limits.to_dict(),
            "version": self.version,
        }
        h.update(json.dumps(payload, sort_keys=True, default=str).encode())
        for root in (self.hidden_dir, self.source_dir and self.source_dir / "files"):
            if root is not None and Path(root).is_dir():
                for p in sorted(Path(root).rglob("*")):
                    if p.is_file():
                        h.update(p.relative_to(root).as_posix().encode())
                        h.update(hashlib.sha256(p.read_bytes()).digest())
        return h.hexdigest()[:16]


@dataclass(frozen=True)
class Suite:
    name: str
    version: str
    tasks: tuple[Task, ...]
    description: str = ""

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for t in self.tasks:
            if t.id in seen:
                raise TaskFormatError(f"duplicate task id in suite {self.name!r}: {t.id}")
            seen.add(t.id)

    @property
    def hash(self) -> str:
        """Task-set hash, used as the suite identity in reports (``capability-v14`` style)."""
        h = hashlib.sha256(f"{self.name}\x00{self.version}".encode())
        for t in sorted(self.tasks, key=lambda t: t.id):
            h.update(f"{t.id}:{t.content_hash()}".encode())
        return h.hexdigest()[:16]

    def by_id(self, task_id: str) -> Task:
        for t in self.tasks:
            if t.id == task_id:
                return t
        raise KeyError(task_id)

    def select(self, *, ids: Iterable[str] | None = None, tags: Iterable[str] | None = None, tier: SuiteTier | None = None) -> "Suite":
        idset = set(ids) if ids is not None else None
        tagset = set(tags or ())
        kept = tuple(
            t for t in self.tasks
            if (idset is None or t.id in idset) and (not tagset or tagset & set(t.tags)) and (tier is None or t.suite is tier)
        )
        return replace(self, tasks=kept)


# ----------------------------------------------------------------------------- parsing


def _enum(cls, value: Any, default, what: str):
    if value is None:
        return default
    try:
        return cls(str(value))
    except ValueError as exc:
        raise TaskFormatError(f"{what}: {value!r} is not one of {[e.value for e in cls]}") from exc


def _setup(d: Any, what: str) -> Setup:
    if d is None:
        return Setup()
    if not isinstance(d, Mapping):
        raise TaskFormatError(f"{what} must be a mapping")
    files = {safe_relpath(k, what=f"{what}.files"): str(v) for k, v in (d.get("files") or {}).items()}
    cmds = d.get("commands") or []
    if isinstance(cmds, str):
        cmds = [cmds]
    copy_dirs = tuple(safe_relpath(p, what=f"{what}.copy_dirs") for p in d.get("copy_dirs") or [])
    return Setup(files, tuple(str(c) for c in cmds), copy_dirs, bool(d.get("git_init", False)))


def parse_task(d: Mapping[str, Any], *, base_dir: Path | None = None) -> Task:
    """Build and validate a :class:`Task` from a mapping (one YAML document or JSONL line)."""
    if not isinstance(d, Mapping):
        raise TaskFormatError("task must be a mapping")
    tid = str(d.get("id") or "")
    if not TASK_ID.match(tid):
        raise TaskFormatError(f"invalid task id {tid!r} (letters, digits, . _ -; max 128 chars)")
    what = f"task {tid}"
    instruction = d.get("instruction", d.get("prompt"))
    if instruction is None and base_dir is not None and (base_dir / "instruction.md").is_file():
        instruction = (base_dir / "instruction.md").read_text(encoding="utf-8")
    if not instruction or not str(instruction).strip():
        raise TaskFormatError(f"{what}: missing instruction/prompt")
    graders = d.get("graders") or []
    if not isinstance(graders, list) or not graders:
        raise TaskFormatError(f"{what}: at least one grader is required")
    for g in graders:
        if not isinstance(g, Mapping) or not g.get("kind"):
            raise TaskFormatError(f"{what}: every grader needs a 'kind'")
    limits_d = d.get("limits") or {}
    timeout = d.get("timeout_s", limits_d.get("timeout_s", Limits.timeout_s))
    limits = Limits(float(timeout), limits_d.get("max_turns"), limits_d.get("max_tool_calls"))
    if limits.timeout_s <= 0:
        raise TaskFormatError(f"{what}: timeout_s must be positive")
    threshold = float(d.get("threshold", 1.0))
    if not 0.0 <= threshold <= 1.0:
        raise TaskFormatError(f"{what}: threshold must be within [0, 1]")
    trials = d.get("trials")
    if trials is not None and int(trials) < 1:
        raise TaskFormatError(f"{what}: trials must be >= 1")
    hidden = {safe_relpath(k, what=f"{what}.hidden_files"): str(v) for k, v in (d.get("hidden_files") or {}).items()}
    hidden_dir = None
    solution_dir = None
    if base_dir is not None:
        if (base_dir / "tests").is_dir():
            hidden_dir = base_dir / "tests"
        if (base_dir / "solution").is_dir():
            solution_dir = base_dir / "solution"
    setup = _setup(d.get("setup"), f"{what}.setup")
    if base_dir is not None and (base_dir / "files").is_dir() and "files" not in setup.copy_dirs:
        setup = replace(setup, copy_dirs=setup.copy_dirs + ("files",))
    for cd in setup.copy_dirs:
        if base_dir is None or not (base_dir / cd).is_dir():
            raise TaskFormatError(f"{what}: copy_dirs entry {cd!r} not found next to the task file")
    isolation = d.get("isolation") or {}
    protected = tuple(str(p) for p in (d.get("protected_paths") or []))
    if isolation.get("tests_writable_by_agent") is True:
        raise TaskFormatError(f"{what}: tests_writable_by_agent=true makes the grader hackable")
    return Task(
        id=tid,
        instruction=str(instruction),
        graders=tuple(dict(g) for g in graders),
        suite=_enum(SuiteTier, d.get("suite"), SuiteTier.CAPABILITY, f"{what}.suite"),
        polarity=_enum(Polarity, d.get("polarity"), Polarity.POSITIVE, f"{what}.polarity"),
        origin=str(d.get("origin") or ""),
        tags=tuple(str(t) for t in d.get("tags") or ()),
        setup=setup,
        hidden_files=hidden,
        hidden_dir=hidden_dir,
        protected_paths=protected,
        scoring=_enum(ScoringMode, d.get("scoring"), ScoringMode.BINARY, f"{what}.scoring"),
        threshold=threshold,
        trials=int(trials) if trials is not None else None,
        limits=limits,
        reference_solution=_setup(d["reference_solution"], f"{what}.reference_solution") if d.get("reference_solution") else None,
        solution_dir=solution_dir,
        canary=str(d["canary"]) if d.get("canary") else None,
        version=str(d.get("version", "1")),
        metadata=dict(d.get("metadata") or {}),
        source_dir=base_dir,
    )


def _read_mapping(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise TaskFormatError(f"{path}: {exc}") from exc


def load_task(path: str | Path) -> Task:
    """Load one task from ``task.yaml``/``.json`` or from a task directory containing ``task.yaml``."""
    p = Path(path)
    if p.is_dir():
        for name in ("task.yaml", "task.yml", "task.json"):
            if (p / name).is_file():
                return parse_task(_read_mapping(p / name), base_dir=p)
        raise TaskFormatError(f"{p}: no task.yaml in task directory")
    if not p.is_file():
        raise TaskFormatError(f"{p}: no such task file")
    return parse_task(_read_mapping(p), base_dir=p.parent)


def load_suite(path: str | Path, *, name: str | None = None, version: str | None = None) -> Suite:
    """Load a suite from a ``suite.yaml`` (``{name, version, tasks: [inline | path]}``), a JSONL file
    (one task per line) or a directory of task directories (optionally with ``suite.yaml``)."""
    p = Path(path)
    if p.is_dir():
        meta: dict[str, Any] = {}
        if (p / "suite.yaml").is_file():
            return load_suite(p / "suite.yaml", name=name, version=version)
        tasks = tuple(load_task(d) for d in sorted(p.iterdir()) if d.is_dir() and any((d / n).is_file() for n in ("task.yaml", "task.yml", "task.json")))
        return Suite(name or meta.get("name", p.name), version or "1", tasks)
    if p.suffix == ".jsonl":
        tasks_l = []
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                tasks_l.append(parse_task(json.loads(line), base_dir=p.parent))
            except json.JSONDecodeError as exc:
                raise TaskFormatError(f"{p}:{i}: {exc}") from exc
        return Suite(name or p.stem, version or "1", tuple(tasks_l))
    doc = _read_mapping(p)
    if not isinstance(doc, Mapping) or "tasks" not in doc:
        raise TaskFormatError(f"{p}: a suite file needs a 'tasks' list")
    tasks_y = []
    for entry in doc["tasks"] or []:
        if isinstance(entry, str):
            tasks_y.append(load_task(p.parent / safe_relpath(entry, what=f"{p} tasks")))
        else:
            tasks_y.append(parse_task(entry, base_dir=p.parent))
    return Suite(name or str(doc.get("name") or p.stem), version or str(doc.get("version", "1")), tuple(tasks_y), str(doc.get("description") or ""))


__all__ = [
    "DEFAULT_TRIALS",
    "Polarity",
    "ScoringMode",
    "Setup",
    "Suite",
    "SuiteTier",
    "Task",
    "load_suite",
    "load_task",
    "parse_task",
    "safe_relpath",
]
