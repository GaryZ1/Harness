"""Long-running task state: the per-task persistence layer (kept separate from learned memory).

Implements Anthropic's "effective harnesses for long-running agents" pattern:

* an **initializer** writes ``feature_list.json`` (every entry ``passes: false``), ``progress.md``,
  ``init.sh`` and an initial git commit;
* every later session runs the **orientation routine** (pwd -> git log + progress -> pick the
  highest-priority failing feature -> init.sh -> smoke test), works on **one feature**, verifies it
  end-to-end, commits and appends to the progress log;
* invariants are **enforced in code**: the feature list is append- and flip-only
  (:func:`validate_feature_list_change`), and :func:`guard_tool_call` is a ready-made PreToolUse
  hook that rejects other edits and test deletions.

Files live with the work, under git: ``<repo>/.harness/task/``.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ._time import Clock, to_iso, utcnow
from .errors import InvariantViolation, NotFoundError, ValidationError
from .locking import atomic_write_text, file_lock

TASK_DIR = ".harness/task"
FEATURE_FILE = "feature_list.json"
PROGRESS_FILE = "progress.md"
INIT_SCRIPT = "init.sh"
NOTES_FILE = "NOTES.md"

_FEATURE_KEYS = ("id", "category", "description", "steps", "passes")
_TEST_PATH = re.compile(r"(^|/)(tests?|__tests__|spec)/|(^|/)test_[^/]+\.py$|_test\.(py|go)$|\.(spec|test)\.[jt]sx?$")


@dataclass
class Feature:
    id: str
    category: str
    description: str
    steps: list[str] = field(default_factory=list)
    passes: bool = False
    priority: int = 100
    evidence: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Feature":
        missing = [k for k in ("id", "category", "description") if k not in d]
        if missing:
            raise ValidationError(f"feature missing keys: {missing}")
        return cls(
            id=str(d["id"]),
            category=str(d["category"]),
            description=str(d["description"]),
            steps=[str(s) for s in d.get("steps", [])],
            passes=bool(d.get("passes", False)),
            priority=int(d.get("priority", 100)),
            evidence=[str(e) for e in d.get("evidence", [])],
        )


def _load_features(text: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvariantViolation(f"feature_list.json is not valid JSON: {exc}") from exc
    if not isinstance(data, list) or not all(isinstance(x, dict) for x in data):
        raise InvariantViolation("feature_list.json must be a JSON array of objects")
    return data


def validate_feature_list_change(old_text: str, new_text: str) -> None:
    """Raise :class:`InvariantViolation` unless ``new`` only flips ``passes``/adds evidence or appends features.

    Rules: no feature may be removed or reordered; ``category``, ``description``, ``steps``,
    ``priority`` and ``id`` are immutable; ``passes`` may flip; ``evidence`` may only grow;
    appended features must start with ``passes: false``.
    """
    old, new = _load_features(old_text), _load_features(new_text)
    if len(new) < len(old):
        raise InvariantViolation("removing features from feature_list.json is not allowed")
    for i, (o, n) in enumerate(zip(old, new, strict=False)):
        if str(o.get("id")) != str(n.get("id")):
            raise InvariantViolation(f"feature #{i} changed id or order ({o.get('id')} -> {n.get('id')})")
        for key in set(o) | set(n):
            if key in ("passes", "evidence"):
                continue
            if o.get(key) != n.get(key):
                raise InvariantViolation(f"feature {o.get('id')}: field '{key}' is immutable; only 'passes' may change")
        old_ev, new_ev = list(o.get("evidence", [])), list(n.get("evidence", []))
        if new_ev[: len(old_ev)] != old_ev:
            raise InvariantViolation(f"feature {o.get('id')}: evidence is append-only")
        if not isinstance(n.get("passes", False), bool):
            raise InvariantViolation(f"feature {o.get('id')}: passes must be a boolean")
    ids = [str(f.get("id")) for f in new]
    if len(ids) != len(set(ids)):
        raise InvariantViolation("feature ids must be unique")
    for f in new[len(old) :]:
        Feature.from_dict(f)
        if f.get("passes"):
            raise InvariantViolation(f"new feature {f.get('id')} must start with passes=false")


def is_test_path(path: str) -> bool:
    return bool(_TEST_PATH.search(path.replace("\\", "/")))


@dataclass(frozen=True)
class HookDecision:
    allow: bool
    reason: str = ""


def guard_tool_call(tool_name: str, tool_input: dict[str, Any], *, repo_root: str | Path) -> HookDecision:
    """PreToolUse hook: block edits to the feature list other than flipping ``passes`` and block test deletion.

    Understands Write/Edit-style inputs (``file_path`` + ``content`` / ``old_string``/``new_string``)
    and Bash ``command`` strings containing ``rm``/``git rm`` of test files.
    """
    root = Path(repo_root).resolve()
    name = tool_name.lower()
    feature_path = (root / TASK_DIR / FEATURE_FILE).resolve()

    if name == "bash":
        cmd = str(tool_input.get("command", ""))
        for m in re.finditer(r"(?:^|[;&|]\s*)(?:git\s+rm|rm)\s+([^;&|]+)", cmd):
            args = [a for a in m.group(1).split() if not a.startswith("-")]
            if any(is_test_path(a) for a in args):
                return HookDecision(False, "deleting tests is not allowed; fix the code instead")
            if any(a.endswith(FEATURE_FILE) for a in args):
                return HookDecision(False, "feature_list.json may not be deleted")
        if FEATURE_FILE in cmd and re.search(r">\s*\S*" + re.escape(FEATURE_FILE), cmd):
            return HookDecision(False, "write feature_list.json through the task-state API, not shell redirection")
        return HookDecision(True)

    file_path = tool_input.get("file_path") or tool_input.get("path")
    if not file_path:
        return HookDecision(True)
    target = (root / file_path).resolve() if not os.path.isabs(file_path) else Path(file_path).resolve()

    if name in ("delete", "remove") and is_test_path(str(target)):
        return HookDecision(False, "deleting tests is not allowed")

    if target == feature_path:
        if not feature_path.exists():
            return HookDecision(True)
        old_text = feature_path.read_text(encoding="utf-8")
        if "content" in tool_input:
            new_text = str(tool_input["content"])
        elif "old_string" in tool_input:
            old_s, new_s = str(tool_input["old_string"]), str(tool_input.get("new_string", ""))
            if old_text.count(old_s) != 1:
                return HookDecision(True)  # the edit tool itself will fail
            new_text = old_text.replace(old_s, new_s, 1)
        else:
            return HookDecision(False, "unsupported edit to feature_list.json")
        try:
            validate_feature_list_change(old_text, new_text)
        except InvariantViolation as exc:
            return HookDecision(False, str(exc))
        return HookDecision(True)

    if name in ("write",) and is_test_path(str(target)) and target.exists() and not str(tool_input.get("content", "")).strip():
        return HookDecision(False, "emptying a test file is not allowed")
    return HookDecision(True)


@dataclass
class OrientationReport:
    cwd: str
    git_log: list[str]
    progress_tail: str
    next_feature: Feature | None
    passing: int
    total: int
    init_result: dict[str, Any] | None = None

    def render(self) -> str:
        lines = [
            "<task-state>",
            f"Working directory: {self.cwd}",
            f"Features passing: {self.passing}/{self.total}",
        ]
        if self.next_feature:
            f = self.next_feature
            lines.append(f"Next feature: [{f.id}] ({f.category}) {f.description}")
            lines.extend(f"  step: {s}" for s in f.steps)
        else:
            lines.append("All features pass.")
        if self.git_log:
            lines.append("Recent commits:")
            lines.extend(f"  {entry}" for entry in self.git_log)
        if self.progress_tail:
            lines.append("Progress log (tail):")
            lines.append(self.progress_tail)
        if self.init_result is not None:
            ok = "ok" if self.init_result.get("returncode") == 0 else f"FAILED (exit {self.init_result.get('returncode')})"
            lines.append(f"init.sh smoke test: {ok}")
        lines.append("Work on ONE feature, verify it end-to-end, then commit and append to the progress log.")
        lines.append("</task-state>")
        return "\n".join(lines)


class TaskState:
    """API over ``<repo>/.harness/task/``."""

    def __init__(self, repo_root: str | Path, *, clock: Clock = utcnow, git: bool = True) -> None:
        self.repo = Path(repo_root).resolve()
        self.dir = self.repo / TASK_DIR
        self.clock = clock
        self.git = git
        self._lock = self.dir / ".task.lock"

    # ------------------------------------------------------------------ paths
    @property
    def feature_path(self) -> Path:
        return self.dir / FEATURE_FILE

    @property
    def progress_path(self) -> Path:
        return self.dir / PROGRESS_FILE

    @property
    def init_path(self) -> Path:
        return self.dir / INIT_SCRIPT

    @property
    def initialized(self) -> bool:
        return self.feature_path.exists()

    # ------------------------------------------------------------------ initializer
    def initialize(self, goal: str, features: Sequence[dict[str, Any] | Feature], init_script: str, *, commit: bool = True) -> None:
        if self.initialized:
            raise ValidationError("task state already initialized; later sessions must not re-run the initializer")
        feats = [f if isinstance(f, Feature) else Feature.from_dict(f) for f in features]
        if not feats:
            raise ValidationError("the initializer must produce at least one feature")
        for f in feats:
            f.passes = False
        ids = [f.id for f in feats]
        if len(ids) != len(set(ids)):
            raise ValidationError("feature ids must be unique")
        self.dir.mkdir(parents=True, exist_ok=True)
        with file_lock(self._lock):
            atomic_write_text(self.feature_path, json.dumps([asdict(f) for f in feats], indent=2) + "\n")
            atomic_write_text(
                self.progress_path,
                f"# Progress log\n\n**Goal:** {goal}\n\n## {to_iso(self.clock())} initializer\n- Created {len(feats)} features (all failing), init.sh and this log.\n",
            )
            script = init_script if init_script.startswith("#!") else "#!/usr/bin/env bash\nset -euo pipefail\n" + init_script
            atomic_write_text(self.init_path, script.rstrip("\n") + "\n")
            self.init_path.chmod(self.init_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
        if commit:
            self.checkpoint("chore(task): initialize feature list, progress log and init.sh")

    # ------------------------------------------------------------------ reads
    def features(self) -> list[Feature]:
        if not self.feature_path.exists():
            raise NotFoundError("task state not initialized")
        return [Feature.from_dict(d) for d in _load_features(self.feature_path.read_text(encoding="utf-8"))]

    def next_feature(self) -> Feature | None:
        failing = [(i, f) for i, f in enumerate(self.features()) if not f.passes]
        if not failing:
            return None
        return min(failing, key=lambda t: (t[1].priority, t[0]))[1]

    def progress_tail(self, max_chars: int = 2_000) -> str:
        if not self.progress_path.exists():
            return ""
        text = self.progress_path.read_text(encoding="utf-8")
        return text[-max_chars:]

    def git_log(self, n: int = 10) -> list[str]:
        if not self.git:
            return []
        proc = subprocess.run(["git", "log", f"-n{n}", "--pretty=format:%h %s"], cwd=self.repo, capture_output=True, text=True)
        return [ln for ln in proc.stdout.splitlines() if ln] if proc.returncode == 0 else []

    def orient(self, *, run_init: bool = False, timeout: float = 300) -> OrientationReport:
        """The session-start routine. Hook this to SessionStart *and* post-compaction."""
        feats = self.features()
        report = OrientationReport(
            cwd=str(self.repo),
            git_log=self.git_log(),
            progress_tail=self.progress_tail(),
            next_feature=self.next_feature(),
            passing=sum(f.passes for f in feats),
            total=len(feats),
        )
        if run_init:
            report.init_result = self.run_init(timeout=timeout)
        return report

    def run_init(self, *, timeout: float = 300) -> dict[str, Any]:
        if not self.init_path.exists():
            raise NotFoundError("init.sh missing")
        try:
            proc = subprocess.run(["bash", str(self.init_path)], cwd=self.repo, capture_output=True, text=True, timeout=timeout)
            return {"returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}
        except subprocess.TimeoutExpired as exc:
            return {
                "returncode": -1,
                "stdout": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                "stderr": f"timed out after {timeout}s",
            }

    # ------------------------------------------------------------------ mutations (flip-only)
    def _rewrite_features(self, mutate) -> None:
        with file_lock(self._lock):
            old_text = self.feature_path.read_text(encoding="utf-8")
            data = _load_features(old_text)
            mutate(data)
            new_text = json.dumps(data, indent=2) + "\n"
            validate_feature_list_change(old_text, new_text)
            atomic_write_text(self.feature_path, new_text)

    def mark_passing(self, feature_id: str, evidence: Sequence[str]) -> None:
        """Flip ``passes`` to true. Requires end-to-end verification evidence (test ids, logs, screenshots)."""
        if not evidence or not all(str(e).strip() for e in evidence):
            raise InvariantViolation("a feature may only be marked passing with end-to-end verification evidence")

        def mutate(data: list[dict[str, Any]]) -> None:
            for d in data:
                if str(d["id"]) == feature_id:
                    d["passes"] = True
                    d["evidence"] = list(d.get("evidence", [])) + [str(e) for e in evidence]
                    return
            raise NotFoundError(f"no feature {feature_id}")

        self._rewrite_features(mutate)

    def mark_failing(self, feature_id: str, reason: str) -> None:
        """Flip a regressed feature back to failing (records the reason in the progress log)."""

        def mutate(data: list[dict[str, Any]]) -> None:
            for d in data:
                if str(d["id"]) == feature_id:
                    d["passes"] = False
                    return
            raise NotFoundError(f"no feature {feature_id}")

        self._rewrite_features(mutate)
        self.append_progress("regression", f"Feature {feature_id} regressed: {reason}")

    def add_feature(self, feature: dict[str, Any] | Feature) -> None:
        f = feature if isinstance(feature, Feature) else Feature.from_dict(feature)
        f.passes = False
        self._rewrite_features(lambda data: data.append(asdict(f)))

    def append_progress(self, session_id: str, text: str) -> None:
        with file_lock(self._lock):
            with open(self.progress_path, "a", encoding="utf-8") as fh:
                fh.write(f"\n## {to_iso(self.clock())} {session_id}\n{text.strip()}\n")

    def checkpoint(self, message: str) -> str | None:
        """``git add -A && git commit`` in the repo. Returns the sha or ``None`` when nothing changed."""
        if not self.git:
            return None
        env_args = ["-c", "user.name=harness", "-c", "user.email=harness@localhost", "-c", "commit.gpgsign=false"]
        subprocess.run(["git", "init", "-q"], cwd=self.repo, capture_output=True) if not (self.repo / ".git").exists() else None
        subprocess.run(["git", "add", "-A"], cwd=self.repo, capture_output=True, check=True)
        staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=self.repo, capture_output=True, text=True).stdout.strip()
        if not staged:
            return None
        subprocess.run(["git", *env_args, "commit", "-q", "-m", message], cwd=self.repo, capture_output=True, check=True)
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.repo, capture_output=True, text=True).stdout.strip()

    def end_session(self, session_id: str, summary: str, *, commit_message: str | None = None) -> str | None:
        """End-of-session checkpoint: append progress, then commit."""
        self.append_progress(session_id, summary)
        return self.checkpoint(commit_message or f"chore(task): session {session_id} checkpoint")
