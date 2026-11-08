"""Trial runner: isolated, parallel, time-boxed trials with infra-only retries.

Behaviour (reference §9.7 steps 1, 4, 7, 9, 17; §9.8):

* **Pinned conditions.** Every run has a :class:`RunConditions` header (harness version, model,
  effort, thinking, prompt/tool SHAs, profile, sandbox image, judge model + prompt version, user
  simulator, safeguards) plus the suite hash and grader versions. Resuming a run with different
  conditions is refused by the store.
* **Isolation.** A fresh :class:`~harness_evals.isolation.TrialSandbox` per attempt - never reused
  across trials - with reset or frozen memory and no shared git history.
* **Concurrency cap.** ``RunConfig.concurrency`` trials run at once (thread pool).
* **Timeouts.** The agent phase runs in a watchdog thread; on timeout the trial is ``timeout``
  (a failure, not retried) and ``ctx.metadata["cancel_event"]`` is set so cooperative agents stop.
* **Retries only for infra errors.** :class:`~harness_evals.errors.InfraError` (including setup
  failures) triggers a retry in a *new* sandbox, up to ``max_infra_retries``; every attempt is
  recorded. Agent exceptions and timeouts are never retried.
* **Error counting is explicit.** ``error_policy`` = ``failure`` (default) counts exhausted infra
  errors and grader errors as failures; ``exclude`` drops them from denominators. Either way the
  policy is in the header and the report (cf. Vals.ai's fallback-counting swing on TB 4.0).
* **Efficiency is first-class.** Each record carries turns, tool calls, token usage, cost,
  wall-clock, tool errors by tool and compaction events.
"""

from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from harness_core.types import Clock, Event, EventSink, EventType, NullTracer, RunContext, Tracer, new_id, utcnow

from .agent import AgentUnderTest, Trajectory, _usage_dict
from .errors import EvalError, InfraError, IsolationError
from .graders.base import GraderDeps, build_graders
from .graders.integrity import hack_pattern_counts
from .graders.judge import JUDGE_PROMPT_VERSION
from .grading import grade_trial
from .isolation import MemoryPolicy, SandboxFactory, TrialSandbox
from .store import ResultStore
from .task import Suite, Task

__version__ = "0.1.0"


class TrialStatus(str, Enum):
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    AGENT_ERROR = "agent_error"
    INFRA_ERROR = "infra_error"      # retries exhausted
    GRADER_ERROR = "grader_error"


@dataclass(frozen=True)
class RunConditions:
    """Everything that must be pinned for a number to be interpretable (§9.7 step 17)."""

    harness_version: str
    model: str
    effort: str | None = None
    thinking: str | None = None
    prompts_sha: str | None = None
    tools_sha: str | None = None
    profile: str | None = None
    sandbox_image: str | None = None
    judge_model: str | None = None
    user_simulator: str | None = None
    safeguards: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.harness_version or not self.model:
            raise EvalError("RunConditions needs harness_version and model")

    def to_dict(self) -> dict[str, Any]:
        return {k: (dict(v) if isinstance(v, Mapping) else v) for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class RunConfig:
    trials_per_task: int | None = None       # None: task.trials, else tier default (capability 5, regression 3)
    concurrency: int = 4
    max_infra_retries: int = 2
    seed: int = 0
    timeout_s: float | None = None           # override every task's limits.timeout_s
    setup_timeout_s: float = 600.0
    grader_timeout_s: float = 600.0
    keep_workspaces: str = "failures"        # never | failures | always
    memory: MemoryPolicy = field(default_factory=MemoryPolicy)
    error_policy: str = "failure"            # failure | exclude
    fail_on_tamper: bool = True
    fail_on_limit_violation: bool = True
    workspace_root: Path | None = None

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise EvalError("concurrency must be >= 1")
        if self.max_infra_retries < 0:
            raise EvalError("max_infra_retries must be >= 0")
        if self.keep_workspaces not in ("never", "failures", "always"):
            raise EvalError("keep_workspaces must be never|failures|always")
        if self.error_policy not in ("failure", "exclude"):
            raise EvalError("error_policy must be failure|exclude")

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k not in ("memory", "workspace_root")}
        d["memory"] = {"mode": self.memory.mode, "snapshot": str(self.memory.snapshot) if self.memory.snapshot else None}
        return d


@dataclass
class RunResult:
    run_id: str
    header: dict[str, Any]
    records: list[dict[str, Any]]

    def summary(self, **kw: Any):
        from .report import summarize

        return summarize(self.header, self.records, **kw)


def derive_seed(base: int, task_id: str, trial_index: int) -> int:
    return int(hashlib.sha256(f"{base}:{task_id}:{trial_index}".encode()).hexdigest()[:8], 16)


class EvalRunner:
    def __init__(
        self,
        agent: AgentUnderTest,
        *,
        store: ResultStore,
        conditions: RunConditions,
        config: RunConfig | None = None,
        judge_client: Any = None,
        judge_model: str | None = None,
        event_sink: EventSink | None = None,
        tracer: Tracer | None = None,
        clock: Clock = utcnow,
    ) -> None:
        self.agent = agent
        self.store = store
        self.conditions = conditions
        self.config = config or RunConfig()
        if judge_model and conditions.judge_model and judge_model != conditions.judge_model:
            raise EvalError("judge_model differs from RunConditions.judge_model")
        self.deps = GraderDeps(judge_client, judge_model or conditions.judge_model)
        self.event_sink = event_sink
        self.tracer = tracer or NullTracer()
        self.clock = clock
        self.factory = SandboxFactory(self.config.workspace_root, memory=self.config.memory)
        self.sink_errors = 0

    # ------------------------------------------------------------------ public
    def header(self, suite: Suite, run_id: str) -> dict[str, Any]:
        graders = {}
        for t in suite.tasks:
            for g in build_graders(t, self.deps):
                graders[g.kind] = g.version
        return {
            "run_id": run_id,
            "suite": {"name": suite.name, "version": suite.version, "hash": suite.hash, "n_tasks": len(suite.tasks)},
            "conditions": self.conditions.to_dict(),
            "config": self.config.to_dict(),
            "graders": dict(sorted(graders.items())),
            "judge": {"model": self.deps.judge_model, "prompt_version": JUDGE_PROMPT_VERSION} if "llm_rubric" in graders else None,
            "harness_evals_version": __version__,
            "created_at": self.clock().isoformat(),
        }

    def run(self, suite: Suite, *, run_id: str | None = None, resume: bool = True) -> RunResult:
        run_id = run_id or new_id("run")
        graders_by_task = {t.id: build_graders(t, self.deps) for t in suite.tasks}  # fail fast on bad specs
        header = self.header(suite, run_id)
        self.store.write_header(run_id, header)
        done = {(r["task_id"], r["trial_index"]) for r in self.store.load_records(run_id)} if resume else set()
        jobs = [(t, i) for t in suite.tasks for i in range(t.trials_for(self.config.trials_per_task)) if (t.id, i) not in done]
        with ThreadPoolExecutor(max_workers=self.config.concurrency, thread_name_prefix="eval-trial") as pool:
            futures = [pool.submit(self._run_trial, run_id, t, i, graders_by_task[t.id]) for t, i in jobs]
            for f in futures:
                f.result()
        records = sorted(self.store.load_records(run_id), key=lambda r: (r["task_id"], r["trial_index"]))
        return RunResult(run_id, self.store.read_header(run_id), records)

    # ------------------------------------------------------------------ one trial
    def _invoke(self, task: Task, sandbox: TrialSandbox, ctx: RunContext, timeout_s: float) -> tuple[Trajectory | None, BaseException | None, bool]:
        box: dict[str, Any] = {}

        def target() -> None:
            try:
                box["traj"] = self.agent.run(task.agent_view(), sandbox.workspace, ctx)
            except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
                box["exc"] = exc

        th = threading.Thread(target=target, name=f"agent-{ctx.session_id}", daemon=True)
        th.start()
        th.join(timeout_s)
        if th.is_alive():
            ctx.metadata["cancel_event"].set()
            return None, None, True
        return box.get("traj"), box.get("exc"), False

    def _run_trial(self, run_id: str, task: Task, idx: int, graders) -> dict[str, Any]:
        cfg = self.config
        trial_id = f"{task.id}.t{idx}"
        seed = derive_seed(cfg.seed, task.id, idx)
        timeout_s = float(cfg.timeout_s or task.limits.timeout_s)
        attempts: list[dict[str, Any]] = []
        started_at = self.clock().isoformat()
        traj: Trajectory | None = None
        status = TrialStatus.INFRA_ERROR
        error: str | None = None
        sandbox: TrialSandbox | None = None
        patterns_before: dict[str, dict[str, int]] = {}
        wall = 0.0
        with self.tracer.span("eval.trial", run_id=run_id, task_id=task.id, trial_index=idx):
            for attempt in range(cfg.max_infra_retries + 1):
                if sandbox is not None:
                    sandbox.cleanup()
                sandbox = self.factory.create(task, trial_id=trial_id, attempt=attempt, seed=seed)
                ctx = RunContext(
                    session_id=f"{run_id}.{trial_id}.a{attempt}", cwd=str(sandbox.workspace), agent_name="eval",
                    metadata={"seed": seed, "trial_id": trial_id, "task_id": task.id, "run_id": run_id, "attempt": attempt,
                              "memory_dir": str(sandbox.memory_dir), "memory_mode": cfg.memory.mode, "env": sandbox.env(),
                              "cancel_event": threading.Event(), "deadline_s": timeout_s},
                )
                try:
                    sandbox.materialise(setup_timeout_s=cfg.setup_timeout_s)
                    patterns_before = hack_pattern_counts(sandbox.workspace)
                    t0 = time.monotonic()
                    traj, exc, timed_out = self._invoke(task, sandbox, ctx, timeout_s)
                    wall = time.monotonic() - t0
                    if isinstance(exc, InfraError):
                        raise exc
                except (InfraError, IsolationError) as e:
                    attempts.append({"attempt": attempt, "status": "infra_error", "infra": True, "error": f"{type(e).__name__}: {e}"})
                    error = f"{type(e).__name__}: {e}"
                    traj = None
                    continue
                if timed_out:
                    status, error = TrialStatus.TIMEOUT, f"agent exceeded {timeout_s}s"
                elif exc is not None:
                    status, error = TrialStatus.AGENT_ERROR, f"{type(exc).__name__}: {exc}"
                else:
                    status, error = TrialStatus.COMPLETED, None
                attempts.append({"attempt": attempt, "status": status.value, "infra": False, "error": error})
                break
            record = self._finish(run_id, task, idx, trial_id, seed, sandbox, traj, status, error, attempts, patterns_before, graders, wall, started_at)
        self.store.append(run_id, record)
        self._emit(run_id, record)
        return record

    def _finish(self, run_id, task, idx, trial_id, seed, sandbox, traj, status, error, attempts, patterns_before, graders, wall, started_at) -> dict[str, Any]:
        cfg = self.config
        traj = traj if traj is not None else Trajectory(error=error)
        grades: list[dict[str, Any]] = []
        signals: list[dict[str, str]] = []
        passed = strict = False
        partial = 0.0
        grading_s = 0.0
        isolation_violations = sandbox.memory_violations() if sandbox is not None and status is not TrialStatus.INFRA_ERROR else []
        limit_violations = []
        if task.limits.max_turns is not None and (traj.turns or 0) > task.limits.max_turns:
            limit_violations.append(f"turns {traj.turns} > {task.limits.max_turns}")
        if task.limits.max_tool_calls is not None and traj.n_tool_calls > task.limits.max_tool_calls:
            limit_violations.append(f"tool calls {traj.n_tool_calls} > {task.limits.max_tool_calls}")
        if status is TrialStatus.COMPLETED and sandbox is not None:
            g0 = time.monotonic()
            outcome = grade_trial(sandbox, traj, graders, patterns_before=patterns_before, fail_on_tamper=cfg.fail_on_tamper, timeout_s=cfg.grader_timeout_s)
            grading_s = time.monotonic() - g0
            grades = [g.to_dict() for g in outcome.grades]
            signals = [s.to_dict() for s in outcome.hack_signals]
            passed, strict, partial = outcome.score.passed, outcome.score.strict_pass, outcome.score.partial
            if outcome.grader_error:
                status = TrialStatus.GRADER_ERROR
                error = "; ".join(f"{g['grader']}: {g['error']}" for g in grades if g.get("error"))
        if isolation_violations or (limit_violations and cfg.fail_on_limit_violation):
            passed = strict = False
        transcript_path = self.store.write_transcript(run_id, trial_id, traj.to_dict())
        keep = cfg.keep_workspaces == "always" or (cfg.keep_workspaces == "failures" and not passed)
        workspace_path = str(sandbox.root) if (keep and sandbox is not None) else None
        if sandbox is not None and not keep:
            sandbox.cleanup()
        return {
            "run_id": run_id, "trial_id": trial_id, "task_id": task.id, "task_hash": task.content_hash(), "suite_tier": task.suite.value,
            "polarity": task.polarity.value, "tags": list(task.tags), "trial_index": idx, "seed": seed, "status": status.value,
            "passed": bool(passed), "strict_pass": bool(strict), "partial": partial, "scoring": task.scoring.value,
            "grades": grades, "hack_signals": signals, "attempts": attempts, "infra_retries": sum(1 for a in attempts if a["infra"]),
            "error": error, "usage": _usage_dict(traj.usage), "cost_usd": traj.cost_usd, "turns": traj.turns, "tool_calls": traj.n_tool_calls,
            "tool_errors_by_tool": traj.tool_errors_by_tool(), "compaction_events": traj.compaction_events, "wall_clock_s": round(wall, 4),
            "grading_s": round(grading_s, 4), "started_at": started_at, "finished_at": self.clock().isoformat(),
            "isolation_violations": isolation_violations, "limit_violations": limit_violations, "transcript_path": transcript_path,
            "workspace_path": workspace_path, "model": traj.model or self.conditions.model,
        }

    def _emit(self, run_id: str, record: Mapping[str, Any]) -> None:
        if self.event_sink is None:
            return
        data = {k: record[k] for k in ("trial_id", "task_id", "status", "passed", "strict_pass", "partial", "cost_usd", "wall_clock_s", "turns", "tool_calls")}
        data["hack_signals"] = len(record["hack_signals"])
        try:
            self.event_sink.emit(Event(session_id=run_id, type=EventType.EVALUATION, data=data))
        except Exception:  # noqa: BLE001 - observability must never break an eval run
            self.sink_errors += 1


__all__ = ["EvalRunner", "RunConditions", "RunConfig", "RunResult", "TrialStatus", "derive_seed"]
