"""Environments and rollouts: reset/step episodes over the same tasks and graders as evals.

* **One task format for evals and RL** (reference §11.7 step 5): environments take
  :class:`harness_evals.Task` and grade through :func:`harness_evals.grade_trial`, so the hidden
  test overlay, tamper detection and scoring are identical in CI evals and in training.
* **Rollout-scoped tool backend** (§11.7 step 7; agent RFT practice): every rollout gets a unique
  ``rollout_id``, its own sandbox, and a :class:`RolloutToolBackend` that stamps the id on every
  call, rejects calls carrying another rollout's id (cross-rollout contamination), enforces a
  per-rollout tool-call cap, and converts unknown tools into profile-mismatch errors.
* **Two driving styles.** ``reset(seed) -> Observation``, ``step(ToolCall) -> StepResult``,
  ``finish() -> Episode`` for a policy that acts one tool call at a time (OpenEnv-style
  reset/step/state); or :class:`RolloutRunner`, which hands the whole episode to a black-box
  :class:`harness_evals.AgentUnderTest` (ClawGym-style RL over an unmodified harness) and exports
  each episode as a :class:`~harness_rl.trajectory.TrainingRecord`.
* **Seeds** are derived deterministically from ``(seed, task id, episode index)``.
* **Readiness gate**: by default an environment refuses tasks that :func:`rl_readiness` flags
  (no deterministic grader, visible tests) - optimising against a hackable verifier teaches hacking.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from harness_core.types import (
    Message,
    RunContext,
    TextBlock,
    ToolCall,
    ToolExecutor,
    ToolOutcome,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    new_id,
)
from harness_evals import (
    AgentTask,
    AgentUnderTest,
    GraderDeps,
    InfraError,
    IsolationError,
    MemoryPolicy,
    SandboxFactory,
    Task,
    Trajectory,
    TrialSandbox,
    build_graders,
    grade_trial,
    hack_pattern_counts,
)
from harness_evals.runner import derive_seed

from .errors import EnvironmentStateError, RewardConfigError
from .profiles import HarnessProfile, ProfileMismatchMonitor
from .rewards import CompositeReward, EpisodeOutcome, RewardBreakdown, default_reward, rl_readiness
from .trajectory import TrainingRecord, TrajectoryExporter, from_trajectory

ToolFn = Callable[[dict[str, Any], RunContext], "str | ToolOutcome"]


class FunctionToolExecutor:
    """Minimal :class:`~harness_core.types.ToolExecutor` over plain functions (tests, simple envs)."""

    def __init__(self, tools: Mapping[str, tuple[ToolSpec, ToolFn]] | None = None) -> None:
        self.tools = dict(tools or {})

    def specs(self) -> list[ToolSpec]:
        return [spec for spec, _ in self.tools.values()]

    def execute(self, call: ToolCall, ctx: RunContext) -> ToolOutcome:
        spec_fn = self.tools.get(call.name)
        if spec_fn is None:
            return ToolOutcome(f"unknown tool {call.name}", is_error=True)
        out = spec_fn[1](dict(call.input), ctx)
        return out if isinstance(out, ToolOutcome) else ToolOutcome(str(out))


class RolloutToolBackend:
    """Rollout-scoped wrapper: id stamping, cross-rollout rejection, call cap, mismatch metric."""

    def __init__(self, inner: ToolExecutor, *, rollout_id: str, max_tool_calls: int | None = None, profile: HarnessProfile | None = None) -> None:
        self.inner = inner
        self.rollout_id = rollout_id
        self.max_tool_calls = max_tool_calls
        self.monitor = ProfileMismatchMonitor(profile) if profile is not None else None
        self.calls = 0
        self.rejected: dict[str, int] = {}
        self.exhausted = False
        self._lock = threading.Lock()

    def specs(self) -> list[ToolSpec]:
        specs = self.inner.specs()
        if self.monitor is not None:
            allowed = set(self.monitor.profile.tool_names)
            specs = [s for s in specs if s.name in allowed]
        return specs

    def _reject(self, why: str, msg: str, **meta: Any) -> ToolOutcome:
        with self._lock:
            self.rejected[why] = self.rejected.get(why, 0) + 1
        return ToolOutcome(msg, is_error=True, metadata={"rejected": why, "rollout_id": self.rollout_id, **meta})

    def execute(self, call: ToolCall, ctx: RunContext) -> ToolOutcome:
        rid = ctx.metadata.get("rollout_id")
        if rid is not None and rid != self.rollout_id:
            return self._reject("cross_rollout", f"tool call carries rollout id {rid!r}, expected {self.rollout_id!r}")
        if self.monitor is not None:
            mismatch = self.monitor.check(call.name)
            if mismatch is not None:
                with self._lock:
                    self.rejected["profile_mismatch"] = self.rejected.get("profile_mismatch", 0) + 1
                return mismatch
        elif call.name not in {s.name for s in self.inner.specs()}:
            return self._reject("unknown_tool", f"Tool '{call.name}' does not exist. Available: {', '.join(s.name for s in self.inner.specs())}.")
        with self._lock:
            if self.max_tool_calls is not None and self.calls >= self.max_tool_calls:
                self.exhausted = True
                over = True
            else:
                self.calls += 1
                over = False
        if over:
            return self._reject("budget", f"Tool call budget exhausted ({self.max_tool_calls} calls). Finish with what you have.")
        ctx.metadata.setdefault("rollout_id", self.rollout_id)
        try:
            out = self.inner.execute(call, ctx)
        except Exception as exc:  # noqa: BLE001 - tool failures are observations, not crashes
            return ToolOutcome(f"{call.name} failed: {type(exc).__name__}: {exc}", is_error=True, metadata={"rollout_id": self.rollout_id})
        out.metadata = {**out.metadata, "rollout_id": self.rollout_id}
        return out

    def stats(self) -> dict[str, Any]:
        return {"calls": self.calls, "rejected": dict(self.rejected), "exhausted": self.exhausted,
                "profile_mismatch": self.monitor.snapshot() if self.monitor else {}}


@dataclass(frozen=True)
class Observation:
    task: AgentTask
    workspace: Path
    rollout_id: str
    seed: int
    tools: list[ToolSpec]
    profile: str | None = None


@dataclass(frozen=True)
class StepResult:
    outcome: ToolOutcome
    done: bool
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    rollout_id: str
    task_id: str
    seed: int
    episode_index: int
    status: str
    trajectory: Trajectory
    reward: RewardBreakdown
    record: TrainingRecord
    grades: list[Any] = field(default_factory=list)
    hack_signals: list[Any] = field(default_factory=list)
    passed: bool = False
    wall_clock_s: float = 0.0
    tool_stats: dict[str, Any] = field(default_factory=dict)


class TaskEnvironment:
    """A single-task episodic environment (one sandbox per rollout, never reused)."""

    def __init__(
        self,
        task: Task,
        *,
        tools_factory: Callable[[TrialSandbox], ToolExecutor] | None = None,
        reward_fn: CompositeReward | None = None,
        profile: HarnessProfile | None = None,
        sandbox_root: str | Path | None = None,
        memory: MemoryPolicy | None = None,
        judge_client: Any = None,
        judge_model: str | None = None,
        grader_timeout_s: float = 600.0,
        fail_on_tamper: bool = True,
        harness_version: str = "",
        model: str = "",
        split: str = "train",
        require_rl_ready: bool = True,
    ) -> None:
        if require_rl_ready:
            problems = rl_readiness(task)
            if problems:
                raise RewardConfigError(f"task {task.id} is not RL-ready: {problems}")
        self.task = task
        self.tools_factory = tools_factory
        self.reward_fn = reward_fn or default_reward()
        self.profile = profile
        self.factory = SandboxFactory(sandbox_root, memory=memory)
        self.graders = build_graders(task, GraderDeps(judge_client, judge_model))
        self.grader_timeout_s, self.fail_on_tamper = grader_timeout_s, fail_on_tamper
        self.harness_version, self.model, self.split = harness_version, model, split
        self.sandbox: TrialSandbox | None = None
        self.backend: RolloutToolBackend | None = None
        self._patterns: dict[str, dict[str, int]] = {}
        self._messages: list[Message] = []
        self._state = "idle"
        self.rollout_id = ""
        self.seed = 0
        self.episode_index = 0

    # ------------------------------------------------------------------ lifecycle
    def reset(self, seed: int, *, rollout_id: str | None = None, episode_index: int = 0) -> Observation:
        self.close()
        self.rollout_id = rollout_id or new_id("rollout")
        self.seed, self.episode_index = seed, episode_index
        self.sandbox = self.factory.create(self.task, trial_id=f"{self.task.id}.e{episode_index}", attempt=0, seed=seed)
        self.sandbox.materialise()
        self._patterns = hack_pattern_counts(self.sandbox.workspace)
        inner = self.tools_factory(self.sandbox) if self.tools_factory else FunctionToolExecutor()
        self.backend = RolloutToolBackend(inner, rollout_id=self.rollout_id, max_tool_calls=self.task.limits.max_tool_calls, profile=self.profile)
        self._messages = [Message.user(self.task.instruction)]
        self._state = "running"
        return Observation(self.task.agent_view(), self.sandbox.workspace, self.rollout_id, seed, self.backend.specs(),
                           self.profile.name if self.profile else None)

    def context(self) -> RunContext:
        self._require("running")
        assert self.sandbox is not None
        return RunContext(session_id=self.rollout_id, cwd=str(self.sandbox.workspace), agent_name="rollout", metadata={
            "rollout_id": self.rollout_id, "seed": self.seed, "task_id": self.task.id, "tools": self.backend, "env": self.sandbox.env(),
            "memory_dir": str(self.sandbox.memory_dir), "profile": self.profile.name if self.profile else None, "cancel_event": threading.Event(),
        })

    def step(self, call: ToolCall) -> StepResult:
        """Execute one tool call (the action); the transcript is recorded with native tool_result blocks."""
        self._require("running")
        assert self.backend is not None
        outcome = self.backend.execute(call, self.context())
        self._messages.append(Message.assistant([ToolUseBlock(call.id, call.name, dict(call.input))]))
        self._messages.append(Message.user([ToolResultBlock(call.id, outcome.content, outcome.is_error)]))
        return StepResult(outcome, self.backend.exhausted, {"calls": self.backend.calls})

    def finish(self, *, trajectory: Trajectory | None = None, final_text: str = "", status: str = "completed", wall_clock_s: float = 0.0) -> Episode:
        """Grade the end state, compute the reward and build the training record."""
        self._require("running")
        assert self.sandbox is not None and self.backend is not None
        if trajectory is None:
            msgs = list(self._messages) + ([Message.assistant([TextBlock(final_text)])] if final_text else [])
            trajectory = Trajectory.from_messages(msgs, model=self.model)
        grades, signals, score = [], [], None
        if status == "completed":
            out = grade_trial(self.sandbox, trajectory, self.graders, patterns_before=self._patterns, fail_on_tamper=self.fail_on_tamper,
                              timeout_s=self.grader_timeout_s)
            grades, signals, score = out.grades, out.hack_signals, out.score
            if out.grader_error:
                status = "grader_error"
        reward = self.reward_fn(EpisodeOutcome(self.task, trajectory, grades, score, signals, status))
        unanswered = {c.id for c in trajectory.tool_calls if c.output is None}
        rec = from_trajectory(
            trajectory, rollout_id=self.rollout_id, task_id=self.task.id, tools=[s.to_api() for s in self.backend.specs()],
            reward=reward.total, reward_components=dict(reward.components), hack_signals=[s.to_dict() for s in signals],
            outcome={"status": status, "passed": bool(score and score.passed), "strict_pass": bool(score and score.strict_pass),
                     "partial": score.partial if score else 0.0, "failing": list(score.failing) if score else []},
            split=self.split, seed=self.seed, episode_index=self.episode_index, profile=self.profile.name if self.profile else None,
            profile_hash=self.profile.content_hash() if self.profile else None, harness_version=self.harness_version,
            metadata={"tool_stats": self.backend.stats(), "truncated": bool(unanswered) or status != "completed", "flags": list(reward.flags)},
        )
        if not rec.model:
            rec.model = self.model
        ep = Episode(self.rollout_id, self.task.id, self.seed, self.episode_index, status, trajectory, reward, rec, list(grades), list(signals),
                     bool(score and score.passed), wall_clock_s, self.backend.stats())
        self._state = "finished"
        return ep

    def close(self) -> None:
        if self.sandbox is not None:
            self.sandbox.cleanup()
        self.sandbox, self.backend, self._state = None, None, "idle"

    def _require(self, state: str) -> None:
        if self._state != state:
            raise EnvironmentStateError(f"environment is {self._state!r}; expected {state!r} (call reset first)")


class RolloutRunner:
    """Run black-box episodes of an :class:`AgentUnderTest` over tasks, with seeds, caps and export."""

    def __init__(
        self,
        agent: AgentUnderTest,
        *,
        reward_fn: CompositeReward | None = None,
        tools_factory: Callable[[TrialSandbox], ToolExecutor] | None = None,
        profile: HarnessProfile | None = None,
        concurrency: int = 4,
        max_infra_retries: int = 2,
        timeout_s: float | None = None,
        sandbox_root: str | Path | None = None,
        exporter: TrajectoryExporter | None = None,
        env_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.agent, self.reward_fn, self.tools_factory, self.profile = agent, reward_fn, tools_factory, profile
        self.concurrency, self.max_infra_retries, self.timeout_s = concurrency, max_infra_retries, timeout_s
        self.sandbox_root, self.exporter, self.env_kwargs = sandbox_root, exporter, dict(env_kwargs or {})

    def run(self, tasks: Sequence[Task], *, n_per_task: int = 1, seed: int = 0, split: str = "train") -> list[Episode]:
        jobs = [(t, i) for t in tasks for i in range(n_per_task)]
        with ThreadPoolExecutor(max_workers=self.concurrency, thread_name_prefix="rollout") as pool:
            eps = list(pool.map(lambda j: self._episode(j[0], j[1], seed, split), jobs))
        return eps

    def _episode(self, task: Task, idx: int, base_seed: int, split: str) -> Episode:
        ep_seed = derive_seed(base_seed, task.id, idx)
        last_err: Exception | None = None
        for _attempt in range(self.max_infra_retries + 1):
            env = TaskEnvironment(task, tools_factory=self.tools_factory, reward_fn=self.reward_fn, profile=self.profile,
                                  sandbox_root=self.sandbox_root, split=split, **self.env_kwargs)
            try:
                obs = env.reset(ep_seed, episode_index=idx)
                ctx = env.context()
                traj, exc, timed_out, wall = self._invoke(obs, ctx, float(self.timeout_s or task.limits.timeout_s))
                if isinstance(exc, InfraError):
                    raise exc
            except (InfraError, IsolationError) as e:
                last_err = e
                env.close()
                continue
            status = "timeout" if timed_out else ("agent_error" if exc is not None else "completed")
            if traj is None:
                traj = Trajectory(error=f"{type(exc).__name__}: {exc}" if exc else "timeout")
            ep = env.finish(trajectory=traj, status=status, wall_clock_s=wall)
            env.close()
            if self.exporter is not None:
                self.exporter.append(ep.record)
            return ep
        # Infra failures exhausted: an explicit, zero-reward episode marked as not trainable.
        traj = Trajectory(error=f"{type(last_err).__name__}: {last_err}")
        reward = (self.reward_fn or default_reward())(EpisodeOutcome(task, traj, [], None, [], "infra_error"))
        rid = new_id("rollout")
        rec = from_trajectory(
            traj, rollout_id=rid, task_id=task.id, reward=reward.total, reward_components=dict(reward.components),
            outcome={"status": "infra_error", "passed": False, "strict_pass": False, "partial": 0.0, "failing": []}, split=split, seed=ep_seed,
            episode_index=idx, profile=self.profile.name if self.profile else None,
            metadata={"exclude_from_training": True, "truncated": True, "attempts": self.max_infra_retries + 1},
        )
        return Episode(rid, task.id, ep_seed, idx, "infra_error", traj, reward, rec)

    def _invoke(self, obs: Observation, ctx: RunContext, timeout_s: float) -> tuple[Trajectory | None, BaseException | None, bool, float]:
        box: dict[str, Any] = {}

        def target() -> None:
            try:
                box["traj"] = self.agent.run(obs.task, obs.workspace, ctx)
            except BaseException as exc:  # noqa: BLE001
                box["exc"] = exc

        t0 = time.monotonic()
        th = threading.Thread(target=target, daemon=True, name=f"rollout-{obs.rollout_id}")
        th.start()
        th.join(timeout_s)
        if th.is_alive():
            ctx.metadata["cancel_event"].set()
            return None, None, True, time.monotonic() - t0
        return box.get("traj"), box.get("exc"), False, time.monotonic() - t0


__all__ = ["Episode", "FunctionToolExecutor", "Observation", "RolloutRunner", "RolloutToolBackend", "StepResult", "TaskEnvironment"]
