# Environments and rollouts: `harness_rl.env`

## Purpose

This module runs episodes for RL (or for large-scale rollout collection) over the same tasks, sandboxes and graders as evals. It supports two driving styles: step-wise policies and black-box agents.

## Research basis

* **Rollout-scoped tool backend** (§11.7 step 7; agent RFT): a unique rollout id on every tool call, one sandbox per rollout, a per-rollout tool-call cap (OpenAI's example is 10), and checkpointable state for long horizons. The same backend serves agent RFT, open RL and evals.
* **OpenEnv** standardises `reset`/`step`/`state`.
* **ClawGym II** runs black-box RL over an *unmodified* harness: a per-rollout sandbox, a proxy for model calls, and a verifier that scores the final workspace.
* **Profile mismatches** (§11.7 step 2): log calls to unknown tools as `profile_mismatch` and return actionable errors.
* **Pitfall: fresh sandboxes per attempt at scale hit creation-rate limits.** Pooling and recycling belong to the sandbox package. Here each rollout gets a fresh directory tree.

## API

Step-wise driving:

```python
from harness_rl import TaskEnvironment, FunctionToolExecutor
env = TaskEnvironment(task, tools_factory=lambda sandbox: my_executor(sandbox.workspace), profile=profile,
                      reward_fn=reward, model=MODEL_ID, harness_version="h@1.8.2", split="train")
obs = env.reset(seed=7)                         # Observation(task: AgentTask, workspace, rollout_id, seed, tools, profile)
res = env.step(ToolCall("c1", "edit", {...}))   # StepResult(outcome: ToolOutcome, done: budget exhausted, info)
ep = env.finish(final_text="done")              # Episode(reward, record, grades, hack_signals, passed, tool_stats…)
env.close()
```

Black-box driving:

```python
from harness_rl import RolloutRunner
runner = RolloutRunner(agent, tools_factory=..., profile=profile, reward_fn=reward, concurrency=8,
                       max_infra_retries=2, timeout_s=None, exporter=TrajectoryExporter("train.jsonl"))
episodes = runner.run(tasks, n_per_task=8, seed=2026, split="train")
```

In black-box mode the agent receives `ctx.metadata["tools"]` (the `RolloutToolBackend`), plus `rollout_id`, `seed`, `task_id`, `env`, `memory_dir`, `profile` and `cancel_event`.

`RolloutToolBackend(inner, rollout_id=…, max_tool_calls=…, profile=…)` works as follows:

* `specs()` is the inner executor's specs filtered to the profile's tools.
* `execute(call, ctx)` checks, in order:
  1. it rejects a call whose `ctx.metadata["rollout_id"]` belongs to another rollout;
  2. it returns the profile-mismatch error for tools the profile lacks (or "does not exist" without a profile);
  3. it enforces the cap;
  4. it stamps `rollout_id` into the outcome metadata.
* Tool exceptions become `ToolOutcome(is_error=True)`.
* `stats()` returns `{calls, rejected{cross_rollout, profile_mismatch, budget, unknown_tool}, exhausted, profile_mismatch{tool: n}}`.

## Defaults

| Setting | Default | Vendor default vs our choice |
|---|---|---|
| Tool-call cap | `task.limits.max_tool_calls` (none if unset) | OpenAI's agent-RFT example is 10; per-task here |
| Seeds | `derive_seed(seed, task.id, episode_index)` | Our choice (deterministic) |
| `concurrency` / `max_infra_retries` | 4 / 2 | Our choice |
| `require_rl_ready` | `True` | Our choice, from the hackable-verifier pitfall |
| `fail_on_tamper` | `True` | Same as evals |

## Invariants

* `step`, `finish` and `context` before `reset` raise `EnvironmentStateError`. `reset` closes any previous episode, so sandboxes are never reused.
* The agent's workspace never contains hidden tests. Grading uses a separate copy.
* Episode transcripts recorded by `step` use native `tool_use`/`tool_result` blocks.
* Infra failures retry in a fresh rollout (new id). Exhausted retries yield an `infra_error` episode with reward 0 and `metadata.exclude_from_training = True`.
* Timeouts set `cancel_event`, score 0 and mark the record `truncated`.

## Failure modes

* **An agent that bypasses `ctx.metadata["tools"]`** (calling tools directly) loses the cap, id stamping and mismatch metric, and its transcript may not reflect reality. The integrator must route through the backend.
* **Long-horizon persistent-state RL** needs environment checkpoints. Not provided: sandboxes are per-episode directories.
* **High `profile_mismatch` counts** mean the profile does not match the model's training harness. Fix the profile rather than the model.
