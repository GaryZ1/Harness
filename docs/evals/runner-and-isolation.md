# Trial runner and isolation: `harness_evals.runner`, `harness_evals.isolation`, `harness_evals.grading`

## Purpose

The runner executes `task × trial` in isolated sandboxes, in parallel up to a cap, with time limits. It retries only infrastructure failures, grades every completed trial through one shared code path, and records everything a report needs.

## Research basis

* **Isolation** (§9.7 step 4; §9.8 "Non-isolated trials"): give each trial a fresh environment with no git history from earlier trials, no shared memory or cache directories, and independent limits. Claude once read git history left by previous trials.
* **Grade from outside the sandbox** (§9.7 step 5): tests are hidden and read-only to the agent.
* **Pinned conditions** (§9.7 steps 1 and 17): each record carries the harness version, prompt/tool SHAs, model, effort/thinking, suite version, seed, sandbox image and grader versions.
* **Fallback counting matters**: TB 4.0 scores 65.15% or 53.54% depending on the counting rule, so the rule for errors is explicit and reported.
* **Efficiency is first-class** (§9.7 step 9).

## API

```python
from harness_evals import EvalRunner, RunConditions, RunConfig, MemoryPolicy, ResultStore

runner = EvalRunner(agent, store=ResultStore(root), conditions=RunConditions("h@1.8.2", MODEL_ID, effort="high"),
                    config=RunConfig(concurrency=8, memory=MemoryPolicy("frozen", snapshot=Path("mem-snapshot"))),
                    judge_client=client, judge_model=JUDGE_ID, event_sink=sink, tracer=tracer)
res = runner.run(suite, run_id="cap-v14")          # resumable: completed (task, trial) pairs are skipped
res.records                                        # list of TrialRecord dicts (also in trials.jsonl)
```

Each trial runs these steps in order:

1. **Seed.** The seed is `derive_seed(config.seed, task.id, trial_index)`.
2. **Attempt loop.** `SandboxFactory.create` makes a new tree `<root>/<trial>.a<attempt>.<rand>/{workspace,home,memory}`. `materialise()` copies seed files, runs `git_init` and setup commands, and calls `verify_isolation()`.
3. **Run the agent** in a watchdog thread with `ctx = RunContext(session_id, cwd=workspace, metadata={seed, trial_id, task_id, run_id, attempt, memory_dir, memory_mode, env, cancel_event, deadline_s})`.
4. **Retry or record.** `InfraError` (raised by setup or by the agent) and `IsolationError` retry in a *new* sandbox. Timeouts and other exceptions are final.
5. **Grade.** `grade_trial`:
   * builds the end-state manifest;
   * creates a grading copy (`prepare_grading`) where hidden tests replace any same-named agent files;
   * runs every grader through `safe_grade`;
   * runs `detect_hack_signals`;
   * scores with `score_task`;
   * forces a failure on tamper.
6. **Record.** The record is appended under a lock, the transcript is written atomically, and an `EVALUATION` event is emitted. Sink errors are counted and never fatal.

Trial statuses are `completed`, `timeout`, `agent_error`, `infra_error` (retries exhausted) and `grader_error`.

## Defaults

| Setting | Default | Vendor default vs our choice |
|---|---|---|
| `concurrency` | 4 | Our choice |
| `max_infra_retries` | 2 | Our choice. Mirrors the "retry infra, never auth/validation" taxonomy |
| `timeout_s` | `task.limits.timeout_s` (1800) | Our choice |
| `setup_timeout_s` / `grader_timeout_s` | 600 / 600 s | Our choice |
| `keep_workspaces` | `failures` | Our choice: keep failure evidence for transcript review |
| `memory` | `MemoryPolicy("reset")` | Checklist: no shared memory between trials |
| `error_policy` | `failure` | Our choice: conservative. `exclude` drops infra/grader errors from denominators. Either way it is reported |
| `fail_on_tamper` | `True` | Our choice, from the hackable-grader pitfall |
| `fail_on_limit_violation` | `True` | Our choice: exceeding the declared `max_turns`/`max_tool_calls` fails the trial |

## Invariants

* A sandbox is never reused across trials or attempts.
* Without `git_init` there is no `.git` in the workspace, and `GIT_CEILING_DIRECTORIES` prevents discovery of enclosing repos. With `git_init` the repo has exactly one commit.
* `HOME`, `XDG_*` and `TMPDIR` point inside the trial. `HARNESS_MEMORY_DIR` is the only memory root.
* Frozen memory is a copy, and any change to it is an `isolation_violation` that fails the trial. The snapshot itself is never touched.
* Graders never run in the agent's workspace, and the agent's workspace never contains hidden tests.
* Retries happen only for `InfraError`/`SetupError`/`IsolationError`. Every attempt is recorded with `infra: true|false`.
* A resumed run must have identical conditions and suite hash, or the store raises.

## Failure modes

| Symptom | Cause | Handling |
|---|---|---|
| Trial hangs past its timeout | Agent ignores `cancel_event` | Recorded as `timeout`. The thread is a daemon and its sandbox is kept for review |
| Many `infra_error` trials | Provider outage or a broken setup command | Shown in the report header (`infra_retries`, `excluded_trials`). Fix the environment before reading scores |
| `grader_error` | Grader crash or unparseable judge output | Audit the grader. These are never counted as agent failures silently: see `error_policy` |
| Pass rate looks inflated | Shared state | Check `isolation_violations` and that the agent honours `memory_dir`/`env` |
