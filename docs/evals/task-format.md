# Task format and the agent boundary: `harness_evals.task`, `harness_evals.agent`

## Purpose

This page covers the on-disk format for tasks and suites, and the boundary between the evaluation stack and the agent. One task artifact serves CI evals, public-benchmark comparisons and RL environments (`harness_rl` consumes the same `Task`).

## Research basis

* **Vocabulary** (Anthropic, "Demystifying evals for AI agents", 2026-01-09):
  * a *task* is inputs plus success criteria;
  * a *trial* is one attempt;
  * a *grader* is scoring logic;
  * a *transcript* is the full record;
  * an *outcome* is the final environment state.
* **Harbor-compatible tasks** (§9.7 step 2, §11.7 step 5) bundle an instruction, a hidden verifier, a reference solution and limits. Harbor's exact file names were not captured in the research, so the directory layout below is our convention.
* **Real-failure seeding**: 20-50 tasks drawn from real failures, with an `origin`, a reference solution, and positive and negative cases balanced (§9.7 step 3).
* **Tiers** (§9.7 step 10): tool (0), capability (1), regression (2), public (3), online (4). The rule is to never merge them into one pass rate.

## Formats

A **task directory** looks like this:

```text
tasks/fix-timezone-parse-001/
  task.yaml          # metadata + graders
  instruction.md     # agent-visible instruction (used when task.yaml has no `instruction`)
  tests/             # HIDDEN: overlaid only into the grading copy, automatically protected
  solution/          # reference solution (Task.solution_dir) – proves solvability
  files/             # optional seed files copied into the workspace
```

```yaml
id: fix-timezone-parse-001
origin: support-queue#4821
suite: capability            # tool | capability | regression | public | online
polarity: positive           # positive | negative
tags: [tz, parsing]
setup:
  files: {"src/tz/__init__.py": ""}
  commands: ["python -m pip install -q -e ."]
  git_init: true             # a fresh one-commit repo per trial; never shared history
protected_paths: ["graders/**", "pyproject.toml"]
graders:
  - kind: unit_tests
    fail_to_pass: [tests/test_tz.py::test_dst_boundary]
    pass_to_pass: [tests/]
  - {kind: diff_scope, allowed: ["src/tz/*"]}
  - {kind: tool_calls, forbid: ["rm -rf"]}
  - {kind: llm_rubric, dimension: code_quality, rubric_file: rubrics/code_quality.md, weight: 0.5}
scoring: hybrid              # binary | weighted | hybrid
threshold: 0.8
trials: 5
limits: {timeout_s: 1800, max_turns: 60, max_tool_calls: 200}
canary: "CANARY-7f3a…"       # contamination probe; never shown to the agent
```

A **suite** can come from any of three sources:

* `suite.yaml` with `{name, version, description, tasks: [<inline task> | <path to task dir or file>]}`;
* a `.jsonl` file with one task per line;
* a directory of task directories.

The `version` is part of the suite hash, so different benchmark versions never mix.

## API

```python
from harness_evals import load_suite, load_task, parse_task, Suite, SuiteTier

suite = load_suite("evals/capability")            # dir | suite.yaml | tasks.jsonl
suite.hash                                        # task-set hash (name, version, every task's content hash)
reg = suite.select(tier=SuiteTier.REGRESSION)     # also ids=..., tags=...
task = suite.by_id("fix-timezone-parse-001")
task.agent_view()                                 # AgentTask(task_id, instruction, limits, tags, metadata["agent"])
task.all_protected()                              # protected_paths ∪ hidden paths
task.trials_for(None)                             # task.trials or the tier default
```

The `AgentUnderTest` protocol is `run(task: AgentTask, workspace: Path, ctx: RunContext) -> Trajectory`. `CallableAgent(fn)` adapts any callable that returns a `Trajectory`, a `list[Message]`, a `str` (the final text) or `None`.

`Trajectory` derives `turns`, `tool_calls` (each `tool_use` paired with its `tool_result`) and `final_text` from `messages` when they are not given. It also provides:

* `tool_errors_by_tool()`;
* `render_text()`, used by judges and reviewers;
* `to_dict()` / `from_dict()`;
* `from_messages()` / `from_events()`.

## Defaults

| Setting | Default | Source |
|---|---|---|
| Trials, capability / public | 5 | TB2 leaderboard convention (vendor practice) |
| Trials, regression / tool | 3 | Checklist minimum for PR-level regression |
| Trials, online | 1 | Our choice (sampled production traces) |
| `limits.timeout_s` | 1800 s | Our choice |
| `scoring` / `threshold` | binary / 1.0 | Binary is the release gate (Anthropic) |
| `polarity` / `suite` | positive / capability | Our choice |

## Invariants (enforced at load)

* Task ids match `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$` and are unique within a suite.
* Every task has an instruction and at least one grader, and every grader spec has a `kind`.
* All setup, hidden and grader paths are relative and traversal-free, following the `harness_memory.PathSandbox` rules.
* `isolation.tests_writable_by_agent: true` is rejected outright.
* Hidden tests (`tests/` and `hidden_files`) never appear in `AgentTask`, the agent's workspace, or its manifest.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `TaskFormatError: copy_dirs entry not found` | Path is relative to the wrong directory | `copy_dirs` resolve next to the task file |
| A task is 0% over many trials | Broken task or grader | See `saturation_report` / `ReviewLog`, and check the reference solution |
| Suite hash changes unexpectedly | Hidden test or seed file edited | Expected: the hash covers file contents. Bump `version` |
