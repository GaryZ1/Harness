# Long-running task state: `harness_memory.task_state`

**Function:** per-task progress for work that spans many context windows. This is **not** learned memory. It lives with the work in `<repo>/.harness/task/`, under git, and is re-read by every session.

| File | Role | Mutability |
|---|---|---|
| `feature_list.json` | `[{id, category, description, steps[], passes, priority, evidence[]}]` | **append and flip only** (enforced) |
| `progress.md` | Free-text log, one `## <timestamp> <session>` section per session | append |
| `init.sh` | How to start, run and smoke-test the app | edited by humans or the initializer |
| `NOTES.md` | Optional working notes (see [working-memory.md](working-memory.md)) | free |

JSON is used for the feature list because models overwrite JSON less readily than Markdown. Invariants are enforced in code, not prompts.

## Lifecycle (Anthropic "effective harnesses for long-running agents")

```python
from harness_memory import TaskState

task = TaskState(repo_root)
# Session 1: the initializer
task.initialize("Build a clone of X", features=[...200 features...], init_script="pnpm i && pnpm dev &\n")

# Every later session: hook to SessionStart AND post-compaction
report = task.orient(run_init=True)        # pwd, git log, progress tail, next failing feature, smoke test
system_prompt += report.render()

# ...implement ONE feature, verify it end to end...
task.mark_passing("f17", evidence=["e2e: tests/e2e/checkout.spec.ts passed (run 8812)"])
task.end_session("s_123", "Implemented checkout; e2e green")   # append progress + git commit
```

* `next_feature()` returns the highest-priority failing feature (lowest `priority`, then file order).
* `mark_passing` **requires evidence**. Premature "done" was the main failure mode this pattern fixed.
* `mark_failing(id, reason)` records regressions in the progress log.
* `add_feature()` appends a new feature, always with `passes: false`.

## Enforcement hook

`validate_feature_list_change(old, new)` raises `InvariantViolation` when a feature is removed or reordered, when any field other than `passes`/`evidence` changes, when `evidence` is not append-only, or when an appended feature starts with `passes: true`.

`guard_tool_call(tool_name, tool_input, repo_root=…)` is a ready-made **PreToolUse** hook:

* `Write` or `Edit` on `feature_list.json` is simulated and validated with the rule above;
* `Bash` commands that `rm` or `git rm` test files, delete the feature list, or redirect into it are blocked;
* `Write` that empties an existing test file is blocked.

```python
decision = guard_tool_call(block.name, block.input, repo_root=repo)
if not decision.allow:
    return {"type": "tool_result", "tool_use_id": block.id, "content": f"Blocked: {decision.reason}", "is_error": True}
```

Keep task state out of learned memory. Progress logs in `MEMORY.md` cause bloat and cross-task leakage.
