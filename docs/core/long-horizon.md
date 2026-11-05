# Long-horizon control: `harness_core.long_horizon`, `harness_core.prompts`

**Purpose:** continuity across many context windows: initializer and coding sessions with fresh
context, Ralph loops, and the turn-scoped prompt templates the loop uses.

## Research basis

* **Anthropic, "Effective harnesses for long-running agents":** an initializer session writes a JSON feature list (`category`, `description`, `steps`, `passes: false`), a progress file, `init.sh` and an initial commit; each later session runs a ritual (`pwd`, git log, progress, pick the highest-priority failing feature, `init.sh`, smoke test), implements one feature, verifies it end to end, commits and logs. The two agents differ *only in the initial user prompt*. "Compaction isn't sufficient" for multi-window work; JSON resists inappropriate rewrites better than Markdown; "It is unacceptable to remove or edit tests".
* **Context anxiety:** fresh-context resets were essential for a model that wrapped up early near its perceived limit and became "dead weight" on later models, so resets are a per-model choice (checklist §2.7 step 15).
* **Ralph loop plugin:** `--max-iterations` plus `--completion-promise`, a state file, and a Stop hook that re-feeds the same prompt; use only with an executable verifier, and run the supervisor outside the agent's sandbox (agents have killed their own loop with `pkill`).
* **OpenAI 2026 guidance:** newer models may stop to ask questions; autonomous runs need persistence prompting.

## `LongRunDriver`

```python
from harness_core import LongRunDriver, LongRunPrompts, SessionStore, AgentLoop
from harness_memory import TaskState            # optional: task state lives in harness_memory

task = TaskState(repo)
driver = LongRunDriver(
    SessionStore("~/.harness/sessions"),
    loop_factory=lambda session: AgentLoop(model, tools, system=SYSTEM, session=session, hooks=hooks,
                                           budgets=Budgets(max_turns=200, max_budget_usd=30)),
    prompts=LongRunPrompts("Build a clone of X", feature_list=".harness/task/feature_list.json",
                           progress_file=".harness/task/progress.md", init_script=".harness/task/init.sh",
                           persistence=True),
    is_initialized=task.exists,                  # any () -> bool
    is_complete=lambda: task.next_feature() is None,
    max_sessions=50,                             # required
    state_path="~/.harness/longrun-x.json",
)
report = driver.run()          # LongRunReport(sessions=[SessionRecord...], completed, reason, total_cost_usd)
```

Per iteration: kind = `initializer` if `is_initialized()` is false, else `coding`; a **new**
session `<prefix>-<index>-<kind>` is created (fresh context); `loop_factory(session).run(prompt)`;
`end_session()` unless the run paused. The driver stops when `is_complete()` is true, when the
initializer did not create task state, when a run ends with a subtype in `TERMINAL_SUBTYPES`
(refusal, denials, goal impossible, during-execution error, dollar budget, prompt blocked, halted,
paused, interrupted), or after `max_sessions`. Turn, token and wall-clock budget results do not stop
it: the next session continues the work. The optional state file (atomic JSON) records every
session so a restarted driver continues the numbering.

Invariants such as "only `passes` may change" are not in these prompts' power to enforce: wire a
`pre_tool` hook (e.g. around `harness_memory.task_state.guard_tool_call`).

## `RalphLoop`

```python
report = RalphLoop(loop, "Make every test in tests/ pass.", max_iterations=20, completion_promise="ALL GREEN",
                   verifier=lambda: run_pytest(), state_path=".harness/ralph.json").run()
report.completed, report.iterations, report.result
```

Runs inside the loop's single session: the first prompt and every re-feed is `RALPH_REFEED`
(the prompt, "Iteration i of at most N", and the instruction to output `<promise>…</promise>` only
when done). A temporary `CompletionPromiseEvaluator` (accepting the promise only if `verifier()`
passes) is added and `max_total_stop_blocks` is set to `max_iterations - 1`; both are restored
afterwards. The state file holds `iteration`, `max_iterations`, `completion_promise`,
`session_id` and `status` (`running`, `completed`, `max_iterations` or an error subtype).
`max_iterations >= 1` is required, and a verifier is required unless `require_verifier=False`.

## Prompt templates (`harness_core.prompts`)

All are appended as messages, never placed in the system prompt.

| Template | Used by |
|---|---|
| `MAX_TOKENS_CONTINUATION`, `TRUNCATED_TOOL_INPUT` | `max_tokens` recovery |
| `NOT_EXECUTED_BUDGET`, `NOT_EXECUTED_INTERRUPT`, `INTERRUPTED_SIDE_EFFECTS_UNKNOWN` | Pairing results for calls that did not (or may not have) run |
| `BUDGET_WRAP_UP`, `BUDGET_WARNING` | Budgets |
| `DOOM_LOOP_NUDGE` | Stuck detector |
| `GOAL_EVALUATOR_SYSTEM`, `GOAL_EVALUATOR_PROMPT`, `GOAL_CONTINUATION` / `render_goal_continuation` | `GoalEvaluator` |
| `RALPH_REFEED` | `RalphLoop` |
| `INITIALIZER_PROMPT`, `CODING_SESSION_PROMPT`, `PERSISTENCE_PROMPT` | `LongRunPrompts` |

## Defaults

| Setting | Default | Source |
|---|---|---|
| Feature list / progress / init file names | `feature_list.json`, `claude-progress.txt`, `init.sh` | Anthropic post |
| `LongRunPrompts.persistence` | False | flag per model (OpenAI guidance) |
| `max_sessions`, `max_iterations` | required | Ralph plugin requires a cap; ours for sessions |
| Ralph verifier | required (waivable) | "use it only with an executable verifier" |

## Failure modes

* **Premature victory:** a coding session that claims success without flipping features leaves `is_complete()` false, so the next session continues; pair with an evaluator for in-session checks.
* **Agent edits tests or the feature list:** prevent with a `pre_tool` hook; the prompt alone is not enforcement.
* **Supervisor killed by the agent:** run the driver outside the sandbox the tools execute in.
* **Obsolete scaffolding:** fresh-context sessions cost a ritual per session; on models that work coherently across compaction, prefer one long session with a compactor and a goal evaluator.
