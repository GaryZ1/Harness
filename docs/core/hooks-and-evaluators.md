# Hooks and stop evaluators: `harness_core.hooks`, `harness_core.evaluators`

**Purpose:** extension points that observe or steer the loop (hooks) and an outer check that can
veto "done" and feed guidance back (evaluators).

## Research basis

* **Claude Code and Codex** share `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `Stop`, `PreCompact`, `PostCompact`, `SessionEnd`; **Gemini CLI** adds `BeforeModel`/`AfterModel`. Exit code `2` blocks with stderr as the reason; `PreToolUse` returns `permissionDecision` plus `updatedInput`; Stop returns `{"decision":"block","reason"}` or `{"continue": false}`; hooks "don't consume context"; command hooks time out after 600 s.
* **Runaway protection:** Claude Code overrides a Stop hook after eight consecutive blocks with no tool call in between and passes `stop_hook_active`; Codex ignores a block that has no continuation prompt.
* **`/goal`** (Claude Code) is a prompt-based Stop hook: a small model judges the condition from the transcript and answers Met / Not yet met (reason becomes guidance) / Impossible; it "does not call tools, so it can only judge what Claude has already surfaced". **Codex goals** inject the objective as user-provided data with a token readout, demand proof of completion, and allow "blocked" only after the same blocker repeats for three consecutive turns.
* **Ralph loop:** re-feed the same prompt until `<promise>…</promise>` appears; use only with an executable verifier.

## Hooks

```python
from harness_core import HookRegistry, HookEvent, HookResult, CommandHook, PermissionBehavior

hooks = HookRegistry()

@hooks.hook(HookEvent.PRE_TOOL, matcher="write|edit")          # regex, full match on tool name
def protect_tests(inp):
    if "tests/" in str(inp.tool_call.input.get("path", "")):
        return HookResult(permission=PermissionBehavior.DENY, reason="tests are immutable")

hooks.on(HookEvent.STOP, CommandHook("python3 .harness/hooks/stop_check.py"))   # Claude Code semantics
hooks.on(HookEvent.SESSION_START, lambda inp: HookResult(additional_context="Repo uses pnpm."))
```

| Event | `HookInput` fields set | Honoured `HookResult` fields |
|---|---|---|
| `session_start` | `data["source"]` = `startup` / `resume` | `additional_context` (joins the next user message) |
| `user_prompt_submit` | `prompt` | `block` (+`reason`) → `error_prompt_blocked`; `additional_context` (prepended to the prompt message) |
| `pre_model` | `request` | `updated_request`; `continue_=False` |
| `post_model` | `request`, `response` | `additional_context`; `continue_=False` (the response is still appended) |
| `pre_tool` | `tool_call`, `tool_spec` | `permission` (`allow`/`deny`/`ask`), `block` (= deny), `reason`, `updated_input`, `additional_context`, `continue_=False` |
| `post_tool` / `post_tool_failure` | `tool_call`, `tool_spec`, `outcome` | `updated_outcome`, `additional_context`, `continue_=False` (after the batch) |
| `stop` | `last_assistant_message`, `stop_hook_active` | `block` + non-empty `reason` → continue with the reason; `continue_=False` → `halted_by_hook` |
| `pre_compact` / `post_compact` | `data["trigger"]="auto"`, `data["reason"]`, (`summary`) | observation only |
| `session_end` | `data["reason"]` | observation only (`AgentLoop.end_session`) |

`HookEvent.claude_name` gives the Claude Code name (`PreToolUse`, …; `BeforeModel`/`AfterModel`
for the model events). A hook that raises is non-blocking: the error becomes an `error` event and
the hook counts as having no opinion.

### `CommandHook`

Runs a shell command with `HookInput.to_json_dict()` on stdin (`hook_event_name`, `session_id`,
`cwd`, `tool_name`, `tool_input`, `tool_use_id`, `tool_response`, `prompt`, `stop_hook_active`,
`last_assistant_message`, …). Exit `0` + JSON stdout is parsed (`hookSpecificOutput.permissionDecision`,
`permissionDecisionReason`, `updatedInput`, `additionalContext`, top-level `decision: "block"` +
`reason`, `continue: false` + `stopReason`); plain stdout is ignored. Exit `2` blocks with stderr
as the reason (deny for `pre_tool`). Other exit codes and timeouts are non-blocking errors.

## Stop evaluators

```python
from harness_core import GoalEvaluator, CompletionPromiseEvaluator, CallableEvaluator

goal = GoalEvaluator(judge_client, "All tests in tests/ pass and the README documents the CLI",
                     impossible_after=3, cost_estimator=prices)
tests_ran = CallableEvaluator(lambda sc: None if "passed" in sc.last_assistant_text else "Run the test suite and show the output.")
loop = AgentLoop(model, tools, evaluators=[tests_ran, goal])
```

| Evaluator | Verdict logic |
|---|---|
| `CallableEvaluator(fn)` | `fn(StopContext)` returns `Evaluation`, `True`/`None` (met), `False` (not met, no feedback) or a string (not met, that feedback) |
| `CompletionPromiseEvaluator(promise, refeed=…, verifier=…)` | Met only if the last assistant text contains `<promise>PROMISE</promise>` and, when given, `verifier()` returns `(True, detail)`; otherwise feeds back `refeed` (string or callable), prefixed with the verifier failure |
| `GoalEvaluator(model, condition)` | Calls a separate model with no tools, `GOAL_EVALUATOR_SYSTEM` (condition is data; prove completion with authoritative evidence) and the rendered transcript (last 40,000 chars). Parses `MET` / `NOT_MET` / `IMPOSSIBLE`. Not met → Codex-style continuation with `<objective>`, the reason and a token readout. Impossible is accepted only after `impossible_after` consecutive verdicts |

`StopContext` carries `messages` (the view), `last_assistant_text`, `num_turns`, `usage`,
`stop_hook_active`, `consecutive_blocks`, `total_blocks`, `token_budget`, `tokens_used`.

### Arbitration (`AgentLoop._arbitrate_stop`)

1. Stop hooks run first; `continue_=False` halts; blocks with a reason are collected; blocks without one are recorded and ignored.
2. Every evaluator runs (inside an `evaluate` span); its usage and cost count toward budgets. `IMPOSSIBLE` ends the run (`error_goal_impossible`). `NOT_MET` with feedback is collected; without feedback it is ignored. An evaluator exception is recorded and, by default, allows the stop.
3. No collected feedback → stop (`success`).
4. If `max_consecutive_stop_blocks` (8) blocks already happened with no tool call in between, or `max_total_stop_blocks` is reached, the stop is allowed (`stop_override=True`).
5. Otherwise all feedback is joined into one continuation user message, `stop_hook_active` becomes true, and the loop continues.

## Defaults

| Setting | Default | Source |
|---|---|---|
| `CommandHook.timeout` | 600 s | Claude Code command hooks |
| `max_consecutive_stop_blocks` | 8 | Claude Code |
| Block without reason | ignored | Codex |
| `GoalEvaluator.impossible_after` | 3 | Codex blocked-after-three rule |
| `GoalEvaluator` condition limit | 4,000 chars | Claude Code `/goal` |
| `GoalEvaluator.max_tokens` / `transcript_chars` | 2,000 / 40,000 | ours |
| `on_evaluator_error` | `allow_stop` | ours |

## Failure modes

* **Evaluator cannot see evidence.** It judges only the transcript; make the agent print proof (test output). The default goal prompt says so.
* **Unparseable or refused judge reply.** Treated as Not met without feedback, which the loop ignores, so a broken judge can never trap the agent.
* **Goal shrinking / injection.** The objective is always restated inside `<objective>` and labelled as user-provided data, never as instructions.
* **Runaway hooks.** Use `stop_hook_active`; the 8-block override is the backstop.
