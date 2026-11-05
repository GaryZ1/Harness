# Agent loop: `harness_core.loop`

**Purpose:** drive one session with a model, tools and controls until the canonical stop
predicate holds, a limit trips, or the run pauses for approval, and return a typed `RunResult`.

## Research basis

* **Claude Agent SDK** runs "the same execution loop that powers Claude Code" and repeats "until it produces a response with no tool calls"; hooks can "intercept, modify, or block" tools; the run ends in a `ResultMessage` with `subtype`, usage, cost, turns and `session_id`.
* **Codex `run_turn`** computes `needs_follow_up = model_needs_follow_up || has_pending_input` and runs Stop hooks that can block with a continuation prompt; it ignores a block that supplies no prompt.
* **Claude Code** overrides a Stop hook after eight consecutive blocks with no intervening tool call; read-only tools run concurrently while `Edit`/`Write`/`Bash` run sequentially; a correction typed mid-run is read "as soon as those calls finish, within the same turn".
* **API rules:** append the full `response.content` (thinking, compaction blocks unchanged); return all `tool_result` blocks for one assistant turn in one user message with `is_error` for failures; `pause_turn` means "resend to continue"; check `refusal` before reading content; `model_context_window_exceeded` means compact.

The canonical predicate implemented by `AgentLoop._loop`:

```text
stop  ⇔  last response has no tool calls  ∧  steering queue empty  ∧  stop hooks and evaluators allow
```

## API

```python
from harness_core import AgentLoop, LoopConfig, Budgets, ResultSubtype

loop = AgentLoop(model, executor, system="...", session=session,
                 config=LoopConfig(context_window_tokens=200_000),
                 budgets=Budgets(max_turns=40, max_budget_usd=5.0), cost_estimator=prices,
                 policy=my_policy, approver=my_approver, hooks=hooks, evaluators=[goal],
                 compactor=my_compactor, sinks=[ui_stream], tracer=otel_tracer)

result = loop.run("Fix the failing test")       # new prompt
loop.steer("also update the changelog")         # from another thread, mid-run
loop.interrupt()                                # hard cancel from another thread
result = loop.run()                             # resume a woken session (crash or pause)
result = loop.run(approvals={"toolu_123": True})  # answer a paused approval
loop.end_session()                              # fire session_end, mark completed
```

`RunResult` fields: `subtype`, `session_id`, `stop_reason` (last model stop reason or `None`),
`final_text`, `num_turns` (model calls this run, excluding retries and the wrap-up call), `usage`,
`total_cost_usd`, `duration_s`, `error`, `reason`, `messages` (canonical history),
`pending_approvals`, `unprocessed_input`, `evaluations`, `wrapped_up`, `stop_override`; plus
`ok` and `is_error`.

### Result subtypes

| Subtype | When |
|---|---|
| `success` | Stop predicate held (also when an empty resume had nothing to do) |
| `error_max_turns`, `error_max_budget_usd`, `error_max_token_budget`, `error_max_wall_clock` | A budget tripped (see [budgets-and-recovery.md](budgets-and-recovery.md)) |
| `error_during_execution` | Terminal model error, retries exhausted, too many `pause_turn`s, or an unexpected harness exception |
| `error_refusal` | `stop_reason == "refusal"` |
| `error_context_window` | Overflow with no compactor, or compaction could not make the context fit |
| `error_doom_loop`, `error_max_consecutive_errors`, `error_permission_denials` | Control-plane limits |
| `error_goal_impossible` | An evaluator returned `IMPOSSIBLE` |
| `error_prompt_blocked` | A `user_prompt_submit` hook blocked the prompt (nothing appended) |
| `halted_by_hook` | A hook returned `continue_=False` |
| `paused_for_approval` | An approver raised `ApprovalRequired` (durable pause) |
| `interrupted` | `interrupt()` was called |
| `error_max_structured_output_retries` | Reserved for a structured-output layer (SDK parity) |

### stop_reason handling

| `stop_reason` | Action |
|---|---|
| `end_turn`, `stop_sequence`, other | Steering queued → append it and continue; otherwise stop arbitration |
| `tool_use` (or any response with `tool_use` blocks) | Tool turn (below) |
| `pause_turn` | Resend with the assistant turn last and no new user message (max `max_pause_continuations`) |
| `max_tokens`, text only | Append `MAX_TOKENS_CONTINUATION` as a user message (max `max_output_continuations`), then normal stop logic |
| `max_tokens`, with a `tool_use` | Discard the response (recorded as a `model_response` event with `discarded`), double `max_tokens` up to `max_tokens_ceiling`, retry; at the ceiling append it and return `TRUNCATED_TOOL_INPUT` errors for its calls |
| `refusal` | Append the turn (append-only), end with `error_refusal`; `final_text` is not taken from it |
| `model_context_window_exceeded` (or `ContextWindowExceededError`) | Discard, call the compactor with `reason="overflow"`, retry |

### A tool turn

1. Emit `tool_call` (`phase: requested`) per call; reset the consecutive stop-block counter.
2. Budget check (needs another call): if tripped, no call runs; each gets a "not executed" error result and the wrap-up turn runs.
3. Stuck detector on the calls (nudge or terminate).
4. Partition into groups: consecutive calls whose trusted spec is `read_only` form one concurrent group (thread pool capped at `max_parallel_tools`); every other call is its own sequential group. Groups run in order.
5. Per call: unknown tool → error result listing available tools; `pre_tool` hooks → policy → approver (see [tools-and-permissions.md](tools-and-permissions.md)); `tool_call` (`phase: executing`, with idempotency key) → `executor.execute` (exceptions become error results) → `post_tool`/`post_tool_failure` hooks → `tool_result` event with the full block.
6. One user message: all `tool_result` blocks in the original call order, then text blocks for carried context (hook context, nudges), drained steering, budget warnings and, if a budget tripped after the batch, the wrap-up instruction.
7. Error limits (`error_doom_loop`, consecutive all-error turns, denials).

### Resume

`run()` on a session whose last message is an assistant turn with unpaired `tool_use` blocks
repairs the batch from the per-call events: completed results are reused; calls that never started
go through the full permission pipeline (an `approvals` entry answers a paused approval);
read-only calls that started but have no result are re-run (retry-safe by definition); mutating
calls that started but have no result get `INTERRUPTED_SIDE_EFFECTS_UNKNOWN`. If the last message
is a user message (crash before the model answered), the model is called. If the session already
ended normally and nothing is queued, `run()` returns `success` with no model call.

## Defaults (`LoopConfig`)

| Field | Default | Vendor default or our choice |
|---|---|---|
| `max_tokens` | 16,000 | Anthropic non-streaming default |
| `max_tokens_ceiling` | 64,000 | ours |
| `max_output_continuations` | 3 | ours |
| `max_pause_continuations` | 10 | ours |
| `parallel_read_only` / `max_parallel_tools` | on / 8 | concurrency rule from Claude Agent SDK; cap is ours |
| `context_window_tokens` | None (no proactive compaction) | Cline pitfall: never auto-compact against an unknown window |
| `compaction_threshold_fraction` | 0.9 | Codex `auto_compact_token_limit` (90% of the window) |
| `max_compaction_attempts` | 3 | "after a few attempts" (Claude Code); number is ours |
| `max_consecutive_stop_blocks` | 8 | Claude Code Stop-hook override |
| `max_total_stop_blocks` | None | ours (Ralph uses it as max iterations) |
| `wrap_up_on_budget` | True | Codex `budget_limit.md` |
| `on_evaluator_error` | `allow_stop` | ours (a broken evaluator must not trap the agent) |
| `stuck` | `StuckConfig()` | OpenHands thresholds, see [budgets-and-recovery.md](budgets-and-recovery.md) |
| `retry` | `RetryPolicy()` | Codex 4 request retries |
| Default policy | `ReadOnlyPolicy` | Agent SDK default mode: callback, else deny |
| Default assembler | `PassthroughAssembler` | ours |

## Invariants (enforced in code)

* History is append-only; assistant content is appended exactly as returned. A response is either appended in full or discarded in full (and recorded as discarded).
* Every `tool_use` gets exactly one `tool_result`, in call order, in the next user message, even when the run ends early (budget, interrupt, halt, doom loop): `_close_dangling` reuses results already recorded and fills the rest with "not executed" errors. Only `paused_for_approval` leaves a batch open, by design.
* The tool list is read from `executor.specs()` once per run and frozen, so the cached prefix cannot change mid-run.
* Turn-scoped text (steering, nudges, warnings, hook context, continuations) is always appended as messages, never written into `system` or the tool list.
* Tool failures never raise out of the loop. Only `SessionLockedError` (lease held elsewhere) and non-`Exception` `BaseException`s escape `run()`; every other failure ends in a typed `RunResult`.
* Runs on one session are serialized by the session lease.

## Events and spans

Events: `session_start` (`source: startup|resume`, tool names, system hash), `user_message`,
`model_request`, `model_response`, `tool_call`, `permission`, `tool_result`, `compaction`,
`evaluation`, `error` (retries, hook failures, evaluator failures), `custom` (budget, stuck,
taint, resume repair), `stop`, `session_end`. Extra `sinks` receive every event too.
Spans: `invoke_agent` → `chat` (attributes `retries`, `stop_reason`) → `execute_tool`, plus
`approval_wait`, `compaction` and `evaluate`.

## Failure modes

| Failure | Result |
|---|---|
| Model keeps returning `pause_turn` | `error_during_execution` after `max_pause_continuations` |
| Truncated tool input even at the ceiling | Error results; the model is told to send smaller input |
| Approver raises | `ApprovalRequired` → pause; any other exception → `error_during_execution` |
| Host calls `run()` while another `run()` holds the session | `SessionLockedError` |
| A new prompt arrives while the history ends in a user message | Appended as another user message (the API merges consecutive user turns) |
