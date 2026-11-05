# Budgets and recovery: `budgets`, `retry`, `stuck`, `context`

**Purpose:** the runaway protection that belongs in harness code: spend and time limits, error and
denial limits, doom-loop detection, model-call retries, and the compaction hook with its thrash guard.

## Research basis

* **Claude Agent SDK:** `max_turns` counts each tool-calling turn plus the final text-only response; `max_budget_usd` is checked after each response, so it can overshoot by one response; both default to no limit, and "Setting a budget is a good default for production agents".
* **Codex:** a token budget with model-visible readouts; at exhaustion a `budget_limit.md` turn tells the model to "wrap up… summarize useful progress… leave the user with a clear next step"; `DEFAULT_REQUEST_MAX_RETRIES = 4`; `ContextWindowExceeded`, `UsageLimitReached` and `Fatal` are not retried; auto-compaction at 90% of the window.
* **LangChain** (secondary): time-budget warnings and per-file edit counters were part of its +13.7-point Terminal-Bench 2.0 gain. **OpenHands `StuckDetector`:** e.g. four repeats of a two-event pattern; the same action failing the same way.
* **Claude Code:** auto mode escalates after 3 consecutive or 20 total denials and terminates headless runs; auto-compaction stops "after a few attempts" when one output refills the window.
* Pitfalls (§2.8): compact *before* overflow; never retry terminal errors; never treat `max_budget_usd` as exact.

## Budgets

```python
from harness_core import Budgets, PricingTable, Price

prices = PricingTable({"<model id from config>": Price(input_per_mtok=..., output_per_mtok=...)})
loop = AgentLoop(model, tools, cost_estimator=prices,
                 budgets=Budgets(max_turns=60, max_total_tokens=3_000_000, max_wall_clock_s=3600, max_budget_usd=20.0))
```

| Field | Default | Checked | Source |
|---|---|---|---|
| `max_turns` | None | before executing a tool batch and before any continuation (trips when another call would be needed) | Agent SDK (no limit by default) |
| `max_budget_usd` | None | after each response, after each tool batch | Agent SDK |
| `max_total_tokens` | None | same (counts `total_input` incl. cache + `output_tokens`, summed over calls) | Codex token budget (harness-counted) |
| `max_wall_clock_s` | None | same | ours |
| `max_consecutive_error_turns` | 5 | after each tool batch (turns where every result is an error) | ours |
| `max_consecutive_denials` / `max_total_denials` | 3 / 20 | after each tool batch | Claude Code auto mode |
| `warning_fraction` | 0.8 | once per budget, appended to the tool-result message | ours (time-budget warnings) |
| `scope` | `run` | `session` seeds usage, cost and turns from the session's recorded responses | ours |

`Price` defaults cache reads to 0.1× and 5-minute cache writes to 1.25× the input price when not
given. `PricingTable` never guesses a price: unknown models cost 0 and are listed in `unpriced`.
Evaluator model calls (`GoalEvaluator(cost_estimator=…)`) count toward the run's spend.

**When a budget trips:** calls that have not run get `Error: not executed - the run's budget is
exhausted (…)`; if `LoopConfig.wrap_up_on_budget` (default on), one extra model call is made with
`BUDGET_WRAP_UP` ("Do not call any more tools… summarize… next step") appended to the same user
message; any tool call in that reply is paired with a "not executed" result. The wrap-up call is
not counted in `num_turns` but its cost is included, so the dollar cap can overshoot by up to two
responses. When the model has just answered without tools and only a continuation would exceed
the budget, the run ends with the budget subtype and no wrap-up (the last answer is the summary).

## Retries (`RetryPolicy`)

| Error class (`harness_core.errors`) | Action |
|---|---|
| `TransientModelError` | Exponential backoff with jitter, up to `max_retries` (4); then switch to `fallback_model` once if configured; else `error_during_execution` |
| `RateLimitError(retry_after=…)` | Wait `retry_after` (capped at `max_retry_after_s`, 900 s) or backoff; up to `max_rate_limit_retries` (6) |
| `ContextWindowExceededError` | No retry; compaction path |
| `TerminalModelError`, any other exception | No retry; `error_during_execution` with the message in `result.error` |

Backoff: `initial_delay_s` 1.0 × `multiplier` 2.0^(attempt−1), capped at `max_delay_s` 60, ±`jitter`
20%. Every retry is an `error` event (`attempt`, `delay_s`); the `chat` span records `retries`.
`sleep`, `rng` and `monotonic` are injectable. An interrupt during a backoff wait is honoured
before and after the sleep. Stream idle timeouts are the adapter's concern (non-streaming here).

## Stuck detection (`StuckDetector`, `StuckConfig`)

| Rule | Default | Source |
|---|---|---|
| Same call (name + canonical JSON input) N times in a row | 4 | OpenHands |
| Cycle of 2..`max_cycle_len` (3) distinct calls repeated N times | 4 | OpenHands (4 × 2) |
| Same call failing with the same error N times in a row | 3 | OpenHands action-error rule |
| Edits to one path (tools in `edit_tools`, path from `path`/`file_path`/`filename`) every N | off | LangChain edit counter |
| Nudges before termination (`max_nudges`) | 2 | ours |

A hard signal appends `DOOM_LOOP_NUDGE` to the tool-result message and resets the history (so it
takes fresh repeats to fire again). A hard signal after `max_nudges` nudges ends the run with
`error_doom_loop`: if it fires on the calls, they are not executed; if on the results, the results
are appended first. Edit-count signals only nudge. Set `LoopConfig(stuck=None)` to disable.

## Compaction hook and thrash guard (`context.Compactor`)

```python
class MyCompactor:
    def compact(self, messages, ctx, *, reason):          # reason: "threshold" | "overflow"
        return CompactionResult([Message.user(handoff_summary(messages)), *messages[-4:]], summary="...")
```

* **Proactive:** if `LoopConfig.compaction_threshold` (min of `compaction_threshold_tokens` and 90% of `context_window_tokens`) is set and the last response's `total_input + output_tokens` reached it, the compactor runs before the next call. No window and no threshold → no proactive compaction.
* **Reactive:** `stop_reason == "model_context_window_exceeded"` or `ContextWindowExceededError` → the response is discarded and the compactor runs with `reason="overflow"`, then the call is retried.
* **Validation:** the new view must pass `validate_tool_pairing` (no orphan `tool_result`, no tool batch followed by a message missing its results) and must be smaller by `estimate_tokens` (UTF-8 JSON bytes / 4); otherwise the attempt fails (`compaction` event with `failed`). Valid views are recorded with `Session.record_compaction` and `pre_compact`/`post_compact` hooks fire.
* **Thrash guard:** after `max_compaction_attempts` (3) consecutive compactions with no successful model response in between, the run ends with `error_context_window`. Without a compactor, overflow ends the run immediately.

## Failure modes

| Pitfall from the research | Mitigation here |
|---|---|
| Treating `max_budget_usd` as exact | Documented overshoot; leave headroom for one response plus the wrap-up |
| Retrying terminal errors / not retrying transient ones | Error classes decide; unknown exceptions are terminal |
| Compaction thrash on one huge output | Thrash guard; cap tool output at ingestion in the executor |
| Compacting only after overflow | Set `context_window_tokens` so the 90% threshold fires first |
| Agents repeating themselves for hours | Stuck detector nudge → terminate; consecutive-error limit |
| Headless runs stuck on denials | 3 consecutive / 20 total denials → `error_permission_denials` |
