# Budgets, cancellation and observability: `harness_multiagent.budget` + `harness_multiagent.cancel`

## Purpose

Multi-agent runs are expensive, and they fail in ways you cannot see. This page covers the code that bounds the cost (hierarchical ledgers), bounds the time (hierarchical cancellation tokens with deadlines), and makes the agent tree visible (`SUBAGENT` events, spans, per-agent accounting, critical path).

## Research basis

* Token usage explained **80%** of BrowseComp variance. Multi-agent systems used ~15x chat tokens. "Treating multi-agent as free quality" is the first pitfall, so "a mandatory run-level `max_usd` that counts subagent spend, with headroom because checks happen after each response" is required (§8.7 item 4).
* The Agent SDK budget is "checked *after* each model response", so spend "can therefore pass the cap by up to the cost of that one response", plus whatever running subagents spend (§8.6).
* "Trace the tree": one `invoke_agent` span per agent linked to the parent tool call. Record per agent tokens (input, output, cache read, cache write), cost, turns, tool calls and wall clock. Compute **critical-path steps** to judge whether a fan-out cut latency. "Monitor structure, not content, by default" (item 19).

## API

```python
from harness_multiagent import BudgetLedger, BudgetLimits, CancellationToken, RunAccounting

run = BudgetLedger("run", BudgetLimits(max_tokens=2_000_000, max_usd=20.0), cost_fn=price, on_exceeded=print)
agent = run.child("explorer:sub_1", BudgetLimits(max_tokens=200_000), on_exceeded=lambda r: token.cancel("budget", r))
agent.charge(response.usage, model=model_id)     # propagates to every ancestor
agent.exceeded(); agent.remaining_tokens(); run.tokens; run.usd

root = CancellationToken()                        # or CancellationToken.with_timeout(3600)
token = root.child(timeout_s=600)                 # deadline = min(own, parent's)
token.raise_if_cancelled()                        # Cancelled / AgentTimeout
root.cancel("cancelled", "user pressed stop")     # cascades to every descendant

mgr.accounting.by_agent()      # {name: AgentTotals(runs, usage, usd, turns, tool_calls, wall_clock_s, failures)}
mgr.accounting.critical_path() # CriticalPath(run_ids, steps, total_steps).parallel_speedup
mgr.accounting.summary()       # JSON-ready dict
```

Budget tokens = `input + cache_read + cache_creation + output` (`usage_tokens`). Dollars come only from a caller-supplied `cost_fn(model_id, usage)`; the package knows no prices.

## Events and spans

Every spawn emits `Event(session_id, EventType.SUBAGENT, data)` to the configured `EventSink`:

| `data.phase` | Fields |
|---|---|
| `start` | `agent, run_id, parent_run_id, depth, parent_tool_use_id, model_role, write_access, tools, objective_sha, workspace` |
| `stop` | `agent, run_id, parent_run_id, depth, parent_tool_use_id, status, model, usage{input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens}, usd, turns, tool_calls, wall_clock_s, warnings (count), error (≤ 200 chars)` |
| `refused` | `agent, run_id, parent_run_id, depth, parent_tool_use_id, code, error` |
| `workflow_phase` | `workflow, name, state (start/end)` |
| `workflow_log` | `workflow, message` |

The manager opens a tracer span `invoke_agent <name>` (attributes `agent, run_id, depth, parent_run_id, parent_tool_use_id`) around each run. Workflows open `workflow_phase <name>`. `parent_tool_use_id` links the subagent to the lead's tool call.

## Defaults

| Setting | Value | Vendor | Note |
|---|---|---|---|
| Run token budget | 2,000,000 | no limit | **Ours** (mandatory budget) |
| Per-agent token budget | 200,000 | no limit | Ours |
| Per-agent timeout | 600 s | none | Ours |
| Budget check point | After each model response | Same | Overshoot bounded (below) |

## Invariants

* A charge propagates to every ancestor. `on_exceeded` fires **once** per ledger, at the first charge that reaches its cap.
* The manager wires the run ledger's `on_exceeded` to cancel the run token and each agent ledger's to cancel that agent's token. Overshoot is therefore bounded by one in-flight response per running agent.
* A child token's deadline is never later than its parent's. Cancelling a parent cancels all descendants, including children created after the cancel.
* Accounting records only finished (not refused) runs. The critical path is the maximum, over root-to-leaf chains of parent links, of the summed turns.

## Failure modes

* **A hung model call cannot be interrupted cooperatively.** The manager runs each runner on a worker thread, waits until the token is cancelled or the deadline passes, gives it `join_grace_s` (5 s), then abandons the thread and reports `error_timeout` with the ledger's usage.
* **A runner that never charges** loses mid-run enforcement. The manager charges its reported usage at the end.
* **Event sink errors** are reported as a `RuntimeWarning` and never break the run.
