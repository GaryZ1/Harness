# Assembler: `harness_context.assembler`

**Purpose:** `LayeredContextAssembler` implements `harness_core.types.ContextAssembler`. It turns the durable log into the model window for the next call, and walks the compaction ladder cheapest rung first.

**Research basis:** the reference report §3.7 fixes the request layout in step 2, puts determinism and immutability in code in step 3, uses one dynamic channel in step 4 and defines the compaction ladder and its guards in step 11. Durable-log projection follows step 16 and the Cline, OpenHands and Managed Agents pattern, where the window is a *projection* of an append-only event log.

## What `prepare()` does, in order

1. **Server compaction check.** If the newest assistant message contains a `compaction` block not seen before, a COMPACTION event (`server_compaction`) is emitted. The re-injection bundle is appended to the log only when that is safe (see [server-side.md](server-side.md)).
2. **Projection.** The window is `[latest compaction marker] + [non-marker messages after marker.covers_through]`, copied with ids kept and old cache markers stripped. The first time a marker is seen, the log prefix it covers is re-hashed and compared with `prefix_sha256`.
3. **L0 safety cap.** Any tool result over `max_result_tokens` (25k) is truncated deterministically in the window, with a coaching note.
4. **Sticky clearing.** Results cleared earlier are replaced by the same placeholders again.
5. **Estimate.** `counter.count_request(...)` is scaled by the session's usage calibration. Messages before a server compaction block are not counted.
6. **L2 clearing pass.** This runs when the estimate is ≥ `clearing.trigger_tokens` and the pass frees ≥ `clear_at_least`. The cleared set grows, a COMPACTION event (`tool_result_clearing`) is emitted, and the next prefix diff is marked as an *expected* rebuild.
7. **L3 auto-compaction.** This runs when the estimate is ≥ `budget.compaction_trigger`, the window is known, a compactor is configured, the session is not thrash-disabled, and the log does not end with an unanswered `tool_use`. The marker is appended to the log through `append_to_log`, the window is projected again, and the thrash guard is updated.
8. **Overflow.** `ContextOverflowError` is raised when the estimate plus `max_tokens` exceeds 95% of the window.
9. **Dynamic tail.** In order: the pre-compaction warning (once per cycle), the loop check, registered providers, the budget note, then recitation last.
10. **Breakpoints.** Priority order is system, tail, mid-history, compaction, then tools, with at most `max_breakpoints`.
11. **Request.** Consecutive user messages are merged (tool results first). `extra` = `request_extra`, plus server params if configured.
12. **Checks.** The volatile-content scan always enforces the breakpoint limit, and enforces volatile-content errors in `strict_cache` mode. The prefix is diffed against the previous call: unexpected breaks emit a CUSTOM `cache_break` event. A `ContextReport` is stored.

## API

```python
asm = LayeredContextAssembler(stack=stack, config=AssemblerConfig(...), counter=..., compactor=..., reinjection=...,
                              dynamic=..., events=sink, tracer=tracer, append_to_log=None, state_dir=None)
req = asm.prepare(system, log, tools, ctx)        # ModelRequest; may append to `log`
asm.observe_response(ctx, response)               # -> True on an unexpected cache miss
asm.compact(system, log, tools, ctx, focus="…")   # manual /compact; resets the thrash guard
asm.append_reminder(log, "nested:src/AGENTS.md", text)   # persistent, de-duplicated reminder
asm.declare_rebuild(ctx, "tool set changed at session boundary")
print(asm.report(ctx.session_id).render())        # /context-style view
```

`ContextReport.categories` gives token estimates per category: `tools`, `system`, `project`, `compaction_summary`, `history`, `tool_results` and `dynamic`. The report also lists breakpoints, actions taken, cache findings, cache breaks and layer lints.

## Defaults (`AssemblerConfig`)

| Setting | Default | Source |
|---|---|---|
| `budget` | `BudgetPolicy()`: trigger 150k, capped at 90% of the window, 95% effective window | vendor default / Codex |
| `cache` | `CachePolicy()`: 1h main, 5m sub-agents, ≤4 breakpoints | recommended / vendor limit |
| `clearing` | `ClearingPolicy()`: trigger 100k, keep 3, `clear_at_least` 5k | vendor defaults; `clear_at_least` is our choice |
| `server` | `None` (client-side ladder) | our choice |
| `auto_compact` | `True` (needs a compactor and a known window) | our choice; Cline skips unknown windows |
| `max_ineffective_compactions` | 3 | Claude Code "stops after a few attempts" |
| `max_compaction_failures` | 3 | our choice |
| `max_tokens` | 16,000 | `ModelRequest` default |
| `max_result_tokens` | 25,000 | vendor default (Claude Code) |
| `request_extra` | `{}`: pin effort and thinking here for the whole session | research: pin per session |
| `sort_tools`, `merge_consecutive_user`, `budget_note`, `pre_compaction_warning`, `stuck_detection` | `True` | our choice |
| `strict_cache` | `False` (findings reported, not raised) | our choice |
| `raise_on_overflow` | `True` | our choice |

## Invariants (enforced in code)

* **The durable log is never mutated.** The only writes are *appends* of a compaction marker, a re-injection bundle or a reminder. Each append happens only when `safe_to_append(log)` holds, meaning the log does not end with an assistant `tool_use` that has no result yet. Tests compare the log with a deep copy before and after `prepare`.
* Derived messages keep their durable ids. Thinking, redacted-thinking and compaction blocks are copied verbatim, and only the compaction block may carry a breakpoint.
* At most 4 breakpoints are sent. `BreakpointLimitError` is raised otherwise.
* Server clearing and client clearing, and server compaction and client auto-compaction, are mutually exclusive (`ConfigurationError`).
* Per-session state lives behind a re-entrant lock. Compaction is single-flight per session.

## Failure modes

| Situation | Behaviour |
|---|---|
| A summarized message was edited in the log | `LogIntegrityError` on the first projection of that marker by an assembler instance |
| Compaction fails (no summary, the model called a tool twice, truncated output) | ERROR event `compaction_failed` and the full window is sent. Auto-compaction is disabled after 3 consecutive failures |
| Compaction does not get the window below the trigger | `ineffective` count rises. After 3 in a row, auto-compaction is disabled and an ERROR event `compaction_thrash` is emitted. Calling `compact()` manually re-enables it |
| The log ends with an unanswered `tool_use` | Compaction is deferred to the next `prepare` |
| Window over 95% | `ContextOverflowError` plus an ERROR event `context_overflow` |
| Process restart | With `state_dir`, the cleared ids, guard counters, last tail and calibration are re-loaded, so placeholders stay byte-identical. The first window after the restart is not diffed |
| Volatile content in the prefix | A finding in the report. `CacheStabilityError` when `strict_cache=True` |
