# Spawn primitive: `harness_multiagent.manager`

## Purpose

`SubagentManager` implements the single delegation primitive: **agent-as-tool**. A caller (the lead model via the `spawn_subagent` tool, or code via `spawn`) names a definition and passes a brief. The manager admits the spawn through harness-enforced caps, prepares an isolated `SubagentTask`, runs it through an `AgentRunner` on a worker thread bounded by a deadline, and returns a condensed `SubagentResult`.

## Research basis

* "Implement one subagent primitive: agent-as-tool with a self-contained prompt. The parent passes only a brief string; only the final message returns" (§8.7 item 2). The OpenAI Agents SDK's `Agent.as_tool()` is the same idea: the manager owns the answer.
* "Enforce caps in the harness, not the prompt": depth 3, 20 concurrent subagents, a mandatory run budget, per-agent `max_turns`. "On cap: refuse new spawns ('Budget limit reached'), stop background agents, and end with a typed terminal status" (item 4).
* "Treat a spawn as a privileged action" and route it through policy (item 18). Hooks follow Claude Code's `SubagentStart` / `SubagentStop`.
* Siblings get byte-identical system prompt, tools, model and effort, so they share the prompt-cache prefix (item 14).

## API

```python
mgr = SubagentManager(
    registry, runner,                       # AgentRegistry, AgentRunner
    tools=parent_tool_executor,             # the pool subagents' tools are selected from
    config=ManagerConfig(...),
    leases=LeaseManager(lock_dir=...),      # optional cross-process leases
    worktrees=WorktreeManager(repo),        # required for own_worktree writers
    memory_store=project_store,             # subagents get a read-only view by default
    event_sink=sink, tracer=tracer, cost_fn=lambda model, usage: usd,
    start_hooks=[policy], stop_hooks=[gate],
)
res = mgr.spawn("code-explorer", brief, ctx)                 # -> SubagentResult (never raises on caps)
res = mgr.spawn("forker", brief, ctx, history=messages)      # context: fork
results = mgr.spawn_many([("a", b1), ("b", b2)], ctx)        # concurrent, results in request order
mgr.specs(); mgr.execute(tool_call, ctx)                     # ToolExecutor protocol: the spawn tool
mgr.cancel_all(); mgr.running; mgr.accounting; mgr.ledger; mgr.worktree(run_id)
```

`StartHook = (definition, brief, parent_ctx) -> deny_reason | None` and `StopHook = (result) -> feedback | None`. Stop-hook feedback is attached to `result.warnings` and `result.metadata["stop_hook_feedback"]`.

### Admission order and refusal codes

| # | Check | `refusal_code` |
|---|---|---|
| 1 | Run (or parent subagent) cancelled; run/parent budget spent | `cancelled` / `budget` |
| 2 | `ctx.depth + 1 <= max_depth` | `depth_limit` |
| 3 | A subagent caller must have `can_spawn: true` | `nesting_disabled` |
| 4 | Start hooks / spawn policy | `policy` |
| 5 | Agents admitted this run `< max_agents_per_run` | `agent_limit` |
| 6 | Concurrency slot (waits while `concurrency_wait_s` allows; cancellable) | `concurrency_limit` / `cancelled` |
| 7 | Writers: workspace lease / ownership claim / worktree | `lease_held` / `ownership` / `no_worktree` / `worktree` |

A refusal returns `SubagentResult(status=REFUSED, refusal_code=..., error=...)` and emits a `refused` event. Unknown agents and invalid briefs raise from `spawn`. The spawn tool turns them into `is_error` tool results.

### What the subagent receives (fresh context)

* **System prompt:** `compose_system_prompt(...)`, which combines the definition's prompt, the subagent preamble (isolated context, no user, read-only or writer rules, delegation rule), the optional `ManagerConfig.shared_context` (project instructions or a repo-status snapshot), the sibling roster and `REPORT_INSTRUCTIONS`. It contains no run ids, timestamps or brief text, so it is byte-identical across siblings.
* **Messages:** the rendered brief only (`fork` prepends the parent history).
* **Tools:** `select_tools(...)`, plus `spawn_subagent` only if `can_spawn` and `depth < max_depth`, all wrapped in a `ScopedToolExecutor`.
* **`RunContext`:** `depth + 1`, `agent_name`, `cwd` (the worktree for `own_worktree` writers), and `metadata` = `subagent_run_id`, `parent_run_id`, `subagent_definition`, `model_role`, `write_access`, `memory_access`, `parent_tool_use_id`. Taint is inherited from the parent.

## Defaults (`ManagerConfig`)

| Setting | Default | Vendor default | Note |
|---|---|---|---|
| `max_depth` | 3 | 3 (`CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH`) | `1` disables nesting |
| `max_concurrent` | 20 | 20 (`CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS`) | |
| `max_agents_per_run` | 1,000 | 1,000 per workflow run | Ours applies to the whole manager run |
| `run_max_tokens` | 2,000,000 | no limit | **Ours**: the run budget is mandatory (pitfall "no hard caps") |
| `run_max_usd` | `None` | no limit | Needs `cost_fn` |
| `default_agent_max_tokens` | 200,000 | no limit | Ours |
| `default_timeout_s` | 600 | none | Ours |
| `result_max_tokens` | 2,000 | ~2k guidance | Hard cap for briefs' `output_max_tokens` |
| `concurrency_wait_s` | `None` (wait) | queue | `0` refuses immediately |
| `lease_wait_s` | 0 | — | A second writer is refused, not queued |
| `join_grace_s` | 5 | — | Time a cancelled runner gets before it is abandoned |
| `spawn_tool_name` | `spawn_subagent` | `Agent` (Claude Code) | Configurable |
| `include_roster` | `True` | roster included | |

## Invariants

* No model call happens before every admission check passes. Leases, ownership claims and slots are released in `finally`, on every path including runner exceptions.
* A subagent's spend rolls into its parent subagent's ledger (nested) and into the run ledger, so a parent's per-agent budget covers its children.
* A run-budget hit cancels the run token, which cancels every running agent (`error_max_budget`), and every later spawn is refused with `budget`.
* Results are post-processed: identity fields are re-stamped, reports are scanned for approval claims and condensed to `output_max_tokens`, and `success` without a report is downgraded to `error_during_execution`.
* Events carry structure, not content: the objective appears only as a 12-hex `objective_sha`.

## Failure modes

| Failure | Result |
|---|---|
| Runner raises | `error_during_execution` with the exception text (never delivered as a report) |
| Model call hangs past the deadline | Token cancelled. After `join_grace_s` the runner thread is abandoned and the result is `error_timeout` |
| Event sink raises | `RuntimeWarning`; the run continues |
| `inherit` role with no `model_role` in the parent ctx | `error_during_execution` |
| `own_worktree` without a `WorktreeManager` | Refused with `no_worktree` (the agent never writes to the shared checkout) |

Spawns are synchronous; for concurrency use `spawn_many` or a [`Workflow`](workflows.md). Claude Code's background "async_launched" mode is not modelled.
