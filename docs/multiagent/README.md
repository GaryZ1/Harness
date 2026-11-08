# Multi-agent orchestration (`harness_multiagent`)

Production Python implementation of the multi-agent design in
[`reports/SOTA agentic harness reference.md` §8](../../reports/SOTA%20agentic%20harness%20reference.md),
[`research_notes/.../multi_agent_orchestration.md`](../../research_notes/SOTA%20agentic%20harness%20build%20plan/multi_agent_orchestration.md)
and the build plan's "Multi-agent: many readers, one writer, scripted fan-out" section. Dependencies: stdlib, PyYAML, `harness_core.types`, `harness_memory`.

| Component | What it does | Module | Doc |
|---|---|---|---|
| **Definitions and briefs** | `.harness/agents/<name>.md` definitions, registry and routing catalog, the validated delegation brief | `definitions.py`, `brief.py` | [definitions-and-briefs.md](definitions-and-briefs.md) |
| **Spawn primitive** | `SubagentManager`: agent-as-tool with isolated context, depth/concurrency/agent caps, hooks, the `spawn_subagent` tool | `manager.py` | [subagent-manager.md](subagent-manager.md) |
| **Runner seam and returns** | `AgentRunner` protocol, `SubagentTask`, `ModelRouter`, built-in `SimpleAgentRunner`; condensed, typed, untrusted reports | `runner.py`, `result.py` | [runner-and-results.md](runner-and-results.md) |
| **Budgets, cancellation, observability** | Hierarchical token/USD ledgers, deadlines, per-agent accounting, critical path, `SUBAGENT` events, spans | `budget.py`, `cancel.py` | [budgets-and-observability.md](budgets-and-observability.md) |
| **One writer** | Write leases (readers unlimited), file ownership, scoped tool executor, git worktrees, single integrator, task board | `leases.py`, `tools.py`, `worktree.py`, `coordination.py` | [single-writer.md](single-writer.md) |
| **Scripted fan-out** | `Workflow`: `agent` / `parallel` / `pipeline` / `map_reduce` / `phase`, retries, ceilings, replay journal | `orchestration.py` | [workflows.md](workflows.md) |
| **Patterns and verification** | Orchestrator-worker, per-claim voting, dedup/rank, best-of-N, convergence loops, generator-evaluator | `patterns.py` | [patterns-and-verification.md](patterns-and-verification.md) |
| **When not to use it** | The delegation gate, effort tiers, fan-out cost warnings, lead system-prompt policy | `gate.py` | [when-not-to-use.md](when-not-to-use.md) |

## Why this shape

The research converges on a handful of rules. This package encodes each one in code, not in prompts.

1. **Single agent by default; fan out only through a gate.** Multi-agent runs gained 81% on parallelizable tasks but lost 39–70% on sequential planning, and returns turn negative above a ~45% single-agent baseline. Gains track token spend (80% of variance) at ~4–20x tokens. `should_delegate` encodes the checklist rule and logs its reasons.
2. **One primitive: agent-as-tool with a self-contained brief.** "The only content you pass from parent to subagent is the Agent tool's prompt string." Fresh subagents get their own system prompt plus the rendered brief, never the parent's history or memory. Briefs are validated, and they can only narrow privileges.
3. **Bounded, typed, untrusted returns.** Only a schema-validated report of at most ~2k tokens comes back (5 repair retries, then `error_max_structured_output_retries`). It is labelled untrusted, scanned for approval claims, and carries no authority.
4. **The harness enforces caps, not the prompt.** Depth 3, 20 concurrent subagents, 16 concurrent workflow agents, 1,000 agents and 4,096 items per run, and a mandatory run budget that counts subagent spend. A cap refuses the spawn with a typed `refusal_code`. A spent run budget also cancels every running agent (`error_max_budget`).
5. **Many readers, one writer.** Readers get no mutating tools, at offer time or at execution time. Writers need an exclusive lease. Parallel writers need disjoint file ownership or their own git worktree, plus a single integrator that runs a test oracle before merging. ("Two teammates editing the same file leads to overwrites.")
6. **Code controls the fan-out.** Deterministic ordering, partial failure as `None`, bounded restarts and replay by prompt hash. Every fan-out ends in verification and a single merge step, because independent agents without a checker amplified errors 17.2x.
7. **Observe structure, not content.** `SUBAGENT` events, one `invoke_agent` span per agent, per-agent token/turn/tool-call/wall-clock accounting, and the critical-path length to judge whether a fan-out actually cut latency.

## Architecture

```text
 lead agent (depth 0) ── tool call spawn_subagent{agent, objective, inputs, scope...}
        │
        ▼
 SubagentManager.spawn ── gate order: cancelled/budget → depth ≤ 3 → can_spawn → start hooks/policy
        │                              → agents/run ≤ 1000 → slot (≤ 20) → write lease / ownership / worktree
        │  builds SubagentTask: cache-stable system prompt + rendered brief (ONLY context)
        │                       ScopedToolExecutor (allowlist, read-only filter, lease + ownership checks)
        │                       child BudgetLedger (per-agent tokens → run ledger) + child CancellationToken (timeout)
        ▼
 AgentRunner.run(task, ctx)  ◀── integrator's real loop, or SimpleAgentRunner(ModelRouter role→model id)
        │      after every response: task.charge(usage) ; task.stop_reason()
        ▼
 SubagentResult{status, report ≤ 2k tokens, usage, turns, tool_calls, wall_clock}
        │──▶ ToolOutcome {"type":"subagent_report","trust":"untrusted",...}   (to the lead's context)
        │──▶ Event(SUBAGENT, start/stop/refused) ; tracer span "invoke_agent <name>" ; RunAccounting
        ▼
 Workflow (code-driven fan-out, 16 concurrent) ── parallel / pipeline / map_reduce ── WorkflowJournal (replay)
        └─ plan → dispatch (readers) → verify (fresh-context checker / votes) → synthesize (single merge)

 writers: WORKSPACE (exclusive lease) | OWN_FILES (disjoint claims + per-path leases) | OWN_WORKTREE
          (git worktree per writer) ──▶ Integrator (workspace lease, ownership check, test oracle, --no-ff merge)
```

## Quick start

```python
from harness_core.types import RunContext
from harness_multiagent import (AgentRegistry, DelegationBrief, ManagerConfig, ModelRouter,
                                SimpleAgentRunner, SubagentManager, Workflow)

registry = AgentRegistry.from_project(".")                 # .harness/agents/*.md, then ~/.harness/agents
router = ModelRouter(config["model_roles"], client)        # e.g. {"lead": ..., "worker": ...}; ids from config only
manager = SubagentManager(registry, SimpleAgentRunner(router), tools=my_tool_executor,
                          config=ManagerConfig(run_max_tokens=1_500_000), event_sink=event_log)

# 1) Model-driven: give the lead the spawn tool (manager is a ToolExecutor).
lead_tools = [*my_tool_executor.specs(), *manager.specs()]
outcome = manager.execute(tool_call, RunContext(session_id="s1", cwd="."))   # condensed, untrusted JSON

# 2) Code-driven: a scripted fan-out that ends in verification and one merge.
wf = Workflow(manager, RunContext(session_id="s1", cwd="."))
reports = wf.pipeline(files, lambda f: wf.agent("code-explorer", DelegationBrief(f"Audit {f} for SQL injection", inputs=(f,))))
print(manager.accounting.summary())
```

A definition file (`.harness/agents/code-explorer.md`):

```markdown
---
name: code-explorer
description: Read-only codebase search; use for "where is X / how does Y work" questions.
tools: [read_file, grep, glob]
model: worker          # a ROLE; the caller maps roles to model ids
max_turns: 20
---
You are a fast, read-only code explorer. Cite file:line for every claim.
```

Run the tests with `python3 -m pytest -p no:cacheprovider tests/multiagent`.

## Integration notes

* **The runner.** The integrator implements `AgentRunner.run(task, ctx)` on top of the real `harness_core` loop. It must: use `task.system_prompt`, `task.initial_messages()`, `task.tools.specs()` and `task.request_extra(model_id)`; execute tools only through `task.tools.execute`; call `task.charge(usage, model)` after every response and stop when `task.stop_reason()` is set; respect `task.max_turns` and `task.max_tool_calls`; and return a `SubagentResult` whose report comes from `parse_report`. See [runner-and-results.md](runner-and-results.md).
* **The model id convention.** The resolved id travels in `ModelRequest.extra["model"]`, because `ModelRequest` has no model field. Effort travels in `extra["effort"]`.
* **The lead's prompt.** When the spawn tool is offered, add `harness_multiagent.DELEGATION_POLICY` to the lead's system prompt (checklist item 5).
