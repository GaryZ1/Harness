# When not to go multi-agent: `harness_multiagent.gate`

## Purpose

This module encodes the decision rule "default to one agent; fan out only through a gate" as a deterministic function whose reasons can be logged per run. It also provides the lead's delegation policy text, effort tiers and fan-out cost warnings.

## Research basis

* Controlled study (arXiv 2512.08296, 260 configurations): +81% on parallelizable finance tasks, **−39% to −70%** on sequential PlanCraft. Returns diminish or turn negative once the single-agent baseline exceeds **~45%**.
* Token spend explained 80% of variance, and multi-agent runs cost ~4–20x tokens (Anthropic). At equal token budgets, single agents match or beat MAS on multi-hop reasoning (arXiv 2604.02460).
* Agents that differ only in instructions compile into one agent plus skills with 54% fewer tokens, until selection degrades beyond ~50–100 skills (arXiv 2601.04748).
* A fresh-context reviewer catches real bugs without a multi-agent build (Cognition's Devin Review: ~2 bugs per PR).
* Over-delegation by newer models: "Claude Opus 5 delegates to subagents more readily", so a policy line is needed (§8.7 item 5).

## API

```python
from harness_multiagent import TaskProfile, should_delegate, effort_tier, fanout_warnings, DELEGATION_POLICY

decision = should_delegate(TaskProfile(
    subtasks_independent=True, subtask_count=4, outputs_mergeable=True,
    single_agent_baseline=0.30, context_degraded=False, value_justifies_cost=True,
    sequential_planning=False, tightly_coupled_edits=False, agents_differ_only_in_instructions=False,
    produces_code_change=False,
))
log.info("delegation gate", extra=decision.to_dict())    # go_multi_agent, recommendation, reasons, suggested_agents
effort_tier("comparison")      # EffortTier(min_agents=2, max_agents=4, tool_calls_per_agent=(10, 15))
fanout_warnings(30, 2_000_000) # two warnings
system_prompt = BASE + "\n\n" + DELEGATION_POLICY
```

## Decision rule

```text
sequential_planning or tightly_coupled_edits   -> single agent (+ reviewer if it produces code)
agents_differ_only_in_instructions             -> single agent + skills (keep under ~50)
go_multi_agent = subtasks_independent and subtask_count >= 2
                 and outputs_mergeable
                 and (single_agent_baseline < 0.45 or context_degraded)
                 and value_justifies_cost
```

An unknown baseline counts as "not shown weak", so the gate stays single-agent unless the context is degraded.

## Defaults

| Constant | Value | Source |
|---|---|---|
| `BASELINE_SATURATION` | 0.45 | arXiv 2512.08296 |
| `WARN_AGENTS` / `WARN_TOKENS` | 25 / 1,500,000 | Claude Code workflow cost guard |
| `MAX_SKILLS` | 50 | arXiv 2601.04748 |
| Effort tiers | simple 1 agent / 3–10 calls; comparison 2–4 / 10–15; complex 5–20 / 10–15 | Anthropic (">10 only for complex research"); our upper bound 20 = concurrency cap |

## Invariants

* The gate is pure and deterministic. It always returns its reasons, so the result can be logged per run (checklist item 1).
* Disqualifiers (sequential or coupled work, instruction-only differences) short-circuit before the positive checks.
* `suggested_agents` never exceeds the tier's `max_agents`.

## Failure modes and limits

The gate is only as good as its inputs. `single_agent_baseline` should come from your own evals, compared at **equal token budget** (item 21). It is a heuristic helper: the manager does not call it automatically, because the lead model or calling code decides *whether* to fan out. Hard caps still apply when the gate is bypassed.
