# Scripted fan-out: `harness_multiagent.orchestration`

## Purpose

`Workflow` lets **code**, not the model, run a large fan-out: which agents run, how many at once, in what order results come back, how failures are retried, and when the run stops. Only the final merged answer needs to reach the lead's context.

## Research basis

* "Build a deterministic workflow runner for large fan-outs (the hybrid model: the LLM decides *whether*, code runs *how*)" (§8.7 item 7). Claude Code dynamic workflows expose `agent(prompt, {schema, label, stallMs, model})`, `parallel()`, `pipeline(items, fn)`, `phase()` and `log()`. A failed `agent()` yields `null` and the run continues. Stalled agents restart up to 5 times. Completed agents replay from cache on relaunch.
* OpenAI: orchestrating "via code makes tasks more deterministic and predictable, in terms of speed, cost and performance".
* "Nondeterministic orchestration scripts": time and randomness break replay and caching. "Cache-hostile fan-out": siblings launched before the first response starts get no cache reads.
* "Always end fan-out in verification, then merge" (item 8). The 17.2x error amplification comes from independent agents without a checker.

## API

```python
wf = Workflow(manager, ctx, config=WorkflowConfig(), journal=WorkflowJournal(".harness/runs/audit.jsonl"), name="audit")

rep = wf.agent("code-explorer", DelegationBrief(f"Audit {f}"), label=f"review:{f}")   # SubagentReport | None
rep = wf.agent("planner", brief, required=True)       # failure raises WorkflowStepFailed: the run ends
res = wf.run_agent("x", brief)                         # full SubagentResult (no journal)

outs = wf.parallel([lambda: ..., lambda: ...])         # input order; a raising thunk -> None
outs = wf.pipeline(files, review_stage, verify_stage)  # each item through the stages; None stops that item
mr   = wf.map_reduce(files, map_fn, reduce_fn, verify_fn=check)   # MapReduceResult(mapped, kept, refuted, failed, reduced)
with wf.phase("verify"): ...                            # SUBAGENT workflow_phase events + span
wf.log("note"); wf.warn_if_large(n_agents, projected_tokens)
wf.stats   # agents_spawned, cache_hits, failures[StepFailure], warnings, log
wf.results # label -> [SubagentResult, ...] (every attempt)
```

`map_reduce` drops only items whose verdict is `REFUTED`. Uncheckable items are kept and labelled `UNVERIFIED`. The reducer receives `VerifiedItem(index, value, verdict)`, never raw concatenated text.

## Defaults (`WorkflowConfig`)

| Setting | Default | Vendor source |
|---|---|---|
| `max_concurrent` | 16 (1–256) | `CLAUDE_CODE_WORKFLOW_MAX_CONCURRENT_AGENTS` |
| `max_items` | 4,096 per `parallel`/`pipeline` | Workflow limits |
| `max_agents` | 1,000 per workflow | Workflow limits |
| `max_restarts` | 5 | Stall auto-restart |
| `prefix_stagger_s` | 5.0 | `CLAUDE_CODE_WORKFLOW_PREFIX_STAGGER_MS` (ours ends the wait early when the first sibling finishes) |
| `warn_agents` / `warn_tokens` | 25 / 1,500,000 | Workflow cost guard |

## Invariants

* **Deterministic ordering.** Results follow input order whatever the completion order.
* **Partial failure.** Ordinary exceptions and failed agents become `None`, recorded in `stats.failures`. `WorkflowLimitError` and `WorkflowStepFailed` propagate, and when several thunks raise, the lowest index wins.
* **Restart policy.** Only `error_timeout` and `error_during_execution` restart. Budget, cancellation, refusals, `error_max_turns` and structured-output failures never restart.
* **Concurrency.** At most `max_concurrent` agents run per workflow (semaphore), nested inside the manager's own cap of 20.
* **Replay.** The journal key is `sha256(agent, definition fingerprint, rendered brief, output cap, tools_allowed, write_permission)`, and only successes are stored. A re-run returns cached reports for unchanged prompts and re-runs changed or failed ones. The journal is append-only JSONL written under a file lock. A torn final line is ignored.
* **Cancellation.** Once the manager's run token is cancelled (including by a run-budget hit), pending thunks are skipped and agents are not restarted.

## Writing deterministic scripts

Pass inputs as arguments: no wall-clock time, no unseeded randomness, no reading files whose content changes between runs. Python cannot make `time.time()` throw the way Claude Code makes `Date.now()` throw, so this rule is a convention. Replay correctness depends on it.

## Failure modes

| Failure | Behaviour |
|---|---|
| More than `max_items` thunks | `WorkflowLimitError` before anything runs |
| `max_agents` reached | `WorkflowLimitError` (fatal; propagates out of `parallel`) |
| Fan-out > `warn_agents` | Warning in `stats.warnings` and a `workflow_log` event; the run continues |
| Process crash mid-run | Re-run the same script with the same journal: completed agents replay |

The replay cache is per key, not Claude Code's "replay until the first changed prompt". With deterministic scripts the two give the same result, and per-key replay also survives reordering.
