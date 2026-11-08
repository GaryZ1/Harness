# Patterns and verification: `harness_multiagent.patterns`

## Purpose

These are reusable building blocks for the patterns that have production evidence: orchestrator-worker fan-out, verification and voting, deduplication and ranking, best-of-N selection, convergence loops and the generator-evaluator gate. All of them are deterministic code around agents. None of them puts the model in charge of the fan-out.

## Research basis

* **Orchestrator-worker** (Anthropic research system): the lead plans, 3–5 subagents run in parallel with explicit briefs, results are synthesised, then citations are checked. The system beat single-agent Opus 4 by 90.2% on breadth research (§8.3).
* **Always end fan-out in verification, then merge.** Use adversarial cross-review or per-claim voting, then a rank/dedup step. "Label uncheckable claims 'unverified,' not 'refuted.'" Independent agents without a checker amplified errors **17.2x** (item 8).
* **Homogeneous debate** does no better than majority voting at 2.1–3.4x the tokens, so voting is offered and debate is not.
* **Best-of-N without a good selector:** equal coverage does not mean an equal final score (CodeMonkeys). Use an executable verifier first and an LLM judge only as tie-breaker (item 20).
* **Self-evaluation** is a known failure mode. The evaluator is a separate, fresh-context agent with hard thresholds (item 9). Workflow loops stop when "two rounds in a row find nothing new".

## API

```python
from harness_multiagent import (OrchestratorWorker, planner_from_agent, agent_verifier, report_verdict, vote,
                                vote_on_claims, dedup_findings, rank_findings, best_of_n, iterate_until_converged,
                                generate_evaluate, Verdict)

ow = OrchestratorWorker(
    workflow=wf, worker="code-explorer",
    plan=planner_from_agent(wf, "planner", max_subtasks=5),   # or any code: task -> [DelegationBrief]
    verify=agent_verifier(wf, "skeptic", accept=0.7, reject=0.3),   # or code: (brief, report) -> Verdict|bool|None
    synthesize=lambda task, kept: merge_agent_or_code(kept),          # the single merge step
    max_workers=5,
)
res = ow.run("Security audit of src/")   # OrchestrationResult(plan, reports, kept, refuted, failed, final)

vote([True, None, False, True])                 # Consensus(verdict, verified, refuted, abstained).agreement
vote_on_claims(findings, [check_a, check_b])     # per-claim ClaimVote; a raising verifier abstains
dedup_findings(findings); rank_findings(findings)
best_of_n(patches, verifier=run_tests, judge=llm_pick)   # Selection(index, candidate, scores, method)
iterate_until_converged(step, max_rounds=10, patience=2) # ConvergenceResult(items, rounds, converged)
generate_evaluate(generate, evaluate, max_rounds=3)      # GenerateEvaluateResult(output, passed, rounds, feedback)
```

`planner_from_agent` turns each finding of the planner's report into one worker brief: `claim` becomes the objective, `evidence` the why-context. It lists each sibling's objective under `out_of_scope`, which stops siblings duplicating each other's work. `agent_verifier` gives a fresh-context checker only the worker's summary, claims and evidence, and maps the checker's report `confidence` to a verdict.

## Defaults

| Helper | Default | Source |
|---|---|---|
| `vote` quorum / `min_decisive` | > 50% of decisive votes / 1 | Majority voting |
| `agent_verifier` accept / reject | ≥ 0.7 verified, ≤ 0.3 refuted, else unverified | Ours |
| `planner_from_agent.max_subtasks` | 5 | 3–5 workers (Anthropic) |
| `iterate_until_converged.patience` | 2 quiet rounds | "two rounds in a row" (workflows docs) |
| `generate_evaluate.max_rounds` | 3 | Ours |
| `best_of_n` tie without judge | Lowest index (`first_of_tied`) | Deterministic |

## Invariants

* `None`, abstentions, ties and too few decisive votes give `UNVERIFIED`, never `REFUTED`.
* `OrchestratorWorker` never concatenates worker output into the answer. Refuted items are dropped, and `synthesize` receives labelled `VerifiedItem`s and runs once. If nothing survives, `final` is `None`.
* `dedup_findings` keeps first-occurrence order, the highest confidence, the union of sources and the longest evidence.
* `best_of_n` asks the judge only about tied top candidates, and only after the verifier.

## Failure modes

| Failure | Behaviour |
|---|---|
| Planner fails | `planner_from_agent` uses `required=True`, so `WorkflowStepFailed` ends the run (a directly awaited step) |
| Worker fails | Its index lands in `failed`; the others proceed |
| Verifier agent fails | Verdict `UNVERIFIED` (the item is kept and labelled) |
| Judge returns an out-of-range index | `ValueError` |
| No candidate passes the verifier | `Selection(index=None, method="none_passed")` |
