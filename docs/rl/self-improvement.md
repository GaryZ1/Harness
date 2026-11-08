# Self-improving harness loop: `harness_rl.improve`

## Purpose

The loop lets an optimizer (a GEPA-style prompt mutator, an ACE-style playbook curator, a tool-description refactorer or a scaffold ablator) propose harness config changes. A change is accepted only when held-out evidence clears the noise band without regressions, budget overruns or reward-hacking signals. Every decision is audited and reversible.

## Research basis

* **GEPA** reflects on traces and keeps a Pareto frontier. It beats GRPO by ~6% on average with up to 35× fewer rollouts, though not uniformly (AIME-2025: GRPO 38% vs GEPA 32%).
* **ACE** makes itemised playbook deltas, which avoids context collapse.
* **DGM** self-modifies code (SWE-bench 20% → 50%) but risks sandbox escape and reward hacking of the eval.
* **Anthropic's tool loop** is validated on held-out sets (§11.4).
* **The recommended composition** (§11.4, §11.7 step 9): run the capability suite, cluster failures, propose diffs, and accept only if the held-out gain exceeds the noise band (≥ 3-5 trials; single runs swing 2-6 points), the regression suite does not lose, and cost and latency stay in budget. Log every accepted diff with its eval delta and harness SHA.
* **Pitfalls** (§9.8, §11.8):
  * accepting changes on single runs;
  * optimising against the test set;
  * running DGM-style self-modification in production;
  * running self-improvement before graders are audited, because it will optimise grader bugs.

## API

```python
from harness_rl import (ImprovementLoop, ConfigStore, AuditLog, GateConfig, ConfigChange,
                        CallableProposer, AblationProposer, EvalSuiteEvaluator, CallableEvaluator, SplitResult)

store = ConfigStore(".harness/config"); store.init(baseline_dict)
loop = ImprovementLoop(store=store, evaluator=evaluator, proposer=proposer,
                       splits={"train": train_ids, "heldout": heldout_ids, "regression": regression_ids},
                       audit=AuditLog(".harness/config/audit.jsonl"), exposure=ExposureLedger(".harness/evals/exposure.jsonl"),
                       gate=GateConfig(), mutable=["prompts", "tools", "playbook", "profile.scaffolds"])
rep = loop.step()            # IterationReport(baseline_version, decisions[CandidateDecision], accepted_version)
loop.rollback(3, reason="prod incident #812"); loop.revert_last(reason="…")
loop.audit.verify()          # raises ConfigStoreError if any entry was edited, removed or reordered
```

* **Proposer**: `propose(config: HarnessConfig, evidence: ProposalEvidence) -> [ConfigChange]`. The evidence holds only train-split means and failures.
* **ConfigChange**: `ConfigChange(description, set={dotted.key: value}, unset=(...), kind="config"|"prompt"|"tool_description"|"playbook"|"profile"|"code")`.
* **Evaluator**: `evaluate(config, task_ids, *, split, trials, seed) -> SplitResult(per_task={id: [trial outcomes]}, guards={cost_per_success_usd, p95_wall_s, hack_rate}, failures=[…])`. `EvalSuiteEvaluator` implements it over `harness_evals.EvalRunner`, suffixing the harness version with the config hash. `CallableEvaluator` wraps a function.
* **AblationProposer**: proposes turning off each enabled `profile.scaffolds` flag.

## One iteration

1. Check the held-out budget: when `heldout_uses() >= max_heldout_uses`, raise `HeldOutExhaustedError` (rotate the split).
2. Evaluate the baseline on **train**. Withhold any held-out failures from the proposer (audited) and record a `proposal` exposure.
3. Gate each proposal (at most `max_candidates`):
   * `kind="code"` becomes `pending_review`;
   * protected or non-mutable keys are rejected;
   * no-op and duplicate changes are rejected;
   * the candidate is **screened** on train and must beat the baseline there.
4. Evaluate the baseline and the survivors on **held-out** and **regression** with the same seed (common random numbers). Record an `acceptance` exposure and a `heldout_evaluated` audit entry.
5. Apply the per-candidate gates:
   * enough trials;
   * a significant `paired_compare` (CI low > 0 and delta ≥ `min_effect`) at confidence 1 − α/n (Bonferroni);
   * regression mean ≥ baseline − tolerance;
   * cost per success and p95 latency within the relative budget plus absolute slack;
   * the hack-signal rate not increased.
6. Commit the best eligible candidate (largest held-out delta) as version N+1. Other eligible candidates are rejected as superseded. Every decision goes to the audit log.

## Defaults

| Setting | Default | Vendor default vs our choice |
|---|---|---|
| `trials` / `min_trials` | 3 / 3 | Checklist: ≥ 3-5 trials |
| `min_effect` | 0.05 | Research noise band (2-6 points single-run; ~5 on ~89 tasks) |
| `confidence`, Bonferroni | 0.95, on | Our choice |
| `require_train_gain` | `True` | Our choice (cheap screen before spending held-out) |
| `regression_tolerance` | 0.0 | Checklist: "no regression-suite loss" |
| `max_cost_increase` / `max_latency_increase` | +10% / +10% | Our choice ("within budget") |
| `latency_abs_slack_s` / `cost_abs_slack_usd` | 1.0 s / 0.0 | Our choice (sub-second p95 jitter must not block) |
| `max_hack_rate_increase` | 0.0 | Our choice |
| `max_heldout_uses` | 20 | Our choice (adaptive-overfitting budget) |
| `max_candidates` | 4 | Our choice |
| Protected prefixes | `graders, eval, evals, tests, suite, rewards, reward, verifier, judge` | Our choice |

## Invariants

* Train and held-out are disjoint, and regression and held-out are disjoint. Held-out tasks exposed to any optimizer make construction fail (`ContaminationError`).
* The proposer never sees held-out results. The evaluator must return exactly the requested task ids.
* Config versions are immutable files with a stored hash (corruption is detected). `HEAD` is an atomic pointer, and a commit fails if HEAD moved since the iteration began.
* The audit log is append-only and hash-chained: `seq`, `prev` and `hash` over the canonical JSON.
* Code changes are never auto-accepted.

## Failure modes

* **Nothing ever gets accepted.** Effects are below the noise band. Add tasks or trials (power), or propose bigger changes.
* **`HeldOutExhaustedError`.** Rotate with `SplitManager.rotated(new_salt)` and start a fresh exposure ledger for the new held-out set.
* **A candidate rejected for the hack-signal rate.** Read the flagged transcripts (`harness_evals.sample_for_review` puts them first). The proposer probably found a grader weakness. Fix the grader before continuing.
* **A post-acceptance production regression.** Run `revert_last(reason=…)`. The audit trail shows which change and which eval delta justified it.
