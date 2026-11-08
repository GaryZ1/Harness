# Metrics, result store and reports: `harness_evals.metrics`, `.store`, `.report`

## Purpose

This page covers the statistics for reliability and harness A/B decisions, the durable record of every trial, and reports that carry their own reporting conditions.

## Research basis

* **pass@k vs pass^k** (Anthropic 2026; Sierra): pass@k = P(≥1 of k succeeds), pass^k = P(all k succeed). With p = 0.75, pass^3 ≈ 42%. GPT-4o's retail agent fell to ~25% at pass^8. Report pass^k with k = 3-8 for user-facing agents. Never compare pass@k against someone else's pass@1 (§9.7 step 8).
* **Noise.** Single-run pass@1 moves 2.2-6.0 points. On ~89 tasks, deltas under ~5 points are noise. Report mean ± SE and say whether the SE is across runs or across tasks (§9.7 step 7; §9.8).
* **Strict vs partial.** OSWorld 2.0 is 81.8% partial vs 48.7% strict.
* **Never merge capability and regression suites.**
* **Harness A/B protocol** (§9.7 step 11): hold the model, effort and suite version fixed, run the same tasks and trial counts, and accept only with a capability gain, no regression, and cost/latency within budget.
* **Efficiency** (§9.7 step 9): cost per success and p50/p95 latency are gates.

## Estimators

| Function | Formula | Notes |
|---|---|---|
| `pass_at_k(n, c, k)` | `1 - C(n-c, k) / C(n, k)` | Unbiased (Chen et al. 2021); needs n ≥ k |
| `pass_hat_k(n, c, k)` | `C(c, k) / C(n, k)` | Unbiased pass^k (tau-bench); converges to p^k |
| `mean_se(values)` | mean, `stdev / sqrt(n)` | SE = 0.0 for n < 2 |
| `bootstrap_ci(values, …)` | percentile bootstrap | Seeded `random.Random`; default 10,000 resamples |
| `cluster_bootstrap_ci(per_task)` | resamples **tasks** | Trials of one task are correlated |
| `paired_compare(a, b, …)` | paired bootstrap of per-task deltas plus exact two-sided sign test | `significant` = CI low > 0 **and** delta ≥ `min_effect`. `verdict` ∈ improved / regressed / within_noise / inconclusive |
| `EfficiencySummary.from_trials` | cost/success, p50/p95 wall clock, mean turns/tool calls, token totals, cache-read share, tool errors by tool, compaction events | |

## Result store

```text
<root>/runs/<run_id>/header.json        atomic; resuming with different conditions or suite hash raises
                    trials.jsonl        one record per line; locked append + fsync; torn trailing line skipped (counted in store.torn_lines)
                    transcripts/<trial>.json
                    reports/<name>
```

## Reports

```python
from harness_evals import summarize, compare_runs
s = summarize(header, records, k_values=(3, 5, 8))   # pass^k / pass@k only for k ≤ min trials per task
s.tiers["capability"].pass1, .pass1_se, .pass1_ci, .strict, .partial, .pass_hat_k, .pass_at_k, .saturation
print(s.to_markdown()); s.to_json()
rep = compare_runs(h_base, r_base, h_cand, r_cand)  # raises ConfoundedComparisonError unless allow_confounded=True
```

The markdown header follows checklist step 17. It lists:

* suite name, version, hash and task count;
* harness version, prompt/tool SHAs and profile;
* model, effort and thinking (`unpinned` when absent);
* sandbox image and user simulator;
* judge model and prompt version;
* safeguards;
* error counting, excluded trials and infra retries;
* `se_unit: across tasks`.

Then come a per-tier table (pass@1 ± SE, CI, strict, partial, pass^k, pass@k labelled "any-of-k"), efficiency, trial statuses and warnings.

**Warnings** fire for:

* fewer than 3 trials per task;
* a capability suite with fewer than 20 tasks;
* 0% tasks with ≥ 5 trials (broken-task suspects);
* tasks ready to graduate (~100%);
* a saturated capability suite (mean ≥ 0.95);
* unpinned effort;
* a judge used without a recorded `judge_human_agreement` in `RunConditions.extra`.

`compare_runs` holds `model`, `effort`, `thinking`, `judge_model`, `user_simulator` and the suite hash fixed. Everything else (harness version, SHAs, profile) is reported as the "variables under test". It warns on regression-tier drops and on deltas inside the noise band.

## Defaults

| Setting | Default | Vendor default vs our choice |
|---|---|---|
| bootstrap resamples | 10,000 | Our choice |
| confidence | 0.95 | Convention |
| `min_effect` / noise band | 0.05 | From the research: ~5-point band |
| `k_values` | 3, 5, 8 | Sierra / checklist range 3-8 |
| SE unit | across tasks | Our choice (stated in every report) |
| error counting | `RunConfig.error_policy` (default `failure`) | Our choice |

## Invariants

* Records are append-only. A resumed run never mixes conditions.
* pass^k and pass@k are only computed for k ≤ the minimum trials per task in that tier.
* Paired comparisons require the same task set (`compare_runs` uses the shared tasks and warns).

## Failure modes

* **A single-run decision.** Warned. Add trials (≥ 3 for PRs, 5 for capability) or tasks.
* **Comparing runs with different effort or model.** Raises. Pass `allow_confounded=True` only for exploratory work, which also adds a warning.
* **Partial credit read as the headline.** The table always shows strict next to partial.
