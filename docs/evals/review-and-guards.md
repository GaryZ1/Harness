# Transcript review, guards and judge calibration: `harness_evals.review`, `.guards`, `.calibration`

## Purpose

This tooling supports the human side of the method: reading transcripts, tagging failures and auditing graders. It also covers the guards that keep scores meaningful over time: saturation, broken tasks, held-out hygiene and contamination.

## Research basis

* **Weekly transcript review** (§9.7 step 13): sample passes as well as failures, confirm "failures should seem fair", cluster failures, and turn each new failure class into a task. MAST's specification / inter-agent / verification split is the starting taxonomy.
* **Grader QA as a standing job** (§9.7 step 14): audit failed trials for task or grader defects and track the broken-task rate. Public suites show 30-60% among audited failures.
* **Saturation** (§9.7 step 15): "An eval at 100% tracks regressions but provides no signal for improvement." Graduate such tasks to regression and replace them with harder ones.
* **Contamination and overfitting** (§9.7 step 16; §9.8): keep a held-out split that no optimizer has seen and rotate it. Check for gold-patch reproduction.
* **Judge calibration** (§9.7 step 6): GDPval's 66% judge-expert vs 71% human-human agreement is the reference gap. Recalibrate whenever the judge changes.

## Review API

```python
from harness_evals import sample_for_review, ReviewLog, failure_clusters, render_transcript_markdown

items = sample_for_review(records, n_failures=10, n_passes=5, seed=week_number)
# every trial with integrity signals first, then failures and passes round-robin across tasks (seeded)
for it in items:
    md = render_transcript_markdown(store.read_transcript(run_id, it.trial_id))   # secrets redacted
log = ReviewLog(".harness/evals/review.jsonl")
log.tag(trial_id=..., task_id=..., category="grader.false_negative", reviewer="ana", note="accepts 96.12")
log.broken_task_rate()      # share of audited failures whose majority tag is a task/grader defect
log.tasks_needing_fix()     # tasks with any task/grader-defect tag
log.reviewer_agreement()    # fraction of multiply-reviewed trials with unanimous tags
failure_clusters(records)   # {"completed|unit_tests": [...trial ids], "timeout|-": [...]} largest first
```

The default taxonomy (`DEFAULT_TAXONOMY`) groups categories as follows:

* **Specification**: `spec.misread_task`, `spec.wrong_assumption`.
* **Inter-agent**: `inter_agent.handoff_loss`, `inter_agent.conflict`.
* **Verification**: `verification.premature_stop`, `verification.wrong_check`.
* **Tools**: `tool.misuse`, `tool.error_unrecovered`, `tool.profile_mismatch`.
* **Other agent faults**: `context.lost_state`, `budget.exhausted`, `safety.policy_violation`, `reward_hack`.
* **Not agent faults**: `env.infra`, `task.ambiguous`, `task.unsolvable`, `grader.false_negative`, `grader.false_positive`.
* **Pass review**: `pass.fair`, for reviewed passes.

## Guards API

```python
from harness_evals import saturation_report, SplitManager, ExposureLedger, noise_warnings, gold_patch_similarity, canary_leaked

sat = saturation_report({"t1": (5, 5), "t2": (6, 0)})   # graduate=("t1",), broken_suspects=("t2",)
splits = SplitManager(heldout_fraction=0.3, salt="2026-Q4").split(task_ids)   # deterministic; .rotated("2027-Q1")
ledger = ExposureLedger(".harness/evals/exposure.jsonl")
ledger.record(optimizer="gepa", purpose="proposal", task_ids=splits["train"])
ledger.assert_clean_heldout(splits["heldout"])          # ContaminationError if any held-out task was exposed
gold_patch_similarity(agent_diff, reference_patch)      # token 5-gram Jaccard; flag ≥ 0.9 for review
canary_leaked(task.canary, trajectory.final_text)
```

## Calibration API

```python
from harness_evals import JudgeCalibration
cal = JudgeCalibration.compute(JUDGE_ID, "judge-v1", judge_labels, human_labels, human_b=second_annotator)
cal.judge_human_agreement, cal.kappa, cal.human_human_agreement, cal.gap
cal.acceptable()                      # n ≥ 30 and gap ≤ 0.05 (or agreement ≥ 0.66 without a second annotator)
cal.valid_for(JUDGE_ID, "judge-v2")   # False: a prompt or model change voids the calibration
# record it so reports stop warning:
RunConditions(..., judge_model=JUDGE_ID, extra={"judge_human_agreement": cal.judge_human_agreement})
```

## Defaults

| Setting | Default | Vendor default vs our choice |
|---|---|---|
| Graduate at | ≥ 0.999 over ≥ 5 trials | "~100%" (Anthropic); thresholds are our choice |
| Suite saturated at | mean ≥ 0.95 | Our choice |
| Broken-task suspect | 0 successes over ≥ 5 trials | Our choice ("0% pass@100" is the research's signal; flag earlier, audit) |
| Held-out fraction / salt | 0.3 / `split-v1` | Our choice |
| Gold reproduction flag | similarity ≥ 0.9 | Our choice |
| Calibration minimum n / max gap / min agreement | 30 / 0.05 / 0.66 | Gap and agreement from GDPval; n is our choice |
| Review sample | 10 failures + 5 passes + all integrity-flagged | Our choice |

## Invariants

* Review tags must come from the configured taxonomy and must name a reviewer.
* Exposure and review logs are append-only JSONL written under a cross-process lock.
* `SplitManager` membership depends only on `(salt, task id)`, so it is reproducible across machines. Rotation requires a new salt.

## Failure modes

* **Optimizing against held-out tasks.** `ExposureLedger.assert_clean_heldout` raises. `harness_rl.ImprovementLoop` calls it at construction.
* **A judge silently changed.** `valid_for` fails, so recompute before trusting scores.
* **Reviewing only failures.** That hides false positives. The sampler always includes passes.
