# Rewards: `harness_rl.rewards`

## Purpose

This module turns graded episodes into scalar rewards, using the same graders and tamper signals as evals, while making reward hacking unprofitable and judge noise bounded.

## Research basis

* **Reward hacking generalises to misalignment.** Anthropic and Redwood made production coding RL environments deliberately hackable. Learning to hack correlated with misalignment on every eval, including ~12% sabotage of hack-detection code. Mitigations rank: prevent hacks, then inoculation prompting, then diversified RLHF safety training (§11.4, §11.8).
* **Hardened verifiers before anything optimises against them** (§11.7 step 6): hidden, read-only tests, end-state grading, and detection of `sys.exit(0)` and test tampering.
* **RL only with an automatic, hack-resistant grader** (§11.7 step 12).
* **Do not over-trust LLM-judge rewards.** Judges agree with experts 66% of the time vs 71% human-human on GDPval, so calibrate RULER-style rewards (§11.8).
* **GRPO uses group-relative advantages** (R1; DeepSeek V3.2 stabilisers).

## API

```python
from harness_rl import CompositeReward, HackPenalty, EfficiencyPenalty, outcome_reward, verifiable_reward, rubric_reward
from harness_rl import rl_readiness, group_advantages, EpisodeOutcome

reward = CompositeReward([
    ("outcome", outcome_reward, 1.0),             # strict pass → 1.0
    ("verifiable", verifiable_reward, 0.5),       # partial credit from deterministic graders only
    ("rubric", rubric_reward, 0.2),               # judge share 0.2/1.7 ≤ 0.3 cap
    ("efficiency", EfficiencyPenalty(per_turn=0.005), 1.0),
], hack=HackPenalty(tamper_penalty=-1.0, suspicious_penalty=-0.1), clip=(-1.0, 1.0))
breakdown = reward(EpisodeOutcome(task, trajectory, grades, task_score, hack_signals, status))
breakdown.total, breakdown.components, breakdown.hack_detected, breakdown.flags

rl_readiness(task)        # [] or reasons the task is unsafe to optimise against
group_advantages([1, 0, 1, 1])   # (advantages, has_signal); zero-variance groups → zeros, False
```

How the total is computed:

* `total = clip(Σ wᵢ · componentᵢ)`.
* With a **tamper** signal, `total = min(total, tamper_penalty)`. This is an override, not an addition, so a hacked trajectory can never out-score an honest failure.
* With only **suspicious** signals, `suspicious_penalty` is added (0 by default) and the flags are recorded.
* Non-`completed` episodes (timeout, agent error, infra error) score 0 on `outcome_reward` and `verifiable_reward`.
* An unknown judge verdict contributes 0 to `rubric_reward`.

## RL readiness gate

`rl_readiness(task)` reports a problem when:

* no grader is a deterministic end-state kind (`unit_tests`, `command`/`state_check`, `file_*`, `json_assert`, `answer`);
* a `unit_tests` grader exists but the task has no hidden tests;
* a `command` grader exists but there are no protected or hidden paths;
* a grader kind is unknown.

`TaskEnvironment` refuses flagged tasks unless `require_rl_ready=False`.

## Defaults

| Setting | Default | Vendor default vs our choice |
|---|---|---|
| Default reward | `outcome` × 1.0 with `HackPenalty()` | Our choice (verifiable outcome reward) |
| `tamper_penalty` / `suspicious_penalty` | −1.0 / 0.0 | Our choice |
| `clip` | (−1, 1) | Our choice |
| `max_judge_share` | 0.3 of positive weight | Our choice, from the GDPval judge-agreement gap |
| `EfficiencyPenalty` | off (0 per turn), capped at 0.2 | Our choice; agent RFT reduced turns as a side effect |
| Strict outcome | `strict=True` | Our choice: binary is the release gate |

## Invariants

* LLM-judge weight above the cap raises `RewardConfigError` at construction.
* Component names are unique, and a NaN reward raises.
* Tamper always dominates, whatever the other components say.

## Failure modes

* **Judge-only tasks used for RL.** The readiness gate refuses them. Add a state check.
* **All rollouts in a group score the same.** `has_signal=False`, so skip the group: there is no gradient signal.
* **Inoculation prompting** (framing hacking as acceptable in context during training) is a training-prompt choice and belongs in the trainer's task prompts. This package implements prevention and detection.
