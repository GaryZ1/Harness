# RL, harness–model co-design and self-improving harnesses (`harness_rl`)

Production Python implementation of
[`reports/SOTA agentic harness reference.md` §11](../../reports/SOTA%20agentic%20harness%20reference.md),
the build plan's "Co-design: match the model's training harness, then train on your own", and the RL and self-improvement parts of
[`research_notes/.../evals_observability_rl.md`](../../research_notes/SOTA%20agentic%20harness%20build%20plan/evals_observability_rl.md).
It builds on [`harness_evals`](../evals/README.md) (same tasks, the same grading path, the same tamper detection) and on `harness_memory` (secret redaction, locking).

| Component | What it does | Module | Doc |
|---|---|---|---|
| **Trajectory export** | Training-ready JSONL from eval trajectories, event streams or message lists. Native tool roles, rollout ids, rewards, token data passthrough, redaction | `trajectory.py` | [trajectories.md](trajectories.md) |
| **Rewards** | Outcome, verifiable and rubric rewards on eval graders. Tamper override penalty, judge-share cap, RL-readiness gate, group advantages | `rewards.py` | [rewards.md](rewards.md) |
| **Environments & rollouts** | `reset`/`step`/`finish` episodes, rollout-scoped tool backend (id stamping, call cap, mismatch metric), black-box `RolloutRunner` over `AgentUnderTest` | `env.py` | [environments.md](environments.md) |
| **Harness profiles** | Per-model tool sets, edit format, prompt, reasoning replay, tool-result role, compaction and scaffold flags, loaded from config. No silent fallback; ablations | `profiles.py` | [profiles.md](profiles.md) |
| **Self-improvement loop** | Propose a config delta, screen on train, accept only on significant held-out gain with no regression and guards in budget. Hash-chained audit log, versioned configs, rollback | `improve.py` | [self-improvement.md](self-improvement.md) |

## Why this shape

1. **Match the model's training harness.** Models are post-trained inside specific harnesses, and mismatches cost several to 20+ points. Codex models call `apply_patch` even where it is absent. Profiles are therefore per-model config, selected by model-id glob, with mismatch detection and no silent generic fallback (§11.7 steps 1-2).
2. **Tool results go in the native tool role.** Sending them as user messages breaks DeepSeek's reasoning retention and native tool-role training generally. Export validates pairing, and the OpenAI-chat export emits `role: "tool"` (§11.7 step 3).
3. **One task format and one grading path for evals and RL.** Environments consume `harness_evals.Task` and grade via `harness_evals.grade_trial`, so CI evals and RL rewards cannot drift apart (§11.7 step 5).
4. **Prevent hacks first.** Anthropic and Redwood found production-RL reward hacking generalised to misalignment. Hidden, read-only tests, end-state grading and a tamper *override* penalty make hacking unprofitable, and tasks without a hack-resistant grader are refused (§11.7 step 6, §11.8).
5. **Rollout-scoped backends.** As in agent RFT: a unique rollout id on every tool call, a sandbox per rollout, and a per-rollout tool-call cap (§11.7 step 7).
6. **Optimise the harness before the weights, and only on held-out evidence.** GEPA matches or beats GRPO with up to 35× fewer rollouts. Self-improvement loops overfit when they accept on single runs or on the tasks the optimizer saw, so acceptance needs ≥ 3 trials, a paired-bootstrap CI above zero plus a 5-point effect, no regression, cost/latency within budget and a clean held-out split (§11.7 step 9, §11.8).
7. **Every change is auditable and reversible.** The audit log is hash-chained and config versions are immutable, so rollback is one call. Code self-modification (DGM-style) is never auto-accepted (§11.7 step 11).

## Architecture

```text
            profiles.yaml ──▶ ProfileRegistry.select(model_id) ──▶ HarnessProfile (tools, edit tool, replay, scaffolds…)
                                                                         │ ablations()
 harness_evals.Task ─▶ TaskEnvironment(task, tools_factory, profile, reward_fn)   [rl_readiness gate]
                         reset(seed) ─▶ fresh TrialSandbox + RolloutToolBackend(rollout_id, cap, ProfileMismatchMonitor)
                         step(ToolCall) ─▶ ToolOutcome (native tool_use/tool_result transcript)
   AgentUnderTest ─▶ RolloutRunner ─▶ (black-box episode; ctx.metadata["tools"] = backend)
                         finish() ─▶ harness_evals.grade_trial ─▶ CompositeReward (+HackPenalty override)
                                    ─▶ TrainingRecord ─▶ TrajectoryExporter (validate ▸ redact ▸ atomic JSONL) ─▶ trainer

 ImprovementLoop(ConfigStore, SplitEvaluator, Proposer, splits{train, heldout, regression}, AuditLog, ExposureLedger)
   baseline@train ─▶ train-only failures ─▶ proposer ─▶ ConfigChange* ─▶ protected-key / code-review gate
   ─▶ screen@train ─▶ paired held-out (Bonferroni) + regression + cost/latency/hack guards ─▶ commit vN+1 | reject
   rollback(v) / revert_last()          every step ─▶ hash-chained audit.jsonl
```

## Quick start

```python
from harness_evals import load_suite, CallableAgent
from harness_rl import ProfileRegistry, RolloutRunner, TrajectoryExporter

profile = ProfileRegistry.from_file("config/profiles.yaml").select(MODEL_ID).profile
runner = RolloutRunner(my_agent, tools_factory=make_tools, profile=profile,
                       exporter=TrajectoryExporter("rollouts/train.jsonl"), concurrency=8)
episodes = runner.run(load_suite("evals/regression").tasks, n_per_task=8, seed=2026)
print(sum(e.reward.total for e in episodes) / len(episodes))
```

Gated self-improvement over a real eval suite:

```python
from harness_rl import ImprovementLoop, ConfigStore, AuditLog, EvalSuiteEvaluator, GateConfig
from harness_evals import ExposureLedger, SplitManager, ResultStore, RunConditions

splits = SplitManager(salt="2026-Q4").split(t.id for t in suite.tasks)
loop = ImprovementLoop(
    store=ConfigStore(".harness/config"), audit=AuditLog(".harness/config/audit.jsonl"),
    evaluator=EvalSuiteEvaluator(suite, agent_for_config, store=ResultStore(".harness/evals"),
                                 conditions=RunConditions("our-harness@1.8.2", MODEL_ID, effort="high")),
    proposer=my_proposer, splits={**splits, "regression": regression_ids},
    exposure=ExposureLedger(".harness/evals/exposure.jsonl"), mutable=["prompts", "tools", "playbook", "profile.scaffolds"],
)
report = loop.step()          # accepted_version or per-candidate reasons; everything is in the audit log
```

## Integration notes

* Black-box agents find the rollout-scoped tool backend in `ctx.metadata["tools"]`, along with `rollout_id`, `seed`, `profile`, `env`, `memory_dir` and `cancel_event`. Route every tool call through the backend so the cap, the id stamping and the mismatch metric apply.
* To feed RL with open models, put token ids and logprobs in `Trajectory.metadata["rollout_details"]`. They pass through to `TrainingRecord.rollout_details`.
* No model ids appear in code. Profiles match ids by glob from your config, and judge/model ids are always passed in.
