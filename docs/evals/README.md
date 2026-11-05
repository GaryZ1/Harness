# Evaluation and benchmarks (`harness_evals`)

Production Python implementation of the evaluation design in
[`reports/SOTA agentic harness reference.md` §9](../../reports/SOTA%20agentic%20harness%20reference.md),
the build plan's "Evaluation and observability: hold the model fixed, grade end states, read transcripts", and
[`research_notes/.../evals_observability_rl.md`](../../research_notes/SOTA%20agentic%20harness%20build%20plan/evals_observability_rl.md).
Stdlib + PyYAML only; it imports `harness_core.types` and `harness_memory` (locking, path rules, secret redaction).

| Component | What it does | Module | Doc |
|---|---|---|---|
| **Agent boundary** | `AgentUnderTest` protocol, agent-visible `AgentTask`, `Trajectory` (transcript + efficiency counters), `CallableAgent` adapter | `agent.py` | [task-format.md](task-format.md) |
| **Task & suite format** | YAML / JSON / JSONL / Harbor-style task directories; tiers, polarity, scoring, hidden tests, limits | `task.py` | [task-format.md](task-format.md) |
| **Trial isolation** | Fresh sandbox per attempt, isolated HOME/caches/memory, no shared git history, hidden-test overlay | `isolation.py` | [runner-and-isolation.md](runner-and-isolation.md) |
| **Trial runner** | Pinned run conditions, concurrency cap, timeouts, infra-only retries, resume, event emission | `runner.py`, `grading.py` | [runner-and-isolation.md](runner-and-isolation.md) |
| **Graders** | End-state (file / JSON / answer / command / unit tests / diff scope), transcript, integrity, LLM-judge rubric, composite scoring | `graders/` | [graders.md](graders.md) |
| **Metrics** | Unbiased pass@k and pass^k, mean ± SE, bootstrap CIs, paired bootstrap + sign test, efficiency | `metrics.py` | [metrics-and-reports.md](metrics-and-reports.md) |
| **Result store & reports** | Append-only JSONL with atomic headers/transcripts; markdown/JSON reports; harness A/B comparison | `store.py`, `report.py` | [metrics-and-reports.md](metrics-and-reports.md) |
| **Transcript review** | Stratified sampling, failure taxonomy, broken-task rate, redacted rendering | `review.py` | [review-and-guards.md](review-and-guards.md) |
| **Guards & calibration** | Saturation / graduation, broken-task suspects, noise warnings, held-out splits, exposure ledger, contamination probes, judge calibration | `guards.py`, `calibration.py` | [review-and-guards.md](review-and-guards.md) |

## Why this shape

Each bullet is a research finding that the code enforces rather than documents.

1. **An eval measures harness + model together.** "When we evaluate 'an agent,' we're evaluating the harness *and* the model working together" (Anthropic, 2026-01-09). Scores are scaffold-conditional, so every run carries a `RunConditions` header (harness version, prompt/tool SHAs, profile, model, effort, thinking, sandbox image, judge, simulator, safeguards) and A/B comparisons refuse to run when the model, effort, thinking, judge, simulator or suite changed.
2. **Grade end states, deterministically first.** Outcome graders run in a separate grading copy with hidden tests overlaid, outside the agent's sandbox. LLM judges are for what code cannot check (§9.7 steps 5-6).
3. **Graders fail too.** CORE-Bench went 42% → 95% after grader fixes, and 59.4% of audited failed SWE-bench Verified tasks were flawed. So grader crashes are `grader_error` (never agent failures), the `answer` grader tolerates format variation ("96.12" vs "96.124991…"), and review tooling tracks the broken-task rate.
4. **Hackable graders get optimised.** Tamper detection (protected/hidden paths changed, deleted or planted; new `sys.exit`/skip markers/`conftest.py`) runs on every trial. Tamper fails the trial by default, and `unit_tests` requires per-test `PASSED` lines, so `sys.exit(0)` does not pass.
5. **Isolate every trial.** Claude once read git history left by earlier trials. Every attempt gets a new directory tree, `GIT_CEILING_DIRECTORIES`, an isolated HOME, and reset or frozen-and-verified memory.
6. **Noise is 2.2-6.0 points per run.** The defaults are 5 trials (capability, public) and 3 (regression), mean pass@1 ± SE *across tasks* plus a task-cluster bootstrap CI, and warnings below 3 trials or for deltas inside the ~5-point band.
7. **Pick the reliability metric by product.** pass^k (k = 3-8) sits next to pass@1. pass@k appears only labelled "any-of-k". Strict and partial scores are always reported together, and tiers are never merged.
8. **Efficiency is a result.** Records carry turns, tool calls, input/cache-read/cache-write/output tokens, cost, wall clock, tool errors by tool and compaction events. Reports show cost per success and p50/p95 latency.
9. **Read transcripts and guard the held-out set.** Review sampling covers passes as well as failures. A failure taxonomy separates agent faults from task and grader defects. A salted split plus an exposure ledger keeps optimizers off the held-out tasks.

## Architecture

```text
 suite.yaml / *.jsonl / tasks/<id>/{task.yaml, instruction.md, tests/, solution/, files/}
        │ load_suite ──▶ Suite(name, version, hash) ── Task.agent_view() ──▶ AgentTask (no graders/tests/canary)
        ▼
 EvalRunner(agent, store, RunConditions, RunConfig)            ThreadPool(concurrency)
   for task × trial (seeded):                                         │
     attempt loop (InfraError / SetupError / IsolationError ⇒ new sandbox, recorded; max_infra_retries)
       SandboxFactory.create ─▶ TrialSandbox{workspace/, home/, memory/(reset|frozen)}  materialise + verify_isolation
       hack_pattern_counts(before) ── watchdog thread: agent.run(task, workspace, ctx) ── timeout ⇒ cancel_event
     grade_trial: manifest(after) ─▶ prepare_grading() = copy(workspace)+hidden tests ─▶ graders (safe_grade)
                  detect_hack_signals ─▶ score_task(binary|weighted|hybrid) ─▶ tamper ⇒ fail
     TrialRecord ─▶ ResultStore.append (locked, fsync)   transcript ─▶ transcripts/<trial>.json   Event(EVALUATION) ─▶ sink
        ▼
 summarize(header, records) ─▶ RunSummary.to_markdown()/to_json()      compare_runs(base, cand) ─▶ paired A/B
 sample_for_review / ReviewLog / failure_clusters        saturation_report / SplitManager / ExposureLedger
```

## Quick start

```python
from harness_evals import CallableAgent, EvalRunner, ResultStore, RunConditions, RunConfig, load_suite

suite = load_suite("evals/capability/suite.yaml")

def my_agent(task, workspace, ctx):          # the integrator wraps the real loop instead
    ...                                      # work only inside `workspace`; honour ctx.metadata["cancel_event"]
    return trajectory                        # harness_evals.Trajectory, a list[Message], a str or None

runner = EvalRunner(
    CallableAgent(my_agent),
    store=ResultStore(".harness/evals"),
    conditions=RunConditions(harness_version="our-harness@1.8.2", model=MODEL_ID, effort="high",
                             prompts_sha=PROMPTS_SHA, tools_sha=TOOLS_SHA, judge_model=JUDGE_MODEL_ID),
    config=RunConfig(concurrency=8),         # trials default to 5 (capability) / 3 (regression)
    judge_client=judge_client,               # any harness_core ModelClient; only needed for llm_rubric graders
)
result = runner.run(suite, run_id="cap-v14-2026-11-05")
summary = result.summary()
print(summary.to_markdown())
runner.store.write_report(result.run_id, "report.md", summary.to_markdown())
```

Harness A/B with the model held fixed:

```python
from harness_evals import compare_runs
store = ResultStore(".harness/evals")
rep = compare_runs(store.read_header("base"), store.load_records("base"),
                   store.read_header("cand"), store.load_records("cand"))
print(rep.to_markdown())   # per-tier paired delta, bootstrap CI, sign test, verdict, cost/latency deltas
```

## Integration notes

* The agent receives `ctx.metadata` with `seed`, `trial_id`, `task_id`, `run_id`, `attempt`, `memory_dir`, `memory_mode`, `env` (isolated environment for subprocesses), `cancel_event` and `deadline_s`.
* `Trajectory.from_events` documents the event payloads it understands. The loop should emit `MODEL_RESPONSE` with `content`/`usage`/`model` and `TOOL_RESULT` with `tool_use_id`/`content`/`is_error`.
* The judge model id is passed in and sent as `ModelRequest.extra["model"]`, because the shared `ModelRequest` has no model field. Adapters bound to a fixed model can ignore it.
