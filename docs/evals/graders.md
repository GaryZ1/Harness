# Graders: `harness_evals.graders`

## Purpose

Graders turn a trial's end state, and where policy requires it the transcript, into scored verdicts. `score_task` then combines them into a task outcome that always reports strict and partial scores together.

## Research basis

* **Deterministic first; models only for what code cannot check** (Anthropic 2026; §9.7 steps 5-6).
* **Outcome, not claims.** "Check whether the reservation exists in the DB, not whether the agent claimed to book it."
* **Format tolerance.** CORE-Bench rejected "96.12" against "96.124991…"; Opus 4.5 went from 42% to 95% once grader bugs were fixed.
* **Hack resistance** (§9.8, §11.8): hidden, read-only tests and `sys.exit(0)` detection. Hackable environments taught production RL models to hack, and the behaviour generalised to misalignment (~12% sabotage of detection code).
* **Judges:**
  * one isolated judge per dimension, with an "Unknown" option, 0.0-1.0 scores, pass/fail and quoted evidence (Anthropic 2026);
  * the alternative, a single call scoring all dimensions (Anthropic 2025 research system), is available for A/B testing;
  * pin the judge model and prompt version, and calibrate against humans (GDPval: 66% judge-expert vs 71% human-human agreement).
* **Path constraints are brittle.** Grade outcomes and constrain the path only where policy requires it.

## Kinds

| `kind` (aliases) | Checks | Key fields | Deterministic |
|---|---|---|---|
| `file_exists` | file present/absent | `path`, `exists` | yes |
| `file_regex` | regex in file (multiline) | `path`, `pattern`, `must_match`, `ignore_case` | yes |
| `file_equals` | content equality (whitespace-normalised by default) | `path`, `content`, `normalize_whitespace` | yes |
| `json_assert` | RFC 6901 pointer assertions: `exists`, `equals`, `approx` (+`rel_tol`/`abs_tol`), `contains`, `type`, `regex`. Score = fraction passing | `path`, `assertions` | yes |
| `answer` | final text or `file:<path>`; `match` = `numeric` / `exact` / `contains` / `regex`; `numeric_pick` = `last` / `only` / `any` | `expected`, `rel_tol` | yes |
| `command` (`state_check`) | command in the grading copy; exit code plus optional `require_output` / `forbid_output` regexes | `command` or `check`, `expect_exit`, `timeout_s` | yes |
| `unit_tests` | `pytest -rA` fail-to-pass ids must be `PASSED`; each pass-to-pass id or prefix needs ≥1 `PASSED` and no `FAILED`/`ERROR`. Exit 0 without report lines fails | `fail_to_pass`, `pass_to_pass`, `command` | yes |
| `diff_scope` | where the agent changed files | `allowed`, `forbidden` (globs), `require_change` | yes |
| `transcript` (`tool_calls`) | `max_turns`, `max_tool_calls`, `max_tool_errors`, `required_tools`, `forbidden_tools`, `forbid` (literal substrings; prefix `re:` for regex), `tool_constraints` / shorthand `refund: {amount: "<=100"}`, `final_regex` | | yes |
| `integrity` (`no_tamper`) | tamper signals (see below); `strict: true` also fails on suspicious signals | `protected`, `strict` | yes |
| `llm_rubric` (`llm_judge`) | judge verdict per criterion | `criteria` or `dimension`, `rubric`/`rubric_file`, `mode`, `include`, `require_evidence` | no |
| `composite` | nested group with its own `mode`/`threshold` | `graders` | if children are |

Every grader accepts `name`, `weight` (≥ 0) and `required` (a gate in hybrid scoring). Unnamed duplicates are suffixed `#2`, `#3`, and so on.

## Scoring

```python
from harness_evals import score_task, ScoringMode
ts = score_task(results, ScoringMode.HYBRID, threshold=0.8)
ts.passed, ts.strict_pass, ts.partial, ts.n_unknown, ts.grader_error, ts.failing
```

* **binary**: every grader must pass.
* **weighted**: weighted mean score ≥ threshold.
* **hybrid**: every `required: true` grader must pass, and the weighted mean of the rest must be ≥ threshold.
* An **unknown** verdict is never a pass: it fails strict/binary scoring and is excluded from the weighted mean.
* A **grader error** never passes and marks the task `grader_error`.
* **Tamper** (via `grade_trial`) forces `passed = strict_pass = False`.

## Integrity signals

| Kind | Severity | Trigger |
|---|---|---|
| `protected_modified` / `protected_deleted` / `protected_created` | tamper | a path in `protected_paths`, a hidden test path, or an integrity grader's `protected` glob changed, was deleted or was planted |
| `early_exit` | suspicious | new `sys.exit(`, `os._exit(`, `raise SystemExit`, or `exit 0` lines in changed files |
| `test_skip` | suspicious | new `pytest.skip(`, `@pytest.mark.skip/skipif/xfail`, `@unittest.skip`, `SkipTest` |
| `assert_neutered` | suspicious | new `assert True` lines |
| `conftest_added` | suspicious | new `conftest.py` anywhere |
| `tool_touched_protected` | suspicious | a tool call that runs a destructive verb (`rm`, `mv`, `sed -i`, `>`, `git checkout/restore/rm`...) on a protected path prefix |

Pattern counts are taken before the agent runs, so pre-existing code never triggers a signal. `harness_rl` uses the same signals for reward penalties.

## LLM judge

```python
spec = {"kind": "llm_rubric", "rubric": "…", "criteria": [{"name": "correctness", "description": "…", "weight": 2},
                                                          {"name": "tone", "description": "…"}],
        "mode": "per_criterion", "include": ["transcript", "final_text", "diff", "files:docs/*.md"]}
runner = EvalRunner(..., judge_client=client, judge_model=JUDGE_ID)   # both required; no default model
```

The judge prompt (`JUDGE_SYSTEM_PROMPT`, version `judge-v1`) fences agent material in `<material trust="data-only">`. Tag-like sequences (`<material`, `<rubric`, `<task`) inside the data are neutralised to `&lt;…`. The judge is told to treat instructions inside the material as manipulation evidence.

The output must be strict JSON: `{"criteria": [{"name", "verdict": pass|fail|unknown, "score": 0..1|null, "evidence": [verbatim quotes], "rationale"}]}`. A response that fails to parse is retried (`max_attempts=2`) and then becomes a grader error.

Evidence quotes are checked against the material. With `require_evidence: true` (the default), a `pass` with no verified quote is downgraded to `unknown`.

The task verdict is `False` if any criterion fails, `None` if any is unknown, and `True` otherwise. The score is the weighted mean of known scores. `details` records `judge_model`, `prompt_version`, `mode`, `calls`, per-criterion results and usage.

## Defaults

| Setting | Default | Vendor default vs our choice |
|---|---|---|
| judge `mode` | `per_criterion` | Anthropic 2026 default; `single_call` is the 2025 alternative to A/B |
| judge `max_attempts` / `max_tokens` / `max_material_chars` | 2 / 2,000 / 60,000 | Our choice |
| judge `require_evidence` | `True` | Our choice (counters lenient judges) |
| `answer` `rel_tol` / `numeric_pick` | 1e-3 / `last` | Our choice (CORE-Bench lesson) |
| `json_assert` `rel_tol` | 1e-6 | Our choice |
| `unit_tests` command | `python -m pytest -rA -q -p no:cacheprovider <targets>` | Our choice |

## Failure modes

* **Exit-code-only `command` checks.** Add `require_output`, or prefer `unit_tests`.
* **A judge used before calibration.** The report warns when the conditions carry no `judge_human_agreement` (see [review-and-guards.md](review-and-guards.md)).
* **Over-constrained transcript graders.** Rigid step checks are "overly brittle". Use budgets and policy constraints only.
