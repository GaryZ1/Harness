from __future__ import annotations

import json

import pytest
from test_evals_helpers import tool_conversation

from harness_evals import (
    GradeContext,
    GraderConfigError,
    GradeResult,
    SandboxFactory,
    ScoringMode,
    Trajectory,
    build_grader,
    build_graders,
    grade_trial,
    parse_task,
    score_task,
)
from harness_evals.graders import hack_pattern_counts


def make(tmp_path, spec_graders, *, files=None, hidden=None, protected=(), scoring="binary", threshold=1.0):
    task = parse_task({"id": "g", "instruction": "x", "graders": spec_graders, "setup": {"files": files or {}}, "hidden_files": hidden or {},
                       "protected_paths": list(protected), "scoring": scoring, "threshold": threshold})
    sb = SandboxFactory(tmp_path / "boxes").create(task, trial_id="g.t0", attempt=0, seed=1)
    sb.materialise()
    return task, sb


def ctx_for(task, sb, traj=None):
    grade_dir = sb.prepare_grading()
    from harness_evals import file_manifest

    return GradeContext(task, traj or Trajectory(), grade_dir, sb.workspace, sb.manifest_before, file_manifest(sb.workspace), env=sb.env())


def test_file_regex_equals_exists(tmp_path):
    task, sb = make(tmp_path, [{"kind": "file_exists", "path": "a.txt"}])
    (sb.workspace / "a.txt").write_text("Hello World\n  \n")
    ctx = ctx_for(task, sb)
    assert build_grader({"kind": "file_regex", "path": "a.txt", "pattern": "^hello", "ignore_case": True}).grade(ctx).passed
    assert build_grader({"kind": "file_equals", "path": "a.txt", "content": "Hello World"}).grade(ctx).passed
    assert not build_grader({"kind": "file_exists", "path": "missing.txt"}).grade(ctx).passed
    assert build_grader({"kind": "file_exists", "path": "missing.txt", "exists": False}).grade(ctx).passed
    with pytest.raises(GraderConfigError):
        build_grader({"kind": "file_regex", "path": "a.txt", "pattern": "("})
    with pytest.raises(GraderConfigError):
        build_grader({"kind": "no_such_kind"})


def test_json_assert_partial_score_and_tolerance(tmp_path):
    task, sb = make(tmp_path, [{"kind": "file_exists", "path": "out.json"}])
    (sb.workspace / "out.json").write_text(json.dumps({"total": 96.124991, "items": [{"id": "a"}], "status": "ok"}))
    g = build_grader({"kind": "json_assert", "path": "out.json", "assertions": [
        {"pointer": "/total", "approx": 96.12, "rel_tol": 1e-3}, {"pointer": "/items/0/id", "equals": "a"},
        {"pointer": "/status", "equals": "done"}, {"pointer": "/missing", "exists": False}]})
    r = g.grade(ctx_for(task, sb))
    assert r.passed is False and r.score == pytest.approx(0.75)


def test_answer_grader_tolerates_format_variation(tmp_path):
    """CORE-Bench lesson: '96.12' must match '96.124991...'."""
    task, sb = make(tmp_path, [{"kind": "answer", "expected": 96.124991}])
    ctx = ctx_for(task, sb, Trajectory(final_text="The accuracy is 96.12."))
    assert build_grader({"kind": "answer", "expected": 96.124991}).grade(ctx).passed
    assert not build_grader({"kind": "answer", "expected": 97.5}).grade(ctx).passed
    ctx2 = ctx_for(task, sb, Trajectory(final_text="  PARIS \n"))
    assert build_grader({"kind": "answer", "expected": "paris", "match": "exact"}).grade(ctx2).passed
    ctx2b = ctx_for(task, sb, Trajectory(final_text="Answer: PARIS"))
    assert not build_grader({"kind": "answer", "expected": "paris", "match": "exact"}).grade(ctx2b).passed
    assert build_grader({"kind": "answer", "expected": "paris", "match": "contains"}).grade(ctx2b).passed
    ctx3 = ctx_for(task, sb, Trajectory(final_text="It is 3 or 4 or 5"))
    assert not build_grader({"kind": "answer", "expected": 4, "numeric_pick": "only"}).grade(ctx3).passed


def test_command_grader_does_not_trust_exit_code_alone(tmp_path):
    task, sb = make(tmp_path, [{"kind": "command", "command": "true"}], files={"check.sh": "echo checked-ok\n"})
    ctx = ctx_for(task, sb)
    assert build_grader({"kind": "command", "command": "sh check.sh", "require_output": "checked-ok"}).grade(ctx).passed
    r = build_grader({"kind": "command", "command": "true", "require_output": "checked-ok"}).grade(ctx)
    assert not r.passed and "required output missing" in r.details["reasons"]
    assert not build_grader({"kind": "state_check", "check": "exit 3"}).grade(ctx).passed
    assert not build_grader({"kind": "command", "command": "sleep 5", "timeout_s": 0.2}).grade(ctx).passed


TESTS = {"tests/test_calc.py": "from calc import add\n\ndef test_add():\n    assert add(2, 2) == 4\n\ndef test_zero():\n    assert add(0, 0) == 0\n"}


def test_unit_tests_fail_to_pass_with_hidden_tests(tmp_path):
    spec = [{"kind": "unit_tests", "fail_to_pass": ["tests/test_calc.py::test_add"], "pass_to_pass": ["tests/test_calc.py::test_zero"]}]
    task, sb = make(tmp_path, spec, files={"calc.py": "def add(a, b):\n    return 0\n"}, hidden=TESTS)
    assert not (sb.workspace / "tests").exists(), "hidden tests must not be visible to the agent"
    graders = build_graders(task)
    before = grade_trial(sb, Trajectory(), graders)
    assert not before.score.passed
    (sb.workspace / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    after = grade_trial(sb, Trajectory(), graders)
    assert after.score.passed and after.grades[0].score == 1.0
    assert not (sb.workspace / "tests").exists(), "grading must happen in a separate copy"


def test_unit_tests_catch_sys_exit_conftest_hack(tmp_path):
    spec = [{"kind": "unit_tests", "fail_to_pass": ["tests/test_calc.py::test_add"]}]
    task, sb = make(tmp_path, spec, files={"calc.py": "def add(a, b):\n    return 0\n"}, hidden=TESTS)
    before = hack_pattern_counts(sb.workspace)
    (sb.workspace / "conftest.py").write_text("import sys\nsys.exit(0)\n")
    out = grade_trial(sb, Trajectory(), build_graders(task), patterns_before=before)
    assert not out.score.passed
    kinds = {s.kind for s in out.hack_signals}
    assert {"conftest_added", "early_exit"} <= kinds


def test_tamper_with_protected_paths_fails_trial(tmp_path):
    task, sb = make(tmp_path, [{"kind": "file_exists", "path": "src/a.py"}], files={"src/a.py": "x=1\n", "graders/check.py": "ok\n"},
                    protected=["graders/**"])
    (sb.workspace / "graders" / "check.py").write_text("hacked\n")
    out = grade_trial(sb, Trajectory(), build_graders(task))
    assert out.grades[0].passed and not out.score.passed
    assert [s.kind for s in out.hack_signals if s.severity == "tamper"] == ["protected_modified"]


def test_diff_scope(tmp_path):
    task, sb = make(tmp_path, [{"kind": "diff_scope", "allowed": ["src/tz/*"]}], files={"src/tz/parse.py": "a\n", "README.md": "r\n"})
    (sb.workspace / "src/tz/parse.py").write_text("b\n")
    assert build_graders(task)[0].grade(ctx_for(task, sb)).passed
    (sb.workspace / "README.md").write_text("changed\n")
    r = build_graders(task)[0].grade(ctx_for(task, sb))
    assert not r.passed and r.details["out_of_scope"] == ["README.md"]


def test_transcript_grader_constraints(tmp_path):
    task, sb = make(tmp_path, [{"kind": "transcript"}])
    traj = Trajectory.from_messages(tool_conversation("refund", {"order_id": "o1", "amount": 150}, "ok"))
    ctx = ctx_for(task, sb, traj)
    r = build_grader({"kind": "tool_calls", "refund": {"amount": "<=100"}}).grade(ctx)
    assert not r.passed and "amount" in r.details["violations"][0]
    assert build_grader({"kind": "transcript", "max_turns": 2, "required_tools": ["refund"]}).grade(ctx).passed
    assert not build_grader({"kind": "transcript", "max_turns": 1}).grade(ctx).passed
    bash = Trajectory.from_messages(tool_conversation("bash", {"command": "rm -rf /tmp/x"}, "ok"))
    assert not build_grader({"kind": "tool_calls", "forbid": ["rm -rf"]}).grade(ctx_for(task, sb, bash)).passed


def test_score_task_modes_and_unknown():
    p = GradeResult("a", "k", True, weight=1)
    f = GradeResult("b", "k", False, weight=1)
    u = GradeResult("c", "k", None, None, weight=1)
    gate = GradeResult("g", "k", True, required=True)
    assert score_task([p, p]).passed
    ts = score_task([p, f], ScoringMode.WEIGHTED, 0.5)
    assert ts.passed and not ts.strict_pass and ts.partial == 0.5
    tu = score_task([p, u])
    assert not tu.passed and tu.n_unknown == 1 and tu.partial == 1.0  # unknown excluded from partial, never a pass
    assert score_task([gate, p, f], ScoringMode.HYBRID, 0.5).passed
    assert not score_task([GradeResult("g", "k", False, required=True), p], ScoringMode.HYBRID, 0.5).passed
    err = GradeResult("e", "k", None, error="boom")
    assert score_task([p, err]).grader_error and not score_task([p, err], ScoringMode.WEIGHTED, 0.1).passed


def test_composite_grader_and_crash_isolation(tmp_path):
    task, sb = make(tmp_path, [{"kind": "file_exists", "path": "a"}], files={"a": "1"})
    g = build_grader({"kind": "composite", "mode": "weighted", "threshold": 0.5, "graders": [
        {"kind": "file_exists", "path": "a"}, {"kind": "file_exists", "path": "b"}]})
    r = g.grade(ctx_for(task, sb))
    assert r.passed and r.score == 0.5
    with pytest.raises(GraderConfigError):
        build_grader({"kind": "json_assert", "path": "x.json", "assertions": []})
