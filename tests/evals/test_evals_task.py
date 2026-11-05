from __future__ import annotations

import json

import pytest
import yaml

from harness_evals import Polarity, ScoringMode, SuiteTier, TaskFormatError, load_suite, load_task, parse_task

BASIC = {"id": "t1", "instruction": "Write hello.txt", "graders": [{"kind": "file_exists", "path": "hello.txt"}]}


def test_parse_task_defaults_and_enums():
    t = parse_task(BASIC | {"suite": "regression", "polarity": "negative", "scoring": "weighted", "threshold": 0.5, "tags": ["io"]})
    assert t.suite is SuiteTier.REGRESSION and t.polarity is Polarity.NEGATIVE and t.scoring is ScoringMode.WEIGHTED
    assert t.trials_for() == 3  # regression default
    assert parse_task(BASIC).trials_for() == 5  # capability default (TB2 convention)
    assert parse_task(BASIC).limits.timeout_s == 1800.0


@pytest.mark.parametrize(
    "bad",
    [
        {"id": "../x", "instruction": "x", "graders": [{"kind": "file_exists", "path": "a"}]},
        {"id": "t", "instruction": "", "graders": [{"kind": "file_exists", "path": "a"}]},
        {"id": "t", "instruction": "x", "graders": []},
        {"id": "t", "instruction": "x", "graders": [{"path": "a"}]},
        {"id": "t", "instruction": "x", "graders": [{"kind": "k"}], "setup": {"files": {"../escape.txt": "x"}}},
        {"id": "t", "instruction": "x", "graders": [{"kind": "k"}], "isolation": {"tests_writable_by_agent": True}},
        {"id": "t", "instruction": "x", "graders": [{"kind": "k"}], "threshold": 2},
        {"id": "t", "instruction": "x", "graders": [{"kind": "k"}], "suite": "nightly"},
    ],
)
def test_parse_task_rejects_invalid(bad):
    with pytest.raises(TaskFormatError):
        parse_task(bad)


def test_task_directory_hides_tests_and_solution(tmp_path):
    d = tmp_path / "fix-001"
    (d / "tests").mkdir(parents=True)
    (d / "solution").mkdir()
    (d / "files").mkdir()
    (d / "tests" / "test_x.py").write_text("def test_x(): pass\n")
    (d / "files" / "app.py").write_text("x = 1\n")
    (d / "instruction.md").write_text("Fix the bug in app.py")
    (d / "task.yaml").write_text(yaml.safe_dump({"id": "fix-001", "graders": [{"kind": "unit_tests", "pass_to_pass": ["tests/"]}], "canary": "CANARY-123"}))
    t = load_task(d)
    assert t.instruction == "Fix the bug in app.py"
    assert t.hidden_dir == d / "tests" and t.solution_dir == d / "solution"
    assert "files" in t.setup.copy_dirs
    assert "tests/test_x.py" in t.all_protected()
    view = t.agent_view()
    assert not hasattr(view, "graders") and "CANARY" not in repr(view)


def test_suite_yaml_jsonl_and_directory(tmp_path):
    (tmp_path / "suite.yaml").write_text(yaml.safe_dump({"name": "cap", "version": "14", "tasks": [BASIC, BASIC | {"id": "t2"}]}))
    s = load_suite(tmp_path / "suite.yaml")
    assert (s.name, s.version, len(s.tasks)) == ("cap", "14", 2)
    (tmp_path / "s.jsonl").write_text("\n".join(json.dumps(BASIC | {"id": f"j{i}"}) for i in range(3)) + "\n")
    assert len(load_suite(tmp_path / "s.jsonl").tasks) == 3
    td = tmp_path / "dir" / "a"
    td.mkdir(parents=True)
    (td / "task.yaml").write_text(yaml.safe_dump(BASIC | {"id": "a"}))
    assert [t.id for t in load_suite(tmp_path / "dir").tasks] == ["a"]


def test_duplicate_ids_rejected_and_suite_hash_tracks_content(tmp_path):
    (tmp_path / "dup.yaml").write_text(yaml.safe_dump({"name": "x", "tasks": [BASIC, BASIC]}))
    with pytest.raises(TaskFormatError):
        load_suite(tmp_path / "dup.yaml")
    (tmp_path / "a.yaml").write_text(yaml.safe_dump({"name": "x", "tasks": [BASIC]}))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"name": "x", "tasks": [BASIC | {"instruction": "changed"}]}))
    h1, h1b, h2 = load_suite(tmp_path / "a.yaml").hash, load_suite(tmp_path / "a.yaml").hash, load_suite(tmp_path / "b.yaml").hash
    assert h1 == h1b != h2


def test_suite_select_by_tier_and_tags():
    from harness_evals import Suite

    s = Suite("s", "1", (parse_task(BASIC | {"tags": ["a"]}), parse_task(BASIC | {"id": "r", "suite": "regression"})))
    assert [t.id for t in s.select(tier=SuiteTier.REGRESSION).tasks] == ["r"]
    assert [t.id for t in s.select(tags=["a"]).tasks] == ["t1"]
