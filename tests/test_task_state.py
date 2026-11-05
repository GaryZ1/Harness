import json

import pytest

from harness_memory import InvariantViolation, TaskState, ValidationError, guard_tool_call, validate_feature_list_change

FEATURES = [
    {"id": "f1", "category": "auth", "description": "Login works", "steps": ["open /login", "submit"], "priority": 1},
    {"id": "f2", "category": "auth", "description": "Logout works", "steps": ["click logout"], "priority": 2},
]


@pytest.fixture
def task(tmp_path, clock):
    t = TaskState(tmp_path, clock=clock)
    t.initialize("Build auth", FEATURES, "echo smoke-ok\n")
    return t


def test_initializer(task, tmp_path):
    feats = task.features()
    assert [f.passes for f in feats] == [False, False]
    assert (tmp_path / ".harness/task/init.sh").stat().st_mode & 0o100
    assert task.git_log()[0].endswith("initialize feature list, progress log and init.sh")
    with pytest.raises(ValidationError):
        task.initialize("again", FEATURES, "x")


def test_orientation(task):
    report = task.orient(run_init=True)
    assert report.next_feature.id == "f1" and report.total == 2 and report.passing == 0
    assert report.init_result["returncode"] == 0 and "smoke-ok" in report.init_result["stdout"]
    text = report.render()
    assert "Next feature: [f1]" in text and "init.sh smoke test: ok" in text


def test_mark_passing_requires_evidence(task):
    with pytest.raises(InvariantViolation):
        task.mark_passing("f1", [])
    task.mark_passing("f1", ["e2e: tests/e2e/login.spec.ts passed"])
    assert task.next_feature().id == "f2"
    task.mark_failing("f1", "broke after refactor")
    assert task.next_feature().id == "f1"
    assert "regressed" in task.progress_tail()


def test_end_session_commits(task):
    task.mark_passing("f1", ["ok"])
    sha = task.end_session("s1", "Implemented login")
    assert sha and "Implemented login" in task.progress_tail()


def test_validate_feature_list_change():
    old = json.dumps([dict(f, passes=False) for f in FEATURES])
    flipped = json.loads(old)
    flipped[0]["passes"] = True
    validate_feature_list_change(old, json.dumps(flipped))
    for mutate in (
        lambda d: d.pop(0),
        lambda d: d[0].__setitem__("description", "easier"),
        lambda d: d[0].__setitem__("steps", []),
        lambda d: d.reverse(),
        lambda d: d.append({"id": "f3", "category": "x", "description": "y", "passes": True}),
    ):
        bad = json.loads(old)
        mutate(bad)
        with pytest.raises(InvariantViolation):
            validate_feature_list_change(old, json.dumps(bad))
    appended = json.loads(old) + [{"id": "f3", "category": "x", "description": "y", "passes": False}]
    validate_feature_list_change(old, json.dumps(appended))


def test_pretooluse_guard(task, tmp_path):
    fl = ".harness/task/feature_list.json"
    text = (tmp_path / fl).read_text()
    edit = guard_tool_call(
        "Edit", {"file_path": fl, "old_string": '"description": "Login works"', "new_string": '"description": "easy"'}, repo_root=tmp_path
    )
    assert not edit.allow and "immutable" in edit.reason
    data = json.loads(text)
    data[0]["passes"] = True
    assert guard_tool_call("Write", {"file_path": fl, "content": json.dumps(data)}, repo_root=tmp_path).allow
    data[1]["description"] = "nothing"
    assert not guard_tool_call("Write", {"file_path": fl, "content": json.dumps(data)}, repo_root=tmp_path).allow
    assert not guard_tool_call("Bash", {"command": "rm -f tests/test_login.py"}, repo_root=tmp_path).allow
    assert not guard_tool_call("Bash", {"command": "echo [] > .harness/task/feature_list.json"}, repo_root=tmp_path).allow
    assert guard_tool_call("Bash", {"command": "rm -rf build/"}, repo_root=tmp_path).allow
