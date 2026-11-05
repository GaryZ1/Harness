import pytest

from harness_memory import WorkingMemory
from harness_memory.errors import LimitExceededError, ValidationError
from harness_memory.working import PRE_COMPACTION_WARNING, ContextBudget, TodoStatus


def test_plan_single_in_progress(clock):
    wm = WorkingMemory(goal="ship auth", clock=clock)
    wm.set_plan(["design", "implement", "test"])
    assert wm.start_next().id == "1"
    with pytest.raises(ValidationError):
        wm.update("2", TodoStatus.IN_PROGRESS)
    wm.update("1", "completed")
    assert wm.start_next().id == "2"
    with pytest.raises(ValidationError):
        wm.set_plan([{"content": "a", "status": "in_progress"}, {"content": "b", "status": "in_progress"}])


def test_recitation_and_notes(clock):
    wm = WorkingMemory(goal="ship", clock=clock, max_notes=2)
    wm.set_plan(["a", "b"])
    wm.update("1", "completed")
    for n in ("one", "two", "three"):
        wm.note(n)
    rec = wm.recitation()
    assert rec.startswith("<working-memory>") and "[x] 1. a" in rec and "(1/2 done)" in rec
    assert "Z] one" not in rec and "Z] three" in rec
    with pytest.raises(LimitExceededError):
        wm.note("x" * 5000)


def test_budget_warning_latch(clock):
    wm = WorkingMemory(clock=clock, budget=ContextBudget(window_tokens=1000, compaction_threshold=0.8))
    assert wm.observe_usage(500) is None
    assert wm.observe_usage(760) == PRE_COMPACTION_WARNING
    assert wm.observe_usage(790) is None
    wm.after_compaction()
    assert wm.observe_usage(780) == PRE_COMPACTION_WARNING


def test_persistence(tmp_path, clock):
    wm = WorkingMemory(goal="g", clock=clock)
    wm.set_plan(["a"])
    wm.note("n")
    p = tmp_path / "wm.json"
    wm.save(p)
    back = WorkingMemory.load(p, clock=clock)
    assert back.goal == "g" and back.todos[0].content == "a" and back.notes == wm.notes
    assert "- [ ] a" in back.to_markdown()
