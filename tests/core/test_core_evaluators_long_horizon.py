"""Stop evaluators (goal, completion promise, callable), runaway guards, Ralph loop, long-run driver."""

from __future__ import annotations

import json

import pytest

from harness_core import (
    AgentLoop,
    AllowAllPolicy,
    Budgets,
    CallableEvaluator,
    CompletionPromiseEvaluator,
    Evaluation,
    EventType,
    FunctionToolExecutor,
    GoalEvaluator,
    LongRunDriver,
    LongRunPrompts,
    LoopConfig,
    RalphLoop,
    ResultSubtype,
    SessionStore,
    StopReason,
    Usage,
    Verdict,
)
from harness_core.evaluators import parse_verdict
from harness_core.testing import ScriptedModelClient, text_response, tool_response


def test_evaluator_feedback_vetoes_stop_then_allows():
    calls = []

    def check(sc):
        calls.append(sc.stop_hook_active)
        return None if "tests pass" in sc.last_assistant_text else "Show the test output before finishing."

    model = ScriptedModelClient([text_response("done"), text_response("tests pass: 12/12")])
    result = AgentLoop(model, evaluators=[CallableEvaluator(check, "tests")]).run("go")
    assert result.ok and model.calls == 2 and calls == [False, True]
    assert model.requests[1].messages[-1].text == "Show the test output before finishing."
    assert [e.verdict for e in result.evaluations] == [Verdict.NOT_MET, Verdict.MET]


def test_evaluator_without_feedback_is_ignored():
    model = ScriptedModelClient([text_response("done")])
    result = AgentLoop(model, evaluators=[CallableEvaluator(lambda sc: False)]).run("go")
    assert result.ok and model.calls == 1


def test_eight_consecutive_blocks_without_tool_calls_are_overridden():
    model = ScriptedModelClient(default=text_response("still done"))
    loop = AgentLoop(model, evaluators=[CallableEvaluator(lambda sc: "keep going")])
    result = loop.run("go")
    assert result.ok and result.stop_override and model.calls == 9


def test_tool_calls_reset_the_consecutive_block_counter():
    ex = FunctionToolExecutor()
    ex.tool("t", "t", read_only=True)(lambda i, c: "x")
    script = []
    for _ in range(3):
        script += [text_response("done")] * 2 + [tool_response(("t", {"n": len(script)}))]
    script.append(text_response("final"))
    model = ScriptedModelClient(script)
    blocks = iter([True] * 6 + [False])
    cfg = LoopConfig(max_consecutive_stop_blocks=3, stuck=None)
    result = AgentLoop(model, ex, config=cfg, evaluators=[CallableEvaluator(lambda sc: "more" if next(blocks) else None)]).run("go")
    assert result.ok and not result.stop_override and model.calls == 10


def test_max_total_stop_blocks_caps_continuations():
    model = ScriptedModelClient(default=text_response("x"))
    result = AgentLoop(model, config=LoopConfig(max_total_stop_blocks=2), evaluators=[CallableEvaluator(lambda sc: "again")]).run("go")
    assert result.stop_override and model.calls == 3


def test_evaluator_exception_allows_stop_by_default_and_can_block():
    def broken(sc):
        raise RuntimeError("evaluator down")

    model = ScriptedModelClient([text_response("done")])
    loop = AgentLoop(model, evaluators=[CallableEvaluator(broken)])
    assert loop.run("go").ok
    assert any(e.data.get("error") == "evaluator_failed" for e in loop.session.events())
    model2 = ScriptedModelClient([text_response("done"), text_response("done")])
    cfg = LoopConfig(on_evaluator_error="block", max_total_stop_blocks=1)
    assert AgentLoop(model2, config=cfg, evaluators=[CallableEvaluator(broken)]).run("go").stop_override


@pytest.mark.parametrize("text,expected", [
    ("MET: all tests pass", (Verdict.MET, "all tests pass")),
    ("NOT_MET: no evidence", (Verdict.NOT_MET, "no evidence")),
    ("**Not yet met** - add docs", (Verdict.NOT_MET, "add docs")),
    ("Thinking...\nIMPOSSIBLE: repo is read-only", (Verdict.IMPOSSIBLE, "repo is read-only")),
    ("I am not sure", None),
])
def test_parse_verdict(text, expected):
    assert parse_verdict(text) == expected


def test_goal_evaluator_injects_objective_as_data_with_budget_readout():
    judge = ScriptedModelClient([text_response("NOT_MET: no test output shown", usage=Usage(10, 5)), text_response("MET: 12 passed")])
    goal = GoalEvaluator(judge, "All tests in tests/ pass", cost_estimator=lambda u, m: 0.01)
    model = ScriptedModelClient([text_response("I think it's done"), text_response("pytest: 12 passed")])
    loop = AgentLoop(model, evaluators=[goal], budgets=Budgets(max_total_tokens=100_000))
    result = loop.run("make tests pass")
    assert result.ok and judge.calls == 2
    cont = model.requests[1].messages[-1].text
    assert "<objective>\nAll tests in tests/ pass\n</objective>" in cont
    assert "user-provided data" in cont and "no test output shown" in cont
    assert "Token budget: 100000" in cont
    assert judge.requests[0].tools == [] and "pytest" not in judge.requests[0].messages[0].text
    assert "I think it's done" in judge.requests[0].messages[0].text
    assert result.total_cost_usd == pytest.approx(0.02)


def test_goal_evaluator_requires_repeated_impossible_verdicts():
    judge = ScriptedModelClient(default=text_response("IMPOSSIBLE: network is disabled"))
    model = ScriptedModelClient(default=text_response("blocked"))
    result = AgentLoop(model, evaluators=[GoalEvaluator(judge, "deploy the app", impossible_after=3)]).run("go")
    assert result.subtype is ResultSubtype.ERROR_GOAL_IMPOSSIBLE and judge.calls == 3 and model.calls == 3
    assert result.reason == "network is disabled"


def test_goal_evaluator_unparseable_or_refused_reply_never_traps_the_agent():
    judge = ScriptedModelClient([text_response("hmm, hard to say"), text_response("no", stop_reason=StopReason.REFUSAL)])
    model = ScriptedModelClient([text_response("done"), text_response("done")])
    assert AgentLoop(model, evaluators=[GoalEvaluator(judge, "x")]).run("go").ok
    assert AgentLoop(model, evaluators=[GoalEvaluator(judge, "x")]).run("again").ok


def test_goal_condition_limits():
    with pytest.raises(ValueError):
        GoalEvaluator(ScriptedModelClient(), "x" * 4001)
    with pytest.raises(ValueError):
        GoalEvaluator(ScriptedModelClient(), "  ")


def test_completion_promise_with_verifier():
    results = iter([(False, "2 tests failing"), (True, "all green")])
    ev = CompletionPromiseEvaluator("DONE", refeed="Keep working on the task.", verifier=lambda: next(results))
    model = ScriptedModelClient([text_response("not yet"), text_response("<promise>DONE</promise>"), text_response("<promise> DONE </promise>")])
    result = AgentLoop(model, evaluators=[ev]).run("go")
    assert result.ok and model.calls == 3
    assert model.requests[1].messages[-1].text == "Keep working on the task."
    assert "verification failed:\n2 tests failing" in model.requests[2].messages[-1].text
    assert result.evaluations[-1].reason == "all green"


def test_ralph_loop_requires_a_verifier_and_max_iterations():
    loop = AgentLoop(ScriptedModelClient())
    with pytest.raises(ValueError, match="verifier"):
        RalphLoop(loop, "p", max_iterations=3, completion_promise="DONE")
    with pytest.raises(ValueError, match="max_iterations"):
        RalphLoop(loop, "p", max_iterations=0, completion_promise="DONE", verifier=lambda: (True, ""))


def test_ralph_loop_refeeds_until_promise_and_writes_state(tmp_path):
    model = ScriptedModelClient([text_response("working"), text_response("<promise>SHIPPED</promise>")])
    loop = AgentLoop(model)
    state = tmp_path / "ralph.json"
    report = RalphLoop(loop, "Build the thing.", max_iterations=5, completion_promise="SHIPPED", verifier=lambda: (True, "ok"),
                       state_path=state).run()
    assert report.completed and report.iterations == 2
    assert "Iteration 2 of at most 5" in model.requests[1].messages[-1].text
    data = json.loads(state.read_text())
    assert data["status"] == "completed" and data["session_id"] == loop.session.id and data["max_iterations"] == 5
    assert loop.evaluators == []  # temporary evaluator removed


def test_ralph_loop_stops_at_max_iterations():
    model = ScriptedModelClient(default=text_response("never done"))
    report = RalphLoop(AgentLoop(model), "p", max_iterations=3, completion_promise="X", require_verifier=False).run()
    assert not report.completed and report.iterations == 3 and model.calls == 3


def test_long_run_driver_initializer_then_coding_sessions(tmp_path):
    state = {"initialized": False, "features_done": 0}
    ex = FunctionToolExecutor()
    ex.tool("init_project", "initializer action")(lambda i, c: state.update(initialized=True) or "initialized")
    ex.tool("finish_feature", "complete one feature")(lambda i, c: state.update(features_done=state["features_done"] + 1) or "ok")
    model = ScriptedModelClient([
        tool_response(("init_project", {})), text_response("set up"),
        tool_response(("finish_feature", {"n": 1})), text_response("feature 1 done"),
        tool_response(("finish_feature", {"n": 2})), text_response("feature 2 done"),
    ])
    store = SessionStore(tmp_path / "sessions")
    driver = LongRunDriver(
        store, lambda s: AgentLoop(model, ex, session=s, policy=AllowAllPolicy(), system="same system prompt"),
        LongRunPrompts("Build a todo app", persistence=True),
        is_initialized=lambda: state["initialized"], is_complete=lambda: state["features_done"] >= 2, max_sessions=10,
        state_path=tmp_path / "longrun.json")
    report = driver.run()
    assert report.completed and [s.kind for s in report.sessions] == ["initializer", "coding", "coding"]
    first_prompts = [r.messages[0].text for r in model.requests if len(r.messages) == 1]
    assert "initializer" in first_prompts[0] and "feature_list.json" in first_prompts[0]
    assert "start every session with this ritual" in first_prompts[1].lower() and "Persist until" in first_prompts[1]
    assert {r.system for r in model.requests} == {"same system prompt"}  # only the user prompt differs
    assert len({s.session_id for s in report.sessions}) == 3  # fresh context per session
    assert json.loads((tmp_path / "longrun.json").read_text())["completed"] is True
    assert any(e.type is EventType.SESSION_END for e in store.open(report.sessions[0].session_id).events())


def test_long_run_driver_stops_on_terminal_result_and_max_sessions(tmp_path):
    store = SessionStore(tmp_path)
    refusing = ScriptedModelClient(default=text_response("no", stop_reason=StopReason.REFUSAL))
    report = LongRunDriver(store, lambda s: AgentLoop(refusing, session=s), LongRunPrompts("x"), is_initialized=lambda: True,
                           is_complete=lambda: False, max_sessions=5).run()
    assert not report.completed and len(report.sessions) == 1 and "error_refusal" in report.reason
    chatty = ScriptedModelClient(default=text_response("progress"))
    report2 = LongRunDriver(store, lambda s: AgentLoop(chatty, session=s), LongRunPrompts("x"), is_initialized=lambda: True,
                            is_complete=lambda: False, max_sessions=2, session_prefix="b").run()
    assert len(report2.sessions) == 2 and "max_sessions" in report2.reason


def test_callable_evaluator_passes_through_evaluations():
    ev = CallableEvaluator(lambda sc: Evaluation(Verdict.MET, reason="fine"), name="x")
    model = ScriptedModelClient([text_response("ok")])
    result = AgentLoop(model, evaluators=[ev]).run("go")
    assert result.evaluations[0].evaluator == "x"
