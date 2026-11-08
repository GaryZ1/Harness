"""Budgets (turns, USD, tokens, wall clock), wrap-up turn, error limits, doom loops, retries, fallback."""

from __future__ import annotations

import random

import pytest

from harness_core import (
    AgentLoop,
    AllowAllPolicy,
    Budgets,
    EventType,
    FunctionToolExecutor,
    LoopConfig,
    Price,
    PricingTable,
    RateLimitError,
    ResultSubtype,
    RetryPolicy,
    StuckConfig,
    TerminalModelError,
    TransientModelError,
    Usage,
)
from harness_core.testing import FakeMonotonic, ScriptedModelClient, text_response, tool_response


def counting_tools(counter: dict, clock: FakeMonotonic | None = None, advance: float = 0.0) -> FunctionToolExecutor:
    ex = FunctionToolExecutor()

    def run(inp, ctx):
        counter["n"] = counter.get("n", 0) + 1
        if clock is not None:
            clock.advance(advance)
        return f"ok {inp}"

    ex.tool("act", "an action", read_only=True)(run)
    ex.tool("fail", "always fails", read_only=True)(lambda inp, ctx: (_ for _ in ()).throw(OSError("disk")))
    ex.tool("write", "mutating")(lambda inp, ctx: "w")
    return ex


def test_max_turns_stops_without_executing_and_runs_a_wrap_up_turn():
    counter: dict = {}
    model = ScriptedModelClient([tool_response(("act", {"i": 1})), tool_response(("act", {"i": 2})), text_response("summary: did 1")])
    result = AgentLoop(model, counting_tools(counter), budgets=Budgets(max_turns=2)).run("go")
    assert result.subtype is ResultSubtype.ERROR_MAX_TURNS and result.is_error
    assert counter["n"] == 1 and result.num_turns == 2  # wrap-up call is not counted
    assert result.wrapped_up and result.final_text == "summary: did 1"
    not_run = result.messages[4].tool_results[0]
    assert not_run.is_error and "budget is exhausted" in not_run.text
    assert "Do not call any more tools" in result.messages[4].content[-1].text


def test_wrap_up_can_be_disabled():
    model = ScriptedModelClient([tool_response(("act", {}))])
    result = AgentLoop(model, counting_tools({}), budgets=Budgets(max_turns=1), config=LoopConfig(wrap_up_on_budget=False)).run("go")
    assert result.subtype is ResultSubtype.ERROR_MAX_TURNS and model.calls == 1 and not result.wrapped_up
    assert result.messages[-1].tool_results[0].is_error  # pairing kept


def test_dollar_budget_is_checked_after_each_response():
    prices = PricingTable({"test-model": Price(input_per_mtok=1000, output_per_mtok=1000)})  # 0.12 USD per scripted call
    model = ScriptedModelClient([tool_response(("act", {"i": 1})), tool_response(("act", {"i": 2})), text_response("wrap")])
    result = AgentLoop(model, counting_tools({}), budgets=Budgets(max_budget_usd=0.2), cost_estimator=prices).run("go")
    assert result.subtype is ResultSubtype.ERROR_MAX_BUDGET_USD
    assert result.total_cost_usd == pytest.approx(0.36)  # overshoots by the wrap-up response, as documented
    assert not prices.unpriced


def test_pricing_table_cache_multipliers_and_unknown_models():
    p = Price(input_per_mtok=10, output_per_mtok=20)
    u = Usage(input_tokens=1_000_000, output_tokens=1_000_000, cache_read_input_tokens=1_000_000, cache_creation_input_tokens=1_000_000)
    assert p.cost(u) == pytest.approx(10 + 20 + 1 + 12.5)
    table = PricingTable({})
    assert table(u, "mystery") == 0.0 and table.unpriced == {"mystery"}


def test_token_budget():
    big = Usage(input_tokens=600, output_tokens=0)
    model = ScriptedModelClient([tool_response(("act", {}), usage=big), tool_response(("act", {"x": 1}), usage=big), text_response("w")])
    result = AgentLoop(model, counting_tools({}), budgets=Budgets(max_total_tokens=1000)).run("go")
    assert result.subtype is ResultSubtype.ERROR_MAX_TOKEN_BUDGET


def test_wall_clock_budget_trips_after_a_slow_tool_batch():
    clock = FakeMonotonic()
    counter: dict = {}
    model = ScriptedModelClient([tool_response(("act", {})), text_response("wrapped")])
    loop = AgentLoop(model, counting_tools(counter, clock, advance=100), budgets=Budgets(max_wall_clock_s=50), monotonic=clock)
    result = loop.run("go")
    assert result.subtype is ResultSubtype.ERROR_MAX_WALL_CLOCK and counter["n"] == 1
    last_user = result.messages[2]
    assert not last_user.tool_results[0].is_error  # the tool ran; the wrap-up text joins its result message
    assert "budget for this run is exhausted" in last_user.content[-1].text
    assert result.duration_s == pytest.approx(100)


def test_budget_warning_is_turn_scoped():
    model = ScriptedModelClient([tool_response(("act", {"i": 1})), tool_response(("act", {"i": 2})), text_response("done")])
    AgentLoop(model, counting_tools({}), budgets=Budgets(max_turns=10, warning_fraction=0.2)).run("go")
    texts = [b.text for b in model.requests[2].messages[-1].content if b.type == "text"]
    assert any("Budget notice: 2 of 10 turns used" in t for t in texts)
    assert "Budget notice" not in model.requests[2].system


def test_consecutive_all_error_turns_terminate():
    model = ScriptedModelClient([tool_response(("fail", {"i": i})) for i in range(5)])
    budgets = Budgets(max_consecutive_error_turns=2)
    result = AgentLoop(model, counting_tools({}), budgets=budgets, config=LoopConfig(stuck=None)).run("go")
    assert result.subtype is ResultSubtype.ERROR_MAX_CONSECUTIVE_ERRORS and model.calls == 2


def test_consecutive_denials_terminate_headless_runs():
    model = ScriptedModelClient([tool_response(("write", {"i": i})) for i in range(5)])
    result = AgentLoop(model, counting_tools({}), budgets=Budgets(max_consecutive_error_turns=None)).run("go")
    assert result.subtype is ResultSubtype.ERROR_PERMISSION_DENIALS and model.calls == 3


def test_doom_loop_nudges_then_terminates():
    counter: dict = {}
    model = ScriptedModelClient(default=tool_response(("act", {"same": True})))
    cfg = LoopConfig(stuck=StuckConfig(identical_call_threshold=2, max_nudges=1))
    loop = AgentLoop(model, counting_tools(counter), config=cfg)
    result = loop.run("go")
    assert result.subtype is ResultSubtype.ERROR_DOOM_LOOP
    assert model.calls == 4 and counter["n"] == 3  # 4th identical call is not executed
    nudge_msg = model.requests[2].messages[-1]
    assert "Loop warning" in nudge_msg.content[-1].text
    assert result.messages[-1].tool_results[0].is_error  # pairing kept on termination
    assert [e.data["stuck"] for e in loop.session.events() if "stuck" in e.data] == ["identical_call", "identical_call"]


def test_edit_count_nudge_is_soft():
    model = ScriptedModelClient([tool_response(("write", {"path": "a.py", "v": i})) for i in range(3)] + [text_response("done")])
    cfg = LoopConfig(stuck=StuckConfig(max_edits_per_file=2, edit_tools=frozenset({"write"})))
    result = AgentLoop(model, counting_tools({}), config=cfg, policy=AllowAllPolicy()).run("go")
    assert result.ok
    assert "a.py has been edited 2 times" in model.requests[2].messages[-1].content[-1].text


def test_transient_errors_retry_with_backoff():
    sleeps: list[float] = []
    model = ScriptedModelClient([TransientModelError("overloaded"), TransientModelError("503"), text_response("ok")])
    loop = AgentLoop(model, sleep=sleeps.append, rng=random.Random(0), config=LoopConfig(retry=RetryPolicy(jitter=0)))
    result = loop.run("go")
    assert result.ok and sleeps == [1.0, 2.0] and result.num_turns == 1
    assert [e.data["attempt"] for e in loop.session.events() if e.type is EventType.ERROR] == [1, 2]


def test_rate_limit_honours_retry_after():
    sleeps: list[float] = []
    model = ScriptedModelClient([RateLimitError("429", retry_after=7), text_response("ok")])
    assert AgentLoop(model, sleep=sleeps.append).run("go").ok and sleeps == [7.0]


def test_terminal_errors_are_not_retried():
    sleeps: list[float] = []
    model = ScriptedModelClient([TerminalModelError("401 invalid key")])
    result = AgentLoop(model, sleep=sleeps.append).run("go")
    assert result.subtype is ResultSubtype.ERROR_DURING_EXECUTION and "invalid key" in result.error
    assert model.calls == 1 and sleeps == []


def test_retries_exhausted_then_fallback_model():
    primary = ScriptedModelClient(default=TransientModelError("down"))
    fallback = ScriptedModelClient([text_response("from fallback")])
    loop = AgentLoop(primary, fallback_model=fallback, sleep=lambda s: None, config=LoopConfig(retry=RetryPolicy(max_retries=2)))
    result = loop.run("go")
    assert result.ok and result.final_text == "from fallback" and primary.calls == 3


def test_retries_exhausted_without_fallback_fail():
    model = ScriptedModelClient(default=TransientModelError("down"))
    result = AgentLoop(model, sleep=lambda s: None, config=LoopConfig(retry=RetryPolicy(max_retries=1))).run("go")
    assert result.subtype is ResultSubtype.ERROR_DURING_EXECUTION and model.calls == 2


def test_retry_policy_backoff_is_capped_and_jittered():
    p = RetryPolicy(initial_delay_s=1, multiplier=2, max_delay_s=5, jitter=0.2)
    rng = random.Random(1)
    delays = [p.backoff(i, rng) for i in range(1, 6)]
    assert all(0.8 <= d / min(5, 2 ** (i - 1)) <= 1.2 for i, d in enumerate(delays, 1))
    assert p.rate_limit_delay(1, 10_000) == p.max_retry_after_s
    with pytest.raises(ValueError):
        RetryPolicy(jitter=1.5)


def test_session_scoped_budget_counts_prior_runs():
    model = ScriptedModelClient([text_response("one"), text_response("two"), tool_response(("act", {}))])
    loop = AgentLoop(model, counting_tools({}), budgets=Budgets(max_turns=3, scope="session"), config=LoopConfig(wrap_up_on_budget=False))
    assert loop.run("a").ok and loop.run("b").ok
    assert loop.run("c").subtype is ResultSubtype.ERROR_MAX_TURNS


def test_budgets_validate():
    with pytest.raises(ValueError):
        Budgets(max_turns=0)
    with pytest.raises(ValueError):
        Budgets(max_budget_usd=-1)
    with pytest.raises(ValueError):
        Budgets(warning_fraction=1.5)
