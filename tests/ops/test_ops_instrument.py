"""GenAI span tree end-to-end: invoke_agent -> chat / execute_tool, cost, cache verdicts, subagents."""

from __future__ import annotations

import itertools

import pytest

from harness_core.types import (
    Message,
    ModelRequest,
    ModelResponse,
    NullTracer,
    PermissionBehavior,
    PermissionDecision,
    StopReason,
    TextBlock,
    ToolCall,
    ToolOutcome,
    Usage,
)
from harness_ops import (
    A,
    AgentInstrumentation,
    CacheMonitor,
    CacheObservation,
    HarnessMetrics,
    HarnessVersion,
    InMemorySpanExporter,
    OpsTracer,
    PriceTable,
)

PRICES = PriceTable.from_dict(
    {
        "version": 1,
        "defaults": {"cache_read_multiplier": 0.1, "cache_write_multiplier": 1.25, "cache_write_1h_multiplier": 2.0},
        "models": {"test-model": {"input_per_mtok": 3.0, "output_per_mtok": 15.0}, "test-model-small": {"input_per_mtok": 1.0, "output_per_mtok": 5.0}},
    }
)


class Ticker:
    """Deterministic monotonic clock: each call advances 10 ms."""

    def __init__(self):
        self._c = itertools.count()

    def __call__(self):
        return next(self._c) * 0.010


def resp(usage, stop=StopReason.END_TURN, model="test-model", text="done"):
    return ModelResponse([TextBlock(text)], stop, usage, model=model, id="msg_1")


@pytest.fixture
def setup(clock):
    exp = InMemorySpanExporter()
    metrics = HarnessMetrics(prices=PRICES)
    inst = AgentInstrumentation(OpsTracer([exp]), metrics=metrics, cache=CacheMonitor(clock=clock), provider="test-provider", monotonic=Ticker())
    return inst, exp, metrics


def test_full_run_span_tree_and_accounting(setup, clock):
    inst, exp, metrics = setup
    version = HarnessVersion("v7", "sha-abc", clock())
    req = ModelRequest(system="be helpful", messages=[Message.user("secret plan: buy ACME")])
    with inst.invoke_agent("coder", session_id="s1", model="test-model", effort="medium", version=version) as run:
        with inst.chat(run, "test-model", request=req) as call:
            rec1 = call.record(resp(Usage(1000, 50, 0, 4000), StopReason.TOOL_USE), ttft_ms=120.0, retry_count=1)
        with inst.execute_tool(run, ToolCall("tu_1", "write_file", {"path": "a.txt", "text": "hello"}), idempotency_key="k1", sandbox_id="sb1") as tool:
            tool.record(ToolOutcome("ok", metadata={"result_tokens": 3}))
        clock.advance(seconds=10)
        with inst.chat(run, "test-model", request=req) as call:
            rec2 = call.record(resp(Usage(200, 80, 5000, 100)))
        run.set_termination("success")

    by_op = {}
    for s in exp.spans:
        by_op.setdefault(s.attributes.get(A.OPERATION_NAME), []).append(s)
    root = by_op["invoke_agent"][0]
    assert root.name == "invoke_agent coder" and root.parent_span_id is None
    assert root.attributes[A.VERSION] == "v7" and root.attributes[A.PROMPTS_TOOLS_SHA] == "sha-abc"
    assert root.attributes[A.CONVERSATION_ID] == "s1" and root.attributes[A.EFFORT] == "medium"
    assert root.attributes[A.TERMINATION] == "success" and root.attributes[A.NUM_TURNS] == 2
    chats = by_op["chat"]
    assert [c.name for c in chats] == ["chat test-model", "chat test-model"] and all(c.parent_span_id == root.span_id for c in chats)
    c1 = chats[0].attributes
    assert c1[A.USAGE_INPUT_TOKENS] == 1000 and c1[A.USAGE_CACHE_CREATION_INPUT_TOKENS] == 4000 and c1[A.TOTAL_INPUT_TOKENS] == 5000
    assert c1[A.RESPONSE_FINISH_REASONS] == ["tool_use"] and c1[A.TTFT_MS] == 120.0 and c1[A.RETRY_COUNT] == 1
    assert c1[A.CACHE_OBSERVATION] == "cold_start" and chats[1].attributes[A.CACHE_OBSERVATION] == "hit"
    # content (prompt, system, output) is hashed by default
    assert A.INPUT_MESSAGES not in c1 and f"{A.INPUT_MESSAGES}.sha256" in c1 and f"{A.OUTPUT_MESSAGES}.sha256" in c1
    assert "ACME" not in str([s.to_dict() for s in exp.spans])
    tool_span = by_op["execute_tool"][0]
    assert tool_span.name == "execute_tool write_file" and tool_span.attributes[A.IDEMPOTENCY_KEY] == "k1"
    assert tool_span.attributes[A.SANDBOX_ID] == "sb1" and tool_span.attributes[A.TOOL_RESULT_TOKENS] == 3
    assert f"{A.TOOL_CALL_ARGUMENTS}.sha256" in tool_span.attributes  # args_hash
    # cost: (1000*3 + 50*15 + 4000*3.75)/1e6 and (200*3 + 80*15 + 5000*0.3 + 100*3.75)/1e6
    assert rec1.cost.total_usd == pytest.approx(0.01875) and rec2.cost.total_usd == pytest.approx(0.003675)
    assert root.attributes[A.TOTAL_COST_USD] == pytest.approx(0.022425)
    assert metrics.cost.sum() == pytest.approx(0.022425)
    assert metrics.cache_hit_rate() == pytest.approx(5000 / 10300)
    assert metrics.tool_error_rate() == 0.0


def test_unexpected_miss_vs_expected_rebuilds(setup, clock):
    inst, exp, metrics = setup
    big_hit, big_miss = Usage(100, 10, 9000, 100), Usage(100, 10, 0, 9000)
    with inst.invoke_agent("main", session_id="s2", model="test-model") as run:
        verdicts = []
        for usage in (big_miss, big_hit, big_miss):  # cold start, hit, UNEXPECTED miss
            with inst.chat(run, "test-model") as c:
                verdicts.append(c.record(resp(usage)).cache.observation)
        with inst.compaction(run, tokens_before=150_000, tokens_after=20_000):
            pass
        with inst.chat(run, "test-model") as c:
            verdicts.append(c.record(resp(big_miss)).cache)
        clock.advance(minutes=6)  # past the 5-minute TTL
        with inst.chat(run, "test-model") as c:
            verdicts.append(c.record(resp(big_miss)).cache)
        with inst.chat(run, "test-model-small") as c:
            verdicts.append(c.record(resp(big_miss, model="test-model-small")).cache)
        with inst.chat(run, "test-model-small") as c:
            verdicts.append(c.record(resp(Usage(10, 1, 0, 0), model="test-model-small")).cache)
    assert verdicts[:3] == [CacheObservation.COLD_START, CacheObservation.HIT, CacheObservation.UNEXPECTED_MISS]
    assert (verdicts[3].observation, verdicts[3].reason) == (CacheObservation.EXPECTED_REBUILD, "compaction")
    assert verdicts[4].reason == "ttl_expired" and verdicts[5].reason == "model_switch"
    assert verdicts[6].observation is CacheObservation.BELOW_MIN_CACHEABLE
    assert metrics.unexpected_miss_rate() == pytest.approx(1 / 7)
    assert metrics.compactions.sum() == 1
    assert 'harness_cache_observations_total{model="test-model",observation="unexpected_miss"} 1' in metrics.render_prometheus()


def test_subagent_carrier_nests_child_root_under_spawn_span(setup):
    inst, exp, _ = setup
    with inst.invoke_agent("lead", session_id="s3") as run:
        with inst.execute_tool(run, ToolCall("tu_9", "delegate", {"objective": "x"})) as tool:
            with inst.subagent_spawn(run, child_agent_name="researcher", parent_tool_call_id="tu_9") as carrier:
                pass
            tool.record(ToolOutcome("done"))
    # child may run in another process: only the carrier crosses the boundary
    child_inst = AgentInstrumentation(OpsTracer([exp]))
    with child_inst.invoke_agent("researcher", session_id="s3-sub", parent=carrier, depth=1):
        pass
    spawn = next(s for s in exp.spans if s.name == "subagent_spawn")
    child_root = next(s for s in exp.spans if s.name == "invoke_agent researcher")
    lead_root = next(s for s in exp.spans if s.name == "invoke_agent lead")
    assert child_root.parent_span_id == spawn.span_id and child_root.trace_id == lead_root.trace_id
    assert spawn.attributes[A.SUBAGENT_PARENT_TOOL_CALL_ID] == "tu_9" and child_root.attributes[A.SUBAGENT_DEPTH] == 1


def test_approval_wait_tool_failure_and_error_termination(setup):
    inst, exp, metrics = setup
    with pytest.raises(RuntimeError):
        with inst.invoke_agent("main", session_id="s4") as run:
            with inst.approval_wait(run, tool="bash") as wait:
                wait.decide(PermissionDecision(PermissionBehavior.ASK))
            with inst.approval_wait(run, tool="bash") as wait:
                wait.decide("deny")
            with inst.execute_tool(run, ToolCall("tu_2", "bash", {"cmd": "ls"})) as t:
                t.record(ToolOutcome("exit 1", is_error=True, metadata={"error_class": "nonzero_exit"}))
            with inst.execute_tool(run, ToolCall("tu_3", "bash", {"cmd": "ls"})):
                raise RuntimeError("sandbox died")
    root = next(s for s in exp.spans if s.name == "invoke_agent main")
    assert root.attributes[A.TERMINATION] == "error_during_execution"
    assert metrics.permission_rates() == {"allow": 0.0, "deny": 0.5, "ask": 0.5}
    assert metrics.tool_error_rate("bash") == 1.0
    tool_spans = [s for s in exp.spans if s.name == "execute_tool bash"]
    assert tool_spans[0].attributes[A.ERROR_TYPE] == "nonzero_exit" and tool_spans[1].attributes[A.ERROR_TYPE] == "RuntimeError"


def test_null_tracer_still_accounts(clock):
    metrics = HarnessMetrics(prices=PRICES)
    inst = AgentInstrumentation(NullTracer(), metrics=metrics, cache=CacheMonitor(clock=clock), max_budget_usd=0.01)
    with inst.invoke_agent("main", session_id="s5") as run:
        with inst.chat(run, "test-model") as c:
            c.record(resp(Usage(10_000, 1_000)), iterations=[Usage(10_000, 1_000), Usage(2_000, 500)])  # compaction iteration billed
    assert run.usage.input_tokens == 12_000
    status = run.ledger.status()
    assert status.exceeded and status.overshoot_usd == pytest.approx(0.0585 - 0.01)


def test_resumed_run_counts_budget_but_never_double_counts_metrics(tmp_path, clock):
    from harness_ops import StepJournal

    class Client:
        calls = 0

        def create(self, request):
            Client.calls += 1
            return resp(Usage(1_000_000, 0))

    metrics = HarnessMetrics(prices=PRICES)
    inst = AgentInstrumentation(NullTracer(), metrics=metrics, cache=CacheMonitor(clock=clock))
    journal = StepJournal(tmp_path / "j.db", clock=clock)
    req = ModelRequest(system="s", messages=[Message.user("go")])
    totals = []
    for _ in range(2):  # first execution, then a resume after a crash
        with journal.open_run("sess-r", heartbeat=False) as run, inst.invoke_agent("main", session_id="sess-r") as agent:
            with inst.chat(agent, "test-model") as c:
                r = run.model_call("turn-0", Client(), req)
                c.record(r, replayed=run.was_replayed("turn-0"))
            totals.append(agent.total_cost_usd)
    assert Client.calls == 1 and totals == [pytest.approx(3.0), pytest.approx(3.0)]  # budget sees prior spend
    assert metrics.cost.sum() == pytest.approx(3.0) and metrics.model_calls.sum() == 1  # reported once
