from __future__ import annotations

import json
import threading
import time

from test_multiagent_support import (
    FakeTools,
    ListSink,
    ScriptedModel,
    agent_of,
    always,
    ctx,
    defn,
    make_manager,
    report,
    text,
    tool_use,
)

from harness_core.types import Message, ModelRequest, RunContext, ToolCall
from harness_memory import FileMemoryStore, Scope
from harness_multiagent import (
    AgentRunner,
    DelegationBrief,
    ManagerConfig,
    MemoryAccess,
    SubagentResult,
    SubagentStatus,
    SubagentTask,
    WriteAccess,
)


def test_spawn_runs_isolated_context_and_returns_only_condensed_report():
    model = always(report("found 2 callers", ["parse_config called in a.py:3"]))
    sink = ListSink()
    mgr = make_manager(model, defn(), sink=sink, config=ManagerConfig(shared_context="Repo uses pytest."))
    parent = ctx(metadata={"secret_parent_note": "do not leak"})
    res = mgr.spawn("explorer", DelegationBrief("Find callers of parse_config", inputs=("src/config.py",)), parent)
    assert res.ok and res.report.summary == "found 2 callers" and res.depth == 1
    req = model.requests[0]
    # Fresh context: exactly one user message (the rendered brief); nothing from the parent conversation.
    assert len(req.messages) == 1 and "Find callers of parse_config" in req.messages[0].text
    assert "do not leak" not in json.dumps([m.to_api() for m in req.messages]) + str(req.system)
    assert "Repo uses pytest." in req.system and 'subagent "explorer"' in req.system
    assert req.extra["model"] == "test-model"
    assert sink.phases() == ["start", "stop"]
    stop = sink.events[-1].data
    assert stop["status"] == "success" and stop["usage"]["input_tokens"] == 100 and "found 2" not in json.dumps(stop)


def test_sibling_system_prompts_are_byte_identical_for_cache_sharing():
    model = always(report())
    mgr = make_manager(model, defn())
    mgr.spawn_many([("explorer", DelegationBrief(f"Audit file {i}")) for i in range(3)], ctx())
    systems = {r.system for r in model.requests}
    assert len(systems) == 1 and len({r.messages[0].text for r in model.requests}) == 3


def test_spawn_tool_executor_round_trip_and_bad_input_is_an_error_result():
    mgr = make_manager(always(report("ok")), defn())
    spec = mgr.specs()[0]
    assert spec.name == "spawn_subagent" and spec.input_schema["properties"]["agent"]["enum"] == ["explorer"]
    assert "Delegation policy" in spec.description and not spec.read_only
    out = mgr.execute(ToolCall("tu_9", "spawn_subagent", {"agent": "explorer", "objective": "Check X"}), ctx())
    payload = json.loads(out.content)
    assert not out.is_error and payload["trust"] == "untrusted" and payload["report"]["summary"] == "ok"
    assert mgr.execute(ToolCall("tu_a", "spawn_subagent", {"agent": "nope", "objective": "x"}), ctx()).is_error
    assert mgr.execute(ToolCall("tu_b", "spawn_subagent", {"agent": "explorer"}), ctx()).is_error


def test_depth_limit_and_nesting_opt_in():
    def script(req: ModelRequest, n: int):
        name = agent_of(req)
        if name == "lead-sub" and n == 0:
            return tool_use("spawn_subagent", {"agent": "leaf", "objective": "nested task"})
        return text(report(f"{name} done"))

    model = ScriptedModel(script)
    mgr = make_manager(model, defn("lead-sub", can_spawn=True), defn("leaf"), config=ManagerConfig(max_depth=2))
    res = mgr.spawn("lead-sub", DelegationBrief("Delegate once"), ctx())
    assert res.ok
    records = mgr.accounting.records()
    leaf = next(r for r in records if r.agent == "leaf")
    assert leaf.depth == 2 and leaf.parent_run_id == res.run_id
    # Direct depth check: a depth-2 caller cannot spawn when max_depth=2.
    deep = mgr.spawn("leaf", DelegationBrief("x"), RunContext("s", depth=2, agent_name="lead-sub", metadata={"subagent_definition": "lead-sub"}))
    assert deep.status is SubagentStatus.REFUSED and deep.refusal_code == "depth_limit"
    # A subagent without can_spawn may not delegate even within depth.
    nested = mgr.spawn("leaf", DelegationBrief("x"), RunContext("s", depth=1, agent_name="leaf", metadata={"subagent_definition": "leaf"}))
    assert nested.refusal_code == "nesting_disabled"
    # Readers without can_spawn are never offered the spawn tool.
    leaf_req = next(r for r in model.requests if agent_of(r) == "leaf")
    assert "spawn_subagent" not in [t.name for t in leaf_req.tools]


def test_read_only_subagent_gets_no_write_tools_and_hallucinated_writes_are_denied():
    calls = iter([
        tool_use("write_file", {"path": "x.py"}, call_id="w1"),
        tool_use("read_file", {"path": "x.py"}, call_id="r1"),
        text(report("read only")),
    ])
    model = ScriptedModel(lambda req, n: next(calls))
    tools = FakeTools()
    mgr = make_manager(model, defn(), tools=tools)
    res = mgr.spawn("explorer", DelegationBrief("Look at x.py"), ctx())
    assert res.ok and res.tool_calls == 2
    assert [t.name for t in model.requests[0].tools] == ["read_file", "grep"]
    assert [c[0] for c in tools.calls] == ["read_file"]  # the write never reached the executor
    denied = model.requests[1].messages[-1].tool_results[0]
    assert denied.is_error and "write" in denied.text


def test_single_writer_invariant_second_workspace_writer_is_refused(tmp_path):
    gate = threading.Event()
    release = threading.Event()

    def script(req, n):
        if "first" in req.messages[0].text and n == 0:
            gate.set()
            release.wait(5)
        return text(report("wrote"))

    mgr = make_manager(ScriptedModel(script), defn("writer", write_access=WriteAccess.WORKSPACE), defn(), tools=FakeTools())
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("a", mgr.spawn("writer", DelegationBrief("first writer"), ctx(str(tmp_path)))))
    t.start()
    assert gate.wait(5)
    second = mgr.spawn("writer", DelegationBrief("second writer"), ctx(str(tmp_path)))
    reader = mgr.spawn("explorer", DelegationBrief("reader meanwhile"), ctx(str(tmp_path)))
    release.set()
    t.join(5)
    assert out["a"].ok and second.status is SubagentStatus.REFUSED and second.refusal_code == "lease_held"
    assert reader.ok  # readers are unlimited and never blocked by the writer
    assert mgr.leases.writer_of(tmp_path) is None  # lease released afterwards


def test_own_files_writers_run_in_parallel_on_disjoint_files_and_cannot_escape(tmp_path):
    def script(req, n):
        if n == 0:
            target = "src/a/x.py" if "alpha" in req.messages[0].text else "src/a/y.py"  # beta tries alpha's file
            return tool_use("write_file", {"path": target}, call_id=f"w{n}")
        return text(report("ok"))

    tools = FakeTools()
    w = defn("filer", write_access=WriteAccess.OWN_FILES, owned_files=("src",))
    mgr = make_manager(ScriptedModel(script), w, tools=tools)
    a = mgr.spawn("filer", DelegationBrief("alpha", owned_files=("src/a",)), ctx(str(tmp_path)))
    b = mgr.spawn("filer", DelegationBrief("beta", owned_files=("src/b",)), ctx(str(tmp_path)))
    assert a.ok and b.ok
    assert [c[1]["path"] for c in tools.calls] == ["src/a/x.py"]  # beta's write outside src/b was denied
    assert mgr.ownership.snapshot() == {}


def test_overlapping_own_files_claim_while_running_is_refused(tmp_path):
    started, release = threading.Event(), threading.Event()

    def script(req, n):
        if "holder" in req.messages[0].text:
            started.set()
            release.wait(5)
        return text(report("ok"))

    mgr = make_manager(ScriptedModel(script), defn("filer", write_access=WriteAccess.OWN_FILES, owned_files=("src",)))
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("a", mgr.spawn("filer", DelegationBrief("holder", owned_files=("src/a",)), ctx(str(tmp_path)))))
    t.start()
    assert started.wait(5)
    clash = mgr.spawn("filer", DelegationBrief("clash", owned_files=("src/a/deep",)), ctx(str(tmp_path)))
    release.set()
    t.join(5)
    assert out["a"].ok and clash.refusal_code == "ownership"


def test_concurrency_cap_is_never_exceeded():
    lock, state = threading.Lock(), {"now": 0, "peak": 0}

    def script(req, n):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.03)
        with lock:
            state["now"] -= 1
        return text(report())

    mgr = make_manager(ScriptedModel(script), defn(), config=ManagerConfig(max_concurrent=3))
    results = mgr.spawn_many([("explorer", DelegationBrief(f"task {i}")) for i in range(10)], ctx())
    assert all(r.ok for r in results) and state["peak"] <= 3
    assert [r.report.summary for r in results] == ["done"] * 10


def test_concurrency_cap_with_zero_wait_refuses_immediately():
    started, release = threading.Event(), threading.Event()

    def script(req, n):
        started.set()
        release.wait(5)
        return text(report())

    mgr = make_manager(ScriptedModel(script), defn(), config=ManagerConfig(max_concurrent=1, concurrency_wait_s=0))
    t = threading.Thread(target=lambda: mgr.spawn("explorer", DelegationBrief("busy"), ctx()))
    t.start()
    assert started.wait(5)
    refused = mgr.spawn("explorer", DelegationBrief("extra"), ctx())
    release.set()
    t.join(5)
    assert refused.refusal_code == "concurrency_limit"


def test_per_agent_token_budget_stops_the_agent():
    model = ScriptedModel(lambda req, n: tool_use("read_file", {"path": "a"}, call_id=f"r{n}", inp=600, out=100))
    mgr = make_manager(model, defn(max_tokens=1500), tools=FakeTools())
    res = mgr.spawn("explorer", DelegationBrief("loop forever"), ctx())
    assert res.status is SubagentStatus.ERROR_MAX_BUDGET and res.turns == 3
    assert "Budget limit reached" in res.error


def test_run_budget_exhaustion_cancels_running_agents_and_refuses_new_spawns():
    hog_started, sibling_running = threading.Event(), threading.Event()

    def script(req, n):
        if "hog" in req.messages[0].text:
            sibling_running.wait(5)  # make sure both agents are admitted before the cap is hit
            hog_started.set()
            return tool_use("read_file", {"path": "x"}, call_id=f"h{n}", inp=900, out=100)
        sibling_running.set()
        time.sleep(0.02)
        return tool_use("read_file", {"path": "y"}, call_id=f"s{n}", inp=1, out=1)

    sink = ListSink()
    mgr = make_manager(ScriptedModel(script), defn(max_turns=10_000), tools=FakeTools(), sink=sink,
                       config=ManagerConfig(run_max_tokens=10_000))
    results = mgr.spawn_many([("explorer", DelegationBrief("hog")), ("explorer", DelegationBrief("slow sibling"))], ctx())
    assert hog_started.is_set()
    assert all(r.status is SubagentStatus.ERROR_MAX_BUDGET for r in results)
    assert mgr.ledger.exceeded()
    later = mgr.spawn("explorer", DelegationBrief("after the cap"), ctx())
    assert later.status is SubagentStatus.REFUSED and later.refusal_code == "budget"
    assert sink.phases()[-1] == "refused"
    # Overshoot is bounded: at most one response per running agent past the cap.
    assert mgr.ledger.tokens < 10_000 + 2 * 1_000


def test_timeout_abandons_a_hung_model_call():
    hang = threading.Event()
    model = ScriptedModel(lambda req, n: (hang.wait(5), text(report()))[1])
    mgr = make_manager(model, defn(timeout_s=0.15), config=ManagerConfig(join_grace_s=0.1))
    t0 = time.monotonic()
    res = mgr.spawn("explorer", DelegationBrief("hang"), ctx())
    hang.set()
    assert res.status is SubagentStatus.ERROR_TIMEOUT and time.monotonic() - t0 < 3
    assert "abandoned" in res.error


def test_cancel_all_stops_running_agents_cooperatively():
    first = threading.Event()

    def script(req, n):
        first.set()
        time.sleep(0.02)
        return tool_use("read_file", {"path": "a"}, call_id=f"c{n}")

    mgr = make_manager(ScriptedModel(script), defn(max_turns=10_000), tools=FakeTools())
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("r", mgr.spawn("explorer", DelegationBrief("long"), ctx())))
    t.start()
    assert first.wait(5)
    mgr.cancel_all()
    t.join(5)
    assert out["r"].status is SubagentStatus.CANCELLED
    assert mgr.spawn("explorer", DelegationBrief("new"), ctx()).refusal_code == "cancelled"


def test_structured_output_retries_then_typed_failure():
    mgr = make_manager(always("I think it is fine."), defn())
    res = mgr.spawn("explorer", DelegationBrief("Return junk"), ctx())
    assert res.status is SubagentStatus.ERROR_MAX_STRUCTURED_OUTPUT_RETRIES and res.turns == 6  # 1 + 5 retries
    ok = make_manager(ScriptedModel(lambda req, n: text("oops" if n == 0 else report("fixed"))), defn())
    assert ok.spawn("explorer", DelegationBrief("x"), ctx()).report.summary == "fixed"


def test_max_turns_and_max_tool_calls_are_enforced():
    loop = ScriptedModel(lambda req, n: tool_use("grep", {"pattern": "x"}, call_id=f"g{n}"))
    tools = FakeTools()
    mgr = make_manager(loop, defn(max_turns=4), tools=tools)
    res = mgr.spawn("explorer", DelegationBrief("loop", max_tool_calls=2), ctx())
    assert res.status is SubagentStatus.ERROR_MAX_TURNS and res.turns == 4 and len(tools.calls) == 2


def test_start_hook_blocks_and_stop_hook_feedback_is_attached():
    mgr = make_manager(always(report()), defn(), start_hooks=[lambda d, b, c: "no network research" if "web" in b.objective else None],
                       stop_hooks=[lambda r: "verify the claims"])
    blocked = mgr.spawn("explorer", DelegationBrief("search the web"), ctx())
    assert blocked.refusal_code == "policy"
    ok = mgr.spawn("explorer", DelegationBrief("read code"), ctx())
    assert ok.ok and ok.metadata["stop_hook_feedback"] == ["verify the claims"]


def test_accounting_aggregates_usage_per_agent_and_critical_path():
    mgr = make_manager(always(report()), defn("a"), defn("b"))
    mgr.spawn_many([("a", DelegationBrief("1")), ("a", DelegationBrief("2")), ("b", DelegationBrief("3"))], ctx())
    totals = mgr.accounting.by_agent()
    assert totals["a"].runs == 2 and totals["a"].usage.input_tokens == 200 and totals["b"].runs == 1
    cp = mgr.accounting.critical_path()
    assert cp.steps == 1 and cp.total_steps == 3 and cp.parallel_speedup == 3.0
    assert mgr.ledger.tokens == 3 * 120
    assert mgr.accounting.summary()["total_tokens"] == 360


def test_memory_is_read_only_by_default(tmp_path):
    store = FileMemoryStore(tmp_path / "mem", scope=Scope.PROJECT)
    seen = {}

    class Probe:
        def run(self, task: SubagentTask, ctx: RunContext) -> SubagentResult:
            seen[task.agent] = task.memory
            from harness_multiagent import SubagentReport
            return SubagentResult(task.agent, task.run_id, SubagentStatus.SUCCESS, report=SubagentReport("ok"))

    assert isinstance(Probe(), AgentRunner)
    from harness_multiagent import AgentRegistry, SubagentManager
    reg = AgentRegistry([defn("r"), defn("n", memory=MemoryAccess.NONE), defn("w", write_access=WriteAccess.WORKSPACE, memory=MemoryAccess.READ_WRITE)])
    mgr = SubagentManager(reg, Probe(), memory_store=store)
    for name in ("r", "n", "w"):
        assert mgr.spawn(name, DelegationBrief("x"), ctx(str(tmp_path))).ok
    assert not seen["r"].writable and seen["n"] is None and seen["w"] is store


def test_fork_mode_inherits_parent_history_and_unknown_role_fails_cleanly():
    model = always(report())
    mgr = make_manager(model, defn("forker", context="fork"), defn("odd", model_role="nonexistent"))
    history = [Message.user("earlier question"), Message.assistant([])]
    assert mgr.spawn("forker", DelegationBrief("continue"), ctx(), history=history).ok
    assert model.requests[0].messages[0].text == "earlier question" and len(model.requests[0].messages) == 3
    res = mgr.spawn("odd", DelegationBrief("x"), ctx())
    assert res.status is SubagentStatus.ERROR_DURING_EXECUTION and "no model configured" in res.error
