"""Crash recovery, wake/resume, hard interrupts and the decoupled sandbox (LazyExecutor)."""

from __future__ import annotations

import pytest

from harness_core import (
    AgentLoop,
    AllowAllPolicy,
    Budgets,
    CompositeExecutor,
    EventType,
    FunctionToolExecutor,
    LazyExecutor,
    LoopConfig,
    Message,
    ResultSubtype,
    Role,
    RunContext,
    SandboxError,
    SessionLockedError,
    SessionStatus,
    SessionStore,
    ToolCall,
    ToolOutcome,
    ToolSpec,
    idempotency_key,
)
from harness_core.testing import ScriptedModelClient, text_response, tool_response


class Crash(BaseException):
    """Simulates the harness process dying (not caught by the loop's ``except Exception``)."""


def tools(counts: dict, crash_on: set | None = None) -> FunctionToolExecutor:
    crash_on = crash_on if crash_on is not None else set()
    ex = FunctionToolExecutor()

    def make(name):
        def fn(inp, ctx):
            counts[name] = counts.get(name, 0) + 1
            if name in crash_on:
                crash_on.discard(name)
                raise Crash()
            return f"{name} ok"
        return fn

    ex.tool("read_a", "ro", read_only=True)(make("read_a"))
    ex.tool("write_b", "mutating")(make("write_b"))
    ex.tool("read_c", "ro", read_only=True)(make("read_c"))
    return ex


def test_crash_mid_batch_resumes_without_repeating_side_effects(tmp_path):
    store = SessionStore(tmp_path)
    counts: dict = {}
    batch = tool_response(("read_a", {}, "t1"), ("write_b", {}, "t2"), ("read_c", {}, "t3"))
    model = ScriptedModelClient([batch])
    with pytest.raises(Crash):
        AgentLoop(model, tools(counts, {"write_b"}), session=store.create("s"), policy=AllowAllPolicy()).run("go")
    assert store.open("s").info.status is SessionStatus.RUNNING  # never cleanly finished

    model2 = ScriptedModelClient([text_response("recovered")])
    loop2 = AgentLoop(model2, tools(counts), session=store.wake("s"), policy=AllowAllPolicy())
    result = loop2.run()
    assert result.ok and result.final_text == "recovered"
    assert counts == {"read_a": 1, "write_b": 1, "read_c": 1}  # nothing re-executed except never-started read_c
    r1, r2, r3 = result.messages[2].tool_results
    assert (r1.text, r1.is_error) == ("read_a ok", False)
    assert r2.is_error and "side effects are unknown" in r2.text
    assert (r3.text, r3.is_error) == ("read_c ok", False)
    starts = [e.data["source"] for e in loop2.session.events() if e.type is EventType.SESSION_START]
    assert starts == ["startup", "resume"]
    assert store.open("s").info.status is SessionStatus.IDLE


def test_crash_in_read_only_tool_is_rerun_on_resume(tmp_path):
    store = SessionStore(tmp_path)
    counts: dict = {}
    with pytest.raises(Crash):
        AgentLoop(ScriptedModelClient([tool_response(("read_a", {}, "t1"))]), tools(counts, {"read_a"}),
                  session=store.create("s")).run("go")
    result = AgentLoop(ScriptedModelClient([text_response("ok")]), tools(counts), session=store.wake("s")).run()
    assert counts["read_a"] == 2 and not result.messages[2].tool_results[0].is_error


def test_crash_before_the_model_answered_resumes_with_a_model_call(tmp_path):
    store = SessionStore(tmp_path)
    with pytest.raises(Crash):
        AgentLoop(ScriptedModelClient([Crash()]), session=store.create("s")).run("hello")
    model = ScriptedModelClient([text_response("hi there")])
    result = AgentLoop(model, session=store.wake("s")).run()
    assert result.ok and model.requests[0].messages[-1].text == "hello"


def test_resume_of_a_finished_session_is_a_no_op_and_new_prompts_continue_it(tmp_path):
    store = SessionStore(tmp_path)
    AgentLoop(ScriptedModelClient([text_response("first")]), session=store.create("s")).run("one")
    idle = ScriptedModelClient([])
    result = AgentLoop(idle, session=store.wake("s")).run()
    assert result.ok and idle.calls == 0 and result.final_text == "first"
    model = ScriptedModelClient([text_response("second")])
    result = AgentLoop(model, session=store.wake("s")).run("two")
    assert [m.text for m in model.requests[0].messages] == ["one", "first", "two"]


def test_two_harnesses_cannot_drive_one_session(tmp_path):
    store = SessionStore(tmp_path)
    store.create("s")
    holder = store.open("s")
    with holder.lease():
        with pytest.raises(SessionLockedError):
            AgentLoop(ScriptedModelClient([text_response("x")]), session=store.open("s")).run("go")


def test_hard_interrupt_records_an_aborted_turn_with_paired_results():
    ex = FunctionToolExecutor()
    holder: dict = {}
    ran = []

    def first(inp, ctx):
        ran.append("first")
        holder["loop"].interrupt()
        return "done"

    ex.tool("first", "mut")(first)
    ex.tool("second", "mut")(lambda i, c: ran.append("second") or "x")
    model = ScriptedModelClient([tool_response(("first", {}, "t1"), ("second", {}, "t2"))])
    loop = AgentLoop(model, ex, policy=AllowAllPolicy())
    holder["loop"] = loop
    result = loop.run("go")
    assert result.subtype is ResultSubtype.INTERRUPTED and ran == ["first"]
    r1, r2 = result.messages[-1].tool_results
    assert not r1.is_error and r2.is_error and "interrupted" in r2.text
    assert loop.session.info.status is SessionStatus.INTERRUPTED
    # the next run starts cleanly
    model.extend([text_response("resumed")])
    assert loop.run("continue").ok


def test_interrupt_cancels_the_executor():
    class Cancellable(FunctionToolExecutor):
        cancelled = False

        def cancel(self):
            self.cancelled = True

    ex = Cancellable()
    loop = AgentLoop(ScriptedModelClient(), ex)
    loop.interrupt()
    assert ex.cancelled


def test_idempotency_key_is_stable_per_session_step_and_call():
    k = idempotency_key("s", 3, "t1")
    assert k == idempotency_key("s", 3, "t1") and k != idempotency_key("s", 4, "t1") and len(k) == 32
    seen = []
    ex = FunctionToolExecutor()
    ex.tool("t", "t", read_only=True)(lambda i, ctx: seen.append(ctx.metadata["idempotency_key"]) or "ok")
    AgentLoop(ScriptedModelClient([tool_response(("t", {})), text_response("x")]), ex).run("go")
    assert len(seen) == 1 and len(seen[0]) == 32


def test_lazy_executor_provisions_on_first_call_and_reprovisions_after_death():
    made = []
    disposed = []

    class Sandbox:
        def __init__(self):
            self.n = 0

        def specs(self):
            return []

        def execute(self, call, ctx):
            self.n += 1
            if call.input.get("kill"):
                raise SandboxError("OOM killed")
            return ToolOutcome(f"sandbox#{len(made)} call {self.n}")

    def factory():
        made.append(Sandbox())
        return made[-1]

    lazy = LazyExecutor([ToolSpec("bash", "run", read_only=True)], factory, on_dispose=disposed.append)
    assert [s.name for s in lazy.specs()] == ["bash"] and not lazy.provisioned  # listing tools provisions nothing
    model = ScriptedModelClient([tool_response(("bash", {"kill": True})), tool_response(("bash", {})), text_response("ok")])
    result = AgentLoop(model, lazy).run("go")
    assert result.ok and lazy.provision_count == 2 and len(disposed) == 1
    died = result.messages[2].tool_results[0]
    assert died.is_error and "sandbox died" in died.text
    assert result.messages[4].tool_results[0].text == "sandbox#2 call 1"


def test_lazy_executor_reports_provisioning_failure_as_tool_error():
    def factory():
        raise RuntimeError("no capacity")

    out = LazyExecutor([ToolSpec("bash", "run")], factory).execute(ToolCall("t", "bash", {}), RunContext("s"))
    assert out.is_error and "could not be provisioned" in out.content


def test_composite_executor_routes_and_rejects_duplicates():
    a, b = FunctionToolExecutor(), FunctionToolExecutor()
    a.tool("x", "x")(lambda i, c: "from a")
    b.tool("y", "y")(lambda i, c: "from b")
    comp = CompositeExecutor(a, b)
    ctx = RunContext("s")
    assert comp.execute(ToolCall("1", "y", {}), ctx).content == "from b"
    assert comp.execute(ToolCall("2", "z", {}), ctx).is_error
    with pytest.raises(ValueError):
        CompositeExecutor(a, a)


def test_dangling_tool_batch_is_closed_when_a_run_ends_early():
    model = ScriptedModelClient([tool_response(("read_a", {}, "t1"))])
    loop = AgentLoop(model, tools({}), config=LoopConfig(wrap_up_on_budget=False), budgets=Budgets(max_turns=1))
    result = loop.run("go")
    roles = [m.role for m in result.messages]
    assert roles == [Role.USER, Role.ASSISTANT, Role.USER]
    assert loop.session.pending_tool_batch() is None
    assert isinstance(result.messages[0], Message)
