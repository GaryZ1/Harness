"""AgentLoop: stop predicate, tool batches, parallelism, stop_reason handling, compaction, steering."""

from __future__ import annotations

import threading

from harness_core import (
    AgentLoop,
    AllowAllPolicy,
    CompactionResult,
    ContextWindowExceededError,
    EventType,
    FunctionToolExecutor,
    LoopConfig,
    Message,
    OpaqueBlock,
    ResultSubtype,
    Role,
    SandboxError,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    Usage,
)
from harness_core.testing import RecordingSink, RecordingTracer, ScriptedModelClient, make_response, text_response, tool_response


def make_tools(log: list | None = None) -> FunctionToolExecutor:
    ex = FunctionToolExecutor()
    log = log if log is not None else []

    @ex.tool("read", "Read a file", read_only=True)
    def read(inp, ctx):
        log.append(("read", inp.get("path")))
        return f"contents of {inp.get('path')}"

    @ex.tool("write", "Write a file")
    def write(inp, ctx):
        log.append(("write", inp.get("path")))
        return "written"

    @ex.tool("boom", "Always fails")
    def boom(inp, ctx):
        raise RuntimeError("kaput")

    return ex


def test_text_only_response_ends_the_run():
    model = ScriptedModelClient([text_response("hello")])
    result = AgentLoop(model, system="sys").run("hi")
    assert result.subtype is ResultSubtype.SUCCESS and result.ok
    assert result.final_text == "hello" and result.num_turns == 1
    assert result.stop_reason is StopReason.END_TURN
    assert model.requests[0].system == "sys"


def test_all_tool_results_return_in_one_user_message_in_call_order():
    log: list = []
    model = ScriptedModelClient([
        tool_response(("read", {"path": "a"}, "t1"), ("write", {"path": "b"}, "t2"), ("read", {"path": "c"}, "t3")),
        text_response("done"),
    ])
    result = AgentLoop(model, make_tools(log), policy=AllowAllPolicy()).run("go")
    tool_msg = result.messages[2]
    assert tool_msg.role is Role.USER
    assert [r.tool_use_id for r in tool_msg.tool_results] == ["t1", "t2", "t3"]
    assert log == [("read", "a"), ("write", "b"), ("read", "c")]  # sequential order preserved around the write
    # the next request carries the full, unchanged history
    assert [m.role for m in model.requests[1].messages] == [Role.USER, Role.ASSISTANT, Role.USER]


def test_read_only_tools_run_concurrently():
    barrier = threading.Barrier(3, timeout=5)
    ex = FunctionToolExecutor()

    @ex.tool("probe", "read-only probe", read_only=True)
    def probe(inp, ctx):
        barrier.wait()  # only passes if all three run at the same time
        return "ok"

    model = ScriptedModelClient([tool_response(("probe", {"i": 1}), ("probe", {"i": 2}), ("probe", {"i": 3})), text_response("done")])
    result = AgentLoop(model, ex).run("go")
    assert [r.is_error for r in result.messages[2].tool_results] == [False, False, False]


def test_mutating_tools_never_overlap():
    active, peak = [0], [0]
    lock = threading.Lock()
    ex = FunctionToolExecutor()

    @ex.tool("mut", "mutating")
    def mut(inp, ctx):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        threading.Event().wait(0.01)
        with lock:
            active[0] -= 1
        return "ok"

    model = ScriptedModelClient([tool_response(*[("mut", {"i": i}) for i in range(4)]), text_response("done")])
    AgentLoop(model, ex, policy=AllowAllPolicy()).run("go")
    assert peak[0] == 1


def test_unknown_tool_and_tool_exceptions_return_errors_to_the_model():
    model = ScriptedModelClient([tool_response(("nope", {}, "t1"), ("boom", {}, "t2")), text_response("recovered")])
    result = AgentLoop(model, make_tools(), policy=AllowAllPolicy()).run("go")
    r1, r2 = result.messages[2].tool_results
    assert r1.is_error and "unknown tool 'nope'" in r1.text and "read" in r1.text
    assert r2.is_error and "RuntimeError: kaput" in r2.text
    assert result.ok


def test_executor_that_raises_is_contained():
    class Exploding:
        def specs(self):
            return [ToolSpec("x", "x", read_only=True)]

        def execute(self, call, ctx):
            raise SandboxError("container died")

    model = ScriptedModelClient([tool_response(("x", {})), text_response("ok")])
    result = AgentLoop(model, Exploding()).run("go")
    assert result.ok and "container died" in result.messages[2].tool_results[0].text


def test_opaque_thinking_blocks_are_replayed_verbatim():
    thinking = OpaqueBlock({"type": "thinking", "thinking": "plan", "signature": "SIG"})
    first = make_response(thinking, TextBlock("let me read"), *tool_response(("read", {"path": "a"}, "t1")).content,
                          stop_reason=StopReason.TOOL_USE)
    model = ScriptedModelClient([first, text_response("done")])
    AgentLoop(model, make_tools()).run("go")
    replayed = model.requests[1].messages[1]
    assert replayed.content[0].to_api() == {"type": "thinking", "thinking": "plan", "signature": "SIG"}
    assert [b.to_api()["type"] for b in replayed.content] == ["thinking", "text", "tool_use"]


def test_pause_turn_is_resent_without_a_user_message():
    server = OpaqueBlock({"type": "server_tool_use", "id": "srv1", "name": "web_search", "input": {"q": "x"}})
    model = ScriptedModelClient([make_response(server, stop_reason=StopReason.PAUSE_TURN), text_response("final")])
    result = AgentLoop(model).run("search")
    assert result.ok and result.final_text == "final" and model.calls == 2
    assert model.requests[1].messages[-1].role is Role.ASSISTANT
    assert model.requests[1].messages[-1].content[0].to_api()["id"] == "srv1"


def test_max_tokens_text_gets_a_continuation_message():
    model = ScriptedModelClient([text_response("part one", stop_reason=StopReason.MAX_TOKENS), text_response("part two")])
    result = AgentLoop(model).run("write an essay")
    assert result.ok and model.calls == 2
    cont = model.requests[1].messages[-1]
    assert cont.role is Role.USER and "cut off" in cont.text


def test_max_tokens_truncated_tool_call_escalates_then_errors_at_ceiling():
    trunc = tool_response(("write", {"path": "big"}), stop_reason=StopReason.MAX_TOKENS)
    model = ScriptedModelClient([trunc, trunc, text_response("smaller edits then")])
    loop = AgentLoop(model, make_tools(), config=LoopConfig(max_tokens=1000, max_tokens_ceiling=2000), policy=AllowAllPolicy())
    result = loop.run("go")
    assert [r.max_tokens for r in model.requests] == [1000, 2000, 2000]
    # first truncated response was discarded (never entered history); the second was paired with an error
    assert len(result.messages) == 4
    err = result.messages[2].tool_results[0]
    assert err.is_error and "truncated" in err.text
    discarded = [e for e in loop.session.events() if e.type is EventType.MODEL_RESPONSE and e.data.get("discarded")]
    assert len(discarded) == 1


def test_refusal_ends_the_run_with_a_typed_result():
    model = ScriptedModelClient([text_response("I can't help with that", stop_reason=StopReason.REFUSAL)])
    result = AgentLoop(model).run("bad")
    assert result.subtype is ResultSubtype.ERROR_REFUSAL and result.stop_reason is StopReason.REFUSAL
    assert result.messages[-1].role is Role.ASSISTANT  # still appended: history is append-only


class SummaryCompactor:
    def __init__(self, fail: bool = False):
        self.calls = 0
        self.fail = fail

    def compact(self, messages, ctx, *, reason):
        self.calls += 1
        if self.fail:
            return None
        return CompactionResult([Message.user(f"SUMMARY ({reason}) of {len(messages)} messages")], summary="s")


def test_context_window_exceeded_triggers_compaction_and_retry():
    model = ScriptedModelClient([make_response(stop_reason=StopReason.MODEL_CONTEXT_WINDOW_EXCEEDED), text_response("fits now")])
    comp = SummaryCompactor()
    loop = AgentLoop(model, compactor=comp)
    result = loop.run("x" * 400)
    assert result.ok and comp.calls == 1
    assert model.requests[1].messages[0].text.startswith("SUMMARY (overflow)")
    assert loop.session.messages()[0].text == "x" * 400  # canonical log untouched
    assert any(e.type is EventType.COMPACTION and "view" in e.data for e in loop.session.events())


def test_context_overflow_exception_also_compacts():
    model = ScriptedModelClient([ContextWindowExceededError("prompt is too long"), text_response("ok")])
    result = AgentLoop(model, compactor=SummaryCompactor()).run("x" * 400)
    assert result.ok


def test_compaction_thrash_guard_terminates():
    over = make_response(stop_reason=StopReason.MODEL_CONTEXT_WINDOW_EXCEEDED)
    comp = SummaryCompactor(fail=True)
    model = ScriptedModelClient(default=over)
    result = AgentLoop(model, compactor=comp, config=LoopConfig(max_compaction_attempts=3)).run("x")
    assert result.subtype is ResultSubtype.ERROR_CONTEXT_WINDOW and comp.calls == 3


def test_overflow_without_compactor_is_terminal():
    model = ScriptedModelClient([make_response(stop_reason=StopReason.MODEL_CONTEXT_WINDOW_EXCEEDED)])
    assert AgentLoop(model).run("x").subtype is ResultSubtype.ERROR_CONTEXT_WINDOW


def test_invalid_compaction_breaking_tool_pairing_is_rejected():
    class Orphaning:
        def compact(self, messages, ctx, *, reason):
            return CompactionResult([Message.user([ToolResultBlock("ghost", "x")])])

    model = ScriptedModelClient([make_response(stop_reason=StopReason.MODEL_CONTEXT_WINDOW_EXCEEDED)] * 3)
    loop = AgentLoop(model, compactor=Orphaning())
    result = loop.run("x" * 100)
    assert result.subtype is ResultSubtype.ERROR_CONTEXT_WINDOW
    assert not loop.session.compacted
    assert any("invalid compaction" in str(e.data.get("failed")) for e in loop.session.events() if e.type is EventType.COMPACTION)


def test_proactive_compaction_at_threshold_before_overflow():
    big = Usage(input_tokens=950, output_tokens=10)
    model = ScriptedModelClient([tool_response(("read", {"path": "a"}), usage=big), text_response("done")])
    comp = SummaryCompactor()
    loop = AgentLoop(model, make_tools(), compactor=comp, config=LoopConfig(context_window_tokens=1000))
    result = loop.run("x" * 400)
    assert result.ok and comp.calls == 1
    assert model.requests[1].messages[0].text.startswith("SUMMARY (threshold)")


def test_steering_input_joins_the_tool_result_message():
    ex = FunctionToolExecutor()
    holder: dict = {}

    @ex.tool("slow", "read-only", read_only=True)
    def slow(inp, ctx):
        holder["loop"].steer("also check the README")
        return "ok"

    model = ScriptedModelClient([tool_response(("slow", {})), text_response("done")])
    loop = AgentLoop(model, ex)
    holder["loop"] = loop
    loop.run("go")
    last = model.requests[1].messages[-1]
    assert [b.type for b in last.content] == ["tool_result", "text"]
    assert last.content[1].text == "also check the README"


def test_queued_input_prevents_stopping():
    model = ScriptedModelClient([text_response("first answer"), text_response("second answer")])
    loop = AgentLoop(model)
    loop.steer("one more thing")  # queued before the model's text-only reply is processed
    result = loop.run("start")
    assert model.calls == 2 and result.final_text == "second answer"
    assert model.requests[1].messages[-1].text == "one more thing"


def test_events_and_spans_are_emitted():
    sink, tracer = RecordingSink(), RecordingTracer()
    model = ScriptedModelClient([tool_response(("read", {"path": "a"})), text_response("done")])
    loop = AgentLoop(model, make_tools(), sinks=[sink], tracer=tracer)
    loop.run("go")
    types = sink.types()
    for t in ("session_start", "user_message", "model_request", "model_response", "tool_call", "permission", "tool_result", "stop"):
        assert t in types
    assert {"invoke_agent", "chat", "execute_tool"} <= set(tracer.names())
    executing = [e for e in sink.events if e.type is EventType.TOOL_CALL and e.data["phase"] == "executing"]
    assert len(executing[0].data["idempotency_key"]) == 32


def test_open_world_tool_taints_the_run_context():
    ex = FunctionToolExecutor()
    ex.tool("fetch", "web fetch", read_only=True, open_world=True)(lambda inp, ctx: "<html>")
    model = ScriptedModelClient([tool_response(("fetch", {})), text_response("done")])
    loop = AgentLoop(model, ex)
    loop.run("go")
    assert loop.ctx.tainted
    assert any(e.data.get("tainted") for e in loop.session.events())


def test_tool_specs_are_frozen_for_the_run():
    ex = make_tools()
    model = ScriptedModelClient([tool_response(("read", {"path": "a"})), text_response("done")])

    def mutate(req):
        ex.add(ToolSpec("late", "added mid-run"), lambda i, c: "x")
        return tool_response(("read", {"path": "b"}))

    model.script.insert(0, mutate)
    AgentLoop(model, ex).run("go")
    assert [len(r.tools) for r in model.requests] == [3, 3, 3]
