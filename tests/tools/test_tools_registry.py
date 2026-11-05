from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from harness_core.types import (
    Message,
    PermissionBehavior,
    PermissionDecision,
    RunContext,
    ToolCall,
    ToolExecutor,
    ToolOutcome,
    ToolSpec,
    ToolUseBlock,
)
from harness_memory import TaintTracker
from harness_tools import OutputPolicy, Tool, ToolDefinitionError, ToolInputError, ToolRegistry, tool_results_message
from harness_tools.catalog import server_tool_search_spec


@pytest.fixture
def ctx() -> RunContext:
    return RunContext(session_id="s1")


@pytest.fixture
def reg() -> ToolRegistry:
    r = ToolRegistry()

    @r.tool(read_only=True, input_examples=[{"user_id": "USR-1", "limit": 2}])
    def lookup(user_id: str, limit: int = 10) -> dict:
        """Look up a user's recent orders by user ID."""
        return {"user_id": user_id, "limit": limit}

    @r.tool
    def explode() -> str:
        """Always raises, to test error conversion."""
        raise RuntimeError("kaboom")

    return r


def call(name: str, inp=None, cid: str = "t1") -> ToolCall:
    return ToolCall(cid, name, {} if inp is None else inp)


def test_registry_implements_tool_executor_and_sorted_specs(reg):
    assert isinstance(reg, ToolExecutor)
    assert [s.name for s in reg.specs()] == ["explode", "lookup"]
    spec = reg.get("lookup").spec
    assert spec.read_only and spec.metadata["concurrency_safe"] and spec.metadata["idempotent"]
    assert reg.api_tools(include_input_examples=True)[1]["input_examples"] == [{"user_id": "USR-1", "limit": 2}]
    assert "input_examples" not in reg.api_tools()[1]


def test_success_applies_defaults_and_serialises_json(reg, ctx):
    out = reg.execute(call("lookup", {"user_id": "u"}), ctx)
    assert not out.is_error and '"limit": 10' in out.content
    assert out.metadata["tool"] == "lookup" and out.metadata["idempotency_key"].startswith("t1:")


def test_validation_error_is_a_tool_result_with_example(reg, ctx):
    out = reg.execute(call("lookup", {"limit": "five", "bogus": 1}), ctx)
    assert out.is_error and out.metadata["error_class"] == "ValidationError"
    assert "/user_id: is a required property" in out.content and "/limit: expected integer" in out.content
    assert "/bogus: unexpected property" in out.content
    assert 'Example of a valid input: {"user_id": "USR-1", "limit": 2}' in out.content


def test_string_input_is_parsed_as_json_never_string_matched(reg, ctx):
    assert not reg.execute(call("lookup", '{"user_id": "u"}'), ctx).is_error  # type: ignore[arg-type]
    bad = reg.execute(call("lookup", '["u"]'), ctx)  # type: ignore[arg-type]
    assert bad.is_error and "must be a JSON object" in bad.content
    assert "not valid JSON" in reg.execute(call("lookup", "{user_id: u}"), ctx).content  # type: ignore[arg-type]


def test_unknown_tool_suggests_close_matches(reg, ctx):
    out = reg.execute(call("lookpu"), ctx)
    assert out.is_error and "Did you mean: lookup" in out.content and out.metadata["error_class"] == "ToolNotFound"


def test_handler_exception_becomes_error_result(reg, ctx):
    out = reg.execute(call("explode"), ctx)
    assert out.is_error and "RuntimeError: kaboom" in out.content and "Traceback" not in out.content


def test_tool_input_error_carries_example(ctx):
    r = ToolRegistry()

    def handler(inp, inv):
        raise ToolInputError("date must be ISO", example={"date": "2026-01-31"})

    r.register(Tool(ToolSpec("book", "Book a meeting room for a date.", {"type": "object", "properties": {"date": {"type": "string"}}}), handler))
    out = r.execute(call("book", {"date": "tomorrow"}), ctx)
    assert out.is_error and '{"date": "2026-01-31"}' in out.content


def test_timeout_returns_error_and_signals_cancellation(ctx):
    r = ToolRegistry()
    seen = threading.Event()

    def slow(inp, inv):
        inv.cancelled.wait(5)
        seen.set()
        return "late"

    r.register(Tool(ToolSpec("slow", "Sleeps until cancelled, to test timeouts."), slow, timeout_s=0.2))
    t0 = time.monotonic()
    out = r.execute(call("slow"), ctx)
    assert out.is_error and "timed out after 0.2s" in out.content and time.monotonic() - t0 < 2
    assert seen.wait(2)


def test_registration_invariants():
    r = ToolRegistry()
    spec = ToolSpec("ok_name", "A perfectly fine description of the tool.")
    r.register(Tool(spec, lambda i, v: "x"))
    with pytest.raises(ToolDefinitionError, match="duplicate"):
        r.register(Tool(spec, lambda i, v: "x"))
    with pytest.raises(ToolDefinitionError, match="invalid tool name"):
        r.register(Tool(ToolSpec("bad name!", "desc"), lambda i, v: "x"))
    with pytest.raises(ToolDefinitionError, match="description"):
        r.register(Tool(ToolSpec("nodesc", ""), lambda i, v: "x"))
    with pytest.raises(ToolDefinitionError, match="additionalProperties"):
        r.register(Tool(ToolSpec("strict_t", "Strict tool without closed schema.", strict=True), lambda i, v: "x"))
    with pytest.raises(ToolDefinitionError, match="input_example"):
        r.register(Tool(ToolSpec("ex", "Tool with a bad example.", {"type": "object", "properties": {"n": {"type": "integer"}}}),
                        lambda i, v: "x", input_examples=[{"n": "x"}]))


def test_freeze_blocks_hot_tool_changes_but_allows_deferred(reg):
    reg.freeze()
    with pytest.raises(ToolDefinitionError, match="frozen"):
        reg.register(Tool(ToolSpec("late", "Late hot tool added mid-session."), lambda i, v: "x"))
    with pytest.raises(ToolDefinitionError, match="frozen"):
        reg.unregister("lookup")
    with pytest.raises(ToolDefinitionError, match="frozen"):
        reg.register_spec(server_tool_search_spec("bm25"))
    reg.register(Tool(ToolSpec("later", "Deferred tool registered after freeze.", defer_loading=True), lambda i, v: "x"))
    assert "later" in reg


def test_output_truncated_in_place_with_hint(ctx):
    r = ToolRegistry(output_policy=OutputPolicy(max_tokens=100))
    r.register(Tool(ToolSpec("big", "Returns a large payload for truncation tests."), lambda i, v: "\n".join(f"line {n}" for n in range(1000))))
    out = r.execute(call("big"), ctx)
    assert out.metadata["truncated"] and len(out.content) <= 100 * 4
    assert "line 0" in out.content and "line 999" in out.content and "result limit" in out.content


def test_output_offloaded_to_file_with_preview(tmp_path, ctx):
    r = ToolRegistry(output_policy=OutputPolicy(max_tokens=200, preview_tokens=50, offload_dir=tmp_path / "out"))
    payload = "".join(f"row {n}\n" for n in range(5000))
    r.register(Tool(ToolSpec("big", "Returns a large payload for offload tests."), lambda i, v: payload))
    out = r.execute(call("big", cid="toolu/../x"), ctx)
    path = out.metadata["offload_path"]
    assert Path(path).parent == tmp_path / "out"  # the call id is sanitised; it cannot pick the directory
    assert open(path).read() == payload
    assert path in out.content and "~" in out.content and len(out.content) < 1000


def test_open_world_output_taints_context_and_tracker():
    tracker = TaintTracker()
    r = ToolRegistry(taint_tracker=tracker)
    r.register(Tool(ToolSpec("web_fetch", "Fetch a web page (third-party content).", open_world=True, read_only=True), lambda i, v: "<html>"))
    r.register(Tool(ToolSpec("local", "Local trusted computation for taint tests."), lambda i, v: "ok"))
    ctx = RunContext(session_id="s")
    assert not r.execute(call("local"), ctx).metadata["untrusted"] and not ctx.tainted
    out = r.execute(call("web_fetch"), ctx)
    assert out.metadata["untrusted"] and ctx.tainted and tracker.polluted and tracker.reasons == ["web_fetch"]


class Policy:
    def __init__(self, behavior, updated=None):
        self.behavior, self.updated = behavior, updated

    def check(self, call, spec, ctx):
        return PermissionDecision(self.behavior, "test policy", self.updated)


def test_permission_deny_ask_and_updated_input(reg, ctx):
    reg.permission_policy = Policy(PermissionBehavior.DENY)
    out = reg.execute(call("lookup", {"user_id": "u"}), ctx)
    assert out.is_error and "Permission denied" in out.content and out.metadata["error_class"] == "PermissionDenied"
    reg.permission_policy = Policy(PermissionBehavior.ASK)
    assert "Permission required" in reg.execute(call("lookup", {"user_id": "u"}), ctx).content
    reg.approver = lambda c, s, x, reason: True
    assert not reg.execute(call("lookup", {"user_id": "u"}), ctx).is_error
    reg.permission_policy = Policy(PermissionBehavior.ALLOW, {"user_id": "rewritten"})
    assert '"rewritten"' in reg.execute(call("lookup", {"user_id": "u"}), ctx).content
    reg.permission_policy = Policy(PermissionBehavior.ALLOW, {"user_id": 5})  # rewritten input is re-validated
    assert reg.execute(call("lookup", {"user_id": "u"}), ctx).is_error


def test_escalation_after_three_consecutive_errors(reg, ctx):
    outs = [reg.execute(call("explode"), ctx) for _ in range(3)]
    assert "escalate" not in outs[1].metadata and "escalate" in outs[2].metadata
    reg.errors.reset()
    reg.execute(call("explode"), ctx)
    assert reg.execute(call("lookup", {"user_id": "u"}), ctx).metadata.get("escalate") is None
    assert reg.errors.consecutive == 0 and reg.errors.total == 1


def test_parallel_batch_runs_read_only_concurrently_and_keeps_order(ctx):
    r = ToolRegistry(max_concurrency=4)
    barrier = threading.Barrier(3, timeout=5)
    order: list[str] = []
    lock = threading.Lock()

    def reader(inp, inv):
        barrier.wait()  # deadlocks (BrokenBarrierError) unless 3 readers run at once
        return f"r{inp['n']}"

    def writer(inp, inv):
        with lock:
            order.append(f"w{inp['n']}")
        return f"w{inp['n']}"

    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    r.register(Tool(ToolSpec("reader", "Concurrency-safe reader for batch tests.", schema, read_only=True), reader))
    r.register(Tool(ToolSpec("writer", "Mutating writer for batch tests.", schema), writer))
    calls = [ToolCall(f"c{i}", "reader", {"n": i}) for i in range(3)] + [ToolCall("c3", "writer", {"n": 3}), ToolCall("c4", "nope", {})]
    outs = r.execute_batch(calls, ctx)
    assert [o.content for o in outs[:4]] == ["r0", "r1", "r2", "w3"] and outs[4].is_error


def test_run_tool_uses_answers_every_tool_use_in_one_message(reg, ctx):
    turn = Message.assistant([ToolUseBlock("a", "lookup", {"user_id": "u"}), ToolUseBlock("b", "explode", {}), ToolUseBlock("c", "missing", {})])
    reply = reg.run_tool_uses(turn, ctx)
    api = reply.to_api()
    assert api["role"] == "user" and [b["tool_use_id"] for b in api["content"]] == ["a", "b", "c"]
    assert [b.get("is_error", False) for b in api["content"]] == [False, True, True]
    with pytest.raises(ValueError):
        tool_results_message([call("x")], [])


def test_telemetry_record_and_provider_tool(ctx):
    records = []
    r = ToolRegistry(on_record=records.append)
    r.register_spec(server_tool_search_spec("regex"))
    r.register(Tool(ToolSpec("echo", "Echo back a ToolOutcome as-is."), lambda i, v: ToolOutcome("hi", metadata={"exit_code": 0})))
    out = r.execute(call("echo", {"x": 1}), ctx)
    assert out.metadata["exit_code"] == 0 and records[-1].tool == "echo" and len(records[-1].args_sha256) == 64
    prov = r.execute(call("tool_search_tool_regex"), ctx)
    assert prov.is_error and "provider-executed" in prov.content and records[-1].error_class == "ProviderTool"
