import copy

import pytest
from context_fakes import tool_turn

from harness_context import (
    ClearingPolicy,
    ConfigurationError,
    OffloadError,
    OffloadingToolExecutor,
    OffloadPolicy,
    ToolOutputOffloader,
    ToolResultClearer,
)
from harness_context.references import extract_locators, find_offload_path
from harness_core.types import Message, RunContext, TextBlock, ToolCall, ToolOutcome, ToolSpec

CTX = RunContext("sess-1")


def test_small_output_passes_through(tmp_path):
    off = ToolOutputOffloader(tmp_path)
    out = ToolOutcome("short")
    assert off.process(ToolCall("t1", "bash", {}), out, CTX) is out


def test_large_output_offloaded_with_preview_and_stats(tmp_path):
    off = ToolOutputOffloader(tmp_path, policy=OffloadPolicy(offload_threshold_tokens=100, max_result_tokens=1000))
    text = "\n".join(f"line {i} " + "y" * 40 for i in range(200))
    res = off.process(ToolCall("toolu/../x", "bash", {"command": "ls"}), ToolOutcome(text, metadata={"untrusted": True}), CTX)
    path = find_offload_path(res.content)
    assert path and path.endswith(".txt") and ".." not in path.split("/")[-1]
    assert (tmp_path / "sess-1").is_dir() and off.restore(path) == text
    assert "line 0" in res.content and "line 199" in res.content and "chars omitted" in res.content
    assert res.metadata["untrusted"] is True and res.metadata["offloaded_path"] == path
    assert "offset/limit" in res.content


def test_cap_without_offload_coaches_narrower_query():
    off = ToolOutputOffloader(None, policy=OffloadPolicy(offload_threshold_tokens=50, max_result_tokens=100))
    res = off.process(ToolCall("t", "grep", {}), ToolOutcome("z" * 4000), CTX)
    assert len(res.content) < 1000 and "truncated" in res.content and "Narrow the request" in res.content
    assert res.metadata["truncated"] is True


def test_offload_exclusions_images_and_sandbox(tmp_path):
    off = ToolOutputOffloader(tmp_path, policy=OffloadPolicy(offload_threshold_tokens=10, max_result_tokens=20))
    big = ToolOutcome("m" * 1000)
    assert off.process(ToolCall("t", "memory", {}), big, CTX) is big
    img = ToolOutcome([{"type": "image", "source": {"data": "x" * 5000}}])
    assert off.process(ToolCall("t", "screenshot", {}), img, CTX) is img
    with pytest.raises(OffloadError):
        off.restore(tmp_path.parent / "elsewhere.txt")
    with pytest.raises(ConfigurationError):
        OffloadPolicy(offload_threshold_tokens=100, max_result_tokens=10)


def test_offloading_executor_wraps_any_executor(tmp_path):
    class Inner:
        def specs(self):
            return [ToolSpec("bash", "run")]

        def execute(self, call, ctx):
            return ToolOutcome("q" * 50_000)

    ex = OffloadingToolExecutor(Inner(), ToolOutputOffloader(tmp_path))
    assert ex.specs()[0].name == "bash"
    assert "<offloaded-tool-output" in ex.execute(ToolCall("c1", "bash", {}), CTX).content


def _window(n=6, size=2000, tool="read_file"):
    log = [Message.user("go")]
    for i in range(n):
        a, r = tool_turn(tool, {"path": f"src/f{i}.py"}, f"body {i} " + "x" * size)
        log += [a, r]
    return log


def test_clearing_plan_keeps_recent_and_respects_threshold():
    win = _window()
    clearer = ToolResultClearer(ClearingPolicy(keep_recent=3, clear_at_least_tokens=100))
    plan = clearer.plan(win)
    ids = [b.tool_use_id for m in win for b in m.tool_results]
    assert plan.applied and plan.cleared_ids == ids[:3]  # oldest first, last 3 kept
    assert plan.freed_tokens > 1000
    stingy = ToolResultClearer(ClearingPolicy(keep_recent=3, clear_at_least_tokens=10**6)).plan(win)
    assert not stingy.applied and "clear_at_least" in stingy.skipped_reason
    excluded = ToolResultClearer(ClearingPolicy(keep_recent=0, clear_at_least_tokens=0)).plan(_window(tool="memory"))
    assert excluded.cleared_ids == []


def test_clearing_placeholders_are_restorable_and_input_untouched(tmp_path):
    win = _window(n=4)
    a, r = tool_turn("bash", {"command": "pytest -q"}, "Traceback ...\nAssertionError: expected 3 got 4\n" + "t" * 3000, is_error=True)
    win += [a, r]
    off_text = ToolOutputOffloader(tmp_path, policy=OffloadPolicy(offload_threshold_tokens=10, max_result_tokens=20)).process(
        ToolCall("big1", "web_fetch", {"url": "https://example.test/x"}), ToolOutcome("w" * 5000), CTX).content
    a2, r2 = tool_turn("web_fetch", {"url": "https://example.test/x"}, off_text)
    win += [a2, r2, Message.user("next")]
    original = copy.deepcopy(win)
    clearer = ToolResultClearer(ClearingPolicy(keep_recent=0, clear_at_least_tokens=0, min_result_tokens=0))
    plan = clearer.plan(win)
    out = clearer.apply(win, plan.cleared_ids)
    assert win == original  # durable messages unchanged
    texts = [b.content for m in out for b in m.tool_results]
    assert "path=src/f0.py" in texts[0] and "re-run read_file" in texts[0]
    assert 'error="true"' in texts[4] and "AssertionError: expected 3 got 4" in texts[4]
    assert "Full output is saved at:" in texts[5]
    assert [b.tool_use_id for m in out for b in m.tool_results] == [b.tool_use_id for m in win for b in m.tool_results]
    assert out[-1] is win[-1]  # untouched messages are reused, not copied
    assert clearer.apply(win, plan.cleared_ids)[1:] == out[1:]  # deterministic


def test_duplicates_cleared_even_inside_keep_window():
    log = [Message.user("go")]
    for _ in range(2):
        a, r = tool_turn("read_file", {"path": "same.py"}, "identical " * 200)
        log += [a, r]
    plan = ToolResultClearer(ClearingPolicy(keep_recent=3, clear_at_least_tokens=0)).plan(log)
    first = log[2].tool_results[0].tool_use_id
    assert plan.cleared_ids == [first] and plan.duplicate_of[first] == log[4].tool_results[0].tool_use_id
    out = ToolResultClearer().apply(log, plan.cleared_ids, duplicate_of=plan.duplicate_of)
    assert "Identical to the later result" in out[2].tool_results[0].content


def test_clear_tool_inputs_never_touches_thinking_turns():
    a1, r1 = tool_turn("bash", {"command": "echo " + "a" * 400}, "o" * 1000)
    a2, r2 = tool_turn("bash", {"command": "echo " + "b" * 400}, "o" * 1000, thinking=True)
    win = [Message.user("go"), a1, r1, a2, r2, Message.assistant([TextBlock("ok")])]
    clearer = ToolResultClearer(ClearingPolicy(keep_recent=0, clear_at_least_tokens=0, clear_tool_inputs=True, dedupe=False))
    out = clearer.apply(win, clearer.plan(win).cleared_ids)
    assert out[1].tool_uses[0].input == {}
    assert out[3] is a2 and out[3].tool_uses[0].input["command"].startswith("echo b")


def test_extract_locators_order_and_truncation():
    loc = extract_locators({"url": "https://a.test", "path": "p.py", "other": 1, "query": "q " * 200})
    assert list(loc) == ["path", "url", "query"] and loc["query"].endswith("...")
    assert extract_locators(None) == {}
