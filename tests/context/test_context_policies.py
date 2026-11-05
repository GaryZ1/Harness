import threading

import pytest
from context_fakes import TOOLS, tool_turn

from harness_context import (
    BudgetLevel,
    BudgetPolicy,
    ConfigurationError,
    ContextReference,
    DelegationBrief,
    ServerContextManagement,
    StuckDetector,
    SubagentLimitError,
    SubagentLimits,
    ToolOutputOffloader,
    UsageCalibration,
    condense_subagent_result,
    render_reference_index,
    tool_count_warning,
)
from harness_context.references import find_offload_path
from harness_core.types import Message, RunContext, ToolSpec, Usage


def test_budget_trigger_is_min_of_threshold_and_90_percent():
    assert BudgetPolicy(window_tokens=1_000_000).compaction_trigger == 150_000
    small = BudgetPolicy(window_tokens=100_000)
    assert small.compaction_trigger == 90_000 and small.effective_window == 95_000 and small.warn_at == 81_000
    assert small.classify(10_000) is BudgetLevel.OK
    assert small.classify(85_000) is BudgetLevel.WARN
    assert small.classify(91_000) is BudgetLevel.COMPACT
    assert small.classify(91_000, max_tokens=8_000) is BudgetLevel.OVERFLOW
    assert BudgetPolicy().classify(10**7) is BudgetLevel.COMPACT  # unknown window never overflows
    assert "next compaction at ~90,000" in small.note(1_234) and "remaining" in small.note(1_234)
    with pytest.raises(ConfigurationError):
        BudgetPolicy(window_tokens=0)
    with pytest.raises(ConfigurationError):
        BudgetPolicy(warn_fraction=1.5)


def test_usage_calibration_is_clamped():
    cal = UsageCalibration()
    assert cal.observe(1000, Usage(input_tokens=100, cache_read_input_tokens=1100, cache_creation_input_tokens=300)) == 1.5
    assert cal.apply(100) == 150
    assert cal.observe(10, Usage(input_tokens=10_000)) == 3.0
    assert cal.observe(10_000, Usage(input_tokens=1)) == 0.5


def test_server_edits_validation_and_defaults():
    cm = ServerContextManagement(clear_tool_uses=True, keep_thinking_turns=2, clear_thinking=True)
    edits = cm.edits()
    assert edits[0] == {"type": "clear_thinking_20251015", "keep": {"type": "thinking_turns", "value": 2}}
    assert edits[1]["trigger"] == {"type": "input_tokens", "value": 100_000} and edits[1]["keep"]["value"] == 3
    assert "memory" in edits[1]["exclude_tools"] and "clear_tool_inputs" not in edits[1]
    compact = ServerContextManagement(compact=True, pause_after_compaction=True).edits()[0]
    assert compact["trigger"]["value"] == 150_000 and compact["pause_after_compaction"] is True
    assert "Do not call any tools" in compact["instructions"]
    assert ServerContextManagement().apply({"x": 1}) == {"x": 1}
    with pytest.raises(ConfigurationError):
        ServerContextManagement(compact=True, compact_trigger_tokens=40_000)


def test_delegation_brief_renders_all_fields():
    brief = DelegationBrief(
        objective="Find where sessions expire", inputs=["src/auth/session.py"], boundaries=["read-only"],
        tools=["grep", "read_file"], budget="20 tool calls", output_schema={"type": "object", "properties": {"file": {}}},
    )
    text = brief.render()
    for tag in ("<objective>", "<inputs>", "<boundaries>", "<tools>", "<budget>", "<output-schema>", "<result-guidance>"):
        assert tag in text
    with pytest.raises(ConfigurationError):
        DelegationBrief(objective="  ")


def test_subagent_limits_depth_and_concurrency():
    lim = SubagentLimits(max_depth=3, max_concurrent=2)
    with pytest.raises(SubagentLimitError):
        lim.check_depth(RunContext("s", depth=3))
    with lim.slot(RunContext("s")) as d1, lim.slot(RunContext("s", depth=1)) as d2:
        assert (d1, d2, lim.active) == (1, 2, 2)
        with pytest.raises(SubagentLimitError):
            with lim.slot(RunContext("s")):
                pass
    assert lim.active == 0
    errors = []

    def worker():
        try:
            with lim.slot(RunContext("s")):
                pass
        except SubagentLimitError as exc:  # pragma: no cover - only under contention
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert lim.active == 0


def test_condensed_subagent_result_is_capped_offloaded_and_untrusted(tmp_path):
    ctx = RunContext("s")
    small = condense_subagent_result("Found it in session.py:42", agent_name="explorer", ctx=ctx)
    assert small.metadata["untrusted"] and 'trust="untrusted"' in small.content and "no user authority" in small.content
    big = condense_subagent_result("detail " * 5000, agent_name="explorer", ctx=ctx, offloader=ToolOutputOffloader(tmp_path))
    assert big.metadata["truncated"] and big.metadata["offloaded_path"].endswith(".txt")
    assert len(big.content) < 10_000 and "Full report" in big.content
    no_off = condense_subagent_result("detail " * 5000, agent_name="explorer", ctx=ctx)
    assert "truncated" in no_off.content and "offloaded_path" not in no_off.metadata


def test_stuck_detector_and_tool_count_warning():
    det = StuckDetector()
    log = [Message.user("go")]
    for i in range(3):
        log += list(tool_turn("bash", {"command": f"try {i}"}, "boom", is_error=True))
    assert "failed with the same error" in det.check(log)
    log += list(tool_turn("bash", {"command": "ok"}, "fine"))
    assert det.check(log) is None
    assert tool_count_warning(TOOLS) is None
    many = [ToolSpec(f"t{i}", "d") for i in range(12)]
    assert "5-10" in tool_count_warning(many)
    assert "30-50" in tool_count_warning([ToolSpec(f"t{i}", "d") for i in range(40)])
    assert tool_count_warning([ToolSpec(f"t{i}", "d", defer_loading=True) for i in range(40)]) is None


def test_reference_index_is_sorted_and_deduplicated():
    refs = [
        ContextReference("url", "https://b.test", "API docs"),
        ContextReference("file", "src/a.py", size_hint="12 KB", how="read_file(path, offset, limit)"),
        ContextReference("url", "https://b.test", "dup"),
    ]
    text = render_reference_index(refs)
    assert text.count("https://b.test") == 1 and text.index("src/a.py") < text.index("https://b.test")
    assert render_reference_index([]) == ""
    assert find_offload_path('<offloaded-tool-output path="/w/x.txt" bytes="1">') == "/w/x.txt"
