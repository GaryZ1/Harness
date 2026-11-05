import pytest
from context_fakes import TOOLS

from harness_context import (
    BreakpointLimitError,
    CacheMonitor,
    CachePolicy,
    CacheStabilityChecker,
    CacheStabilityError,
    ConfigurationError,
    PrefixSnapshot,
)
from harness_core.types import Message, ModelRequest, RunContext, TextBlock, Usage


def test_policy_ttl_main_vs_subagent():
    pol = CachePolicy()
    assert pol.cache_control(RunContext("s")) == {"type": "ephemeral", "ttl": "1h"}
    assert pol.cache_control(RunContext("s", depth=1)) == {"type": "ephemeral"}
    with pytest.raises(ConfigurationError):
        CachePolicy(main_ttl="2h")
    with pytest.raises(ConfigurationError):
        CachePolicy(max_breakpoints=5)


@pytest.mark.parametrize(
    "text,kind",
    [
        ("Current time: 2026-11-05T10:00:00Z", "iso_timestamp"),
        ("session 3f2a9c1e-1111-4abc-9def-0123456789ab", "uuid"),
        ("request id req_ab12cd34ef56", "run_id"),
        ("Started at 1762336800", "unix_epoch"),
    ],
)
def test_checker_flags_silent_invalidators(text, kind):
    req = ModelRequest(system=[{"type": "text", "text": "You are helpful. " + text}], messages=[], tools=TOOLS)
    findings = CacheStabilityChecker().scan_request(req)
    assert any(f.kind == kind and f.segment == "system" for f in findings)
    with pytest.raises(CacheStabilityError):
        CacheStabilityChecker(strict=True).enforce(findings)


def test_checker_allow_list_and_clean_prompt():
    req = ModelRequest(system=[{"type": "text", "text": "Policy version 2026-01-01 applies."}], messages=[], tools=TOOLS)
    assert CacheStabilityChecker(allow=["Policy version"]).scan_request(req) == []
    clean = ModelRequest(system=[{"type": "text", "text": "You are a careful agent."}], messages=[], tools=TOOLS)
    assert CacheStabilityChecker(strict=True).scan_request(clean) == []


def test_breakpoint_limit_always_enforced():
    cc = {"type": "ephemeral"}
    msgs = [Message.user([TextBlock(f"m{i}", cache_control=cc)]) for i in range(4)]
    req = ModelRequest(system=[{"type": "text", "text": "s", "cache_control": cc}], messages=msgs)
    findings = CacheStabilityChecker().scan_request(req)
    with pytest.raises(BreakpointLimitError):
        CacheStabilityChecker().enforce(findings)


def test_breakpoint_on_volatile_message_block_flagged():
    cc = {"type": "ephemeral"}
    req = ModelRequest(system="s", messages=[Message.user([TextBlock("now 2026-11-05T10:00:00", cache_control=cc)])])
    assert any(f.kind == "volatile_breakpoint" for f in CacheStabilityChecker().scan_request(req))


def test_compare_locates_divergence():
    m1, m2 = Message.user("a"), Message.assistant([TextBlock("b")])
    base = ModelRequest(system="s", messages=[m1, m2], tools=TOOLS)
    snap = PrefixSnapshot.of(base)
    grown = PrefixSnapshot.of(ModelRequest(system="s", messages=[m1, m2, Message.user("c")], tools=TOOLS))
    assert CacheStabilityChecker.compare(snap, grown) == []  # append-only growth is cache-safe
    edited = PrefixSnapshot.of(ModelRequest(system="s", messages=[m1, Message.assistant([TextBlock("B")])], tools=TOOLS))
    br = CacheStabilityChecker.compare(snap, edited)
    assert [(b.segment, b.index, b.expected) for b in br] == [("messages", 1, False)]
    tools_changed = PrefixSnapshot.of(ModelRequest(system="s2", messages=[m1, m2], tools=TOOLS[:1]))
    segs = {b.segment for b in CacheStabilityChecker.compare(snap, tools_changed, expected_reason="tool change")}
    assert segs == {"tools", "system"}
    assert all(b.expected for b in CacheStabilityChecker.compare(snap, tools_changed, expected_reason="x"))


def test_monitor_separates_expected_rebuilds():
    mon = CacheMonitor()
    assert mon.observe(Usage(input_tokens=50, cache_creation_input_tokens=5000)) is False  # first write
    assert mon.observe(Usage(input_tokens=50, cache_read_input_tokens=5000)) is False
    assert mon.observe(Usage(input_tokens=50, cache_creation_input_tokens=5000), expected_rebuild="compaction") is False
    assert mon.observe(Usage(input_tokens=50, cache_creation_input_tokens=5000)) is True
    s = mon.stats
    assert (s.requests, s.expected_rebuilds, s.unexpected_misses) == (4, 1, 1)
    assert 0.2 < s.hit_rate < 0.3
