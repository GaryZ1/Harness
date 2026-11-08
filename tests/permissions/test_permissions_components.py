"""Exec policy, MCP pins, modes, denial counters and hooks as standalone components."""

import sys

import pytest

from harness_core.types import ToolSpec
from harness_permissions import (
    CommandHook,
    DenialTracker,
    ExecDecision,
    ExecPolicy,
    HookDecision,
    HookInput,
    HookRunner,
    PermissionMode,
    PinStatus,
    PolicyValidationError,
    SandboxAttestation,
    ToolPinStore,
    host_executable,
    least_permissive,
    prefix_rule,
    require_bypass_allowed,
)
from harness_permissions.errors import BypassRefusedError, ModeError


def test_exec_policy_examples_run_at_load_and_strictest_wins():
    allow = prefix_rule(["git", ["status", "diff", "log"]], "allow", "read-only inspection", match=["git status", ["git", "log", "-1"]], not_match=["git push"])
    with pytest.raises(PolicyValidationError, match="should not match"):
        prefix_rule(["git"], "allow", "too broad", not_match=["git push"])
    with pytest.raises(PolicyValidationError, match="justification"):
        prefix_rule(["rm"], "forbidden", "  ")
    ep = ExecPolicy([allow, prefix_rule(["git", "status"], "prompt", "audit this one")])
    ep.add(prefix_rule(["rm", "-rf"], "forbidden", "destructive"))
    assert ep.evaluate_argv(["git", "status"]).decision is ExecDecision.PROMPT
    assert ep.evaluate_argv(["git", "log"]).decision is ExecDecision.ALLOW
    assert ep.evaluate_argv(["git", "push"]) is None
    assert [(t, m.decision.value if m else None) for t, m in ep.check("git diff && rm -rf build")] == [("git diff", "allow"), ("rm -rf build", "forbidden")]


def test_host_executable_pinning():
    ep = ExecPolicy([prefix_rule(["git", "status"], "allow", "ro"), prefix_rule(["git", "reset", "--hard"], "forbidden", "loses work")])
    assert ep.evaluate_argv(["./git", "status"], executable_path="./git") is None          # shim never matches allow
    assert ep.evaluate_argv(["./git", "reset", "--hard"], executable_path="./git").decision is ExecDecision.FORBIDDEN
    ep.pin(host_executable("git", ["/usr/bin/git"]))
    assert ep.evaluate_argv(["/usr/bin/git", "status"], executable_path="/usr/bin/git").decision is ExecDecision.ALLOW
    assert ep.evaluate_argv(["/opt/git", "status"], executable_path="/opt/git") is None
    with pytest.raises(PolicyValidationError):
        host_executable("git", ["relative/git"])


def test_tool_pins_detect_rug_pulls_and_persist(tmp_path):
    path = tmp_path / "pins.json"
    store = ToolPinStore(path)
    spec = ToolSpec("mcp__wa__send", "Send a message.", metadata={"mcp_server_version": "1.0.0"})
    assert store.check(spec).status is PinStatus.NEW
    assert store.check(spec).status is PinStatus.MATCH
    bumped = ToolSpec("mcp__wa__send", "Send a message.", metadata={"mcp_server_version": "1.0.1"})
    assert store.check(bumped).status is PinStatus.CHANGED
    assert ToolPinStore(path).check(spec).status is PinStatus.MATCH


def test_modes_parse_aliases_and_order():
    assert PermissionMode.parse("acceptEdits") is PermissionMode.ACCEPT_EDITS
    assert PermissionMode.parse("bypassPermissions") is PermissionMode.BYPASS
    assert PermissionMode.parse("dontAsk") is PermissionMode.DONT_ASK
    with pytest.raises(ModeError):
        PermissionMode.parse("yolo-max")
    assert least_permissive(PermissionMode.AUTO, PermissionMode.PLAN, PermissionMode.BYPASS) is PermissionMode.PLAN


def test_bypass_guard_and_explicit_attestation(monkeypatch):
    monkeypatch.delenv("HARNESS_EXTERNAL_SANDBOX", raising=False)
    assert not SandboxAttestation.detect().externally_sandboxed      # container markers alone are not enough
    monkeypatch.setenv("HARNESS_EXTERNAL_SANDBOX", "1")
    assert SandboxAttestation.detect().externally_sandboxed
    with pytest.raises(BypassRefusedError, match="managed policy"):
        require_bypass_allowed(SandboxAttestation(externally_sandboxed=True), admin_allows=False)
    require_bypass_allowed(SandboxAttestation(externally_sandboxed=True), admin_allows=True)


def test_denial_tracker_thresholds():
    t = DenialTracker(3, 5)
    t.record_denial()
    t.record_denial()
    t.record_allow()
    assert t.state.consecutive == 0 and t.state.total == 2
    for _ in range(3):
        s = t.record_denial("classifier")
    assert s.limit_reached and "3 consecutive" in s.reason
    t.record_human_resolution()
    assert t.record_denial().limit_reached and "total" in t.state.reason
    with pytest.raises(ValueError):
        DenialTracker(0, 1)


def test_command_hook_protocol():
    hi = HookInput("PreToolUse", "s", "Bash", {"command": "ls"})
    ask = CommandHook("ask", [sys.executable, "-c",
                              "import json,sys; d=json.load(sys.stdin); print(json.dumps({'hookSpecificOutput': {'permissionDecision': 'ask', 'permissionDecisionReason': d['tool_input']['command']}}))"])
    out = HookRunner([ask]).run("PreToolUse", hi)
    assert out.decision is HookDecision.ASK and out.reason == "ls"
    bad = CommandHook("garbage", [sys.executable, "-c", "print('not json')"])
    assert HookRunner([bad]).run("PreToolUse", hi).decision is HookDecision.DENY
    assert HookRunner([bad], on_failure="ignore").run("PreToolUse", hi).decision is HookDecision.DEFER
    scoped = CommandHook("edit-only", [sys.executable, "-c", "import sys; sys.exit(2)"], matcher="Edit|Write")
    assert HookRunner([scoped]).run("PreToolUse", hi).decision is HookDecision.DEFER
    with pytest.raises(ValueError):
        CommandHook("shell-string", "rm -rf /")  # argv lists only, never a shell string
