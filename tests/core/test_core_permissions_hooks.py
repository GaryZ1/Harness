"""Permission pipeline (policy -> hooks -> approver), durable approval pauses, lifecycle hooks."""

from __future__ import annotations

import json
import shlex
import sys

from harness_core import (
    AgentLoop,
    AllowAllPolicy,
    ApprovalRequired,
    CommandHook,
    DenyAllPolicy,
    EventType,
    FunctionToolExecutor,
    HookEvent,
    HookInput,
    HookRegistry,
    HookResult,
    PermissionBehavior,
    PermissionDecision,
    ResultSubtype,
    RunContext,
    SessionStatus,
    SessionStore,
    ToolCall,
    ToolOutcome,
)
from harness_core.testing import ScriptedModelClient, text_response, tool_response


def tools(log: list) -> FunctionToolExecutor:
    ex = FunctionToolExecutor()
    ex.tool("read", "read", read_only=True)(lambda inp, ctx: log.append(("read", inp)) or "R")
    ex.tool("write", "write")(lambda inp, ctx: log.append(("write", inp)) or "W")
    return ex


def test_default_policy_runs_read_only_and_denies_mutations_without_an_approver():
    log: list = []
    model = ScriptedModelClient([tool_response(("read", {}, "t1"), ("write", {"p": 1}, "t2")), text_response("ok")])
    loop = AgentLoop(model, tools(log))
    result = loop.run("go")
    r1, r2 = result.messages[2].tool_results
    assert not r1.is_error and r2.is_error and "Permission denied" in r2.text
    assert log == [("read", {})]
    perms = [e.data["behavior"] for e in loop.session.events() if e.type is EventType.PERMISSION]
    assert perms == ["allow", "deny"]


def test_approver_allows_denies_and_rewrites_input():
    log: list = []
    asked = []

    def approver(call, spec, ctx, decision):
        asked.append(call.id)
        if call.id == "t1":
            return PermissionDecision(PermissionBehavior.ALLOW, updated_input={"p": "sanitized"})
        return False

    model = ScriptedModelClient([tool_response(("write", {"p": "raw"}, "t1"), ("write", {"p": 2}, "t2")), text_response("ok")])
    result = AgentLoop(model, tools(log), approver=approver).run("go")
    assert asked == ["t1", "t2"]
    assert log == [("write", {"p": "sanitized"})]
    assert result.messages[2].tool_results[1].is_error


def test_deny_all_policy_blocks_even_read_only():
    log: list = []
    model = ScriptedModelClient([tool_response(("read", {})), text_response("ok")])
    AgentLoop(model, tools(log), policy=DenyAllPolicy()).run("go")
    assert log == []


def test_approval_pause_is_durable_and_resumable(tmp_path):
    log: list = []
    store = SessionStore(tmp_path)

    def pause(call, spec, ctx, decision):
        raise ApprovalRequired("needs a human")

    model = ScriptedModelClient([tool_response(("read", {}, "t1"), ("write", {"p": 1}, "t2"))])
    session = store.create("s")
    result = AgentLoop(model, tools(log), session=session, approver=pause).run("go")
    assert result.subtype is ResultSubtype.PAUSED_FOR_APPROVAL
    assert [p.tool_use_id for p in result.pending_approvals] == ["t2"]
    assert store.open("s").info.status is SessionStatus.PAUSED
    assert log == [("read", {})]

    # a new harness process wakes the session and supplies the decision
    model2 = ScriptedModelClient([text_response("wrote it")])
    resumed = AgentLoop(model2, tools(log), session=store.wake("s"), approver=pause).run(approvals={"t2": True})
    assert resumed.ok and resumed.final_text == "wrote it"
    assert log == [("read", {}), ("write", {"p": 1})]  # read was not re-run
    results = resumed.messages[2].tool_results
    assert [r.tool_use_id for r in results] == ["t1", "t2"] and not any(r.is_error for r in results)


def test_pre_tool_hook_can_deny_rewrite_and_allow_past_ask():
    log: list = []
    hooks = HookRegistry()

    @hooks.hook(HookEvent.PRE_TOOL, matcher="write")
    def guard(inp: HookInput):
        if inp.tool_call.input.get("p") == "tests/":
            return HookResult(permission=PermissionBehavior.DENY, reason="tests are immutable")
        return HookResult(permission=PermissionBehavior.ALLOW, updated_input={**inp.tool_call.input, "sandbox": True})

    model = ScriptedModelClient([tool_response(("write", {"p": "tests/"}, "t1"), ("write", {"p": "src/"}, "t2")), text_response("ok")])
    result = AgentLoop(model, tools(log), hooks=hooks).run("go")  # default policy would ASK -> deny; hook allow overrides
    r1, r2 = result.messages[2].tool_results
    assert r1.is_error and "tests are immutable" in r1.text
    assert not r2.is_error
    assert log == [("write", {"p": "src/", "sandbox": True})]


def test_policy_deny_beats_hook_allow():
    log: list = []
    hooks = HookRegistry()
    hooks.on(HookEvent.PRE_TOOL, lambda inp: HookResult(permission=PermissionBehavior.ALLOW))
    model = ScriptedModelClient([tool_response(("read", {})), text_response("ok")])
    AgentLoop(model, tools(log), hooks=hooks, policy=DenyAllPolicy()).run("go")
    assert log == []


def test_post_tool_hook_rewrites_outcome_and_adds_context():
    hooks = HookRegistry()
    hooks.on(HookEvent.POST_TOOL, lambda inp: HookResult(updated_outcome=ToolOutcome("[redacted]"), additional_context="note: redacted"))
    model = ScriptedModelClient([tool_response(("read", {})), text_response("ok")])
    AgentLoop(model, tools([]), hooks=hooks).run("go")
    last = model.requests[1].messages[-1]
    assert last.tool_results[0].text == "[redacted]"
    assert last.content[-1].text == "note: redacted"


def test_failing_hook_is_non_blocking_and_recorded():
    hooks = HookRegistry()

    def bad(inp):
        raise ValueError("oops")

    hooks.on(HookEvent.PRE_TOOL, bad)
    model = ScriptedModelClient([tool_response(("read", {})), text_response("ok")])
    loop = AgentLoop(model, tools([]), hooks=hooks)
    assert loop.run("go").ok
    assert any(e.data.get("error") == "hook_failed" for e in loop.session.events() if e.type is EventType.ERROR)


def test_session_start_and_prompt_hooks_add_context_to_the_first_message():
    hooks = HookRegistry()
    hooks.on(HookEvent.SESSION_START, lambda inp: HookResult(additional_context=f"source={inp.data['source']}"))
    hooks.on(HookEvent.USER_PROMPT_SUBMIT, lambda inp: HookResult(additional_context="today is test day"))
    model = ScriptedModelClient([text_response("ok")])
    AgentLoop(model, hooks=hooks).run("hello")
    first = model.requests[0].messages[0]
    assert [b.text for b in first.content] == ["source=startup", "today is test day", "hello"]
    assert model.requests[0].system == ""  # never injected into the cached prefix


def test_user_prompt_submit_can_block():
    hooks = HookRegistry()
    hooks.on(HookEvent.USER_PROMPT_SUBMIT, lambda inp: HookResult(block=True, reason="contains a secret"))
    model = ScriptedModelClient([])
    result = AgentLoop(model, hooks=hooks).run("my key is ...")
    assert result.subtype is ResultSubtype.ERROR_PROMPT_BLOCKED and model.calls == 0 and result.messages == []


def test_pre_model_hook_can_rewrite_the_request_and_post_model_can_halt():
    hooks = HookRegistry()
    hooks.on(HookEvent.PRE_MODEL, lambda inp: HookResult(updated_request=type(inp.request)(**{**inp.request.__dict__, "max_tokens": 123})))
    hooks.on(HookEvent.POST_MODEL, lambda inp: HookResult(continue_=False, stop_reason="operator stop"))
    model = ScriptedModelClient([tool_response(("read", {}))])
    result = AgentLoop(model, tools([]), hooks=hooks).run("go")
    assert model.requests[0].max_tokens == 123
    assert result.subtype is ResultSubtype.HALTED_BY_HOOK and result.reason == "operator stop"
    # the halted batch was still paired with results
    assert result.messages[-1].tool_results and result.messages[-1].tool_results[0].is_error


def test_stop_hook_block_continues_with_reason_and_sees_stop_hook_active():
    seen = []
    hooks = HookRegistry()

    def stop(inp):
        seen.append(inp.stop_hook_active)
        return None if inp.stop_hook_active else HookResult(block=True, reason="run the tests first")

    hooks.on(HookEvent.STOP, stop)
    model = ScriptedModelClient([text_response("done?"), text_response("tests pass")])
    result = AgentLoop(model, hooks=hooks).run("go")
    assert seen == [False, True] and result.ok
    assert model.requests[1].messages[-1].text == "run the tests first"


def test_stop_hook_block_without_reason_is_ignored():
    hooks = HookRegistry()
    hooks.on(HookEvent.STOP, lambda inp: HookResult(block=True))
    model = ScriptedModelClient([text_response("done")])
    assert AgentLoop(model, hooks=hooks).run("go").ok and model.calls == 1


def test_stop_hook_continue_false_halts():
    hooks = HookRegistry()
    hooks.on(HookEvent.STOP, lambda inp: HookResult(continue_=False, stop_reason="halt"))
    result = AgentLoop(ScriptedModelClient([text_response("x")]), hooks=hooks).run("go")
    assert result.subtype is ResultSubtype.HALTED_BY_HOOK


def _py(code: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def test_command_hook_exit_code_2_denies_with_stderr_reason(tmp_path):
    hook = CommandHook(_py("import sys,json; d=json.load(sys.stdin); sys.stderr.write('no ' + d['tool_name']); sys.exit(2)"))
    ctx = RunContext("s", cwd=str(tmp_path))
    res = hook(HookInput(HookEvent.PRE_TOOL, "s", ctx, tool_call=ToolCall("t", "Bash", {"command": "rm -rf /"})))
    assert res.permission is PermissionBehavior.DENY and res.reason == "no Bash"


def test_command_hook_parses_claude_code_json(tmp_path):
    out = {"hookSpecificOutput": {"permissionDecision": "ask", "permissionDecisionReason": "risky", "updatedInput": {"x": 1},
                                  "additionalContext": "ctx"}}
    hook = CommandHook(_py(f"print({json.dumps(json.dumps(out))})"))
    res = hook(HookInput(HookEvent.PRE_TOOL, "s", RunContext("s", cwd=str(tmp_path)), tool_call=ToolCall("t", "Edit", {})))
    assert res.permission is PermissionBehavior.ASK and res.reason == "risky"
    assert res.updated_input == {"x": 1} and res.additional_context == "ctx"
    stop = CommandHook(_py("print('{\"decision\": \"block\", \"reason\": \"keep going\"}')"))
    sres = stop(HookInput(HookEvent.STOP, "s", RunContext("s", cwd=str(tmp_path))))
    assert sres.block and sres.reason == "keep going"


def test_command_hook_payload_uses_claude_code_field_names(tmp_path):
    out = tmp_path / "payload.json"
    hook = CommandHook(_py(f"import sys; open({str(out)!r}, 'w').write(sys.stdin.read())"))
    hook(HookInput(HookEvent.STOP, "s1", RunContext("s1", cwd=str(tmp_path)), last_assistant_message="bye", stop_hook_active=True))
    payload = json.loads(out.read_text())
    assert payload["hook_event_name"] == "Stop" and payload["session_id"] == "s1"
    assert payload["stop_hook_active"] is True and payload["last_assistant_message"] == "bye"


def test_allow_all_policy_runs_mutations():
    log: list = []
    model = ScriptedModelClient([tool_response(("write", {})), text_response("ok")])
    AgentLoop(model, tools(log), policy=AllowAllPolicy()).run("go")
    assert log == [("write", {})]


def test_harness_memory_task_guard_as_pre_tool_hook(tmp_path):
    from harness_memory import guard_tool_call

    hooks = HookRegistry()

    def task_guard(inp):
        decision = guard_tool_call(inp.tool_call.name, inp.tool_call.input, repo_root=tmp_path)
        if not decision.allow:
            return HookResult(permission=PermissionBehavior.DENY, reason=decision.reason)
        return None

    hooks.on(HookEvent.PRE_TOOL, task_guard)
    ran = []
    ex = FunctionToolExecutor()
    ex.tool("Bash", "shell")(lambda inp, ctx: ran.append(inp["command"]) or "ok")
    model = ScriptedModelClient([tool_response(("Bash", {"command": "rm tests/test_app.py"}, "t1"),
                                               ("Bash", {"command": "ls"}, "t2")), text_response("ok")])
    result = AgentLoop(model, ex, hooks=hooks, policy=AllowAllPolicy()).run("clean up")
    r1, r2 = result.messages[2].tool_results
    assert r1.is_error and "Permission denied" in r1.text and not r2.is_error
    assert ran == ["ls"]
