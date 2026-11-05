"""Sub-agents: SubagentManager -> LoopAgentRunner -> AgentLoop, standalone and through the harness spawn tool."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

from integration_support import build, events_of, explorer_registry, report_json, text_response, tool_response, tool_results

from harness import LoopAgentRunner
from harness.config import FeaturesConfig
from harness_core.anthropic_adapter import AnthropicModelClient
from harness_core.results import ResultSubtype
from harness_core.testing import ScriptedModelClient
from harness_core.types import EventType, RunContext
from harness_multiagent import (
    DELEGATION_POLICY,
    AgentRegistry,
    DelegationBrief,
    ManagerConfig,
    SubagentDefinition,
    SubagentManager,
    SubagentStatus,
)
from harness_tools import ToolRegistry

logging.getLogger("harness_sandbox").setLevel(logging.ERROR)

ROLES = {"worker": "test-model-worker"}


def _tools() -> ToolRegistry:
    reg = ToolRegistry()

    def lookup(term: str) -> str:
        """Look up a term in the project glossary and return its definition."""
        return f"{term}: a thing"

    def save(text: str) -> str:
        """Save text to the scratch area (mutating)."""
        return "saved"

    reg.register_function(lookup, read_only=True)
    reg.register_function(save)
    return reg


def _manager(client, *, definition: dict | None = None, config: ManagerConfig | None = None, **runner_kw) -> SubagentManager:
    d = dict(name="worker-agent", description="Answers glossary questions.", system_prompt="You answer glossary questions.",
             tools=("lookup",), model_role="worker")
    d.update(definition or {})
    runner = LoopAgentRunner(client, model_for_role=ROLES, **runner_kw)
    return SubagentManager(AgentRegistry([SubagentDefinition(**d)]), runner, tools=_tools(), config=config or ManagerConfig())


def test_loop_agent_runner_follows_the_runner_contract():
    client = ScriptedModelClient([tool_response(("lookup", {"term": "lease"})), text_response(report_json("lease explained"))])
    mgr = _manager(client)
    res = mgr.spawn("worker-agent", DelegationBrief("Explain what a lease is"), RunContext(session_id="lead", cwd="."))

    assert res.status is SubagentStatus.SUCCESS and res.report.summary == "lease explained"
    assert res.turns == 2 and res.tool_calls == 1 and res.model == "test-model-worker"
    first = client.requests[0]
    assert first.extra == {"model": "test-model-worker"}  # task.request_extra(model_id)
    assert [t.name for t in first.tools] == ["lookup"]  # task.tools.specs(): scoped, read-only
    assert "You are the subagent" in first.system and "reply with ONLY one JSON object" in first.system
    assert len(first.messages) == 1 and "Explain what a lease is" in first.messages[0].text  # brief only
    assert mgr.accounting.summary()["agents"]["worker-agent"]["tokens"] == 240  # charged after every response


def test_structured_report_is_repaired_then_parsed():
    client = ScriptedModelClient([text_response("I think it is fine."), text_response(report_json("ok after repair"))])
    res = _manager(client).spawn("worker-agent", DelegationBrief("Check it"), RunContext(session_id="lead"))
    assert res.status is SubagentStatus.SUCCESS and res.report.summary == "ok after repair"
    assert "not a valid report" in client.requests[1].messages[-1].text


def test_tool_call_cap_and_mutating_tools_are_enforced():
    client = ScriptedModelClient([
        tool_response(("lookup", {"term": "a"}), ("lookup", {"term": "b"}), ("save", {"text": "x"})),
        text_response(report_json("capped")),
    ])
    mgr = _manager(client, definition={"tools": None, "max_tool_calls": 1})
    res = mgr.spawn("worker-agent", DelegationBrief("Look things up"), RunContext(session_id="lead"))
    assert res.status is SubagentStatus.SUCCESS and res.tool_calls == 1
    results = [b for m in client.requests[1].messages for b in m.tool_results]
    assert not results[0].is_error and "budget (1) exhausted" in results[1].text
    assert results[2].is_error  # a read-only agent never gets the mutating tool
    assert "save" not in [t.name for t in client.requests[0].tools]


def test_budget_stop_halts_the_loop_after_the_charging_response():
    client = ScriptedModelClient([tool_response(("lookup", {"term": "a"})), text_response(report_json("never reached"))])
    mgr = _manager(client, definition={"max_tokens": 100})
    res = mgr.spawn("worker-agent", DelegationBrief("Look it up"), RunContext(session_id="lead"))
    assert res.status is SubagentStatus.ERROR_MAX_BUDGET
    assert client.calls == 1  # stopped as soon as task.stop_reason() was set


def test_max_turns_is_respected():
    client = ScriptedModelClient([tool_response(("lookup", {"term": str(i)})) for i in range(5)])
    res = _manager(client, definition={"max_turns": 2}).spawn("worker-agent", DelegationBrief("Loop"), RunContext(session_id="lead"))
    assert res.status is SubagentStatus.ERROR_MAX_TURNS and client.calls == 2


def test_effort_and_model_reach_the_anthropic_api_correctly():
    sent = []

    def create(**kw):
        sent.append(kw)
        return {"content": [{"type": "text", "text": report_json("done")}], "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5}, "model": kw["model"], "id": "msg_1"}

    sdk = SimpleNamespace(messages=SimpleNamespace(create=create))
    client = AnthropicModelClient("test-model", client=sdk)
    res = _manager(client, definition={"effort": "low"}).spawn("worker-agent", DelegationBrief("Go"), RunContext(session_id="lead"))
    assert res.status is SubagentStatus.SUCCESS
    kw = sent[0]
    assert kw["model"] == "test-model-worker"  # extra["model"] overrides the client's model
    assert kw["output_config"] == {"effort": "low"} and "effort" not in kw  # extra["effort"] mapped, not top-level


def test_spawn_tool_through_the_harness(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "greet.py").write_text("print('hello')\n")
    script = [
        tool_response(("spawn_subagent", {"agent": "explorer", "objective": "Find where the greeting is printed"})),
        tool_response(("grep", {"pattern": "hello"})),
        text_response(report_json("greeting is printed in greet.py", files=["greet.py"])),
        text_response("The greeting is printed in greet.py."),
    ]
    h, client = build(tmp_path, script, features=FeaturesConfig(memory_git=False, subagents=True), agents=explorer_registry(),
                      model_roles=ROLES)
    result = h.run("Where is the greeting printed?")

    assert result.subtype is ResultSubtype.SUCCESS
    lead_tools = [t.name for t in client.requests[0].tools]
    assert "spawn_subagent" in lead_tools
    assert DELEGATION_POLICY.strip() in "\n".join(b["text"] for b in client.requests[0].system)  # lead prompt policy
    child = client.requests[1]
    assert child.extra == {"model": "test-model-worker"} and sorted(t.name for t in child.tools) == ["glob", "grep", "read"]
    # the child's grep found the file inside the workspace
    assert "greet.py" in [b for m in client.requests[2].messages for b in m.tool_results][0].text
    report = tool_results(result)[0].text
    payload = json.loads(report[report.index("{"):])
    assert payload["type"] == "subagent_report" and payload["trust"] == "untrusted"
    assert payload["status"] == "success" and "greet.py" in payload["report"]["summary"]
    assert [e.data["phase"] for e in events_of(h, EventType.SUBAGENT)] == ["start", "stop"]
    # the child's calls were checked by the same PolicyEngine as a depth-1 session, then released
    child_sessions = [sid for sid, st in h.engine._sessions.items() if st.depth == 1]
    assert len(child_sessions) == 1 and not h.engine.session(h.session.id).active_children
    h.end_session()
