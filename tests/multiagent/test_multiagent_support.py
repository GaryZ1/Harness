"""Shared scripted fakes for the multiagent tests (no tests in this module)."""

from __future__ import annotations

import json
import re
import threading
from typing import Callable

from harness_core.types import (
    ModelRequest,
    ModelResponse,
    RunContext,
    StopReason,
    TextBlock,
    ToolCall,
    ToolOutcome,
    ToolSpec,
    ToolUseBlock,
    Usage,
)
from harness_multiagent import (
    AgentRegistry,
    ManagerConfig,
    ModelRouter,
    SimpleAgentRunner,
    SubagentDefinition,
    SubagentManager,
    WriteAccess,
)

ROLES = {"lead": "test-model", "worker": "test-model", "reviewer": "test-model"}
_AGENT = re.compile(r'You are the subagent "([^"]+)"')


def report(summary: str = "done", findings=(), confidence: float | None = 0.9, files=()) -> str:
    return json.dumps({
        "summary": summary,
        "findings": [f if isinstance(f, dict) else {"claim": f, "evidence": "e", "source": "s", "confidence": 0.8} for f in findings],
        "files": list(files), "open_questions": [], "confidence": confidence,
    })


def text(t: str, inp: int = 100, out: int = 20) -> ModelResponse:
    return ModelResponse([TextBlock(t)], StopReason.END_TURN, Usage(inp, out), model="test-model")


def tool_use(name: str, input: dict, *, call_id: str = "tu_1", inp: int = 100, out: int = 20) -> ModelResponse:
    return ModelResponse([ToolUseBlock(call_id, name, input)], StopReason.TOOL_USE, Usage(inp, out), model="test-model")


def agent_of(request: ModelRequest) -> str:
    system = request.system if isinstance(request.system, str) else json.dumps(request.system)
    m = _AGENT.search(system)
    return m.group(1) if m else "?"


def brief_of(request: ModelRequest) -> str:
    return request.messages[0].text if request.messages else ""


class ScriptedModel:
    """Deterministic fake ModelClient. ``script(request, n)`` where ``n`` = prior calls for this agent+brief."""

    def __init__(self, script: Callable[[ModelRequest, int], ModelResponse]) -> None:
        self.script = script
        self.requests: list[ModelRequest] = []
        self._counts: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()

    def create(self, request: ModelRequest) -> ModelResponse:
        key = (agent_of(request), brief_of(request))
        with self._lock:
            n = self._counts.get(key, 0)
            self._counts[key] = n + 1
            self.requests.append(request)
        return self.script(request, n)


def always(t: str) -> ScriptedModel:
    return ScriptedModel(lambda req, n: text(t))


class FakeTools:
    """ToolExecutor with one read-only and one mutating tool; records executed calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, str]] = []
        self._lock = threading.Lock()

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec("read_file", "Read a file", {"type": "object", "properties": {"path": {"type": "string"}}}, read_only=True),
            ToolSpec("grep", "Search", {"type": "object", "properties": {"pattern": {"type": "string"}}}, read_only=True),
            ToolSpec("write_file", "Write a file", {"type": "object", "properties": {"path": {"type": "string"}}}, read_only=False),
        ]

    def execute(self, call: ToolCall, ctx: RunContext) -> ToolOutcome:
        with self._lock:
            self.calls.append((call.name, dict(call.input), ctx.agent_name))
        return ToolOutcome(f"ok:{call.name}")


class ListSink:
    def __init__(self) -> None:
        self.events = []
        self._lock = threading.Lock()

    def emit(self, event):
        with self._lock:
            event.seq = len(self.events)
            self.events.append(event)
        return event

    def phases(self) -> list[str]:
        return [e.data["phase"] for e in self.events]


def defn(name: str = "explorer", **kw) -> SubagentDefinition:
    base = dict(description=f"{name} agent", system_prompt=f"You are {name}.")
    base.update(kw)
    return SubagentDefinition(name=name, **base)


def make_manager(model: ScriptedModel, *defs: SubagentDefinition, tools=None, sink=None, config: ManagerConfig | None = None, **kw) -> SubagentManager:
    registry = AgentRegistry(defs or [defn()])
    runner = SimpleAgentRunner(ModelRouter(ROLES, model))
    return SubagentManager(registry, runner, tools=tools, event_sink=sink, config=config or ManagerConfig(), **kw)


def ctx(cwd: str = ".", **kw) -> RunContext:
    return RunContext(session_id="s_test", cwd=cwd, **kw)


__all__ = [
    "ROLES", "report", "text", "tool_use", "agent_of", "brief_of", "ScriptedModel", "always", "FakeTools", "ListSink", "defn",
    "make_manager", "ctx", "WriteAccess",
]
