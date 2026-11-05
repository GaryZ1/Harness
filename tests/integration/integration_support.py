"""Shared helpers for the integration tests (offline: every model call is scripted)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

import pytest

from harness import HarnessConfig, build_harness
from harness.config import FeaturesConfig, PermissionsConfig
from harness_core.testing import ScriptedModelClient, make_response, text_response, tool_response, tool_use
from harness_core.types import EventType, ModelRequest, ModelResponse, StopReason, TextBlock, Usage
from harness_multiagent import AgentRegistry, SubagentDefinition

MODEL = "test-model"

GOOD_SUMMARY = """<summary>
## Task overview
Inspect the workspace and report.
## User preferences and learnings
None.
## Current state
Several outputs were inspected.
## Files touched
None.
## Decisions and rationale
None.
## Errors and failed approaches
None.
## Open tasks and next steps
Finish the report.
## Critical references
None.
</summary>"""


def is_summary_request(request: ModelRequest) -> bool:
    last = request.messages[-1] if request.messages else None
    return last is not None and "context checkpoint" in last.text


class RoutingClient:
    """Scripted main conversation; compaction (summary) requests are answered with ``summary``."""

    def __init__(self, script: Iterable[Any], *, summary: str = GOOD_SUMMARY) -> None:
        self.main = ScriptedModelClient(script)
        self.summary = summary
        self.summary_requests: list[ModelRequest] = []

    @property
    def requests(self) -> list[ModelRequest]:
        return self.main.requests

    def create(self, request: ModelRequest) -> ModelResponse:
        if is_summary_request(request):
            self.summary_requests.append(request)
            return text_response(self.summary, usage=Usage(input_tokens=500, output_tokens=200))
        return self.main.create(request)


def make_config(tmp_path: Path, *, workspace: Path | None = None, **kw: Any) -> HarnessConfig:
    ws = workspace or (tmp_path / "ws")
    ws.mkdir(parents=True, exist_ok=True)
    features = kw.pop("features", FeaturesConfig(memory_git=False))
    permissions = kw.pop("permissions", PermissionsConfig())
    return HarnessConfig(model=MODEL, workspace=str(ws), base_dir=str(tmp_path / "base"), features=features, permissions=permissions, **kw)


def build(tmp_path: Path, script: Iterable[Any] | Any, **kw: Any):
    build_kw = {k: kw.pop(k) for k in list(kw) if k in ("approver", "approval_ui", "session_id", "tool_executor", "agents", "sinks", "hooks")}
    cfg = kw.pop("config", None) or make_config(tmp_path, **kw)
    client = script if hasattr(script, "create") else ScriptedModelClient(script)
    return build_harness(cfg, model_client=client, **build_kw), client


def events_of(harness, type_: EventType) -> list:
    return [e for e in harness.session.events() if e.type is type_]


def permission_events(harness) -> list[dict[str, Any]]:
    return [e.data for e in events_of(harness, EventType.PERMISSION)]


def tool_results(result) -> list:
    return [b for m in result.messages for b in m.tool_results]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def explorer_registry(**overrides: Any) -> AgentRegistry:
    d = dict(name="explorer", description="Read-only code explorer for where-is questions.",
             system_prompt="You are a fast, read-only code explorer.", tools=("read", "grep", "glob"), model_role="worker")
    d.update(overrides)
    return AgentRegistry([SubagentDefinition(**d)])


def report_json(summary: str = "found it", **extra: Any) -> str:
    return json.dumps({"summary": summary, "findings": [], "files": [], "open_questions": [], "confidence": 0.9, **extra})


def big_output_tool_turn(i: int, chars: int = 2_400) -> ModelResponse:
    """A bash call whose (read-only) output is about ``chars`` characters."""
    count = max(1, chars // 5)
    return tool_response(("bash", {"command": f"seq {10000 + i * 1000} {10000 + i * 1000 + count - 1}"}))


@pytest.fixture(name="approve_all")
def approve_all_fixture() -> Callable[..., bool]:
    calls: list[str] = []

    def approver(call, spec, ctx, decision):
        calls.append(call.name)
        return True

    approver.calls = calls  # type: ignore[attr-defined]
    return approver


__all__ = [
    "GOOD_SUMMARY", "MODEL", "RoutingClient", "ScriptedModelClient", "StopReason", "TextBlock", "Usage", "approve_all_fixture", "big_output_tool_turn",
    "build", "events_of", "explorer_registry", "is_summary_request", "make_config", "make_response", "permission_events", "read_jsonl",
    "report_json", "text_response", "tool_response", "tool_results", "tool_use",
]
