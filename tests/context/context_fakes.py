"""Offline fakes shared by the harness_context tests (not collected: no ``test_`` prefix)."""

from __future__ import annotations

import itertools
from typing import Callable

from harness_core.types import (
    Event,
    Message,
    ModelRequest,
    ModelResponse,
    OpaqueBlock,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)

_ids = itertools.count(1)

GOOD_SUMMARY = """<summary>
## Task overview
Add OAuth login to the web app.
## User preferences and learnings
User prefers small commits; tests run with pytest -q.
## Current state
Provider config added.
## Files touched
- src/auth.py
## Decisions and rationale
Use PKCE because the client is public.
## Errors and failed approaches
Implicit flow rejected by provider.
## Open tasks and next steps
Write the e2e test.
## Critical references
https://example.test/docs/oauth
</summary>"""


class ScriptedClient:
    """ModelClient returning queued responses (or callables of the request) and recording requests."""

    def __init__(self, *responses: ModelResponse | Callable[[ModelRequest], ModelResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    def create(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        nxt = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return nxt(request) if callable(nxt) else nxt


def text_response(text: str = GOOD_SUMMARY, stop: StopReason = StopReason.END_TURN) -> ModelResponse:
    return ModelResponse([TextBlock(text)], stop, Usage(input_tokens=100, output_tokens=50))


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event) -> Event:
        event.seq = len(self.events)
        self.events.append(event)
        return event

    def kinds(self) -> list[str]:
        return [str(e.data.get("kind")) for e in self.events]


def tool_turn(name: str, tool_input: dict, result: str, *, is_error: bool = False, thinking: bool = False) -> tuple[Message, Message]:
    uid = f"toolu_{next(_ids):04d}"
    blocks = [OpaqueBlock({"type": "thinking", "thinking": "hmm", "signature": "sig"})] if thinking else []
    blocks.append(ToolUseBlock(uid, name, tool_input))
    return Message.assistant(blocks), Message.user([ToolResultBlock(uid, result, is_error)])


def conversation(turns: int, *, result_chars: int = 400, tool: str = "read_file", calls_per_turn: int = 1) -> list[Message]:
    """``turns`` user turns, each with tool loops and a final assistant answer."""
    log: list[Message] = []
    for t in range(turns):
        log.append(Message.user(f"request {t}: please look at file_{t}.py"))
        for c in range(calls_per_turn):
            a, r = tool_turn(tool, {"path": f"src/file_{t}_{c}.py"}, f"contents {t}.{c} " + "x" * result_chars)
            log += [a, r]
        log.append(Message.assistant([TextBlock(f"done with request {t}")]))
    return log


TOOLS = [
    ToolSpec("read_file", "Read a file.", {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
    ToolSpec("bash", "Run a shell command.", {"properties": {"command": {"type": "string"}}, "type": "object"}),
]
