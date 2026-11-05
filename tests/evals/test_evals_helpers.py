"""Shared fakes for the evals tests (no tests here)."""

from __future__ import annotations

import json
import threading
from typing import Callable

from harness_core.types import Message, ModelRequest, ModelResponse, StopReason, TextBlock, ToolResultBlock, ToolUseBlock, Usage

TEST_MODEL = "test-model"


class ScriptedClient:
    """ModelClient fake: returns scripted texts in order (or computes them from the request)."""

    def __init__(self, replies: list[str] | Callable[[ModelRequest], str]) -> None:
        self.replies = replies
        self.requests: list[ModelRequest] = []
        self._lock = threading.Lock()

    def create(self, request: ModelRequest) -> ModelResponse:
        with self._lock:
            self.requests.append(request)
            text = self.replies(request) if callable(self.replies) else self.replies.pop(0)
        return ModelResponse([TextBlock(text)], StopReason.END_TURN, Usage(100, 20), model=TEST_MODEL)


def verdict_json(*criteria: tuple[str, str, float | None, list[str]]) -> str:
    return json.dumps({"criteria": [{"name": n, "verdict": v, "score": s, "evidence": e, "rationale": "r"} for n, v, s, e in criteria]})


def tool_conversation(name: str, inp: dict, output: str, final: str = "done", is_error: bool = False) -> list[Message]:
    return [
        Message.user("do it"),
        Message.assistant([ToolUseBlock("tu_1", name, inp)]),
        Message.user([ToolResultBlock("tu_1", output, is_error)]),
        Message.assistant([TextBlock(final)]),
    ]
