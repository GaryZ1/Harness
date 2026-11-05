"""Deterministic test doubles for harness code (public: sibling packages' tests may reuse them).

* :class:`ScriptedModelClient` - a :class:`~harness_core.types.ModelClient` that replays a script
  of responses, exceptions or callables and records every request it received;
* :func:`text_response`, :func:`tool_response`, :func:`make_response` - response builders;
* :class:`FakeMonotonic` - an injectable ``monotonic`` clock for wall-clock budgets;
* :class:`RecordingTracer`, :class:`RecordingSink` - capture spans and events.

No network, no randomness: tests stay offline and repeatable. Model ids default to "test-model".
"""

from __future__ import annotations

import itertools
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Iterator, Union

from .errors import ScriptExhaustedError
from .types import (
    ContentBlock,
    Event,
    ModelRequest,
    ModelResponse,
    StopReason,
    TextBlock,
    ToolUseBlock,
    Usage,
)

ScriptItem = Union[ModelResponse, BaseException, Callable[[ModelRequest], Union[ModelResponse, BaseException]]]

_ids = itertools.count(1)


def _usage(usage: Usage | None) -> Usage:
    return usage if usage is not None else Usage(input_tokens=100, output_tokens=20)


def make_response(*blocks: ContentBlock, stop_reason: StopReason | str = StopReason.END_TURN, usage: Usage | None = None,
                  model: str = "test-model") -> ModelResponse:
    sr = stop_reason if isinstance(stop_reason, StopReason) else StopReason.parse(stop_reason)
    return ModelResponse(content=list(blocks), stop_reason=sr, usage=_usage(usage), model=model, id=f"resp_test_{next(_ids)}")


def text_response(text: str, *, stop_reason: StopReason | str = StopReason.END_TURN, usage: Usage | None = None,
                  model: str = "test-model") -> ModelResponse:
    return make_response(TextBlock(text), stop_reason=stop_reason, usage=usage, model=model)


def tool_use(name: str, input: dict[str, Any] | None = None, id: str | None = None) -> ToolUseBlock:
    return ToolUseBlock(id or f"toolu_test_{next(_ids)}", name, dict(input or {}))


def tool_response(*calls: ToolUseBlock | tuple[Any, ...], text: str = "", stop_reason: StopReason | str = StopReason.TOOL_USE,
                  usage: Usage | None = None, model: str = "test-model") -> ModelResponse:
    """``tool_response(("read", {"path": "a"}), ("bash", {"cmd": "ls"}, "tu_2"))``."""
    blocks: list[ContentBlock] = [TextBlock(text)] if text else []
    for c in calls:
        blocks.append(c if isinstance(c, ToolUseBlock) else tool_use(*c))
    return make_response(*blocks, stop_reason=stop_reason, usage=usage, model=model)


class ScriptedModelClient:
    """Replay ``script`` in order. Each item is a response, an exception to raise, or a callable
    ``f(request) -> response | exception``. ``default`` (if set) answers once the script is used up;
    otherwise :class:`~harness_core.errors.ScriptExhaustedError` is raised."""

    def __init__(self, script: Iterable[ScriptItem] = (), *, default: ScriptItem | None = None) -> None:
        self.script: list[ScriptItem] = list(script)
        self.default = default
        self.requests: list[ModelRequest] = []
        self._lock = threading.Lock()

    def extend(self, items: Iterable[ScriptItem]) -> None:
        with self._lock:
            self.script.extend(items)

    @property
    def calls(self) -> int:
        return len(self.requests)

    @property
    def remaining(self) -> int:
        return len(self.script)

    def create(self, request: ModelRequest) -> ModelResponse:
        with self._lock:
            self.requests.append(replace(request, messages=list(request.messages), tools=list(request.tools), extra=dict(request.extra)))
            if self.script:
                item = self.script.pop(0)
            elif self.default is not None:
                item = self.default
            else:
                raise ScriptExhaustedError(f"scripted model called {len(self.requests)} times; script exhausted")
        if callable(item) and not isinstance(item, (ModelResponse, BaseException)):
            item = item(request)
        if isinstance(item, BaseException):
            raise item
        return replace(item, content=list(item.content))


class FakeMonotonic:
    """Injectable monotonic clock: ``loop = AgentLoop(..., monotonic=clock)``; ``clock.advance(30)``."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class RecordedSpan:
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value


class RecordingTracer:
    """Tracer that records span names and attributes (thread-safe)."""

    def __init__(self) -> None:
        self.spans: list[RecordedSpan] = []
        self._lock = threading.Lock()

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[RecordedSpan]:
        s = RecordedSpan(name, dict(attributes))
        with self._lock:
            self.spans.append(s)
        yield s

    def names(self) -> list[str]:
        return [s.name for s in self.spans]


class RecordingSink:
    """EventSink that keeps every event in memory."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self._lock = threading.Lock()

    def emit(self, event: Event) -> Event:
        with self._lock:
            self.events.append(event)
        return event

    def types(self) -> list[str]:
        return [e.type.value for e in self.events]
