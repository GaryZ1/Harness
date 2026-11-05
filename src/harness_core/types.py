"""Shared contract for every ``harness_*`` package.

All packages exchange data through these provider-neutral types and protocols. They mirror the
Anthropic Messages API shapes (content blocks, ``tool_use`` / ``tool_result``, ``stop_reason``,
``usage``) so an adapter is a thin translation, but nothing here imports a provider SDK.

Rules for this module: additive changes only; never remove or rename a field.
"""

from __future__ import annotations

import json
import uuid
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, ContextManager, Iterable, Literal, Protocol, Sequence, Union, runtime_checkable

Clock = Callable[[], datetime]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


# ----------------------------------------------------------------------------- content blocks


@dataclass
class TextBlock:
    text: str
    type: Literal["text"] = "text"
    cache_control: dict[str, Any] | None = None

    def to_api(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": "text", "text": self.text}
        if self.cache_control:
            d["cache_control"] = self.cache_control
        return d


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: Literal["tool_use"] = "tool_use"

    def to_api(self) -> dict[str, Any]:
        return {"type": "tool_use", "id": self.id, "name": self.name, "input": self.input}


@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: str | list[dict[str, Any]]
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"
    cache_control: dict[str, Any] | None = None

    def to_api(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": "tool_result", "tool_use_id": self.tool_use_id, "content": self.content}
        if self.is_error:
            d["is_error"] = True
        if self.cache_control:
            d["cache_control"] = self.cache_control
        return d

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "".join(str(c.get("text", "")) for c in self.content if isinstance(c, dict))


@dataclass
class OpaqueBlock:
    """Any other block (thinking, redacted_thinking, compaction, server tool results...).

    Kept verbatim and passed back unchanged - never edit or drop provider blocks you do not own.
    """

    raw: dict[str, Any]

    @property
    def type(self) -> str:
        return str(self.raw.get("type", "unknown"))

    def to_api(self) -> dict[str, Any]:
        return dict(self.raw)


ContentBlock = Union[TextBlock, ToolUseBlock, ToolResultBlock, OpaqueBlock]


def block_from_api(d: dict[str, Any] | Any) -> ContentBlock:
    """Build a block from an API dict or an SDK object (anything with ``model_dump``/attributes)."""
    if not isinstance(d, dict):
        d = d.model_dump() if hasattr(d, "model_dump") else dict(vars(d))
    t = d.get("type")
    if t == "text":
        return TextBlock(text=d.get("text", ""), cache_control=d.get("cache_control"))
    if t == "tool_use":
        return ToolUseBlock(id=d["id"], name=d["name"], input=dict(d.get("input") or {}))
    if t == "tool_result":
        return ToolResultBlock(
            tool_use_id=d["tool_use_id"], content=d.get("content", ""), is_error=bool(d.get("is_error", False)),
            cache_control=d.get("cache_control"),
        )
    return OpaqueBlock(raw={k: v for k, v in d.items() if v is not None})


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"  # mid-conversation operator message (supported models only)


@dataclass
class Message:
    role: Role
    content: list[ContentBlock]
    id: str = field(default_factory=lambda: new_id("msg"))
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def user(cls, text_or_blocks: str | Sequence[ContentBlock], **meta: Any) -> "Message":
        blocks = [TextBlock(text_or_blocks)] if isinstance(text_or_blocks, str) else list(text_or_blocks)
        return cls(Role.USER, blocks, metadata=dict(meta))

    @classmethod
    def assistant(cls, blocks: Sequence[ContentBlock], **meta: Any) -> "Message":
        return cls(Role.ASSISTANT, list(blocks), metadata=dict(meta))

    @classmethod
    def system(cls, text: str, **meta: Any) -> "Message":
        return cls(Role.SYSTEM, [TextBlock(text)], metadata=dict(meta))

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]

    @property
    def tool_results(self) -> list[ToolResultBlock]:
        return [b for b in self.content if isinstance(b, ToolResultBlock)]

    def to_api(self) -> dict[str, Any]:
        return {"role": self.role.value, "content": [b.to_api() for b in self.content]}

    @classmethod
    def from_api(cls, d: dict[str, Any]) -> "Message":
        content = d.get("content", "")
        blocks = [TextBlock(content)] if isinstance(content, str) else [block_from_api(b) for b in content]
        return cls(Role(d["role"]), blocks)


# ----------------------------------------------------------------------------- model I/O


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_input_tokens + other.cache_read_input_tokens,
            self.cache_creation_input_tokens + other.cache_creation_input_tokens,
        )

    @property
    def total_input(self) -> int:
        return self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens


class StopReason(str, Enum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    PAUSE_TURN = "pause_turn"
    REFUSAL = "refusal"
    MODEL_CONTEXT_WINDOW_EXCEEDED = "model_context_window_exceeded"
    OTHER = "other"

    @classmethod
    def parse(cls, value: str | None) -> "StopReason":
        try:
            return cls(value or "other")
        except ValueError:
            return cls.OTHER


@dataclass
class ToolSpec:
    """A tool as offered to the model. ``input_schema`` is JSON Schema."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    read_only: bool = False          # no side effects outside the session (safe to auto-allow / parallelise)
    open_world: bool = False         # output may contain untrusted third-party content (taints the session)
    defer_loading: bool = False      # hidden until discovered via tool search
    strict: bool = False
    provider_type: str | None = None  # e.g. "memory_20250818" for provider-defined (schema-less) tools
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_api(self) -> dict[str, Any]:
        if self.provider_type:
            d: dict[str, Any] = {"type": self.provider_type, "name": self.name}
        else:
            d = {"name": self.name, "description": self.description, "input_schema": self.input_schema}
            if self.strict:
                d["strict"] = True
        if self.defer_loading:
            d["defer_loading"] = True
        return d


@dataclass
class ModelRequest:
    system: str | list[dict[str, Any]]
    messages: list[Message]
    tools: list[ToolSpec] = field(default_factory=list)
    max_tokens: int = 16_000
    extra: dict[str, Any] = field(default_factory=dict)  # provider-specific params (effort, betas, context_management...)


@dataclass
class ModelResponse:
    content: list[ContentBlock]
    stop_reason: StopReason
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    id: str = ""
    raw: Any = None

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]


@runtime_checkable
class ModelClient(Protocol):
    """Provider-neutral model call. Implementations: Anthropic adapter, scripted fakes for tests."""

    def create(self, request: ModelRequest) -> ModelResponse: ...


# ----------------------------------------------------------------------------- tools, permissions, context


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]

    @classmethod
    def from_block(cls, b: ToolUseBlock) -> "ToolCall":
        return cls(b.id, b.name, dict(b.input))


@dataclass
class ToolOutcome:
    content: str | list[dict[str, Any]]
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)  # e.g. {"untrusted": True, "exit_code": 1, "bytes": 1234}

    def to_block(self, tool_use_id: str) -> ToolResultBlock:
        return ToolResultBlock(tool_use_id, self.content, self.is_error)


@dataclass
class RunContext:
    """Per-call context handed to tools, permission policies and hooks."""

    session_id: str
    cwd: str = "."
    agent_name: str = "main"
    depth: int = 0                      # 0 = main agent; >0 = sub-agent
    tainted: bool = False               # session ingested untrusted content
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ToolExecutor(Protocol):
    def specs(self) -> list[ToolSpec]: ...

    def execute(self, call: ToolCall, ctx: RunContext) -> ToolOutcome: ...


class PermissionBehavior(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass
class PermissionDecision:
    behavior: PermissionBehavior
    reason: str = ""
    updated_input: dict[str, Any] | None = None  # policy may rewrite input (e.g. add sandbox flags)

    @property
    def allowed(self) -> bool:
        return self.behavior is PermissionBehavior.ALLOW


@runtime_checkable
class PermissionPolicy(Protocol):
    def check(self, call: ToolCall, spec: ToolSpec | None, ctx: RunContext) -> PermissionDecision: ...


@runtime_checkable
class ContextAssembler(Protocol):
    """Turns the durable message log into the model-visible window for the next call."""

    def prepare(self, system: str, messages: list[Message], tools: list[ToolSpec], ctx: RunContext) -> ModelRequest: ...


# ----------------------------------------------------------------------------- session log & tracing


class EventType(str, Enum):
    SESSION_START = "session_start"
    USER_MESSAGE = "user_message"
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    PERMISSION = "permission"
    COMPACTION = "compaction"
    EVALUATION = "evaluation"
    SUBAGENT = "subagent"
    ERROR = "error"
    STOP = "stop"
    SESSION_END = "session_end"
    CUSTOM = "custom"


@dataclass
class Event:
    session_id: str
    type: EventType
    data: dict[str, Any]
    seq: int = -1
    ts: str = ""
    id: str = field(default_factory=lambda: new_id("evt"))

    def to_json(self) -> str:
        d = asdict(self)
        d["type"] = self.type.value
        return json.dumps(d, ensure_ascii=False, default=str, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> "Event":
        d = json.loads(line)
        d["type"] = EventType(d["type"])
        return cls(**d)


@runtime_checkable
class EventSink(Protocol):
    def emit(self, event: Event) -> Event: ...


@runtime_checkable
class Tracer(Protocol):
    def span(self, name: str, **attributes: Any) -> ContextManager[Any]: ...


class NullTracer:
    """Tracer that records nothing."""

    def span(self, name: str, **attributes: Any) -> ContextManager[Any]:
        return nullcontext()


def messages_to_api(messages: Iterable[Message]) -> list[dict[str, Any]]:
    return [m.to_api() for m in messages]


__all__ = [
    "Clock", "utcnow", "new_id",
    "TextBlock", "ToolUseBlock", "ToolResultBlock", "OpaqueBlock", "ContentBlock", "block_from_api",
    "Role", "Message", "Usage", "StopReason", "ToolSpec", "ModelRequest", "ModelResponse", "ModelClient",
    "ToolCall", "ToolOutcome", "RunContext", "ToolExecutor", "PermissionBehavior", "PermissionDecision",
    "PermissionPolicy", "ContextAssembler", "EventType", "Event", "EventSink", "Tracer", "NullTracer",
    "messages_to_api",
]
