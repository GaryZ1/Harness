"""The agent-under-test boundary and the trajectory record graders read.

The agent loop is built elsewhere, so this package depends only on a tiny protocol:

``AgentUnderTest.run(task: AgentTask, workspace: Path, ctx: RunContext) -> Trajectory``

* :class:`AgentTask` is the *agent-visible* view of a task. Graders, hidden tests, reference
  solutions and canaries are never part of it ("tests hidden from the agent", Anthropic 2026;
  reference §9.7 step 5).
* :class:`Trajectory` is the transcript (messages and/or the raw :class:`~harness_core.types.Event`
  stream) plus the efficiency counters the checklist makes first-class (§9.7 step 9: turns, tool
  calls, input / cache-read / cache-write / output tokens, cost, tool errors by tool, compaction
  events).
* :class:`CallableAgent` wraps any callable so the evaluation stack is testable without the real
  loop.

``Trajectory.from_events`` documents the event payload conventions it understands; it is tolerant
of missing fields because the loop's exact payloads are owned by another package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from harness_core.types import (
    Event,
    EventType,
    Message,
    OpaqueBlock,
    Role,
    RunContext,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
    block_from_api,
)


@dataclass(frozen=True)
class Limits:
    """Per-task budgets. ``timeout_s`` is wall clock for the agent phase only (grading is separate)."""

    timeout_s: float = 1800.0
    max_turns: int | None = None
    max_tool_calls: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"timeout_s": self.timeout_s, "max_turns": self.max_turns, "max_tool_calls": self.max_tool_calls}


@dataclass(frozen=True)
class AgentTask:
    """Everything the agent is allowed to know about a task."""

    task_id: str
    instruction: str
    limits: Limits = field(default_factory=Limits)
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallRecord:
    """One tool invocation paired with its result (``output is None`` = no result was recorded)."""

    id: str
    name: str
    input: dict[str, Any]
    output: str | None = None
    is_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "input": self.input, "output": self.output, "is_error": self.is_error}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ToolCallRecord":
        return cls(str(d["id"]), str(d["name"]), dict(d.get("input") or {}), d.get("output"), bool(d.get("is_error", False)))


def _usage_from(d: Mapping[str, Any] | None) -> Usage:
    d = d or {}
    return Usage(
        int(d.get("input_tokens", 0) or 0),
        int(d.get("output_tokens", 0) or 0),
        int(d.get("cache_read_input_tokens", 0) or 0),
        int(d.get("cache_creation_input_tokens", 0) or 0),
    )


def _usage_dict(u: Usage) -> dict[str, int]:
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_read_input_tokens": u.cache_read_input_tokens,
        "cache_creation_input_tokens": u.cache_creation_input_tokens,
    }


@dataclass
class Trajectory:
    """One trial's transcript and counters.

    ``turns`` and ``tool_calls`` are derived from ``messages`` when not given explicitly, so a loop
    only has to return its message log. ``metadata`` is free-form; RL export reads
    ``metadata["rollout_details"]`` (token ids / logprobs, Harbor ``collect_rollout_details``).
    """

    messages: list[Message] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    final_text: str = ""
    system: str = ""
    usage: Usage = field(default_factory=Usage)
    cost_usd: float | None = None
    turns: int | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    compaction_events: int = 0
    model: str = ""
    stop_reason: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tool_calls and self.messages:
            self.tool_calls = pair_tool_calls(self.messages)
        if self.turns is None:
            self.turns = sum(1 for m in self.messages if m.role is Role.ASSISTANT)
        if not self.final_text:
            for m in reversed(self.messages):
                if m.role is Role.ASSISTANT and m.text:
                    self.final_text = m.text
                    break

    # ------------------------------------------------------------------ derived views
    @property
    def n_tool_calls(self) -> int:
        return len(self.tool_calls)

    def tool_errors_by_tool(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.tool_calls:
            if c.is_error:
                out[c.name] = out.get(c.name, 0) + 1
        return dict(sorted(out.items()))

    def render_text(self, *, max_block_chars: int = 4_000) -> str:
        """Render the transcript as numbered lines for judges and human reviewers."""
        lines: list[str] = []
        if self.system:
            lines.append(f"[system] {_clip(self.system, max_block_chars)}")
        for i, m in enumerate(self.messages):
            for b in m.content:
                if isinstance(b, TextBlock):
                    lines.append(f"[{i}] {m.role.value}: {_clip(b.text, max_block_chars)}")
                elif isinstance(b, ToolUseBlock):
                    lines.append(f"[{i}] tool_use {b.name}: {_clip(json.dumps(b.input, ensure_ascii=False, sort_keys=True), max_block_chars)}")
                elif isinstance(b, ToolResultBlock):
                    tag = "tool_error" if b.is_error else "tool_result"
                    lines.append(f"[{i}] {tag}: {_clip(b.text, max_block_chars)}")
                elif isinstance(b, OpaqueBlock):
                    lines.append(f"[{i}] <{b.type} block omitted>")
        if not self.messages and self.final_text:
            lines.append(f"[final] assistant: {_clip(self.final_text, max_block_chars)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ (de)serialisation
    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "messages": [m.to_api() | {"id": m.id, "metadata": m.metadata} for m in self.messages],
            "final_text": self.final_text,
            "usage": _usage_dict(self.usage),
            "cost_usd": self.cost_usd,
            "turns": self.turns,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "compaction_events": self.compaction_events,
            "model": self.model,
            "stop_reason": self.stop_reason,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Trajectory":
        messages = []
        for md in d.get("messages") or []:
            m = Message.from_api(md)
            if md.get("id"):
                m.id = md["id"]
            m.metadata = dict(md.get("metadata") or {})
            messages.append(m)
        return cls(
            messages=messages,
            final_text=str(d.get("final_text") or ""),
            system=str(d.get("system") or ""),
            usage=_usage_from(d.get("usage")),
            cost_usd=d.get("cost_usd"),
            turns=d.get("turns"),
            tool_calls=[ToolCallRecord.from_dict(c) for c in d.get("tool_calls") or []],
            compaction_events=int(d.get("compaction_events", 0) or 0),
            model=str(d.get("model") or ""),
            stop_reason=str(d.get("stop_reason") or ""),
            error=d.get("error"),
            metadata=dict(d.get("metadata") or {}),
        )

    @classmethod
    def from_messages(cls, messages: Sequence[Message], **kw: Any) -> "Trajectory":
        return cls(messages=list(messages), **kw)

    @classmethod
    def from_events(cls, events: Iterable[Event], **kw: Any) -> "Trajectory":
        """Rebuild a trajectory from a session event stream.

        Understood payloads (extra keys are ignored):

        * ``USER_MESSAGE``: ``{"message": <api message>}`` or ``{"text": str}`` or ``{"content": [...]}``
        * ``MODEL_RESPONSE``: ``{"content": [<api blocks>], "usage": {...}, "stop_reason": str, "model": str}``
          or ``{"message": <api message>}``
        * ``TOOL_RESULT``: ``{"tool_use_id": str, "content": str|list, "is_error": bool}`` - consecutive
          results are grouped into one user message (native ``tool_result`` blocks, never plain text)
        * ``COMPACTION``: counted; ``SESSION_START`` may carry ``{"system": str}``; ``STOP`` /
          ``SESSION_END`` may carry ``{"stop_reason": str, "cost_usd": float}``.
        """
        evs = sorted(list(events), key=lambda e: e.seq)
        messages: list[Message] = []
        pending_results: list[ToolResultBlock] = []
        usage = Usage()
        out: dict[str, Any] = {"compaction_events": 0}

        def flush() -> None:
            if pending_results:
                messages.append(Message(Role.USER, list(pending_results)))
                pending_results.clear()

        for e in evs:
            d = e.data or {}
            if e.type is EventType.SESSION_START and d.get("system"):
                out["system"] = str(d["system"])
            elif e.type is EventType.USER_MESSAGE:
                flush()
                if isinstance(d.get("message"), dict):
                    messages.append(Message.from_api(d["message"]))
                elif "content" in d:
                    messages.append(Message.from_api({"role": "user", "content": d["content"]}))
                elif "text" in d:
                    messages.append(Message.user(str(d["text"])))
            elif e.type is EventType.MODEL_RESPONSE:
                flush()
                if isinstance(d.get("message"), dict):
                    msg = Message.from_api(d["message"])
                else:
                    msg = Message(Role.ASSISTANT, [block_from_api(b) for b in d.get("content") or []])
                messages.append(msg)
                usage = usage + _usage_from(d.get("usage"))
                if d.get("model"):
                    out["model"] = str(d["model"])
                if d.get("stop_reason"):
                    out["stop_reason"] = str(d["stop_reason"])
            elif e.type is EventType.TOOL_RESULT and d.get("tool_use_id"):
                pending_results.append(ToolResultBlock(str(d["tool_use_id"]), d.get("content", ""), bool(d.get("is_error", False))))
            elif e.type is EventType.COMPACTION:
                out["compaction_events"] += 1
            elif e.type in (EventType.STOP, EventType.SESSION_END):
                if d.get("stop_reason"):
                    out["stop_reason"] = str(d["stop_reason"])
                if d.get("cost_usd") is not None:
                    out["cost_usd"] = float(d["cost_usd"])
        flush()
        out.update(kw)
        out.setdefault("usage", usage)
        return cls(messages=messages, events=evs, **out)


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + f"... [truncated {len(text) - n} chars]"


def pair_tool_calls(messages: Sequence[Message]) -> list[ToolCallRecord]:
    """Pair every ``tool_use`` with its ``tool_result`` (by id) across the message log."""
    results: dict[str, ToolResultBlock] = {}
    for m in messages:
        for r in m.tool_results:
            results.setdefault(r.tool_use_id, r)
    calls: list[ToolCallRecord] = []
    for m in messages:
        if m.role is not Role.ASSISTANT:
            continue
        for u in m.tool_uses:
            r = results.get(u.id)
            calls.append(ToolCallRecord(u.id, u.name, dict(u.input), r.text if r else None, bool(r.is_error) if r else False))
    return calls


@runtime_checkable
class AgentUnderTest(Protocol):
    """The integrator implements this over the real agent loop.

    Contract: work only inside ``workspace`` (``ctx.cwd``), honour ``ctx.metadata["cancel_event"]``
    (a ``threading.Event`` set on timeout), use ``ctx.metadata["memory_dir"]`` as the only memory
    root, and raise :class:`~harness_evals.errors.InfraError` for failures outside the agent's
    control (provider 5xx after retries, sandbox loss). Any other exception is an agent error.
    """

    def run(self, task: AgentTask, workspace: Path, ctx: RunContext) -> Trajectory: ...


class CallableAgent:
    """Adapter: wrap ``fn(task, workspace, ctx)`` returning a Trajectory, a message list, a string or None."""

    def __init__(self, fn: Callable[[AgentTask, Path, RunContext], Any], *, name: str = "callable-agent") -> None:
        self.fn = fn
        self.name = name

    def run(self, task: AgentTask, workspace: Path, ctx: RunContext) -> Trajectory:
        out = self.fn(task, workspace, ctx)
        if isinstance(out, Trajectory):
            return out
        if out is None:
            return Trajectory()
        if isinstance(out, str):
            return Trajectory(final_text=out)
        if isinstance(out, (list, tuple)) and all(isinstance(m, Message) for m in out):
            return Trajectory.from_messages(list(out))
        raise TypeError(f"agent callable returned unsupported type {type(out).__name__}")


__all__ = [
    "AgentTask",
    "AgentUnderTest",
    "CallableAgent",
    "Limits",
    "ToolCallRecord",
    "Trajectory",
    "pair_tool_calls",
]
