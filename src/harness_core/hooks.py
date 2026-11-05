"""Lifecycle hooks: the extension points that turn a thin loop into a controllable one.

Research basis (reference report section 1.7 step 4 and section 2.7 step 9):

* Claude Code and Codex share the event vocabulary ``SessionStart``, ``UserPromptSubmit``,
  ``PreToolUse``, ``PostToolUse``, ``PostToolUseFailure``, ``Stop``, ``PreCompact``,
  ``PostCompact`` and ``SessionEnd``; Gemini CLI adds model-level ``BeforeModel``/``AfterModel``.
  :class:`HookEvent` implements all of them (``pre_model``/``post_model`` for the Gemini pair).
* Semantics copied from Claude Code: a ``PreToolUse`` hook returns ``permissionDecision``
  (``allow``/``deny``/``ask``) plus an optional ``updatedInput``; a Stop hook returns
  ``{"decision": "block", "reason": ...}`` to force continuation or ``{"continue": false}`` to
  halt; a command hook that exits with code ``2`` blocks with stderr as the reason. Command hooks
  default to a 600 s timeout.
* Hooks "don't consume context": only an explicit ``additional_context`` reaches the model, and it
  is appended as a message, never written into the cached prefix.

Hook failures are non-blocking: an exception (or a command exiting with a code other than 0 or 2)
is recorded as an ``error`` event and the hook is treated as having no opinion.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .types import Message, ModelRequest, ModelResponse, PermissionBehavior, RunContext, ToolCall, ToolOutcome, ToolSpec


class HookEvent(str, Enum):
    SESSION_START = "session_start"
    USER_PROMPT_SUBMIT = "user_prompt_submit"
    PRE_MODEL = "pre_model"
    POST_MODEL = "post_model"
    PRE_TOOL = "pre_tool"
    POST_TOOL = "post_tool"
    POST_TOOL_FAILURE = "post_tool_failure"
    STOP = "stop"
    PRE_COMPACT = "pre_compact"
    POST_COMPACT = "post_compact"
    SESSION_END = "session_end"

    @property
    def claude_name(self) -> str:
        """The Claude Code / Codex ``hook_event_name`` for this event."""
        return _CLAUDE_NAMES[self]


_CLAUDE_NAMES = {
    HookEvent.SESSION_START: "SessionStart",
    HookEvent.USER_PROMPT_SUBMIT: "UserPromptSubmit",
    HookEvent.PRE_MODEL: "BeforeModel",
    HookEvent.POST_MODEL: "AfterModel",
    HookEvent.PRE_TOOL: "PreToolUse",
    HookEvent.POST_TOOL: "PostToolUse",
    HookEvent.POST_TOOL_FAILURE: "PostToolUseFailure",
    HookEvent.STOP: "Stop",
    HookEvent.PRE_COMPACT: "PreCompact",
    HookEvent.POST_COMPACT: "PostCompact",
    HookEvent.SESSION_END: "SessionEnd",
}

_TOOL_EVENTS = {HookEvent.PRE_TOOL, HookEvent.POST_TOOL, HookEvent.POST_TOOL_FAILURE}


@dataclass
class HookInput:
    """What a hook sees. Only the fields relevant to ``event`` are set."""

    event: HookEvent
    session_id: str
    ctx: RunContext
    tool_call: ToolCall | None = None
    tool_spec: ToolSpec | None = None
    outcome: ToolOutcome | None = None
    request: ModelRequest | None = None
    response: ModelResponse | None = None
    prompt: Message | None = None
    last_assistant_message: str = ""
    stop_hook_active: bool = False
    data: dict[str, Any] = field(default_factory=dict)  # e.g. {"source": "startup"|"resume"}, {"trigger": "auto"}

    def to_json_dict(self) -> dict[str, Any]:
        """Claude Code-compatible JSON payload for command hooks (stdin)."""
        d: dict[str, Any] = {"hook_event_name": self.event.claude_name, "session_id": self.session_id, "cwd": self.ctx.cwd,
                             "agent_name": self.ctx.agent_name}
        if self.tool_call is not None:
            d.update(tool_name=self.tool_call.name, tool_input=self.tool_call.input, tool_use_id=self.tool_call.id)
        if self.outcome is not None:
            d["tool_response"] = {"content": self.outcome.content, "is_error": self.outcome.is_error}
        if self.prompt is not None:
            d["prompt"] = self.prompt.text
        if self.event is HookEvent.STOP:
            d.update(stop_hook_active=self.stop_hook_active, last_assistant_message=self.last_assistant_message)
        if self.response is not None:
            d["stop_reason"] = self.response.stop_reason.value
        d.update(self.data)
        return d


@dataclass
class HookResult:
    """A hook's opinion. ``None`` / defaults mean "no opinion".

    * ``permission`` / ``updated_input`` - ``pre_tool`` only (``updated_input`` rewrites arguments);
    * ``block`` + ``reason`` - ``stop``: veto the stop and feed ``reason`` back as guidance;
      ``user_prompt_submit``: reject the prompt; ``pre_tool``: deny the call;
    * ``continue_=False`` - halt the run now (any event), with ``stop_reason`` as the explanation;
    * ``additional_context`` - text appended for the model (as a message, never in the prefix);
    * ``updated_request`` - ``pre_model`` only: replace the request about to be sent;
    * ``updated_outcome`` - ``post_tool`` only: replace the tool's result (e.g. redact secrets).
    """

    permission: PermissionBehavior | None = None
    reason: str = ""
    updated_input: dict[str, Any] | None = None
    block: bool = False
    continue_: bool = True
    stop_reason: str = ""
    additional_context: str | None = None
    updated_request: ModelRequest | None = None
    updated_outcome: ToolOutcome | None = None


HookFn = Callable[[HookInput], "HookResult | None"]


@dataclass
class _Registration:
    event: HookEvent
    fn: HookFn
    matcher: re.Pattern[str] | None
    name: str


@dataclass
class HookError:
    hook: str
    event: HookEvent
    error: str


class HookRegistry:
    """Ordered hook registrations per event, with optional tool-name regex matchers."""

    def __init__(self) -> None:
        self._hooks: list[_Registration] = []
        self.errors: list[HookError] = []

    def on(self, event: HookEvent | str, fn: HookFn, *, matcher: str | None = None, name: str | None = None) -> HookFn:
        ev = HookEvent(event)
        if matcher is not None and ev not in _TOOL_EVENTS:
            raise ValueError("matcher is only valid for tool events")
        self._hooks.append(_Registration(ev, fn, re.compile(matcher) if matcher else None,
                                         name or getattr(fn, "__name__", type(fn).__name__)))
        return fn

    def hook(self, event: HookEvent | str, *, matcher: str | None = None) -> Callable[[HookFn], HookFn]:
        def deco(fn: HookFn) -> HookFn:
            return self.on(event, fn, matcher=matcher)
        return deco

    def has(self, event: HookEvent) -> bool:
        return any(r.event is event for r in self._hooks)

    def run(self, inp: HookInput) -> list[HookResult]:
        """Run matching hooks in registration order. For ``pre_tool``, each hook sees the input
        as rewritten by the previous hook's ``updated_input``."""
        results: list[HookResult] = []
        for reg in self._hooks:
            if reg.event is not inp.event:
                continue
            if reg.matcher is not None and (inp.tool_call is None or not reg.matcher.fullmatch(inp.tool_call.name)):
                continue
            try:
                res = reg.fn(inp)
            except Exception as exc:  # noqa: BLE001 - hook failures are non-blocking
                self.errors.append(HookError(reg.name, inp.event, f"{type(exc).__name__}: {exc}"))
                continue
            if res is None:
                continue
            results.append(res)
            if inp.event is HookEvent.PRE_TOOL and res.updated_input is not None and inp.tool_call is not None:
                inp.tool_call = ToolCall(inp.tool_call.id, inp.tool_call.name, dict(res.updated_input))
            if not res.continue_:
                break
        return results

    def drain_errors(self) -> list[HookError]:
        errs, self.errors = self.errors, []
        return errs


def combine_permission(results: list[HookResult]) -> tuple[PermissionBehavior | None, str]:
    """Deny beats ask beats allow (deny precedence); ``block`` on a pre_tool hook means deny."""
    best: PermissionBehavior | None = None
    reason = ""
    rank = {PermissionBehavior.ALLOW: 0, PermissionBehavior.ASK: 1, PermissionBehavior.DENY: 2}
    for r in results:
        p = PermissionBehavior.DENY if r.block else r.permission
        if p is None:
            continue
        if best is None or rank[p] > rank[best]:
            best, reason = p, r.reason
    return best, reason


class CommandHook:
    """Run a shell command as a hook with Claude Code semantics.

    The :meth:`HookInput.to_json_dict` payload is written to stdin. Exit code ``0`` with JSON on
    stdout is parsed (``hookSpecificOutput.permissionDecision``/``permissionDecisionReason``/
    ``updatedInput``/``additionalContext``, top-level ``decision: "block"`` + ``reason``,
    ``continue: false`` + ``stopReason``); plain stdout is ignored. Exit code ``2`` blocks, with
    stderr as the reason. Any other exit code, or a timeout, is a non-blocking error.
    """

    def __init__(self, command: str, *, timeout: float = 600.0, env: dict[str, str] | None = None, cwd: str | None = None) -> None:
        self.command = command
        self.timeout = timeout
        self.env = env
        self.cwd = cwd
        self.__name__ = f"command:{command[:40]}"

    def __call__(self, inp: HookInput) -> HookResult | None:
        env = {**os.environ, **(self.env or {})}
        try:
            proc = subprocess.run(self.command, shell=True, input=json.dumps(inp.to_json_dict(), default=str),
                                  capture_output=True, text=True, timeout=self.timeout, env=env, cwd=self.cwd or inp.ctx.cwd)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"hook timed out after {self.timeout}s") from exc
        if proc.returncode == 2:
            reason = proc.stderr.strip() or "blocked by hook"
            if inp.event is HookEvent.PRE_TOOL:
                return HookResult(permission=PermissionBehavior.DENY, reason=reason)
            return HookResult(block=True, reason=reason)
        if proc.returncode != 0:
            raise RuntimeError(f"hook exited {proc.returncode}: {proc.stderr.strip()[:500]}")
        out = proc.stdout.strip()
        if not out.startswith("{"):
            return None
        try:
            data = json.loads(out)
        except ValueError:
            return None
        return parse_hook_json(data)


def parse_hook_json(data: dict[str, Any]) -> HookResult:
    """Translate Claude Code hook JSON output into a :class:`HookResult`."""
    spec = data.get("hookSpecificOutput") or {}
    res = HookResult()
    decision = spec.get("permissionDecision")
    if decision in ("allow", "deny", "ask"):
        res.permission = PermissionBehavior(decision)
        res.reason = str(spec.get("permissionDecisionReason", ""))
    if isinstance(spec.get("updatedInput"), dict):
        res.updated_input = dict(spec["updatedInput"])
    ctx = spec.get("additionalContext")
    if isinstance(ctx, str) and ctx:
        res.additional_context = ctx
    if data.get("decision") == "block":
        res.block = True
        res.reason = str(data.get("reason", "")) or res.reason
    if data.get("continue") is False:
        res.continue_ = False
        res.stop_reason = str(data.get("stopReason", ""))
    return res
