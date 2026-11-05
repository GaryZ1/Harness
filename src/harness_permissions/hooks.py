"""PreToolUse / PermissionRequest hooks: the programmable policy layer (L5), fail closed.

Claude Code and Codex share the vocabulary (research §7.2, §7.3 technique 6, checklist §7.7 step 6):

* ``PreToolUse`` returns ``hookSpecificOutput.permissionDecision`` in ``allow|deny|ask|defer``
  plus optional ``updatedInput`` (rewrites the arguments) and ``permissionDecisionReason``;
* a ``command`` hook receives the event as JSON on stdin; exit code ``2`` blocks with stderr as
  the reason; any other non-zero exit, a timeout, or unparsable stdout is a *failure*;
* ``onFailure: "block"`` (our default ``on_failure = "block"``) turns failures into denials;
* default timeouts: 600 s for ``command``/``http``/``mcp_tool``, 30 s for ``prompt``, 60 s for
  ``agent``;
* ``prompt``/``agent`` hooks are LLM judgements, so they are **advisory**: their ``deny``/``ask``
  are honoured, their ``allow`` is ignored ("do not treat them as hard guarantees").

Rewritten input is re-checked by the engine's deterministic layers (a hook cannot rewrite a
call into something a deny rule forbids).
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

DEFAULT_TIMEOUTS_S: dict[str, float] = {"command": 600, "http": 600, "mcp_tool": 600, "prompt": 30, "agent": 60, "callable": 600}
ADVISORY_KINDS = frozenset({"prompt", "agent"})


class HookDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"
    DEFER = "defer"


@dataclass
class HookInput:
    hook_event_name: str
    session_id: str
    tool_name: str
    tool_input: dict[str, Any]
    tool_use_id: str = ""
    cwd: str = "."
    permission_mode: str = "default"
    agent_name: str = "main"
    depth: int = 0
    tainted: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, default=str)


@dataclass
class HookOutput:
    decision: HookDecision = HookDecision.DEFER
    reason: str = ""
    updated_input: dict[str, Any] | None = None
    failed: bool = False
    hook: str = ""

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None, hook: str = "") -> "HookOutput":
        if not data:
            return cls(hook=hook)
        spec = data.get("hookSpecificOutput", data)
        if not isinstance(spec, Mapping):
            raise ValueError("hookSpecificOutput must be an object")
        raw = spec.get("permissionDecision", "defer")
        decision = HookDecision(str(raw).lower())
        updated = spec.get("updatedInput")
        if updated is not None and not isinstance(updated, dict):
            raise ValueError("updatedInput must be an object")
        reason = spec.get("permissionDecisionReason") or spec.get("reason") or data.get("reason") or ""
        return cls(decision=decision, reason=str(reason), updated_input=updated, hook=hook)


class Hook:
    """Base class. Subclasses implement :meth:`_run`."""

    kind = "callable"

    def __init__(self, name: str, *, matcher: str | None = None, events: Sequence[str] = ("PreToolUse",), timeout_s: float | None = None, advisory: bool | None = None) -> None:
        self.name = name
        self.matcher = matcher
        self.events = tuple(events)
        self.timeout_s = timeout_s
        self.advisory = (self.kind in ADVISORY_KINDS) if advisory is None else advisory
        self._rx = None if matcher in (None, "", "*") else re.compile(matcher)

    def applies(self, event: str, tool_name: str) -> bool:
        return event in self.events and (self._rx is None or bool(self._rx.fullmatch(tool_name)))

    def run(self, hook_input: HookInput, timeout_s: float) -> HookOutput:
        result: list[HookOutput] = []
        error: list[BaseException] = []

        def target() -> None:
            try:
                result.append(self._run(hook_input, timeout_s))
            except BaseException as exc:  # noqa: BLE001 - reported as a hook failure
                error.append(exc)

        t = threading.Thread(target=target, name=f"hook-{self.name}", daemon=True)
        t.start()
        t.join(timeout_s)
        if t.is_alive():
            return HookOutput(failed=True, reason=f"hook {self.name!r} timed out after {timeout_s:g}s", hook=self.name)
        if error:
            return HookOutput(failed=True, reason=f"hook {self.name!r} failed: {error[0]}", hook=self.name)
        out = result[0] if result else HookOutput()
        out.hook = self.name
        return out

    def _run(self, hook_input: HookInput, timeout_s: float) -> HookOutput:  # pragma: no cover - abstract
        raise NotImplementedError


class CallableHook(Hook):
    """In-process hook: ``fn(HookInput) -> HookOutput | dict | None``. ``kind`` may be set to ``prompt``/``agent``."""

    def __init__(self, name: str, fn: Callable[[HookInput], Any], *, kind: str = "callable", **kw: Any) -> None:
        self.kind = kind
        super().__init__(name, **kw)
        self.fn = fn

    def _run(self, hook_input: HookInput, timeout_s: float) -> HookOutput:
        out = self.fn(hook_input)
        if out is None:
            return HookOutput()
        if isinstance(out, HookOutput):
            return out
        if isinstance(out, Mapping):
            return HookOutput.from_json(out, self.name)
        raise TypeError(f"hook {self.name!r} returned {type(out).__name__}")


class CommandHook(Hook):
    """External command hook (argv list, never a shell string). JSON in on stdin, JSON out on stdout."""

    kind = "command"

    def __init__(self, name: str, argv: Sequence[str], *, env: Mapping[str, str] | None = None, cwd: str | None = None, **kw: Any) -> None:
        if isinstance(argv, str) or not argv:
            raise ValueError("CommandHook argv must be a non-empty list of strings")
        super().__init__(name, **kw)
        self.argv = list(argv)
        self.env = dict(env) if env is not None else None
        self.cwd = cwd

    def _run(self, hook_input: HookInput, timeout_s: float) -> HookOutput:
        try:
            proc = subprocess.run(self.argv, input=hook_input.to_json(), capture_output=True, text=True, timeout=timeout_s, env=self.env, cwd=self.cwd, check=False)
        except subprocess.TimeoutExpired:
            return HookOutput(failed=True, reason=f"hook {self.name!r} timed out after {timeout_s:g}s")
        except OSError as exc:
            return HookOutput(failed=True, reason=f"hook {self.name!r} could not start: {exc}")
        if proc.returncode == 2:
            return HookOutput(decision=HookDecision.DENY, reason=(proc.stderr or "blocked by hook").strip())
        if proc.returncode != 0:
            return HookOutput(failed=True, reason=f"hook {self.name!r} exited {proc.returncode}: {proc.stderr.strip()[:300]}")
        stdout = proc.stdout.strip()
        if not stdout:
            return HookOutput()
        try:
            return HookOutput.from_json(json.loads(stdout), self.name)
        except (ValueError, TypeError) as exc:
            return HookOutput(failed=True, reason=f"hook {self.name!r} produced invalid output: {exc}")


class HookRunner:
    """Runs matching hooks in order; aggregates deny > failure(block) > ask > allow > defer."""

    def __init__(self, hooks: Sequence[Hook] = (), *, on_failure: str = "block", timeouts_s: Mapping[str, float] | None = None) -> None:
        if on_failure not in {"block", "ignore"}:
            raise ValueError("on_failure must be 'block' or 'ignore'")
        self.hooks = list(hooks)
        self.on_failure = on_failure
        self.timeouts = {**DEFAULT_TIMEOUTS_S, **(timeouts_s or {})}

    def add(self, hook: Hook) -> None:
        self.hooks.append(hook)

    def run(self, event: str, hook_input: HookInput) -> HookOutput:
        current = dict(hook_input.tool_input)
        decision = HookDecision.DEFER
        reasons: list[str] = []
        updated: dict[str, Any] | None = None
        for hook in self.hooks:
            if not hook.applies(event, hook_input.tool_name):
                continue
            hi = HookInput(**{**asdict(hook_input), "hook_event_name": event, "tool_input": current})
            out = hook.run(hi, hook.timeout_s or self.timeouts.get(hook.kind, 600))
            if out.failed:
                if self.on_failure == "block":
                    return HookOutput(HookDecision.DENY, out.reason or "hook failed", None, True, hook.name)
                continue
            if out.decision is HookDecision.DENY:
                return HookOutput(HookDecision.DENY, out.reason or f"denied by hook {hook.name!r}", None, False, hook.name)
            if out.decision is HookDecision.ASK:
                decision = HookDecision.ASK
                reasons.append(out.reason or f"hook {hook.name!r} asks for confirmation")
            elif out.decision is HookDecision.ALLOW and not hook.advisory and decision is not HookDecision.ASK:
                decision = HookDecision.ALLOW
                reasons.append(out.reason or f"allowed by hook {hook.name!r}")
            if out.updated_input is not None and not hook.advisory:
                current = dict(out.updated_input)
                updated = current
        return HookOutput(decision, "; ".join(reasons), updated, False)
