"""Minimal permission policies and the approval-callback contract used by the loop.

The full layered permission pipeline (modes, ``Tool(pattern)`` rules, classifier) belongs to a
dedicated package; the loop only needs the shared :class:`~harness_core.types.PermissionPolicy`
protocol plus these defaults.

Research basis (reference report section 1.7 step 5, Agent SDK ``permission_mode`` table):
``default`` mode asks a ``canUseTool`` callback, and "no callback means deny". The loop follows
that: an ``ASK`` decision goes to the configured :data:`Approver`; with no approver it becomes a
denial. Denials come back to the model as the tool result.
"""

from __future__ import annotations

from typing import Callable, Union

from .types import PermissionBehavior, PermissionDecision, RunContext, ToolCall, ToolSpec

#: ``approver(call, spec, ctx, decision) -> bool | PermissionDecision``. Return ``True``/``False``
#: (or a decision, optionally with ``updated_input``), or raise
#: :class:`~harness_core.errors.ApprovalRequired` to durably pause the run.
Approver = Callable[[ToolCall, Union[ToolSpec, None], RunContext, PermissionDecision], Union[bool, PermissionDecision]]


class AllowAllPolicy:
    """Allow every call. Only for isolated environments (the ``bypassPermissions`` analogue)."""

    def check(self, call: ToolCall, spec: ToolSpec | None, ctx: RunContext) -> PermissionDecision:
        return PermissionDecision(PermissionBehavior.ALLOW, "allow-all policy")


class DenyAllPolicy:
    """Deny every call (useful for evaluators, dry runs and plan-only sub-agents)."""

    def check(self, call: ToolCall, spec: ToolSpec | None, ctx: RunContext) -> PermissionDecision:
        return PermissionDecision(PermissionBehavior.DENY, "deny-all policy")


class AskAllPolicy:
    """Route every call to the approver."""

    def check(self, call: ToolCall, spec: ToolSpec | None, ctx: RunContext) -> PermissionDecision:
        return PermissionDecision(PermissionBehavior.ASK, "approval required for every tool")


class ReadOnlyPolicy:
    """Allow tools whose *trusted* spec says ``read_only``; ask for everything else (loop default).

    With no approver configured this means "read-only tools run, everything else is denied",
    which matches the Agent SDK's default mode without a ``canUseTool`` callback.
    """

    def check(self, call: ToolCall, spec: ToolSpec | None, ctx: RunContext) -> PermissionDecision:
        if spec is not None and spec.read_only:
            return PermissionDecision(PermissionBehavior.ALLOW, "read-only tool")
        return PermissionDecision(PermissionBehavior.ASK, "tool may have side effects")


class CallbackPolicy:
    """Adapt a plain function ``fn(call, spec, ctx) -> PermissionDecision`` to the protocol."""

    def __init__(self, fn: Callable[[ToolCall, Union[ToolSpec, None], RunContext], PermissionDecision]) -> None:
        self.fn = fn

    def check(self, call: ToolCall, spec: ToolSpec | None, ctx: RunContext) -> PermissionDecision:
        return self.fn(call, spec, ctx)
