"""Exception hierarchy for :mod:`harness_multiagent`.

Runtime failures of a *subagent* (timeouts, budget exhaustion, bad report JSON) are normally reported
as a typed :class:`~harness_multiagent.result.SubagentStatus` on the result rather than raised, so a
fan-out can continue past one failed worker (Claude Code workflows turn a failed ``agent()`` into
``null``). The exceptions below are raised for programmer errors (unknown agent, invalid brief or
definition) and inside the machinery (cancellation, leases, worktrees).
"""

from __future__ import annotations


class MultiAgentError(Exception):
    """Base class for every error raised by this package."""


class DefinitionError(MultiAgentError):
    """A subagent definition (frontmatter file or programmatic) is malformed or violates an invariant."""


class UnknownAgentError(MultiAgentError):
    """The requested subagent name is not registered."""


class BriefValidationError(MultiAgentError):
    """A delegation brief is missing required fields or tries to widen the definition's privileges."""


class SpawnRefused(MultiAgentError):
    """A spawn was refused by a harness-enforced cap. ``code`` is a stable machine-readable reason."""

    code = "refused"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class DepthLimitExceeded(SpawnRefused):
    code = "depth_limit"


class BudgetExceeded(SpawnRefused):
    """The run-level or per-agent budget is spent (Claude Agent SDK wording: ``Budget limit reached``)."""

    code = "budget"


class ConcurrencyLimitExceeded(SpawnRefused):
    code = "concurrency_limit"


class AgentLimitExceeded(SpawnRefused):
    code = "agent_limit"


class PolicyDenied(SpawnRefused):
    """A spawn policy or ``SubagentStart`` hook blocked the spawn."""

    code = "policy"


class Cancelled(MultiAgentError):
    """Cooperative cancellation was observed. ``reason`` is ``cancelled``, ``timeout`` or ``budget``."""

    def __init__(self, reason: str = "cancelled", detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


class AgentTimeout(Cancelled):
    def __init__(self, detail: str = "") -> None:
        super().__init__("timeout", detail or "deadline exceeded")


class LeaseError(MultiAgentError):
    """Base class for single-writer lease violations."""


class LeaseHeldError(LeaseError):
    """Another writer already holds the write lease on this resource (or an overlapping one)."""


class LeaseNotHeldError(LeaseError):
    """A mutating action was attempted without holding the required write lease."""


class OwnershipConflict(LeaseError):
    """Two writers claimed overlapping files, or a writer touched a file it does not own."""


class StructuredOutputError(MultiAgentError):
    """A subagent's final message was not a valid report object."""


class WorktreeError(MultiAgentError):
    """A git worktree operation failed."""


class MergeConflictError(WorktreeError):
    """Merging a writer's branch back through the integrator conflicted (the merge was aborted)."""


class WorkflowLimitError(MultiAgentError):
    """A scripted workflow exceeded a hard ceiling (items per call, agents per run)."""


class WorkflowStepFailed(MultiAgentError):
    """A directly awaited (``required=True``) workflow step failed; the run ends here."""


__all__ = [
    "MultiAgentError", "DefinitionError", "UnknownAgentError", "BriefValidationError", "SpawnRefused",
    "DepthLimitExceeded", "BudgetExceeded", "ConcurrencyLimitExceeded", "AgentLimitExceeded", "PolicyDenied",
    "Cancelled", "AgentTimeout", "LeaseError", "LeaseHeldError", "LeaseNotHeldError", "OwnershipConflict",
    "StructuredOutputError", "WorktreeError", "MergeConflictError", "WorkflowLimitError", "WorkflowStepFailed",
]
