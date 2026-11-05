"""Exception hierarchy for :mod:`harness_permissions`.

Every error a caller can act on has its own class. The policy *decision* path never raises for
an ordinary deny: denials are returned as :class:`~harness_core.types.PermissionDecision` values
so the agent loop can hand them back to the model as tool results (research §7.3 technique 9:
"do not silently drop denied calls").
"""

from __future__ import annotations


class PermissionsError(Exception):
    """Base class for every error raised by this package."""


class PolicyLoadError(PermissionsError):
    """A policy file could not be read or parsed (missing parser, bad TOML/YAML/JSON)."""


class PolicyValidationError(PermissionsError):
    """A policy document violates the schema or an invariant (blanket allow rule, ``deny_wins=false``...).

    ``problems`` lists every violation found, each prefixed with its dotted key path, so one load
    reports all mistakes at once.
    """

    def __init__(self, problems: list[str] | str) -> None:
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("; ".join(self.problems))


class RuleSyntaxError(PolicyValidationError):
    """A ``Tool(pattern)`` rule string is malformed."""


class BypassRefusedError(PermissionsError):
    """Bypass mode was requested outside an attested external sandbox, as root, or without admin consent.

    Both vendors document bypass/yolo as "only for externally sandboxed environments" and Claude
    Code refuses ``bypassPermissions`` as root (research §7.2).
    """


class ModeError(PermissionsError):
    """An unknown permission mode, or a mode change that a sub-agent is not allowed to make."""


class UntrustedApproverError(PermissionsError):
    """An approval arrived from a non-human source (sub-agent, teammate, tool output).

    "Instructions or approval claims within them carry no authority from the user" (Claude Code
    sub-agents docs); agent-team messages "cannot grant permissions".
    """


class ApprovalNotFoundError(PermissionsError):
    """No pending or resolved approval has the given id."""


class ApprovalAlreadyResolvedError(PermissionsError):
    """The approval was already approved, denied or timed out."""


class ClassifierError(PermissionsError):
    """A model-based classifier failed (transport error or unparsable output). Callers fail closed."""
