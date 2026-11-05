"""Exception hierarchy for :mod:`harness_sandbox`.

Errors split along the line the research draws between *configuration* problems (a policy that is
invalid or that tries to weaken isolation without admin approval) and *runtime* problems (a backend
that is unavailable, a sandbox that died). Blocked operations are deliberately **not** exceptions:
they come back to the model as structured :class:`~harness_sandbox.violations.Violation` records
inside the command result (srt's ``<sandbox_violations>`` loop), so the model can pick an
in-boundary alternative instead of the harness crashing.
"""

from __future__ import annotations


class SandboxError(Exception):
    """Base class for every error raised by this package."""


class PolicyError(SandboxError):
    """A sandbox policy failed validation (unknown key, bad value, contradictory settings)."""


class EscapeHatchError(PolicyError):
    """A weakening knob or ``full-access`` mode was requested without admin approval.

    Checklist item 18: weakening knobs are off by default and settable only by admin policy.
    """


class SandboxUnavailableError(SandboxError):
    """The requested isolation backend is missing or unusable on this host (no ``bwrap``, WSL1...)."""


class SandboxCrashedError(SandboxError):
    """The execution environment died. Environments convert this into a tool-call error result
    (Managed Agents: "a dead sandbox becomes a tool-call error the model can retry")."""


class SnapshotError(SandboxError):
    """A workspace snapshot could not be created, restored or diffed."""


class ProxyError(SandboxError):
    """The egress proxy could not start (bad bind address, socket in use...)."""
