"""Exception hierarchy for :mod:`harness_ops`.

Every error raised on purpose by this package derives from :class:`OpsError`, so a harness can
separate "the ops layer refused" from provider or tool failures with a single ``except``.
"""

from __future__ import annotations

from datetime import datetime


class OpsError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(OpsError, ValueError):
    """A configuration document (pricing table, alert rules, flags) failed validation.

    ``path`` names the offending field, e.g. ``models.test-model.input_per_mtok``.
    """

    def __init__(self, message: str, path: str = "") -> None:
        super().__init__(f"{path}: {message}" if path else message)
        self.path = path


class PricingError(OpsError):
    """A cost could not be computed (unknown model in a strict price table)."""


class LeaseLostError(OpsError):
    """This worker no longer holds the lease (fencing token superseded or lease released).

    The worker must stop writing immediately: another worker owns the run now.
    """


class LeaseHeldError(OpsError):
    """The resource is leased by another live worker."""

    def __init__(self, resource: str, owner: str, expires_at: float) -> None:
        super().__init__(f"{resource!r} is leased by {owner!r} until {expires_at:.0f}")
        self.resource = resource
        self.owner = owner
        self.expires_at = expires_at


class DuplicateStepError(OpsError):
    """The same step key was used twice in one execution of a durable run."""


class NonDeterminismError(OpsError):
    """A replayed step's recorded input differs from the input supplied now (strict mode only)."""


class StepSerializationError(OpsError):
    """A durable step's result could not be serialised by the journal codec."""


class NoActiveRunError(OpsError):
    """:func:`harness_ops.durable.durable_step` was called outside ``with run:``."""


class IdempotencyConflictError(OpsError):
    """An idempotency key was reused with a different request fingerprint."""


class IdempotencyInProgressError(OpsError):
    """Another live worker is executing the side effect for this key; retry later."""


class RetryExhaustedError(OpsError):
    """A retryable operation still failed after the policy's last attempt.

    ``last_error`` is the final underlying exception (also chained as ``__cause__``).
    """

    def __init__(self, last_error: BaseException, attempts: int) -> None:
        super().__init__(f"gave up after {attempts} attempt(s): {type(last_error).__name__}")
        self.last_error = last_error
        self.attempts = attempts


class SessionPaused(OpsError):
    """Usage or long rate limit: pause the session and resume at ``resume_at`` instead of failing it.

    Mirrors Claude Code ``/goal``: "Rate or usage limits pause the goal, which resumes when the limit resets."
    """

    def __init__(self, reason: str, resume_at: datetime | None) -> None:
        super().__init__(f"session paused ({reason}); resume at {resume_at.isoformat() if resume_at else 'unknown'}")
        self.reason = reason
        self.resume_at = resume_at


class DeployError(OpsError):
    """A deployment operation violated an invariant (unknown version, retire with in-flight sessions...)."""


class SessionAbortedError(DeployError):
    """The session's pinned version was rolled back with ``abort_inflight=True``; end the session."""


class OTelUnavailableError(OpsError, ImportError):
    """The optional OpenTelemetry dependency is not installed."""
