"""Exception hierarchy for :mod:`harness_core`.

Model errors are split by *harness action*, following the error-class table in the build plan
("The loop" section) and reference report section 2.7 step 3:

=============================  ===============================================  =====================
Class                          Examples                                         Loop action
=============================  ===============================================  =====================
:class:`TransientModelError`   dropped connection, overload, 5xx                retry with backoff
:class:`RateLimitError`        429, usage limit with a reset time               wait ``retry_after``
:class:`ContextWindowExceeded` prompt too long, ``model_context_window_...``    compact, then retry
:class:`TerminalModelError`    auth failure, exhausted credit, bad request      stop, surface error
=============================  ===============================================  =====================

Tool failures are never exceptions at the loop level: they come back to the model as
``is_error`` tool results ("never crash the loop", Managed Agents).
"""

from __future__ import annotations


class CoreError(Exception):
    """Base class for every error raised by :mod:`harness_core`."""


class ConfigError(CoreError, ValueError):
    """Invalid configuration (e.g. a forbidden provider parameter or a missing model id)."""


# ----------------------------------------------------------------------------- session log


class SessionError(CoreError):
    """Base class for session-store errors."""


class SessionNotFoundError(SessionError, KeyError):
    """No session with the requested id exists in the store."""


class SessionExistsError(SessionError):
    """A session with the requested id already exists."""


class SessionLockedError(SessionError):
    """Another live harness process holds this session's lease."""


class CorruptLogError(SessionError):
    """The append-only log has a malformed line or a gap in sequence numbers (not a torn tail)."""


class InvalidSessionIdError(SessionError, ValueError):
    """A session id contains characters that are unsafe in a file name."""


# ----------------------------------------------------------------------------- model calls


class ModelError(CoreError):
    """Base class for errors raised by a :class:`~harness_core.types.ModelClient`."""


class TransientModelError(ModelError):
    """Retryable failure: overload, dropped connection, server error, idle stream."""


class RateLimitError(TransientModelError):
    """Rate or usage limit. ``retry_after`` (seconds) is honoured by the retry policy when set."""

    def __init__(self, message: str = "rate limited", *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class TerminalModelError(ModelError):
    """Non-retryable failure: authentication, exhausted credit, invalid request, unavailable model."""


class ContextWindowExceededError(ModelError):
    """The request (input + ``max_tokens``) does not fit in the model's context window."""


class ModelRefusalError(ModelError):
    """Raised by helpers (not the loop) when a model returns ``stop_reason == "refusal"``."""


# ----------------------------------------------------------------------------- tools & control


class SandboxError(CoreError):
    """The execution environment behind a :class:`~harness_core.types.ToolExecutor` died or is unusable.

    :class:`~harness_core.tools.LazyExecutor` converts it into an ``is_error`` tool result and
    re-provisions the sandbox on the next call; the session is never lost.
    """


class ApprovalRequired(CoreError):
    """Raised by an approver to *durably pause* the run until a human decides.

    The loop records a pending ``permission`` event, returns ``paused_for_approval`` and resumes
    the same tool batch when ``AgentLoop.run(approvals={tool_use_id: True|False})`` is called.
    """

    def __init__(self, reason: str = "approval required") -> None:
        super().__init__(reason)
        self.reason = reason


class CompactionError(CoreError):
    """A compactor returned a view that breaks tool_use/tool_result pairing or other invariants."""


class ScriptExhaustedError(CoreError, AssertionError):
    """:class:`~harness_core.testing.ScriptedModelClient` was called more times than scripted."""
