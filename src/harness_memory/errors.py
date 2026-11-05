"""Exception hierarchy for :mod:`harness_memory`."""

from __future__ import annotations


class MemoryError_(Exception):
    """Base class for every error raised by this package.

    Named with a trailing underscore so it never shadows the builtin ``MemoryError``.
    """


class PathSecurityError(MemoryError_):
    """A path escaped its sandbox, was malformed, or targeted a protected location."""


class AccessDeniedError(MemoryError_):
    """A write was attempted against a read-only scope, or a restricted-mode rule blocked it."""


class LimitExceededError(MemoryError_):
    """A size, count or budget cap would be exceeded."""


class ConcurrencyError(MemoryError_):
    """An optimistic-concurrency precondition (``content_sha256``) did not match."""


class NotFoundError(MemoryError_):
    """The requested memory, file, skill or feature does not exist."""


class AlreadyExistsError(MemoryError_):
    """The target already exists and the operation refuses to overwrite it."""


class ValidationError(MemoryError_):
    """Input failed schema or invariant validation."""


class InvariantViolation(ValidationError):
    """A protected invariant (e.g. flip-only feature list, no test deletion) was violated."""


class LockUnavailableError(MemoryError_):
    """A required exclusive lock is held by another writer."""
