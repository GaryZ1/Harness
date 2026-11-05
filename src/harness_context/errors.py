"""Exception hierarchy for :mod:`harness_context`.

Every invariant this package enforces in code (breakpoint limit, append-only log, cut safety,
single in-flight compaction, sub-agent caps) fails with a dedicated exception type instead of a
silent fallback, so the integrator can tell configuration bugs from runtime conditions.
"""

from __future__ import annotations


class ContextError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(ContextError, ValueError):
    """A policy or layer was configured with invalid values (e.g. compaction trigger < 50,000)."""


class CacheStabilityError(ContextError):
    """Strict mode found a silent cache invalidator (timestamp, UUID...) in the cached prefix."""


class BreakpointLimitError(CacheStabilityError):
    """More than the provider maximum (4) ``cache_control`` breakpoints would be sent."""


class CompactionError(ContextError):
    """Client-side compaction failed (no summary, tool call instead of text, request too large...)."""


class CompactionInProgressError(CompactionError):
    """A second compaction was requested for a session while one is still running."""


class NothingToCompactError(CompactionError):
    """There is no safe cut point that would summarize at least one message."""


class ContextOverflowError(ContextError):
    """The assembled window (plus ``max_tokens``) does not fit the model's context window."""


class LogIntegrityError(ContextError):
    """The durable log no longer matches a compaction marker (a summarized message was edited or removed)."""


class OffloadError(ContextError):
    """A large tool output could not be offloaded or restored (I/O error, path outside the sandbox)."""


class SubagentLimitError(ContextError):
    """Spawning a sub-agent would exceed the depth or concurrency cap."""


__all__ = [
    "BreakpointLimitError",
    "CacheStabilityError",
    "CompactionError",
    "CompactionInProgressError",
    "ConfigurationError",
    "ContextError",
    "ContextOverflowError",
    "LogIntegrityError",
    "NothingToCompactError",
    "OffloadError",
    "SubagentLimitError",
]
