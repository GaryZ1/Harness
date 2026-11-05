"""Exception hierarchy for :mod:`harness_rl`."""

from __future__ import annotations


class RLError(Exception):
    """Base class for every error raised by this package."""


class TrajectorySchemaError(RLError):
    """A trajectory violates the training schema (unpaired tool calls, tool results in plain user text...)."""


class ProfileError(RLError):
    """A harness profile is invalid."""


class ProfileNotFoundError(ProfileError):
    """No profile matches the model id and no explicit fallback was allowed (never a silent generic fallback)."""


class ProfileAmbiguityError(ProfileError):
    """Several equally specific profiles match the same model id."""


class EnvironmentStateError(RLError):
    """An environment method was called in the wrong lifecycle state (step before reset...)."""


class RewardConfigError(RLError):
    """A reward function is misconfigured (e.g. LLM-judge share above the cap)."""


class ProtectedChangeError(RLError):
    """A proposed harness change touches a protected key (graders, eval config, tests)."""


class HeldOutExhaustedError(RLError):
    """The held-out split has been used for too many acceptance decisions; rotate it."""


class ConfigStoreError(RLError):
    """The harness config store or audit log is inconsistent."""
