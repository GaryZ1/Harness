"""Exception hierarchy for :mod:`harness_evals`.

The split between :class:`InfraError` and everything else is load-bearing: the trial runner
retries *only* infrastructure failures (sandbox creation, setup, transient provider outages) and
records every retry, so a flaky environment can never be laundered into an agent success or an
agent failure (Anthropic, "Demystifying evals for AI agents": isolate trials, make failures fair).
"""

from __future__ import annotations


class EvalError(Exception):
    """Base class for every error raised by this package."""


class TaskFormatError(EvalError):
    """A task or suite file is malformed or violates a schema invariant."""


class GraderConfigError(EvalError):
    """A grader spec is invalid (unknown kind, missing field, missing judge client)."""


class GraderError(EvalError):
    """A grader could not produce a verdict (crashed, judge output unparseable).

    A grader error is *not* an agent failure; the runner records it as ``grader_error`` so grader QA
    can find it (CORE-Bench went 42% -> 95% after grader fixes).
    """


class InfraError(EvalError):
    """Infrastructure failure outside the agent's control. The only error class that is retried."""


class SetupError(InfraError):
    """Workspace materialisation or a task setup command failed."""


class IsolationError(EvalError):
    """A trial could observe state from outside its sandbox (shared git history, memory, caches)."""


class ResultStoreError(EvalError):
    """The result store is unreadable or an id is invalid."""


class ConfoundedComparisonError(EvalError):
    """Two runs differ in more than the variable under test (model, effort, suite version...)."""


class ContaminationError(EvalError):
    """A held-out task was exposed to an optimizer, or a canary leaked."""


class TaxonomyError(EvalError):
    """A review tag is not part of the configured failure taxonomy."""
