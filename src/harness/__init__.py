"""harness: the top-level integration of every ``harness_*`` package into one working agent harness.

=====================  ==================================================================
Module                 Role
=====================  ==================================================================
:mod:`harness.config`  ``HarnessConfig`` (YAML; the model id is required, never defaulted)
:mod:`harness.build`   ``build_harness(config, ...) -> Harness`` wiring every package
:mod:`harness.adapters` ``LoopAgentRunner`` (multi-agent), ``LoopAgentUnderTest`` (evals/RL),
                       the core ``Compactor`` over harness_context, event plumbing
=====================  ==================================================================

Quick start (offline)::

    from harness import HarnessConfig, build_harness
    from harness_core.testing import ScriptedModelClient, text_response

    cfg = HarnessConfig.from_yaml("harness.yaml")          # model: ${HARNESS_MODEL}
    with build_harness(cfg, model_client=ScriptedModelClient([text_response("done")])) as h:
        print(h.run("Say done").final_text, h.metrics()["stop_reasons"])
"""

from .adapters import (
    OVERFLOW_KEY,
    ContextCompactorAdapter,
    ExcludingExecutor,
    LoopAgentRunner,
    LoopAgentUnderTest,
    OpsEventAdapter,
    OverflowGuardClient,
    SessionBoundAssembler,
    SessionEventSink,
    StepIndexSink,
    ToolCallCap,
    trajectory_from_session,
)
from .build import ApprovalUI, EnginePolicy, Harness, build_harness, make_approver
from .config import (
    DEFAULT_SYSTEM_PROMPT,
    BudgetsConfig,
    FeaturesConfig,
    HarnessConfig,
    HarnessConfigError,
    McpServerConfig,
    PermissionsConfig,
    SandboxConfig,
)

__version__ = "0.1.0"

__all__ = [
    "OVERFLOW_KEY",
    "ApprovalUI",
    "BudgetsConfig",
    "ContextCompactorAdapter",
    "DEFAULT_SYSTEM_PROMPT",
    "EnginePolicy",
    "ExcludingExecutor",
    "FeaturesConfig",
    "Harness",
    "HarnessConfig",
    "HarnessConfigError",
    "LoopAgentRunner",
    "LoopAgentUnderTest",
    "McpServerConfig",
    "OpsEventAdapter",
    "OverflowGuardClient",
    "PermissionsConfig",
    "SandboxConfig",
    "SessionBoundAssembler",
    "SessionEventSink",
    "StepIndexSink",
    "ToolCallCap",
    "build_harness",
    "make_approver",
    "trajectory_from_session",
]
