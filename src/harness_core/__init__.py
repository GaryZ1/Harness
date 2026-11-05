"""harness_core: shared types (``harness_core.types``) plus the agent architecture and loop.

=========================  ===========================================  ======================================
Component                  Module                                       Role
=========================  ===========================================  ======================================
Shared contract            :mod:`harness_core.types`                    blocks, messages, protocols, events
Session log                :mod:`harness_core.session`                  append-only JSONL events, replay, fork
Agent loop                 :mod:`harness_core.loop`                     stop predicate, tools, budgets, resume
Tool executors (hands)     :mod:`harness_core.tools`                    function / composite / lazy sandbox
Permissions                :mod:`harness_core.permissions`              allow-all, deny-all, read-only, ask
Hooks                      :mod:`harness_core.hooks`                    Claude Code-compatible lifecycle hooks
Stop evaluators            :mod:`harness_core.evaluators`               goal / completion-promise / callable
Budgets & retries          :mod:`harness_core.budgets`, ``retry``       turns, tokens, USD, wall clock, backoff
Stuck detection            :mod:`harness_core.stuck`                    doom-loop and edit-count signals
Context contract           :mod:`harness_core.context`                  pass-through assembler, compactor
Long-horizon drivers       :mod:`harness_core.long_horizon`             initializer/coding sessions, Ralph
Anthropic adapter          :mod:`harness_core.anthropic_adapter`        optional ``ModelClient`` (lazy import)
Test doubles               :mod:`harness_core.testing`                  ``ScriptedModelClient`` and friends
=========================  ===========================================  ======================================
"""

from . import types as _types
from .budgets import Budgets, BudgetTracker, CostEstimator, Price, PricingTable
from .context import CompactionResult, Compactor, PassthroughAssembler, estimate_tokens, validate_tool_pairing
from .errors import (
    ApprovalRequired,
    CompactionError,
    ConfigError,
    ContextWindowExceededError,
    CoreError,
    CorruptLogError,
    InvalidSessionIdError,
    ModelError,
    ModelRefusalError,
    RateLimitError,
    SandboxError,
    ScriptExhaustedError,
    SessionError,
    SessionExistsError,
    SessionLockedError,
    SessionNotFoundError,
    TerminalModelError,
    TransientModelError,
)
from .evaluators import (
    CallableEvaluator,
    CompletionPromiseEvaluator,
    Evaluation,
    GoalEvaluator,
    StopContext,
    StopEvaluator,
    Verdict,
)
from .hooks import CommandHook, HookEvent, HookInput, HookRegistry, HookResult
from .long_horizon import LongRunDriver, LongRunPrompts, LongRunReport, RalphLoop, RalphReport
from .loop import AgentLoop, LoopConfig, SteeringQueue, idempotency_key
from .permissions import AllowAllPolicy, Approver, AskAllPolicy, CallbackPolicy, DenyAllPolicy, ReadOnlyPolicy
from .results import PendingApproval, ResultSubtype, RunResult
from .retry import RetryPolicy
from .session import (
    EventLog,
    FanoutSink,
    InMemoryEventLog,
    JsonlEventLog,
    PendingToolBatch,
    Session,
    SessionInfo,
    SessionStatus,
    SessionStore,
    message_from_dict,
    message_to_dict,
)
from .stuck import StuckConfig, StuckDetector, StuckSignal
from .tools import CompositeExecutor, FunctionTool, FunctionToolExecutor, LazyExecutor
from .types import *  # noqa: F401,F403

__version__ = "0.1.0"

__all__ = [
    *_types.__all__,
    "AgentLoop",
    "AllowAllPolicy",
    "ApprovalRequired",
    "Approver",
    "AskAllPolicy",
    "BudgetTracker",
    "Budgets",
    "CallableEvaluator",
    "CallbackPolicy",
    "CommandHook",
    "CompactionError",
    "CompactionResult",
    "Compactor",
    "CompletionPromiseEvaluator",
    "CompositeExecutor",
    "ConfigError",
    "ContextWindowExceededError",
    "CoreError",
    "CorruptLogError",
    "CostEstimator",
    "DenyAllPolicy",
    "Evaluation",
    "EventLog",
    "FanoutSink",
    "FunctionTool",
    "FunctionToolExecutor",
    "GoalEvaluator",
    "HookEvent",
    "HookInput",
    "HookRegistry",
    "HookResult",
    "InMemoryEventLog",
    "InvalidSessionIdError",
    "JsonlEventLog",
    "LazyExecutor",
    "LongRunDriver",
    "LongRunPrompts",
    "LongRunReport",
    "LoopConfig",
    "ModelError",
    "ModelRefusalError",
    "PassthroughAssembler",
    "PendingApproval",
    "PendingToolBatch",
    "Price",
    "PricingTable",
    "RalphLoop",
    "RalphReport",
    "RateLimitError",
    "ReadOnlyPolicy",
    "ResultSubtype",
    "RetryPolicy",
    "RunResult",
    "SandboxError",
    "ScriptExhaustedError",
    "Session",
    "SessionError",
    "SessionExistsError",
    "SessionInfo",
    "SessionLockedError",
    "SessionNotFoundError",
    "SessionStatus",
    "SessionStore",
    "SteeringQueue",
    "StopContext",
    "StopEvaluator",
    "StuckConfig",
    "StuckDetector",
    "StuckSignal",
    "TerminalModelError",
    "TransientModelError",
    "Verdict",
    "estimate_tokens",
    "idempotency_key",
    "message_from_dict",
    "message_to_dict",
    "validate_tool_pairing",
    "__version__",
]
