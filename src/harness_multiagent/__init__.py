"""harness-multiagent: multi-agent orchestration for agentic harnesses.

"Many readers, one writer, scripted fan-out" (``reports/SOTA agentic harness reference.md`` section 8):

======================  ===============================================  ==========================================
Component               Module                                           Purpose
======================  ===============================================  ==========================================
Definitions             :mod:`harness_multiagent.definitions`            ``.harness/agents/<name>.md`` + registry
Delegation brief        :mod:`harness_multiagent.brief`                  the only context a fresh subagent gets
Spawn primitive         :mod:`harness_multiagent.manager`                agent-as-tool with depth/budget/concurrency caps
Runner seam             :mod:`harness_multiagent.runner`                 ``AgentRunner`` protocol + built-in loop
Returns                 :mod:`harness_multiagent.result`                 condensed, typed, untrusted reports
Budgets & accounting    :mod:`harness_multiagent.budget`                 hierarchical ledgers, per-agent usage
Cancellation            :mod:`harness_multiagent.cancel`                 hierarchical tokens with deadlines
Single writer           :mod:`harness_multiagent.leases`, ``tools``      write leases, ownership, scoped tools
Worktrees               :mod:`harness_multiagent.worktree`               per-writer worktrees + one integrator
Scripted fan-out        :mod:`harness_multiagent.orchestration`          ``Workflow``: parallel / pipeline / map-reduce
Patterns                :mod:`harness_multiagent.patterns`               orchestrator-worker, voting, best-of-N
Coordination            :mod:`harness_multiagent.coordination`           file-locked task board
When not to             :mod:`harness_multiagent.gate`                   delegation gate, effort tiers, policy text
======================  ===============================================  ==========================================
"""

from .brief import DelegationBrief
from .budget import AgentRecord, AgentTotals, BudgetLedger, BudgetLimits, CriticalPath, RunAccounting, usage_tokens
from .cancel import CancellationToken
from .coordination import BoardTask, TaskBoard, TaskStatus
from .definitions import AgentRegistry, ContextMode, MemoryAccess, SubagentDefinition, WriteAccess, estimate_tokens, select_tools
from .errors import (
    AgentLimitExceeded,
    AgentTimeout,
    BriefValidationError,
    BudgetExceeded,
    Cancelled,
    ConcurrencyLimitExceeded,
    DefinitionError,
    DepthLimitExceeded,
    LeaseError,
    LeaseHeldError,
    LeaseNotHeldError,
    MergeConflictError,
    MultiAgentError,
    OwnershipConflict,
    PolicyDenied,
    SpawnRefused,
    StructuredOutputError,
    UnknownAgentError,
    WorkflowLimitError,
    WorkflowStepFailed,
    WorktreeError,
)
from .gate import DELEGATION_POLICY, EffortTier, GateDecision, Recommendation, TaskProfile, effort_tier, fanout_warnings, should_delegate
from .leases import FileOwnership, LeaseManager, ReadLease, WriteLease
from .manager import SPAWN_TOOL_NAME, ManagerConfig, SubagentManager
from .orchestration import MapReduceResult, Verdict, VerifiedItem, Workflow, WorkflowConfig, WorkflowJournal
from .patterns import (
    Consensus,
    OrchestrationResult,
    OrchestratorWorker,
    Selection,
    agent_verifier,
    best_of_n,
    dedup_findings,
    generate_evaluate,
    iterate_until_converged,
    planner_from_agent,
    rank_findings,
    report_verdict,
    vote,
    vote_on_claims,
)
from .result import REPORT_INSTRUCTIONS, Finding, SubagentReport, SubagentResult, SubagentStatus, condense, parse_report, scan_untrusted
from .runner import AgentRunner, ModelRouter, SimpleAgentRunner, SubagentTask, compose_system_prompt
from .tools import ScopedToolExecutor, ToolHandler, as_handler
from .worktree import Integrator, MergeResult, Worktree, WorktreeManager

__version__ = "0.1.0"

__all__ = [
    # definitions & briefs
    "SubagentDefinition", "AgentRegistry", "WriteAccess", "MemoryAccess", "ContextMode", "select_tools", "estimate_tokens",
    "DelegationBrief",
    # spawning
    "SubagentManager", "ManagerConfig", "SPAWN_TOOL_NAME", "AgentRunner", "SubagentTask", "ModelRouter", "SimpleAgentRunner",
    "compose_system_prompt", "ScopedToolExecutor", "ToolHandler", "as_handler",
    # results
    "SubagentResult", "SubagentReport", "SubagentStatus", "Finding", "REPORT_INSTRUCTIONS", "parse_report", "condense", "scan_untrusted",
    # budgets, cancellation, accounting
    "BudgetLedger", "BudgetLimits", "RunAccounting", "AgentRecord", "AgentTotals", "CriticalPath", "usage_tokens", "CancellationToken",
    # single writer
    "LeaseManager", "WriteLease", "ReadLease", "FileOwnership", "WorktreeManager", "Worktree", "Integrator", "MergeResult",
    # orchestration
    "Workflow", "WorkflowConfig", "WorkflowJournal", "Verdict", "VerifiedItem", "MapReduceResult",
    "OrchestratorWorker", "OrchestrationResult", "planner_from_agent", "agent_verifier", "report_verdict",
    "vote", "vote_on_claims", "Consensus", "dedup_findings", "rank_findings", "best_of_n", "Selection",
    "iterate_until_converged", "generate_evaluate",
    # coordination & gate
    "TaskBoard", "BoardTask", "TaskStatus",
    "should_delegate", "TaskProfile", "GateDecision", "Recommendation", "effort_tier", "EffortTier", "fanout_warnings", "DELEGATION_POLICY",
    # errors
    "MultiAgentError", "DefinitionError", "UnknownAgentError", "BriefValidationError", "SpawnRefused", "DepthLimitExceeded",
    "BudgetExceeded", "ConcurrencyLimitExceeded", "AgentLimitExceeded", "PolicyDenied", "Cancelled", "AgentTimeout", "LeaseError",
    "LeaseHeldError", "LeaseNotHeldError", "OwnershipConflict", "StructuredOutputError", "WorktreeError", "MergeConflictError",
    "WorkflowLimitError", "WorkflowStepFailed",
]
