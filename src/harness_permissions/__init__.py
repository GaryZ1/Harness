"""harness-permissions: layered permissions, human-in-the-loop approvals and injection defence.

"Auto-allow inside the boundary, classify every crossing." One module per component:

=========================  ==============================================  =================================
Component                  Module                                          Pipeline layer
=========================  ==============================================  =================================
Policy engine              :mod:`harness_permissions.engine`               L0-L8 orchestration
Modes + bypass guard       :mod:`harness_permissions.modes`                L2, bypass
Rules (``Tool(pattern)``)  :mod:`harness_permissions.rules`                L1 deny / ask, L4 allow
Shell analyzer             :mod:`harness_permissions.shell`                normalize (every sub-command)
Boundary classification    :mod:`harness_permissions.boundary`             normalize, L1 hard deny, L3
Paths / network            :mod:`~.paths`, :mod:`~.network`                normalize
Exec policy                :mod:`harness_permissions.execpolicy`           L1 forbidden, L1b prompt, L4 allow
Hooks                      :mod:`harness_permissions.hooks`                L5, PermissionRequest
Taint + injection probes   :mod:`~.taint`, :mod:`~.injection`              L6, PostToolUse
Auto-reviewer              :mod:`harness_permissions.reviewer`             L7
Approval broker            :mod:`harness_permissions.broker`               L8 (human)
Denials / fatigue          :mod:`harness_permissions.denials`              escalation, fatigue
MCP pinning                :mod:`harness_permissions.mcp`                  L0
Policy file                :mod:`harness_permissions.config`               configuration layers
Audit                      :mod:`harness_permissions.audit`                every decision
=========================  ==============================================  =================================
"""

from .audit import AuditTrail, MemoryEventSink
from .boundary import HARD_DENY, OUTWARD, ActionProfile, BoundaryConfig, Crossing, ToolKind, classify, tool_kind
from .broker import (
    ApprovalBroker,
    ApprovalScope,
    ApprovalStatus,
    Approver,
    ApproverKind,
    HumanInTheLoopPolicy,
    PendingApproval,
    denial_outcome,
)
from .config import (
    SCHEMA,
    Layer,
    PolicyDocument,
    ResolvedPolicy,
    default_policy,
    discover_policy_files,
    load_layers,
    load_policy_file,
    parse_policy,
    resolve,
)
from .denials import DenialTracker, FatigueMonitor, FatigueReport
from .engine import DENIAL_GUIDANCE, DecisionRecord, PolicyEngine, SessionState, SubagentLimitError, suggest_rules
from .errors import (
    ApprovalAlreadyResolvedError,
    ApprovalNotFoundError,
    BypassRefusedError,
    ClassifierError,
    ModeError,
    PermissionsError,
    PolicyLoadError,
    PolicyValidationError,
    RuleSyntaxError,
    UntrustedApproverError,
)
from .execpolicy import ExecDecision, ExecPolicy, PrefixRule, host_executable, prefix_rule
from .hooks import CallableHook, CommandHook, HookDecision, HookInput, HookOutput, HookRunner
from .injection import HeuristicInjectionDetector, InjectionProbe, ModelInjectionClassifier, ProbeResult
from .mcp import PinStatus, ToolPinStore, description_hash
from .modes import PermissionMode, SandboxAttestation, least_permissive, require_bypass_allowed
from .paths import Workspace
from .reviewer import DEFAULT_BLOCK_RULES, ActionReviewer, ReviewRequest, ReviewVerdict, TwoStageClassifier, review_inputs
from .rules import Rule, RuleEffect, is_blanket, lint_rule
from .shell import ShellAnalysis, SimpleCommand, analyze
from .taint import SessionTaint, TaintSource

__version__ = "0.1.0"

__all__ = [
    "AuditTrail", "MemoryEventSink",
    "HARD_DENY", "OUTWARD", "ActionProfile", "BoundaryConfig", "Crossing", "ToolKind", "classify", "tool_kind",
    "ApprovalBroker", "ApprovalScope", "ApprovalStatus", "Approver", "ApproverKind", "HumanInTheLoopPolicy", "PendingApproval",
    "denial_outcome",
    "SCHEMA", "Layer", "PolicyDocument", "ResolvedPolicy", "default_policy", "discover_policy_files", "load_layers",
    "load_policy_file", "parse_policy", "resolve",
    "DenialTracker", "FatigueMonitor", "FatigueReport",
    "DENIAL_GUIDANCE", "DecisionRecord", "PolicyEngine", "SessionState", "SubagentLimitError", "suggest_rules",
    "ApprovalAlreadyResolvedError", "ApprovalNotFoundError", "BypassRefusedError", "ClassifierError", "ModeError",
    "PermissionsError", "PolicyLoadError", "PolicyValidationError", "RuleSyntaxError", "UntrustedApproverError",
    "ExecDecision", "ExecPolicy", "PrefixRule", "host_executable", "prefix_rule",
    "CallableHook", "CommandHook", "HookDecision", "HookInput", "HookOutput", "HookRunner",
    "HeuristicInjectionDetector", "InjectionProbe", "ModelInjectionClassifier", "ProbeResult",
    "PinStatus", "ToolPinStore", "description_hash",
    "PermissionMode", "SandboxAttestation", "least_permissive", "require_bypass_allowed",
    "Workspace",
    "DEFAULT_BLOCK_RULES", "ActionReviewer", "ReviewRequest", "ReviewVerdict", "TwoStageClassifier", "review_inputs",
    "Rule", "RuleEffect", "is_blanket", "lint_rule",
    "ShellAnalysis", "SimpleCommand", "analyze",
    "SessionTaint", "TaintSource",
]
