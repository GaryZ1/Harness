"""harness-context: context engineering for agentic harnesses.

Turns the append-only durable message log into the model-visible window, following section 3 of
``reports/SOTA agentic harness reference.md`` ("every token is a cost, a cache key and a
distraction"):

=================================  =========================================  ==============================
Component                          Module                                     Ladder rung / checklist step
=================================  =========================================  ==============================
Assembler (``ContextAssembler``)   :mod:`harness_context.assembler`           step 2 layout, L0-L4 driver
Stable prompt layers               :mod:`harness_context.layers`              steps 2, 8, 9
Cache policy + stability checker   :mod:`harness_context.cache`               steps 1-3
Token counting                     :mod:`harness_context.tokens`              step 1
Budget thresholds                  :mod:`harness_context.budget`              steps 11 (guards), 15
Ingest cap + file offload          :mod:`harness_context.offload`             step 6; L0, L1
Tool-result clearing               :mod:`harness_context.clearing`            L2
Client compaction + summary        :mod:`harness_context.compaction`          steps 11-12; L3
Post-compaction re-injection       :mod:`harness_context.reinjection`         L4
Server-side context management     :mod:`harness_context.server`              L2/L3 (provider)
Dynamic tail + recitation          :mod:`harness_context.dynamic`             step 4
Just-in-time references            :mod:`harness_context.references`          technique 3, 21
Sub-agent isolation                :mod:`harness_context.subagent`            step 14
Context-rot guards                 :mod:`harness_context.guards`              step 17
=================================  =========================================  ==============================
"""

from .assembler import AssemblerConfig, ContextReport, LayeredContextAssembler, canonical_tools
from .budget import PRE_COMPACTION_WARNING, BudgetLevel, BudgetPolicy, UsageCalibration
from .cache import (
    CacheBreak,
    CacheFinding,
    CacheMonitor,
    CachePolicy,
    CacheStabilityChecker,
    CacheStats,
    PrefixSnapshot,
)
from .clearing import ClearingPlan, ClearingPolicy, ToolResultClearer
from .compaction import (
    SUMMARY_PROMPT,
    CompactionConfig,
    CompactionResult,
    CompactionSummary,
    Compactor,
    choose_cut,
    parse_summary,
)
from .dynamic import DynamicContext, DynamicItem, safe_to_append
from .errors import (
    BreakpointLimitError,
    CacheStabilityError,
    CompactionError,
    CompactionInProgressError,
    ConfigurationError,
    ContextError,
    ContextOverflowError,
    LogIntegrityError,
    NothingToCompactError,
    OffloadError,
    SubagentLimitError,
)
from .guards import StuckDetector, tool_count_warning
from .layers import LayerKind, LayerLint, PromptLayer, PromptStack, SkillEntry
from .offload import OffloadedOutput, OffloadingToolExecutor, OffloadPolicy, ToolOutputOffloader
from .references import ContextReference, extract_locators, extract_paths, render_reference_index
from .reinjection import (
    FileActivityTracker,
    RecentFilesSource,
    ReinjectionBuilder,
    ReinjectionBundle,
    SkillBodiesSource,
    TextSource,
)
from .server import ServerContextManagement
from .subagent import SUBAGENT_RESULT_GUIDANCE, DelegationBrief, SubagentLimits, condense_subagent_result
from .tokens import AnthropicTokenCounter, HeuristicTokenCounter, TokenCounter

__version__ = "0.1.0"

__all__ = [
    "AnthropicTokenCounter",
    "AssemblerConfig",
    "BreakpointLimitError",
    "BudgetLevel",
    "BudgetPolicy",
    "CacheBreak",
    "CacheFinding",
    "CacheMonitor",
    "CachePolicy",
    "CacheStabilityChecker",
    "CacheStabilityError",
    "CacheStats",
    "ClearingPlan",
    "ClearingPolicy",
    "CompactionConfig",
    "CompactionError",
    "CompactionInProgressError",
    "CompactionResult",
    "CompactionSummary",
    "Compactor",
    "ConfigurationError",
    "ContextError",
    "ContextOverflowError",
    "ContextReference",
    "ContextReport",
    "DelegationBrief",
    "DynamicContext",
    "DynamicItem",
    "FileActivityTracker",
    "HeuristicTokenCounter",
    "LayerKind",
    "LayerLint",
    "LayeredContextAssembler",
    "LogIntegrityError",
    "NothingToCompactError",
    "OffloadError",
    "OffloadPolicy",
    "OffloadedOutput",
    "OffloadingToolExecutor",
    "PRE_COMPACTION_WARNING",
    "PrefixSnapshot",
    "PromptLayer",
    "PromptStack",
    "RecentFilesSource",
    "ReinjectionBuilder",
    "ReinjectionBundle",
    "SUBAGENT_RESULT_GUIDANCE",
    "SUMMARY_PROMPT",
    "ServerContextManagement",
    "SkillBodiesSource",
    "SkillEntry",
    "StuckDetector",
    "SubagentLimitError",
    "SubagentLimits",
    "TextSource",
    "TokenCounter",
    "ToolOutputOffloader",
    "ToolResultClearer",
    "UsageCalibration",
    "canonical_tools",
    "choose_cut",
    "condense_subagent_result",
    "extract_locators",
    "extract_paths",
    "parse_summary",
    "render_reference_index",
    "safe_to_append",
    "tool_count_warning",
]
