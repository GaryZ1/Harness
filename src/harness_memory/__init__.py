"""harness-memory: production memory subsystem for agentic harnesses.

One module per memory type:

=====================  ==========================================  ===============================
Memory type            Module                                      Loading policy
=====================  ==========================================  ===============================
Instruction memory     :mod:`harness_memory.instructions`          always loaded
Working memory         :mod:`harness_memory.working`               in context + recitation
Semantic memory        :mod:`harness_memory.semantic`              index always, topics on demand
Episodic memory        :mod:`harness_memory.episodic`              evidence only (on demand)
Procedural memory      :mod:`harness_memory.procedural`            catalog always, body on demand
Task state             :mod:`harness_memory.task_state`            re-read at session start
=====================  ==========================================  ===============================

Write paths: :mod:`harness_memory.tool` (inline ``memory_20250818`` tool) and
:mod:`harness_memory.pipeline` (Phase-1 extraction, Phase-2 consolidation). Cross-cutting:
:mod:`~harness_memory.store`, :mod:`~harness_memory.security`, :mod:`~harness_memory.retrieval`,
:mod:`~harness_memory.usage`, :mod:`~harness_memory.forgetting`, :mod:`~harness_memory.scopes`.
"""

from .episodic import RolloutSummaryStore, SessionMeta, SessionRegistry, TranscriptLog
from .errors import (
    AccessDeniedError,
    AlreadyExistsError,
    ConcurrencyError,
    InvariantViolation,
    LimitExceededError,
    LockUnavailableError,
    MemoryError_,
    NotFoundError,
    PathSecurityError,
    ValidationError,
)
from .forgetting import ForgettingPolicy, expire_unused, forget_derived
from .frontmatter import Epistemic, MemoryDocument, MemoryMeta, MemoryType, Provenance, Scope, Status
from .index import MemoryIndex, MemorySummary
from .instructions import InstructionLoader
from .llm import LLM, AnthropicLLM, CallableLLM, run_memory_agent
from .pipeline.consolidation import Consolidator, inbox_note
from .pipeline.extraction import ExtractionConfig, Extractor, JobStore
from .procedural import Playbook, SkillLibrary
from .retrieval import BM25Index, HybridRetriever
from .scopes import MemoryLayout, Mount, MountSet, open_store
from .security import PathSandbox, SecretRedactor, TaintTracker
from .semantic import SemanticMemory
from .store import Access, FileMemoryStore, StoreLimits
from .system import MemorySystem, Session
from .task_state import TaskState, guard_tool_call, validate_feature_list_change
from .tool import TOOL_DEFINITION, MemoryToolHandler, ToolResult
from .usage import UsageLog, parse_citations, strip_citations
from .working import WorkingMemory

__version__ = "0.1.0"

__all__ = [
    "Access",
    "AccessDeniedError",
    "AlreadyExistsError",
    "AnthropicLLM",
    "BM25Index",
    "CallableLLM",
    "ConcurrencyError",
    "Consolidator",
    "Epistemic",
    "ExtractionConfig",
    "Extractor",
    "FileMemoryStore",
    "ForgettingPolicy",
    "HybridRetriever",
    "InstructionLoader",
    "InvariantViolation",
    "JobStore",
    "LLM",
    "LimitExceededError",
    "LockUnavailableError",
    "MemoryDocument",
    "MemoryError_",
    "MemoryIndex",
    "MemoryLayout",
    "MemoryMeta",
    "MemorySummary",
    "MemorySystem",
    "MemoryToolHandler",
    "MemoryType",
    "Mount",
    "MountSet",
    "NotFoundError",
    "PathSandbox",
    "PathSecurityError",
    "Playbook",
    "Provenance",
    "RolloutSummaryStore",
    "Scope",
    "SecretRedactor",
    "SemanticMemory",
    "Session",
    "SessionMeta",
    "SessionRegistry",
    "SkillLibrary",
    "Status",
    "StoreLimits",
    "TOOL_DEFINITION",
    "TaintTracker",
    "TaskState",
    "ToolResult",
    "TranscriptLog",
    "UsageLog",
    "ValidationError",
    "WorkingMemory",
    "expire_unused",
    "forget_derived",
    "guard_tool_call",
    "inbox_note",
    "open_store",
    "parse_citations",
    "run_memory_agent",
    "strip_citations",
    "validate_feature_list_change",
    "__version__",
]
