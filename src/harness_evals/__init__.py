"""harness-evals: evaluation for agentic harnesses - hold the model fixed, grade end states, read transcripts.

=====================  ==============================================  =================================
Component              Module                                          Research anchor (reference §9)
=====================  ==============================================  =================================
Agent boundary         :mod:`harness_evals.agent`                      task / trial / transcript vocabulary
Task & suite format    :mod:`harness_evals.task`                       Harbor-style tasks, tiers, polarity
Trial isolation        :mod:`harness_evals.isolation`                  fresh sandbox, no shared git/memory
Graders                :mod:`harness_evals.graders`                    deterministic first, judges second
Trial runner           :mod:`harness_evals.runner`                     pinned conditions, infra-only retry
Metrics                :mod:`harness_evals.metrics`                    pass@k, pass^k, SE, paired bootstrap
Result store           :mod:`harness_evals.store`                      append-only JSONL, atomic writes
Reports & A/B          :mod:`harness_evals.report`                     strict+partial, tiers never merged
Transcript review      :mod:`harness_evals.review`                     sampling, taxonomy, grader QA
Guards                 :mod:`harness_evals.guards`                     saturation, held-out, contamination
Judge calibration      :mod:`harness_evals.calibration`                judge-human vs human-human agreement
=====================  ==============================================  =================================
"""

from .agent import AgentTask, AgentUnderTest, CallableAgent, Limits, ToolCallRecord, Trajectory, pair_tool_calls
from .calibration import JudgeCalibration, agreement, cohens_kappa
from .errors import (
    ConfoundedComparisonError,
    ContaminationError,
    EvalError,
    GraderConfigError,
    GraderError,
    InfraError,
    IsolationError,
    ResultStoreError,
    SetupError,
    TaskFormatError,
    TaxonomyError,
)
from .graders import (
    JUDGE_PROMPT_VERSION,
    Criterion,
    GradeContext,
    Grader,
    GraderDeps,
    GradeResult,
    HackSignal,
    LLMJudgeGrader,
    TaskScore,
    build_grader,
    build_graders,
    detect_hack_signals,
    grader_kinds,
    hack_pattern_counts,
    has_tamper,
    register_grader,
    score_task,
)
from .grading import GradingOutcome, grade_trial
from .guards import ExposureLedger, SaturationReport, SplitManager, canary_leaked, gold_patch_similarity, noise_warnings, saturation_report
from .isolation import MemoryPolicy, SandboxFactory, TrialSandbox, diff_manifests, file_manifest, run_command
from .metrics import (
    EfficiencySummary,
    PairedComparison,
    bootstrap_ci,
    cluster_bootstrap_ci,
    mean_se,
    paired_compare,
    pass_at_k,
    pass_hat_k,
    percentile,
    sign_test,
)
from .report import ComparisonReport, RunSummary, compare_runs, summarize
from .review import DEFAULT_TAXONOMY, FailureCategory, ReviewLog, failure_clusters, render_transcript_markdown, sample_for_review
from .runner import EvalRunner, RunConditions, RunConfig, RunResult, TrialStatus, __version__
from .store import ResultStore
from .task import Polarity, ScoringMode, Setup, Suite, SuiteTier, Task, load_suite, load_task, parse_task

__all__ = [
    "AgentTask", "AgentUnderTest", "CallableAgent", "ComparisonReport", "ConfoundedComparisonError", "ContaminationError", "Criterion",
    "DEFAULT_TAXONOMY", "EfficiencySummary", "EvalError", "EvalRunner", "ExposureLedger", "FailureCategory", "GradeContext", "GradeResult",
    "Grader", "GraderConfigError", "GraderDeps", "GraderError", "GradingOutcome", "HackSignal", "InfraError", "IsolationError",
    "JUDGE_PROMPT_VERSION", "JudgeCalibration", "LLMJudgeGrader", "Limits", "MemoryPolicy", "PairedComparison", "Polarity", "ResultStore",
    "ResultStoreError", "ReviewLog", "RunConditions", "RunConfig", "RunResult", "RunSummary", "SandboxFactory", "SaturationReport",
    "ScoringMode", "Setup", "SetupError", "SplitManager", "Suite", "SuiteTier", "Task", "TaskFormatError", "TaskScore", "TaxonomyError",
    "ToolCallRecord", "Trajectory", "TrialSandbox", "TrialStatus", "__version__", "agreement", "bootstrap_ci", "build_grader", "build_graders",
    "canary_leaked", "cluster_bootstrap_ci", "cohens_kappa", "compare_runs", "detect_hack_signals", "diff_manifests", "failure_clusters",
    "file_manifest", "gold_patch_similarity", "grade_trial", "grader_kinds", "hack_pattern_counts", "has_tamper", "load_suite", "load_task",
    "mean_se", "noise_warnings", "paired_compare", "pair_tool_calls", "parse_task", "pass_at_k", "pass_hat_k", "percentile",
    "register_grader", "render_transcript_markdown", "run_command", "sample_for_review", "saturation_report", "score_task", "sign_test",
    "summarize",
]
