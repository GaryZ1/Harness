"""harness-rl: harness-model co-design, training-ready trajectories and a gated self-improving harness.

=====================  ===========================================  ======================================
Component              Module                                       Research anchor (reference §11)
=====================  ===========================================  ======================================
Trajectory export      :mod:`harness_rl.trajectory`                 native tool roles, rollout ids, redaction
Rewards                :mod:`harness_rl.rewards`                    verifiable rewards, hack penalties
Environments           :mod:`harness_rl.env`                        reset/step, rollout-scoped tool backend
Harness profiles       :mod:`harness_rl.profiles`                   "match the model's training harness"
Self-improvement loop  :mod:`harness_rl.improve`                    held-out gates, audit log, rollback
=====================  ===========================================  ======================================

Escalation order (§11.7 step 12): prompt/tool/config optimisation (:mod:`~harness_rl.improve`)
-> per-model profile tuning (:mod:`~harness_rl.profiles`) -> RL on our own harness
(:mod:`~harness_rl.env` + :mod:`~harness_rl.trajectory` feeding an external trainer), and only for
tasks with an automatic, hack-resistant grader (:func:`~harness_rl.rewards.rl_readiness`).
"""

from .env import Episode, FunctionToolExecutor, Observation, RolloutRunner, RolloutToolBackend, StepResult, TaskEnvironment
from .errors import (
    ConfigStoreError,
    EnvironmentStateError,
    HeldOutExhaustedError,
    ProfileAmbiguityError,
    ProfileError,
    ProfileNotFoundError,
    ProtectedChangeError,
    RewardConfigError,
    RLError,
    TrajectorySchemaError,
)
from .improve import (
    AblationProposer,
    AuditLog,
    CallableEvaluator,
    CallableProposer,
    CandidateDecision,
    ConfigChange,
    ConfigStore,
    EvalSuiteEvaluator,
    GateConfig,
    HarnessConfig,
    ImprovementLoop,
    IterationReport,
    ProposalEvidence,
    SplitResult,
    check_change,
)
from .profiles import CompactionProtocol, HarnessProfile, ProfileMismatchMonitor, ProfileRegistry, ProfileSelection, parse_profile
from .rewards import (
    CompositeReward,
    EfficiencyPenalty,
    EpisodeOutcome,
    HackPenalty,
    RewardBreakdown,
    default_reward,
    group_advantages,
    outcome_reward,
    rl_readiness,
    rubric_reward,
    verifiable_reward,
)
from .trajectory import SCHEMA_VERSION, TrainingRecord, TrajectoryExporter, from_events, from_messages, from_trajectory, load_jsonl

__version__ = "0.1.0"

__all__ = [
    "AblationProposer", "AuditLog", "CallableEvaluator", "CallableProposer", "CandidateDecision", "CompactionProtocol", "CompositeReward",
    "ConfigChange", "ConfigStore", "ConfigStoreError", "EfficiencyPenalty", "EnvironmentStateError", "Episode", "EpisodeOutcome",
    "EvalSuiteEvaluator", "FunctionToolExecutor", "GateConfig", "HackPenalty", "HarnessConfig", "HarnessProfile", "HeldOutExhaustedError",
    "ImprovementLoop", "IterationReport", "Observation", "ProfileAmbiguityError", "ProfileError", "ProfileMismatchMonitor",
    "ProfileNotFoundError", "ProfileRegistry", "ProfileSelection", "ProposalEvidence", "ProtectedChangeError", "RLError", "RewardBreakdown",
    "RewardConfigError", "RolloutRunner", "RolloutToolBackend", "SCHEMA_VERSION", "SplitResult", "StepResult", "TaskEnvironment",
    "TrainingRecord", "TrajectoryExporter", "TrajectorySchemaError", "__version__", "check_change", "default_reward", "from_events",
    "from_messages", "from_trajectory", "group_advantages", "load_jsonl", "outcome_reward", "parse_profile", "rl_readiness", "rubric_reward",
    "verifiable_reward",
]
