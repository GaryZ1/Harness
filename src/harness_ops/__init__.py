"""harness-ops: observability and production operations for agentic harnesses.

=============================  =====================================  ==========================================
Component                      Module                                 What it gives the harness
=============================  =====================================  ==========================================
Tracing (OTel GenAI)           :mod:`harness_ops.tracing`,            ``Tracer`` impl, nested spans, W3C
                               :mod:`harness_ops.semconv`,            ``traceparent``, semconv rename layer,
                               :mod:`harness_ops.instrument`          ``invoke_agent``/``chat``/``execute_tool``
Content policy                 :mod:`harness_ops.redaction`           structure-only by default, opt-in content
Exporters                      :mod:`harness_ops.exporters`,          JSONL (stdlib) + optional OTel/OTLP bridge
                               :mod:`harness_ops.otel`
Event fan-out                  :mod:`harness_ops.events`              ``EventSink`` that feeds every exporter
Metrics                        :mod:`harness_ops.metrics`,            counters/histograms, Prometheus text,
                               :mod:`harness_ops.cache`               dashboard set, cache-miss classification
Cost accounting                :mod:`harness_ops.pricing`             caller-supplied price table, cost ledger
Durable execution              :mod:`harness_ops.durable`,            step journal, leases + fencing, heartbeats,
                               :mod:`harness_ops.leases`,             idempotency keys, retry taxonomy
                               :mod:`harness_ops.idempotency`,
                               :mod:`harness_ops.retry`
Rainbow deploys                :mod:`harness_ops.deploy`,             version pinning, hash traffic split,
                               :mod:`harness_ops.flags`               drain/rollback, per-model flags
Health, SLOs, alerts           :mod:`harness_ops.health`,             error budgets, burn rates, rules as data
                               :mod:`harness_ops.alerts`
Review sampling                :mod:`harness_ops.sampling`            stratified transcript/trace sampling
=============================  =====================================  ==========================================

Everything works with the standard library + PyYAML; OpenTelemetry is optional (lazy imports).
"""

from .alerts import Alert, AlertEvaluator, AlertRule, AlertState, default_alert_rules, load_alert_rules, render_prometheus_rules
from .cache import CacheMonitor, CacheObservation, CacheTracker, CacheVerdict, hit_ratio
from .deploy import DeploymentRegistry, HarnessVersion, SessionPin, VersionState, prompts_tools_sha
from .durable import (
    JSON_CODEC,
    MODEL_RESPONSE_CODEC,
    TOOL_OUTCOME_CODEC,
    Codec,
    DurableRun,
    RunStatus,
    StepJournal,
    StepRecord,
    StepStatus,
    current_run,
    durable_step,
)
from .errors import (
    ConfigError,
    DeployError,
    DuplicateStepError,
    IdempotencyConflictError,
    IdempotencyInProgressError,
    LeaseHeldError,
    LeaseLostError,
    NoActiveRunError,
    NonDeterminismError,
    OpsError,
    OTelUnavailableError,
    PricingError,
    RetryExhaustedError,
    SessionAbortedError,
    SessionPaused,
    StepSerializationError,
)
from .events import EventExporter, FanoutEventSink, InMemoryEventExporter, JsonlEventExporter, SpanEventExporter, read_events
from .exporters import CriticalPath, InMemorySpanExporter, JsonlSpanExporter, critical_path, read_spans, span_tree
from .flags import FeatureFlags, FlagRule, FlagSnapshot
from .health import (
    DEFAULT_BURN_RATE_POLICIES,
    SLO,
    BurnAlert,
    BurnRatePolicy,
    HealthRegistry,
    HealthReport,
    HealthStatus,
    SLOTracker,
    flatten_signals,
    task_success_sli,
    tool_success_sli,
)
from .idempotency import IdempotencyStore, IdempotentResult, IdempotentToolExecutor, idempotency_key
from .instrument import AgentInstrumentation, AgentRun, ChatCall, ChatRecord, ToolCallSpan
from .leases import Heartbeat, Lease, LeaseStore
from .metrics import PROMETHEUS_CONTENT_TYPE, Counter, Gauge, HarnessMetrics, Histogram, MetricsRegistry
from .otel import OTelSpanExporter, OTelTracer, otel_available, otlp_exporter
from .pricing import PRICING_SCHEMA, BudgetStatus, CostBreakdown, CostLedger, ModelPrice, PriceTable
from .redaction import ContentPolicy
from .retry import (
    DEFAULT_REQUEST_MAX_RETRIES,
    DEFAULT_STREAM_IDLE_TIMEOUT_S,
    DEFAULT_STREAM_MAX_RETRIES,
    ErrorClassification,
    ErrorKind,
    RetryingModelClient,
    RetryPolicy,
    RetryStats,
    StreamIdleTimeout,
    classify_error,
    retry_call,
)
from .sampling import ReviewItem, ReviewQueue, ReviewSampler, SamplingPolicy
from .semconv import A, HarnessOp, Op, SemConv, agent_span_name
from .tracing import (
    Link,
    OpsTracer,
    Sampler,
    Span,
    SpanContext,
    SpanData,
    SpanKind,
    StatusCode,
    current_span,
    extract,
    inject,
    parse_traceparent,
    use_span,
)

__version__ = "0.1.0"

__all__ = [
    "A",
    "AgentInstrumentation",
    "AgentRun",
    "Alert",
    "AlertEvaluator",
    "AlertRule",
    "AlertState",
    "BudgetStatus",
    "BurnAlert",
    "BurnRatePolicy",
    "CacheMonitor",
    "CacheObservation",
    "CacheTracker",
    "CacheVerdict",
    "ChatCall",
    "ChatRecord",
    "Codec",
    "ConfigError",
    "ContentPolicy",
    "CostBreakdown",
    "CostLedger",
    "CriticalPath",
    "Counter",
    "DEFAULT_BURN_RATE_POLICIES",
    "DEFAULT_REQUEST_MAX_RETRIES",
    "DEFAULT_STREAM_IDLE_TIMEOUT_S",
    "DEFAULT_STREAM_MAX_RETRIES",
    "DeployError",
    "DeploymentRegistry",
    "DuplicateStepError",
    "DurableRun",
    "ErrorClassification",
    "ErrorKind",
    "EventExporter",
    "FanoutEventSink",
    "FeatureFlags",
    "FlagRule",
    "FlagSnapshot",
    "Gauge",
    "HarnessMetrics",
    "HarnessOp",
    "HarnessVersion",
    "HealthRegistry",
    "HealthReport",
    "HealthStatus",
    "Heartbeat",
    "Histogram",
    "IdempotencyConflictError",
    "IdempotencyInProgressError",
    "IdempotencyStore",
    "IdempotentResult",
    "IdempotentToolExecutor",
    "InMemoryEventExporter",
    "InMemorySpanExporter",
    "JSON_CODEC",
    "JsonlEventExporter",
    "JsonlSpanExporter",
    "Lease",
    "LeaseHeldError",
    "LeaseLostError",
    "LeaseStore",
    "Link",
    "MODEL_RESPONSE_CODEC",
    "MetricsRegistry",
    "ModelPrice",
    "NoActiveRunError",
    "NonDeterminismError",
    "OTelSpanExporter",
    "OTelTracer",
    "OTelUnavailableError",
    "Op",
    "OpsError",
    "OpsTracer",
    "PRICING_SCHEMA",
    "PROMETHEUS_CONTENT_TYPE",
    "PriceTable",
    "PricingError",
    "RetryExhaustedError",
    "RetryPolicy",
    "RetryStats",
    "RetryingModelClient",
    "ReviewItem",
    "ReviewQueue",
    "ReviewSampler",
    "RunStatus",
    "SLO",
    "SLOTracker",
    "Sampler",
    "SamplingPolicy",
    "SemConv",
    "SessionAbortedError",
    "SessionPaused",
    "SessionPin",
    "Span",
    "SpanContext",
    "SpanData",
    "SpanEventExporter",
    "SpanKind",
    "StatusCode",
    "StepJournal",
    "StepRecord",
    "StepSerializationError",
    "StepStatus",
    "StreamIdleTimeout",
    "TOOL_OUTCOME_CODEC",
    "ToolCallSpan",
    "VersionState",
    "agent_span_name",
    "classify_error",
    "critical_path",
    "current_run",
    "current_span",
    "default_alert_rules",
    "durable_step",
    "extract",
    "flatten_signals",
    "hit_ratio",
    "idempotency_key",
    "inject",
    "load_alert_rules",
    "otel_available",
    "otlp_exporter",
    "parse_traceparent",
    "prompts_tools_sha",
    "read_events",
    "read_spans",
    "render_prometheus_rules",
    "retry_call",
    "span_tree",
    "task_success_sli",
    "tool_success_sli",
    "use_span",
]
