"""Alert rules as data: evaluate in-process and/or render as Prometheus alerting rules.

Research basis (reference §10.7 checklist item 4: "Alert on unexpected cache-miss spikes as
regressions"; cache-hit target ">80-90% of input tokens"; §10.8 pitfalls). Rules are plain data so
they can be versioned with the harness, reviewed, and loaded from YAML.

A rule compares one *signal* (a key of a flat snapshot such as
``flatten_signals(metrics.dashboard())`` plus ``SLOTracker.signals()``) to a threshold and fires
after the condition has held for ``for_seconds`` (Prometheus ``for:`` semantics). A missing or
``None`` signal is "no data", never a firing alert. Rules that carry ``promql`` can also be
rendered into a Prometheus rule file for server-side evaluation.

YAML::

    rules:
      - name: CacheHitRateLow
        signal: cache_hit_rate
        op: "<"
        threshold: 0.8
        for_seconds: 900
        severity: ticket
        summary: Prompt-cache hit rate below 80% of input tokens
        promql: 'sum(rate(harness_tokens_total{type="cache_read"}[15m])) / ... < 0.8'
"""

from __future__ import annotations

import operator
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from harness_core.types import Clock, utcnow

from .errors import ConfigError

_OPS: dict[str, Callable[[float, float], bool]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}
SEVERITIES = ("page", "ticket", "info")


@dataclass(frozen=True)
class AlertRule:
    name: str
    signal: str
    op: str
    threshold: float
    for_seconds: float = 0.0
    severity: str = "ticket"
    summary: str = ""
    promql: str | None = None
    runbook: str = ""

    def __post_init__(self) -> None:
        if self.op not in _OPS:
            raise ConfigError(f"unknown op {self.op!r} (use one of {sorted(_OPS)})", f"rules.{self.name}.op")
        if self.severity not in SEVERITIES:
            raise ConfigError(f"unknown severity {self.severity!r}", f"rules.{self.name}.severity")
        if self.for_seconds < 0:
            raise ConfigError("must be >= 0", f"rules.{self.name}.for_seconds")

    def holds(self, value: float | None) -> bool:
        return value is not None and _OPS[self.op](float(value), self.threshold)


class AlertState(str, Enum):
    PENDING = "pending"
    FIRING = "firing"


@dataclass(frozen=True)
class Alert:
    rule: AlertRule
    state: AlertState
    value: float
    since: datetime


def load_alert_rules(source: Mapping[str, Any] | str | Path) -> list[AlertRule]:
    """Load rules from a mapping, a YAML string, or a ``.yaml``/``.json`` path."""
    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source and source.endswith((".yaml", ".yml", ".json"))):
        source = Path(source).read_text(encoding="utf-8")
    if isinstance(source, str):
        try:
            source = yaml.safe_load(source)
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML: {exc}") from exc
    if not isinstance(source, Mapping) or not isinstance(source.get("rules"), list):
        raise ConfigError("expected a mapping with a 'rules' list")
    rules, names = [], set()
    for i, r in enumerate(source["rules"]):
        path = f"rules[{i}]"
        if not isinstance(r, Mapping):
            raise ConfigError("must be a mapping", path)
        missing = {"name", "signal", "op", "threshold"} - set(r)
        if missing:
            raise ConfigError(f"missing {sorted(missing)}", path)
        unknown = set(r) - {"name", "signal", "op", "threshold", "for_seconds", "severity", "summary", "promql", "runbook"}
        if unknown:
            raise ConfigError(f"unknown keys {sorted(unknown)}", path)
        if r["name"] in names:
            raise ConfigError(f"duplicate rule name {r['name']!r}", path)
        names.add(r["name"])
        thr = r["threshold"]
        if isinstance(thr, bool) or not isinstance(thr, (int, float)):
            raise ConfigError("must be a number", f"{path}.threshold")
        rules.append(
            AlertRule(
                str(r["name"]), str(r["signal"]), str(r["op"]), float(thr), float(r.get("for_seconds", 0)), str(r.get("severity", "ticket")),
                str(r.get("summary", "")), r.get("promql"), str(r.get("runbook", "")),
            )
        )
    return rules


def default_alert_rules() -> list[AlertRule]:
    """The research-tied default rule set (thresholds marked "our choice" in docs/ops/health-and-alerts.md)."""
    return [
        AlertRule(
            "UnexpectedCacheMissSpike", "unexpected_cache_miss_rate", ">", 0.05, 600, "page",
            "More than 5% of model calls missed the prompt cache without a declared rebuild (regression: prefix instability).",
            'sum(rate(harness_cache_observations_total{observation="unexpected_miss"}[10m])) / sum(rate(harness_cache_observations_total[10m])) > 0.05',
            "Diff the system prompt / tool list between the last hit and the first miss; look for timestamps or reordered tools.",
        ),
        AlertRule(
            "CacheHitRateLow", "cache_hit_rate", "<", 0.8, 900, "ticket",
            "Prompt-cache read share of input tokens below the 80% target.",
            'sum(rate(harness_tokens_total{type="cache_read"}[15m])) / sum(rate(harness_tokens_total{type=~"input|cache_read|cache_creation"}[15m])) < 0.8',
        ),
        AlertRule(
            "ToolErrorRateHigh", "tool_error_rate", ">", 0.10, 900, "ticket",
            "More than 10% of tool calls returned errors.",
            'sum(rate(harness_tool_calls_total{outcome="error"}[15m])) / sum(rate(harness_tool_calls_total[15m])) > 0.1',
        ),
        AlertRule(
            "ModelLatencyP95High", "model_latency_p95_ms", ">", 60_000, 900, "ticket",
            "p95 model-call latency above 60 s.",
            "histogram_quantile(0.95, sum by (le) (rate(harness_model_latency_ms_bucket[15m]))) > 60000",
        ),
        AlertRule(
            "TaskSuccessBudgetFastBurn", "slo.task_success.burn_rate_1h", ">=", 14.4, 0, "page",
            "Task-success SLO burning its 30-day error budget at >= 14.4x (2% of budget per hour).",
        ),
    ]


class AlertEvaluator:
    """Stateful evaluation with ``for_seconds`` pending periods. Thread-safe."""

    def __init__(self, rules: Sequence[AlertRule], *, clock: Clock = utcnow) -> None:
        names = [r.name for r in rules]
        if len(names) != len(set(names)):
            raise ConfigError("duplicate rule names")
        self.rules = list(rules)
        self.clock = clock
        self._since: dict[str, datetime] = {}
        self._lock = threading.Lock()

    def evaluate(self, signals: Mapping[str, float | None]) -> list[Alert]:
        now = self.clock()
        out = []
        with self._lock:
            for rule in self.rules:
                value = signals.get(rule.signal)
                if not rule.holds(value):
                    self._since.pop(rule.name, None)
                    continue
                since = self._since.setdefault(rule.name, now)
                state = AlertState.FIRING if (now - since).total_seconds() >= rule.for_seconds else AlertState.PENDING
                out.append(Alert(rule, state, float(value), since))  # type: ignore[arg-type]
        return out

    def firing(self, signals: Mapping[str, float | None]) -> list[Alert]:
        return [a for a in self.evaluate(signals) if a.state is AlertState.FIRING]


def _duration(seconds: float) -> str:
    s = int(seconds)
    if s % 3600 == 0 and s:
        return f"{s // 3600}h"
    if s % 60 == 0 and s:
        return f"{s // 60}m"
    return f"{s}s"


def render_prometheus_rules(rules: Sequence[AlertRule], *, group: str = "harness") -> str:
    """Prometheus rule-file YAML for every rule that carries ``promql``."""
    items = []
    for r in rules:
        if not r.promql:
            continue
        item: dict[str, Any] = {"alert": r.name, "expr": r.promql}
        if r.for_seconds:
            item["for"] = _duration(r.for_seconds)
        item["labels"] = {"severity": r.severity}
        ann = {"summary": r.summary} if r.summary else {}
        if r.runbook:
            ann["runbook"] = r.runbook
        if ann:
            item["annotations"] = ann
        items.append(item)
    return yaml.safe_dump({"groups": [{"name": group, "rules": items}]}, sort_keys=False, width=200)
