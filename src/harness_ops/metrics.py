"""Counters, gauges and histograms with a Prometheus text-format renderer, plus the harness dashboard set.

Research basis (reference §10.3 "Derived ops dashboards", §10.2 Claude Code telemetry, checklist item 4):

* Token usage is labelled by type exactly like ``claude_code.token.usage``'s ``token.type``
  (input / output / cacheRead / cacheCreation; we use snake-case label values), cost like
  ``claude_code.cost.usage`` (Prometheus: ``*_cost_usage_USD_total``-style ``_total`` counters).
* Default dashboards: cache hit rate (target >80-90% of input tokens, split expected rebuilds vs
  unexpected misses), cost per *successful* task, tool error rate by tool, turns per task,
  p50/p95 wall-clock and TTFT, compaction frequency, permission asks/denials, stop reasons.
* Label cardinality is capped per metric (tool names come from the model and can be anything):
  extra label sets collapse into ``__overflow__`` instead of exploding the backend.

Quantiles come from a bounded sliding reservoir per label set (the most recent ``reservoir``
observations, default 2,048), so p50/p95 describe *current* behaviour; Prometheus gets cumulative
buckets for server-side ``histogram_quantile``.
"""

from __future__ import annotations

import math
import re
import threading
from collections import deque
from typing import Any, Iterable, Mapping, Sequence

from harness_core.types import Event, EventType, Usage

from .cache import CacheObservation
from .errors import PricingError
from .pricing import PriceTable

_NAME = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_LABEL = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
OVERFLOW = "__overflow__"
DEFAULT_MAX_LABEL_SETS = 1000
DEFAULT_LATENCY_BUCKETS_MS: tuple[float, ...] = (50, 100, 250, 500, 1000, 2500, 5000, 10_000, 30_000, 60_000, 120_000, 300_000)
DEFAULT_TURN_BUCKETS: tuple[float, ...] = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89)

LabelKey = tuple[str, ...]


def _fmt(v: float) -> str:
    if math.isinf(v):
        return "+Inf" if v > 0 else "-Inf"
    if math.isnan(v):
        return "NaN"
    if float(v).is_integer() and abs(v) < 1e15:
        return str(int(v))
    return repr(float(v))


def _escape_label(v: str) -> str:
    return v.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _escape_help(v: str) -> str:
    return v.replace("\\", "\\\\").replace("\n", "\\n")


class _Metric:
    type_name = "untyped"

    def __init__(self, name: str, help: str, labelnames: Sequence[str] = (), *, max_label_sets: int = DEFAULT_MAX_LABEL_SETS) -> None:
        if not _NAME.match(name):
            raise ValueError(f"invalid metric name {name!r}")
        for ln in labelnames:
            if not _LABEL.match(ln) or ln.startswith("__") or ln == "le":
                raise ValueError(f"invalid label name {ln!r}")
        self.name = name
        self.help = help
        self.labelnames = tuple(labelnames)
        self.max_label_sets = max_label_sets
        self._lock = threading.Lock()

    def _key(self, labels: Mapping[str, Any], existing: Mapping[LabelKey, Any]) -> LabelKey:
        extra = set(labels) - set(self.labelnames)
        if extra:
            raise ValueError(f"{self.name}: unknown labels {sorted(extra)}")
        key = tuple(str(labels.get(n, "")) for n in self.labelnames)
        if key not in existing and len(existing) >= self.max_label_sets:
            return tuple(OVERFLOW for _ in self.labelnames)
        return key

    def _labelstr(self, key: LabelKey, extra: Mapping[str, str] | None = None) -> str:
        pairs = [(n, v) for n, v in zip(self.labelnames, key, strict=True) if v != ""]
        if extra:
            pairs.extend(extra.items())
        if not pairs:
            return ""
        return "{" + ",".join(f'{n}="{_escape_label(v)}"' for n, v in pairs) + "}"

    def _header(self) -> list[str]:
        return [f"# HELP {self.name} {_escape_help(self.help)}", f"# TYPE {self.name} {self.type_name}"]

    def render(self) -> list[str]:  # pragma: no cover - overridden
        raise NotImplementedError


class Counter(_Metric):
    type_name = "counter"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._values: dict[LabelKey, float] = {}

    def inc(self, amount: float = 1.0, **labels: Any) -> None:
        if amount < 0:
            raise ValueError("counters only go up")
        with self._lock:
            k = self._key(labels, self._values)
            self._values[k] = self._values.get(k, 0.0) + amount

    def value(self, **labels: Any) -> float:
        """Exact label match; pass a subset to :meth:`sum` instead for aggregation."""
        key = tuple(str(labels.get(n, "")) for n in self.labelnames)
        with self._lock:
            return self._values.get(key, 0.0)

    def sum(self, **match: Any) -> float:
        idx = {n: i for i, n in enumerate(self.labelnames)}
        with self._lock:
            return sum(v for k, v in self._values.items() if all(k[idx[n]] == str(want) for n, want in match.items()))

    def items(self) -> list[tuple[dict[str, str], float]]:
        with self._lock:
            return [(dict(zip(self.labelnames, k, strict=True)), v) for k, v in sorted(self._values.items())]

    def render(self) -> list[str]:
        lines = self._header()
        with self._lock:
            for k, v in sorted(self._values.items()):
                lines.append(f"{self.name}{self._labelstr(k)} {_fmt(v)}")
        return lines


class Gauge(_Metric):
    type_name = "gauge"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._values: dict[LabelKey, float] = {}

    def set(self, value: float, **labels: Any) -> None:
        with self._lock:
            self._values[self._key(labels, self._values)] = float(value)

    def inc(self, amount: float = 1.0, **labels: Any) -> None:
        with self._lock:
            k = self._key(labels, self._values)
            self._values[k] = self._values.get(k, 0.0) + amount

    def value(self, **labels: Any) -> float:
        key = tuple(str(labels.get(n, "")) for n in self.labelnames)
        with self._lock:
            return self._values.get(key, 0.0)

    def render(self) -> list[str]:
        lines = self._header()
        with self._lock:
            for k, v in sorted(self._values.items()):
                lines.append(f"{self.name}{self._labelstr(k)} {_fmt(v)}")
        return lines


class _HistState:
    __slots__ = ("counts", "sum", "count", "reservoir")

    def __init__(self, nbuckets: int, reservoir: int) -> None:
        self.counts = [0] * nbuckets
        self.sum = 0.0
        self.count = 0
        self.reservoir: deque[float] = deque(maxlen=reservoir)


class Histogram(_Metric):
    type_name = "histogram"

    def __init__(self, name: str, help: str, labelnames: Sequence[str] = (), *, buckets: Sequence[float] = DEFAULT_LATENCY_BUCKETS_MS, reservoir: int = 2048, **kwargs: Any) -> None:
        super().__init__(name, help, labelnames, **kwargs)
        bs = sorted(float(b) for b in buckets)
        if not bs or len(set(bs)) != len(bs):
            raise ValueError("buckets must be non-empty and unique")
        if not math.isinf(bs[-1]):
            bs.append(math.inf)
        self.buckets = tuple(bs)
        self.reservoir_size = reservoir
        self._states: dict[LabelKey, _HistState] = {}

    def observe(self, value: float, **labels: Any) -> None:
        v = float(value)
        if math.isnan(v):
            return
        with self._lock:
            k = self._key(labels, self._states)
            st = self._states.get(k)
            if st is None:
                st = self._states[k] = _HistState(len(self.buckets), self.reservoir_size)
            for i, b in enumerate(self.buckets):
                if v <= b:
                    st.counts[i] += 1
                    break
            st.sum += v
            st.count += 1
            st.reservoir.append(v)

    def _samples(self, match: Mapping[str, Any]) -> list[float]:
        idx = {n: i for i, n in enumerate(self.labelnames)}
        out: list[float] = []
        with self._lock:
            for k, st in self._states.items():
                if all(k[idx[n]] == str(want) for n, want in match.items()):
                    out.extend(st.reservoir)
        return out

    def quantile(self, q: float, **match: Any) -> float | None:
        """Linear-interpolated quantile over the reservoir(s) of matching label sets."""
        if not 0.0 <= q <= 1.0:
            raise ValueError("q must be within [0, 1]")
        data = sorted(self._samples(match))
        if not data:
            return None
        pos = q * (len(data) - 1)
        lo, hi = math.floor(pos), math.ceil(pos)
        return data[lo] + (data[hi] - data[lo]) * (pos - lo)

    def count(self, **match: Any) -> int:
        idx = {n: i for i, n in enumerate(self.labelnames)}
        with self._lock:
            return sum(st.count for k, st in self._states.items() if all(k[idx[n]] == str(w) for n, w in match.items()))

    def total(self, **match: Any) -> float:
        idx = {n: i for i, n in enumerate(self.labelnames)}
        with self._lock:
            return sum(st.sum for k, st in self._states.items() if all(k[idx[n]] == str(w) for n, w in match.items()))

    def render(self) -> list[str]:
        lines = self._header()
        with self._lock:
            for k, st in sorted(self._states.items()):
                cum = 0
                for b, c in zip(self.buckets, st.counts, strict=True):
                    cum += c
                    lines.append(f"{self.name}_bucket{self._labelstr(k, {'le': _fmt(b)})} {cum}")
                lines.append(f"{self.name}_sum{self._labelstr(k)} {_fmt(st.sum)}")
                lines.append(f"{self.name}_count{self._labelstr(k)} {st.count}")
        return lines


class MetricsRegistry:
    """Named metric registry. ``counter``/``gauge``/``histogram`` are get-or-create (same type required)."""

    def __init__(self, *, namespace: str = "harness", max_label_sets: int = DEFAULT_MAX_LABEL_SETS) -> None:
        self.namespace = namespace
        self.max_label_sets = max_label_sets
        self._metrics: dict[str, _Metric] = {}
        self._lock = threading.Lock()

    def _full(self, name: str) -> str:
        return f"{self.namespace}_{name}" if self.namespace else name

    def _get(self, cls: type, name: str, help: str, labelnames: Sequence[str], **kw: Any) -> Any:
        full = self._full(name)
        with self._lock:
            m = self._metrics.get(full)
            if m is None:
                m = self._metrics[full] = cls(full, help, labelnames, max_label_sets=self.max_label_sets, **kw)
            elif not isinstance(m, cls) or m.labelnames != tuple(labelnames):
                raise ValueError(f"metric {full!r} already registered with a different type or labels")
            return m

    def counter(self, name: str, help: str, labelnames: Sequence[str] = ()) -> Counter:
        return self._get(Counter, name, help, labelnames)

    def gauge(self, name: str, help: str, labelnames: Sequence[str] = ()) -> Gauge:
        return self._get(Gauge, name, help, labelnames)

    def histogram(self, name: str, help: str, labelnames: Sequence[str] = (), *, buckets: Sequence[float] = DEFAULT_LATENCY_BUCKETS_MS) -> Histogram:
        return self._get(Histogram, name, help, labelnames, buckets=buckets)

    def metrics(self) -> list[_Metric]:
        with self._lock:
            return [self._metrics[k] for k in sorted(self._metrics)]

    def render_prometheus(self) -> str:
        """Prometheus text exposition format 0.0.4 (``Content-Type: text/plain; version=0.0.4``)."""
        lines: list[str] = []
        for m in self.metrics():
            lines.extend(m.render())
        return "\n".join(lines) + "\n"


PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def _safe_div(a: float, b: float) -> float | None:
    return a / b if b else None


class HarnessMetrics:
    """The standard harness metric set and the derived dashboard values.

    Also an :class:`~harness_ops.events.EventExporter` (``export_event``), so it can sit behind the
    fan-out event sink and update itself from session events.
    """

    def __init__(self, registry: MetricsRegistry | None = None, *, prices: PriceTable | None = None, latency_buckets_ms: Sequence[float] = DEFAULT_LATENCY_BUCKETS_MS) -> None:
        r = self.registry = registry or MetricsRegistry()
        self.prices = prices
        self.tokens = r.counter("tokens_total", "Model tokens by type (input, output, cache_read, cache_creation).", ("model", "type"))
        self.cost = r.counter("cost_usd_total", "Model spend in USD from the caller-supplied price table.", ("model",))
        self.unpriced = r.counter("unpriced_calls_total", "Model calls with no configured price.", ("model",))
        self.model_calls = r.counter("model_calls_total", "Model calls by stop reason.", ("model", "stop_reason"))
        self.model_latency = r.histogram("model_latency_ms", "Model call wall-clock latency in ms.", ("model",), buckets=latency_buckets_ms)
        self.ttft = r.histogram("ttft_ms", "Time to first token in ms.", ("model",), buckets=latency_buckets_ms)
        self.cache_obs = r.counter("cache_observations_total", "Model calls by cache verdict (hit, expected_rebuild, unexpected_miss...).", ("model", "observation", "reason"))
        self.retries = r.counter("retries_total", "Retried operations by error kind.", ("kind",))
        self.tool_calls = r.counter("tool_calls_total", "Tool calls by tool and outcome (ok / error).", ("tool", "outcome"))
        self.tool_duration = r.histogram("tool_duration_ms", "Tool execution duration in ms.", ("tool",), buckets=latency_buckets_ms)
        self.permissions = r.counter("permission_decisions_total", "Permission decisions by behavior (allow / deny / ask).", ("behavior", "tool"))
        self.stops = r.counter("stop_reasons_total", "Loop stop reasons.", ("reason",))
        self.compactions = r.counter("compactions_total", "Context compactions (expected cache rebuilds).")
        self.errors = r.counter("errors_total", "Harness errors by class.", ("error_class",))
        self.tasks = r.counter("tasks_total", "Finished tasks/sessions by outcome (success or error subtype).", ("outcome",))
        self.task_cost = r.counter("task_cost_usd_total", "Spend of finished tasks by outcome.", ("outcome",))
        self.task_turns = r.histogram("task_turns", "Turns per finished task.", ("outcome",), buckets=DEFAULT_TURN_BUCKETS)
        self.task_duration = r.histogram("task_duration_ms", "Wall-clock per finished task in ms.", ("outcome",), buckets=latency_buckets_ms)
        self.exporter_errors = r.counter("telemetry_export_errors_total", "Failed telemetry exports.", ("exporter",))

    # ------------------------------------------------------------------ recording

    def record_model_call(
        self,
        model: str,
        usage: Usage,
        *,
        latency_ms: float | None = None,
        ttft_ms: float | None = None,
        stop_reason: str | None = None,
        cost_usd: float | None = None,
        cache_observation: CacheObservation | str | None = None,
        cache_reason: str = "",
        retry_count: int = 0,
    ) -> float | None:
        """Record one model response; returns the cost used (computed from ``prices`` if not given)."""
        self.tokens.inc(usage.input_tokens, model=model, type="input")
        self.tokens.inc(usage.output_tokens, model=model, type="output")
        self.tokens.inc(usage.cache_read_input_tokens, model=model, type="cache_read")
        self.tokens.inc(usage.cache_creation_input_tokens, model=model, type="cache_creation")
        if cost_usd is None and self.prices is not None:
            try:
                br = self.prices.cost(model, usage)
            except PricingError:  # metrics never raise on a missing price; the unpriced counter shows it
                br = None
            if br is not None and br.priced:
                cost_usd = br.total_usd
        if cost_usd is None:
            self.unpriced.inc(model=model)
        else:
            self.cost.inc(cost_usd, model=model)
        self.model_calls.inc(model=model, stop_reason=stop_reason or "unknown")
        if latency_ms is not None:
            self.model_latency.observe(latency_ms, model=model)
        if ttft_ms is not None:
            self.ttft.observe(ttft_ms, model=model)
        if cache_observation is not None:
            obs = cache_observation.value if isinstance(cache_observation, CacheObservation) else str(cache_observation)
            self.cache_obs.inc(model=model, observation=obs, reason=cache_reason)
        if retry_count:
            self.retries.inc(retry_count, kind="model")
        return cost_usd

    def record_tool_call(self, tool: str, *, is_error: bool, duration_ms: float | None = None) -> None:
        self.tool_calls.inc(tool=tool, outcome="error" if is_error else "ok")
        if duration_ms is not None:
            self.tool_duration.observe(duration_ms, tool=tool)

    def record_permission(self, behavior: str, tool: str = "") -> None:
        self.permissions.inc(behavior=str(getattr(behavior, "value", behavior)), tool=tool)

    def record_stop(self, reason: str) -> None:
        self.stops.inc(reason=str(getattr(reason, "value", reason)) or "unknown")

    def record_compaction(self) -> None:
        self.compactions.inc()

    def record_retry(self, kind: str) -> None:
        self.retries.inc(kind=kind)

    def record_error(self, error_class: str) -> None:
        self.errors.inc(error_class=error_class)

    def record_task(self, outcome: str, *, cost_usd: float = 0.0, turns: int | None = None, duration_ms: float | None = None) -> None:
        """``outcome`` is ``success`` or an Agent-SDK-style subtype (``error_max_turns``...)."""
        self.tasks.inc(outcome=outcome)
        if cost_usd:
            self.task_cost.inc(cost_usd, outcome=outcome)
        if turns is not None:
            self.task_turns.observe(turns, outcome=outcome)
        if duration_ms is not None:
            self.task_duration.observe(duration_ms, outcome=outcome)

    # ------------------------------------------------------------------ EventExporter

    def export_event(self, event: Event) -> None:
        """Update metrics from a (sanitised) session event. See ``docs/ops/tracing-and-events.md`` for payload keys."""
        d = event.data or {}
        t = event.type
        if t is EventType.MODEL_RESPONSE:
            u = d.get("usage") or {}
            usage = Usage(
                int(u.get("input_tokens", 0) or 0),
                int(u.get("output_tokens", 0) or 0),
                int(u.get("cache_read_input_tokens", 0) or 0),
                int(u.get("cache_creation_input_tokens", 0) or 0),
            )
            self.record_model_call(
                str(d.get("model") or "unknown"),
                usage,
                latency_ms=d.get("latency_ms"),
                ttft_ms=d.get("ttft_ms"),
                stop_reason=d.get("stop_reason"),
                cost_usd=d.get("cost_usd"),
                cache_observation=d.get("cache_observation"),
                cache_reason=str(d.get("cache_reason") or ""),
                retry_count=int(d.get("retry_count", 0) or 0),
            )
        elif t is EventType.TOOL_RESULT:
            self.record_tool_call(str(d.get("tool") or d.get("name") or "unknown"), is_error=bool(d.get("is_error")), duration_ms=d.get("duration_ms"))
        elif t is EventType.PERMISSION:
            self.record_permission(str(d.get("behavior") or "unknown"), str(d.get("tool") or ""))
        elif t is EventType.STOP:
            self.record_stop(str(d.get("reason") or d.get("stop_reason") or "unknown"))
        elif t is EventType.COMPACTION:
            self.record_compaction()
        elif t is EventType.ERROR:
            self.record_error(str(d.get("error_class") or "unknown"))
        elif t is EventType.SESSION_END:
            self.record_task(
                str(d.get("outcome") or d.get("termination") or "unknown"),
                cost_usd=float(d.get("total_cost_usd") or 0.0),
                turns=d.get("num_turns"),
                duration_ms=d.get("duration_ms"),
            )

    # ------------------------------------------------------------------ derived dashboard values

    def cache_hit_rate(self, model: str | None = None) -> float | None:
        """cache_read / (input + cache_read + cache_creation)."""
        m = {"model": model} if model else {}
        read = self.tokens.sum(type="cache_read", **m)
        total = read + self.tokens.sum(type="cache_creation", **m) + self.tokens.sum(type="input", **m)
        return _safe_div(read, total)

    def unexpected_miss_rate(self) -> float | None:
        """Share of model calls classified ``unexpected_miss`` (alert on spikes: they are regressions)."""
        total = self.cache_obs.sum()
        return _safe_div(self.cache_obs.sum(observation=CacheObservation.UNEXPECTED_MISS.value), total)

    def tool_error_rate(self, tool: str | None = None) -> float | None:
        m = {"tool": tool} if tool else {}
        return _safe_div(self.tool_calls.sum(outcome="error", **m), self.tool_calls.sum(**m))

    def tool_error_rates(self) -> dict[str, float]:
        tools = {labels["tool"] for labels, _ in self.tool_calls.items()}
        return {t: r for t in sorted(tools) if (r := self.tool_error_rate(t)) is not None}

    def latency_quantile(self, q: float, model: str | None = None) -> float | None:
        return self.model_latency.quantile(q, **({"model": model} if model else {}))

    def cost_per_successful_task(self) -> float | None:
        """Total task spend (all outcomes) divided by successful tasks: failures are part of the price of success."""
        return _safe_div(self.task_cost.sum(), self.tasks.sum(outcome="success"))

    def permission_rates(self) -> dict[str, float | None]:
        total = self.permissions.sum()
        return {b: _safe_div(self.permissions.sum(behavior=b), total) for b in ("allow", "deny", "ask")}

    def dashboard(self) -> dict[str, Any]:
        """The default dashboard set (checklist item 4) as plain numbers; ``None`` = no data yet."""
        tasks = self.tasks.sum()
        return {
            "cache_hit_rate": self.cache_hit_rate(),
            "unexpected_cache_miss_rate": self.unexpected_miss_rate(),
            "cost_usd_total": self.cost.sum(),
            "cost_per_successful_task_usd": self.cost_per_successful_task(),
            "task_success_rate": _safe_div(self.tasks.sum(outcome="success"), tasks),
            "tool_error_rate": self.tool_error_rate(),
            "tool_error_rate_by_tool": self.tool_error_rates(),
            "turns_per_task_p50": self.task_turns.quantile(0.5),
            "model_latency_p50_ms": self.latency_quantile(0.5),
            "model_latency_p95_ms": self.latency_quantile(0.95),
            "ttft_p50_ms": self.ttft.quantile(0.5),
            "ttft_p95_ms": self.ttft.quantile(0.95),
            "task_duration_p50_ms": self.task_duration.quantile(0.5),
            "task_duration_p95_ms": self.task_duration.quantile(0.95),
            "compactions_per_task": _safe_div(self.compactions.sum(), tasks),
            "permission_rates": self.permission_rates(),
            "stop_reasons": {labels["reason"]: v for labels, v in self.stops.items()},
            "retries_total": self.retries.sum(),
        }

    def render_prometheus(self) -> str:
        return self.registry.render_prometheus()


def merge_counts(pairs: Iterable[tuple[Mapping[str, str], float]], label: str) -> dict[str, float]:
    """Sum counter items by one label (helper for custom dashboards)."""
    out: dict[str, float] = {}
    for labels, v in pairs:
        out[labels.get(label, "")] = out.get(labels.get(label, ""), 0.0) + v
    return out
