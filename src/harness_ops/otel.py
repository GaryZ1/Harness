"""Optional OpenTelemetry bridge (lazy imports; nothing here is needed for the stdlib path).

Research basis (reference §10.1, checklist item 5): export to an OTel-native backend - self-hosted
Langfuse (MIT) or Phoenix (ELv2) by default, Braintrust/LangSmith via OTLP - while keeping exporters
pluggable. Two bridges are offered:

* :class:`OTelSpanExporter` - plug into :class:`~harness_ops.tracing.OpsTracer` next to the JSONL
  exporter. Finished spans keep **our** trace/span ids and are converted to SDK ``ReadableSpan``
  objects for any OTel ``SpanExporter`` (OTLP, console...). Needs ``opentelemetry-sdk``.
* :class:`OTelTracer` - implements the shared ``Tracer`` protocol directly on an OTel API tracer,
  applying the same content policy and semconv mapping. Needs ``opentelemetry-api`` only.

``OTEL_EXPORTER_OTLP_ENDPOINT`` / ``OTEL_EXPORTER_OTLP_HEADERS`` are read by the OTLP exporter
itself at construction time (Claude Code: "Headers are read only at startup").
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, MutableMapping, Sequence

from .errors import OTelUnavailableError
from .redaction import ContentPolicy
from .semconv import SemConv
from .tracing import SpanData, SpanKind, StatusCode

INSTALL_HINT = (
    "pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp "
    "(the stdlib JSONL tracer in harness_ops works without them)"
)


def _import(module: str, feature: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise OTelUnavailableError(f"{feature} requires the optional dependency {module!r}: {INSTALL_HINT}") from exc


def otel_available() -> bool:
    try:
        importlib.import_module("opentelemetry.trace")
    except ImportError:
        return False
    return True


def _load_sdk() -> SimpleNamespace:
    trace = _import("opentelemetry.trace", "OTelSpanExporter")
    sdk_trace = _import("opentelemetry.sdk.trace", "OTelSpanExporter")
    resources = _import("opentelemetry.sdk.resources", "OTelSpanExporter")
    instr = _import("opentelemetry.sdk.util.instrumentation", "OTelSpanExporter")
    return SimpleNamespace(
        ReadableSpan=sdk_trace.ReadableSpan,
        Event=sdk_trace.Event,
        SpanContext=trace.SpanContext,
        TraceFlags=trace.TraceFlags,
        TraceState=trace.TraceState,
        Link=trace.Link,
        SpanKind=trace.SpanKind,
        Status=trace.Status,
        StatusCode=trace.StatusCode,
        Resource=resources.Resource,
        InstrumentationScope=instr.InstrumentationScope,
    )


def _otel_tracestate(mods: SimpleNamespace, raw: str) -> Any:
    pairs = []
    for member in (raw or "").split(","):
        if "=" in member:
            k, v = member.strip().split("=", 1)
            pairs.append((k, v))
    try:
        return mods.TraceState(pairs)
    except Exception:  # noqa: BLE001 - invalid upstream tracestate is dropped, not fatal
        return mods.TraceState()


class OTelSpanExporter:
    """Forward finished :class:`SpanData` to an OpenTelemetry SDK ``SpanExporter``.

    ``otel_modules`` is an injection seam for tests; leave it ``None`` in production.
    """

    def __init__(self, exporter: Any, *, scope_name: str = "harness_ops", scope_version: str = "", otel_modules: Any = None) -> None:
        self._mods = otel_modules or _load_sdk()
        self._exporter = exporter
        self._scope = self._mods.InstrumentationScope(scope_name, scope_version or None)

    def to_readable(self, s: SpanData) -> Any:
        m = self._mods
        flags = m.TraceFlags(1 if s.context.sampled else 0)
        state = _otel_tracestate(m, s.context.tracestate)
        ctx = m.SpanContext(int(s.trace_id, 16), int(s.span_id, 16), False, flags, state)
        parent = m.SpanContext(int(s.trace_id, 16), int(s.parent_span_id, 16), False, flags, state) if s.parent_span_id else None
        status = m.Status(
            {StatusCode.OK: m.StatusCode.OK, StatusCode.ERROR: m.StatusCode.ERROR}.get(s.status, m.StatusCode.UNSET),
            s.status_message or None,
        )
        return m.ReadableSpan(
            name=s.name,
            context=ctx,
            parent=parent,
            resource=m.Resource(dict(s.resource)),
            attributes=dict(s.attributes),
            events=[m.Event(e.name, dict(e.attributes), e.time_ns) for e in s.events],
            links=[m.Link(m.SpanContext(int(lk.context.trace_id, 16), int(lk.context.span_id, 16), False), dict(lk.attributes)) for lk in s.links],
            kind=getattr(m.SpanKind, s.kind.name),
            status=status,
            start_time=s.start_ns,
            end_time=s.end_ns,
            instrumentation_scope=self._scope,
        )

    def export(self, spans: Sequence[SpanData]) -> None:
        self._exporter.export([self.to_readable(s) for s in spans])

    def shutdown(self) -> None:
        self._exporter.shutdown()


def otlp_exporter(protocol: str = "http/protobuf", **kwargs: Any) -> OTelSpanExporter:
    """Build an :class:`OTelSpanExporter` around the OTLP span exporter (endpoint/headers from env or kwargs)."""
    if protocol == "grpc":
        mod = _import("opentelemetry.exporter.otlp.proto.grpc.trace_exporter", "OTLP gRPC export")
    elif protocol == "http/protobuf":
        mod = _import("opentelemetry.exporter.otlp.proto.http.trace_exporter", "OTLP HTTP export")
    else:
        raise ValueError(f"unsupported OTLP protocol {protocol!r} (use 'http/protobuf' or 'grpc')")
    return OTelSpanExporter(mod.OTLPSpanExporter(**kwargs))


class _OTelSpanAdapter:
    """Wraps an OTel span so callers get the same sanitising ``set_attribute(s)`` surface."""

    def __init__(self, span: Any, tracer: "OTelTracer") -> None:
        self._span = span
        self._tracer = tracer

    @property
    def raw(self) -> Any:
        return self._span

    def set_attribute(self, key: str, value: Any) -> "_OTelSpanAdapter":
        return self.set_attributes({key: value})

    def set_attributes(self, attributes: Mapping[str, Any]) -> "_OTelSpanAdapter":
        for k, v in self._tracer._clean(attributes).items():
            self._span.set_attribute(k, v)
        return self

    def add_event(self, name: str, attributes: Mapping[str, Any] | None = None) -> "_OTelSpanAdapter":
        self._span.add_event(name, self._tracer._clean(attributes or {}))
        return self

    def record_exception(self, exc: BaseException) -> "_OTelSpanAdapter":
        self.set_attribute("error.type", type(exc).__qualname__)
        return self


class OTelTracer:
    """``Tracer`` protocol implementation on top of an OpenTelemetry API tracer."""

    supports_span_kind = True  # accepts span_kind= / parent= keywords (see AgentInstrumentation)

    def __init__(
        self,
        otel_tracer: Any = None,
        *,
        tracer_name: str = "harness_ops",
        policy: ContentPolicy | None = None,
        semconv: SemConv | None = None,
    ) -> None:
        self._trace_mod = _import("opentelemetry.trace", "OTelTracer") if otel_tracer is None else None
        self._tracer = otel_tracer if otel_tracer is not None else self._trace_mod.get_tracer(tracer_name)
        self.policy = policy or ContentPolicy()
        self.semconv = semconv or SemConv()

    def _clean(self, attributes: Mapping[str, Any]) -> dict[str, Any]:
        return self.semconv.apply(self.policy.sanitize_attributes(attributes))

    def _kind(self, kind: SpanKind) -> Any:
        mod = self._trace_mod
        if mod is None:
            try:
                mod = importlib.import_module("opentelemetry.trace")
            except ImportError:
                return None
        return getattr(mod.SpanKind, SpanKind(kind).name)

    @contextmanager
    def span(self, name: str, *, span_kind: SpanKind = SpanKind.INTERNAL, parent: Any = None, links: Any = (), **attributes: Any) -> Iterator[Any]:
        kwargs: dict[str, Any] = {"attributes": self._clean(attributes)}
        kind = self._kind(span_kind)
        if kind is not None:
            kwargs["kind"] = kind
        if isinstance(parent, Mapping):
            prop = _import("opentelemetry.trace.propagation.tracecontext", "traceparent extraction")
            kwargs["context"] = prop.TraceContextTextMapPropagator().extract(dict(parent))
        with self._tracer.start_as_current_span(name, **kwargs) as s:
            yield _OTelSpanAdapter(s, self)

    @staticmethod
    def inject(carrier: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        prop = _import("opentelemetry.trace.propagation.tracecontext", "traceparent injection")
        prop.TraceContextTextMapPropagator().inject(carrier)
        return carrier
