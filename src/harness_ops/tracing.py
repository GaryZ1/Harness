"""Stdlib tracer implementing the shared :class:`harness_core.types.Tracer` protocol.

Research basis (reference §10.3, checklist items 1-2 and 5):

* Spans carry W3C trace-context ids (32-hex trace id, 16-hex span id) and nest through a
  :mod:`contextvars` current-span slot, so nesting is correct across threads *and* asyncio tasks.
* ``traceparent`` / ``tracestate`` propagate across process boundaries; MCP 2026-07-28 (SEP-414)
  carries them in a request's ``_meta``, and child agents link back to the parent tool-call span.
* Every attribute passes the :class:`~harness_ops.redaction.ContentPolicy` (structure-only by
  default, secrets always redacted) and the :class:`~harness_ops.semconv.SemConv` rename layer.
* Exporters are pluggable (JSONL here, OTLP via :mod:`harness_ops.otel`); an exporter failure is
  counted and logged but never breaks the agent loop.

Head sampling is deterministic on the trace id (OTel ``TraceIdRatioBased`` semantics) and
parent-based, so a whole trace is either kept or dropped. Unsampled spans still have valid ids
for propagation but record nothing.
"""

from __future__ import annotations

import contextvars
import logging
import re
import secrets
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator, Mapping, MutableMapping, Protocol, Sequence

from .redaction import ContentPolicy
from .semconv import A, SemConv

log = logging.getLogger("harness_ops.tracing")

_TRACEPARENT = re.compile(r"^([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})(-.*)?$")
_ZERO_TRACE = "0" * 32
_ZERO_SPAN = "0" * 16
MAX_TRACESTATE_CHARS = 512  # W3C: vendors SHOULD propagate at least 512 chars
MAX_SPAN_EVENTS = 128  # OTel SDK default event count limit
MAX_SPAN_ATTRIBUTES = 128  # OTel SDK default attribute count limit


class SpanKind(str, Enum):
    INTERNAL = "internal"
    CLIENT = "client"
    SERVER = "server"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class StatusCode(str, Enum):
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass(frozen=True)
class SpanContext:
    trace_id: str
    span_id: str
    sampled: bool = True
    tracestate: str = ""
    remote: bool = False

    @property
    def is_valid(self) -> bool:
        return (
            len(self.trace_id) == 32 and self.trace_id != _ZERO_TRACE and len(self.span_id) == 16 and self.span_id != _ZERO_SPAN
        )

    @property
    def traceparent(self) -> str:
        return f"00-{self.trace_id}-{self.span_id}-{'01' if self.sampled else '00'}"


def format_traceparent(ctx: SpanContext) -> str:
    return ctx.traceparent


def parse_traceparent(header: str | None, tracestate: str | None = None) -> SpanContext | None:
    """Parse a W3C ``traceparent``. Returns ``None`` for anything invalid (never raises).

    Per the W3C spec: version ``ff`` is invalid; all-zero ids are invalid; a future version may
    append fields, which are ignored; version ``00`` must have exactly four fields.
    """
    if not header:
        return None
    m = _TRACEPARENT.match(header.strip().lower())
    if not m:
        return None
    version, trace_id, span_id, flags, rest = m.groups()
    if version == "ff" or (version == "00" and rest):
        return None
    if trace_id == _ZERO_TRACE or span_id == _ZERO_SPAN:
        return None
    state = (tracestate or "").strip()[:MAX_TRACESTATE_CHARS]
    return SpanContext(trace_id, span_id, sampled=bool(int(flags, 16) & 0x01), tracestate=state, remote=True)


def inject(carrier: MutableMapping[str, Any], ctx: SpanContext | None = None) -> MutableMapping[str, Any]:
    """Write ``traceparent`` (and ``tracestate``) for ``ctx`` (default: current span) into ``carrier``.

    For MCP, pass the request's ``_meta`` dict (SEP-414).
    """
    ctx = ctx or current_context()
    if ctx is not None and ctx.is_valid:
        carrier["traceparent"] = ctx.traceparent
        if ctx.tracestate:
            carrier["tracestate"] = ctx.tracestate
    return carrier


def extract(carrier: Mapping[str, Any] | None) -> SpanContext | None:
    if not carrier:
        return None
    lowered = {str(k).lower(): v for k, v in carrier.items()}
    tp = lowered.get("traceparent")
    ts = lowered.get("tracestate")
    return parse_traceparent(tp if isinstance(tp, str) else None, ts if isinstance(ts, str) else None)


# ----------------------------------------------------------------------------- span data


@dataclass(frozen=True)
class Link:
    context: SpanContext
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpanEvent:
    name: str
    time_ns: int
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpanData:
    """An ended span, as handed to exporters. Immutable."""

    name: str
    context: SpanContext
    parent_span_id: str | None
    kind: SpanKind
    start_ns: int
    end_ns: int
    attributes: Mapping[str, Any]
    events: tuple[SpanEvent, ...] = ()
    links: tuple[Link, ...] = ()
    status: StatusCode = StatusCode.UNSET
    status_message: str = ""
    resource: Mapping[str, Any] = field(default_factory=dict)
    dropped_attributes: int = 0

    @property
    def trace_id(self) -> str:
        return self.context.trace_id

    @property
    def span_id(self) -> str:
        return self.context.span_id

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1e6

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "kind": self.kind.value,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status.value,
            "status_message": self.status_message,
            "attributes": dict(self.attributes),
            "events": [{"name": e.name, "time_ns": e.time_ns, "attributes": dict(e.attributes)} for e in self.events],
            "links": [
                {"trace_id": lk.context.trace_id, "span_id": lk.context.span_id, "attributes": dict(lk.attributes)} for lk in self.links
            ],
            "resource": dict(self.resource),
            "sampled": self.context.sampled,
            "tracestate": self.context.tracestate,
            "dropped_attributes": self.dropped_attributes,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SpanData":
        ctx = SpanContext(d["trace_id"], d["span_id"], bool(d.get("sampled", True)), d.get("tracestate", ""))
        return cls(
            name=d["name"],
            context=ctx,
            parent_span_id=d.get("parent_span_id"),
            kind=SpanKind(d.get("kind", "internal")),
            start_ns=int(d["start_ns"]),
            end_ns=int(d["end_ns"]),
            attributes=dict(d.get("attributes", {})),
            events=tuple(SpanEvent(e["name"], int(e["time_ns"]), dict(e.get("attributes", {}))) for e in d.get("events", [])),
            links=tuple(Link(SpanContext(lk["trace_id"], lk["span_id"]), dict(lk.get("attributes", {}))) for lk in d.get("links", [])),
            status=StatusCode(d.get("status", "unset")),
            status_message=d.get("status_message", ""),
            resource=dict(d.get("resource", {})),
            dropped_attributes=int(d.get("dropped_attributes", 0)),
        )


class SpanExporter(Protocol):
    def export(self, spans: Sequence[SpanData]) -> None: ...

    def shutdown(self) -> None: ...


# ----------------------------------------------------------------------------- sampling


@dataclass(frozen=True)
class Sampler:
    """Parent-based, trace-id-ratio head sampler (OTel ``ParentBased(TraceIdRatioBased(ratio))``)."""

    ratio: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.ratio <= 1.0:
            raise ValueError("sampling ratio must be within [0, 1]")

    def should_sample(self, trace_id: str, parent: SpanContext | None) -> bool:
        if parent is not None:
            return parent.sampled
        if self.ratio >= 1.0:
            return True
        if self.ratio <= 0.0:
            return False
        bound = int(self.ratio * (1 << 64))
        return int(trace_id[16:], 16) < bound


class IdGenerator(Protocol):
    def trace_id(self) -> str: ...

    def span_id(self) -> str: ...


class RandomIdGenerator:
    def trace_id(self) -> str:
        while True:
            t = secrets.token_hex(16)
            if t != _ZERO_TRACE:
                return t

    def span_id(self) -> str:
        while True:
            s = secrets.token_hex(8)
            if s != _ZERO_SPAN:
                return s


# ----------------------------------------------------------------------------- live span

_CURRENT: contextvars.ContextVar["Span | None"] = contextvars.ContextVar("harness_ops_current_span", default=None)


def current_span() -> "Span | None":
    return _CURRENT.get()


def current_context() -> SpanContext | None:
    s = _CURRENT.get()
    return s.context if s is not None else None


class Span:
    """A live span. Thread-safe; attribute writes after :meth:`end` are ignored."""

    def __init__(
        self,
        tracer: "OpsTracer",
        name: str,
        context: SpanContext,
        parent_span_id: str | None,
        kind: SpanKind,
        start_ns: int,
        links: Sequence[Link] = (),
    ) -> None:
        self._tracer = tracer
        self.name = name
        self.context = context
        self.parent_span_id = parent_span_id
        self.kind = kind
        self.start_ns = start_ns
        self.end_ns: int | None = None
        self._attributes: dict[str, Any] = {}
        self._events: list[SpanEvent] = []
        self._links: list[Link] = list(links)
        self._status = StatusCode.UNSET
        self._status_message = ""
        self._dropped = 0
        self._lock = threading.Lock()

    # Tracer-protocol consumers may treat the yielded object as opaque; everything is optional.

    @property
    def is_recording(self) -> bool:
        return self.context.sampled and self.end_ns is None

    @property
    def trace_id(self) -> str:
        return self.context.trace_id

    @property
    def span_id(self) -> str:
        return self.context.span_id

    @property
    def traceparent(self) -> str:
        return self.context.traceparent

    @property
    def attributes(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._attributes)

    def set_attribute(self, key: str, value: Any) -> "Span":
        return self.set_attributes({key: value})

    def set_attributes(self, attributes: Mapping[str, Any]) -> "Span":
        if not self.is_recording:
            return self
        clean = self._tracer._clean(attributes)
        with self._lock:
            if self.end_ns is not None:
                return self
            for k, v in clean.items():
                if k not in self._attributes and len(self._attributes) >= MAX_SPAN_ATTRIBUTES:
                    self._dropped += 1
                    continue
                self._attributes[k] = v
        return self

    def add_event(self, name: str, attributes: Mapping[str, Any] | None = None) -> "Span":
        if not self.is_recording:
            return self
        clean = self._tracer._clean(attributes or {})
        with self._lock:
            if self.end_ns is None and len(self._events) < MAX_SPAN_EVENTS:
                self._events.append(SpanEvent(name, self._tracer._now_ns(), clean))
        return self

    def add_link(self, context: SpanContext, attributes: Mapping[str, Any] | None = None) -> "Span":
        if self.is_recording and context.is_valid:
            with self._lock:
                self._links.append(Link(context, self._tracer._clean(attributes or {})))
        return self

    def set_status(self, status: StatusCode, message: str = "") -> "Span":
        with self._lock:
            # OK is final; ERROR overrides UNSET (OTel status precedence).
            if self._status is StatusCode.OK and status is not StatusCode.OK:
                return self
            self._status = status
            self._status_message = self._tracer.policy.redact(message) if message else ""
        return self

    def record_exception(self, exc: BaseException) -> "Span":
        """Record the exception *class*; the message only when content capture is on (it may quote content)."""
        self.set_attribute(A.ERROR_TYPE, type(exc).__qualname__)
        event: dict[str, Any] = {"exception.type": type(exc).__qualname__}
        if self._tracer.policy.capture_content:
            event["exception.message"] = str(exc)
        self.add_event("exception", event)
        return self.set_status(StatusCode.ERROR, type(exc).__qualname__)

    def end(self, end_ns: int | None = None) -> None:
        with self._lock:
            if self.end_ns is not None:
                return
            self.end_ns = end_ns if end_ns is not None else self._tracer._now_ns()
        if self.context.sampled:
            self._tracer._export(self.to_data())

    def to_data(self) -> SpanData:
        with self._lock:
            return SpanData(
                name=self.name,
                context=self.context,
                parent_span_id=self.parent_span_id,
                kind=self.kind,
                start_ns=self.start_ns,
                end_ns=self.end_ns if self.end_ns is not None else self._tracer._now_ns(),
                attributes=dict(self._attributes),
                events=tuple(self._events),
                links=tuple(self._links),
                status=self._status,
                status_message=self._status_message,
                resource=self._tracer.resource,
                dropped_attributes=self._dropped,
            )


# ----------------------------------------------------------------------------- tracer


class OpsTracer:
    """The built-in tracer. ``span(name, **attributes)`` satisfies :class:`harness_core.types.Tracer`.

    Reserved keyword arguments to :meth:`span` (never recorded as attributes): ``span_kind``,
    ``parent`` (a :class:`SpanContext`, :class:`Span`, carrier dict, or ``False`` for a new root)
    and ``links``.
    """

    supports_span_kind = True  # accepts span_kind= / parent= keywords (see AgentInstrumentation)

    def __init__(
        self,
        exporters: Sequence[SpanExporter] = (),
        *,
        policy: ContentPolicy | None = None,
        semconv: SemConv | None = None,
        sampler: Sampler | None = None,
        resource: Mapping[str, Any] | None = None,
        id_generator: IdGenerator | None = None,
        time_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self._exporters: list[SpanExporter] = list(exporters)
        self.policy = policy or ContentPolicy()
        self.semconv = semconv or SemConv()
        self.sampler = sampler or Sampler()
        self.resource: Mapping[str, Any] = dict(resource or {"service.name": "harness"})
        self.ids = id_generator or RandomIdGenerator()
        self._now_ns = time_ns
        self.export_errors = 0
        self._lock = threading.Lock()

    def add_exporter(self, exporter: SpanExporter) -> None:
        with self._lock:
            self._exporters.append(exporter)

    def _clean(self, attributes: Mapping[str, Any]) -> dict[str, Any]:
        return self.semconv.apply(self.policy.sanitize_attributes(attributes))

    def _export(self, data: SpanData) -> None:
        with self._lock:
            exporters = list(self._exporters)
        for exp in exporters:
            try:
                exp.export([data])
            except Exception as exc:  # noqa: BLE001 - telemetry must never break the agent
                with self._lock:
                    self.export_errors += 1
                log.warning("span exporter %s failed: %s", type(exp).__name__, type(exc).__name__)

    def _resolve_parent(self, parent: Any) -> SpanContext | None:
        if parent is False:
            return None
        if parent is None:
            return current_context()
        if isinstance(parent, Span):
            return parent.context
        if isinstance(parent, SpanContext):
            return parent if parent.is_valid else None
        if isinstance(parent, Mapping):
            return extract(parent)
        raise TypeError(f"unsupported parent type: {type(parent).__name__}")

    def start_span(
        self,
        name: str,
        *,
        span_kind: SpanKind = SpanKind.INTERNAL,
        parent: Any = None,
        links: Sequence[Link] = (),
        attributes: Mapping[str, Any] | None = None,
    ) -> Span:
        """Start a span without making it current. Call :meth:`Span.end` yourself."""
        parent_ctx = self._resolve_parent(parent)
        trace_id = parent_ctx.trace_id if parent_ctx else self.ids.trace_id()
        sampled = self.sampler.should_sample(trace_id, parent_ctx)
        ctx = SpanContext(trace_id, self.ids.span_id(), sampled, parent_ctx.tracestate if parent_ctx else "")
        span = Span(self, name, ctx, parent_ctx.span_id if parent_ctx else None, SpanKind(span_kind), self._now_ns(), links)
        if attributes:
            span.set_attributes(attributes)
        return span

    @contextmanager
    def span(
        self,
        name: str,
        *,
        span_kind: SpanKind = SpanKind.INTERNAL,
        parent: Any = None,
        links: Sequence[Link] = (),
        **attributes: Any,
    ) -> Iterator[Span]:
        """Start a span, make it current for the ``with`` body, end it on exit.

        An exception escaping the body is recorded (``error.type`` + ERROR status) and re-raised.
        """
        s = self.start_span(name, span_kind=span_kind, parent=parent, links=links, attributes=attributes)
        token = _CURRENT.set(s)
        try:
            yield s
        except BaseException as exc:
            s.record_exception(exc)
            raise
        finally:
            _CURRENT.reset(token)
            s.end()

    def shutdown(self) -> None:
        with self._lock:
            exporters = list(self._exporters)
        for exp in exporters:
            try:
                exp.shutdown()
            except Exception:  # noqa: BLE001
                log.warning("span exporter %s failed to shut down", type(exp).__name__)


@contextmanager
def use_span(span: Span) -> Iterator[Span]:
    """Make an already-started span current (e.g. inside a worker thread) without ending it."""
    token = _CURRENT.set(span)
    try:
        yield span
    finally:
        _CURRENT.reset(token)


def set_span_attributes(span: Any, attributes: Mapping[str, Any]) -> None:
    """Duck-typed attribute setter: works for our :class:`Span`, an OTel span, or ``None`` (NullTracer)."""
    if span is None:
        return
    setter = getattr(span, "set_attributes", None)
    if callable(setter):
        setter({k: v for k, v in attributes.items() if v is not None})
        return
    one = getattr(span, "set_attribute", None)
    if callable(one):
        for k, v in attributes.items():
            if v is not None:
                one(k, v)
