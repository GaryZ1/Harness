"""Fan-out :class:`~harness_core.types.EventSink`: one ``emit`` reaches every telemetry exporter.

Research basis (reference §10.3 "Append-only session log", "Structure-only production tracing";
checklist items 2, 4 and 6):

* The durable session log (owned by the session package) is the source of truth; this sink is
  the *telemetry* tap next to it. ``emit`` returns the caller's event unchanged (only ``seq``/``ts``
  filled in when missing), so the log keeps full fidelity, while every exporter receives a
  **sanitised copy**: content keys replaced by digests unless capture is opted in, secrets
  redacted always.
* Exporters are isolated: one failing exporter is counted and logged (class names only, never
  payloads) and the others still receive the event - telemetry must never break the agent.

Built-in exporters: :class:`JsonlEventExporter`, :class:`InMemoryEventExporter`,
:class:`SpanEventExporter` (attaches events to the current span) and anything with an
``export_event(event)`` method - :class:`~harness_ops.metrics.HarnessMetrics` and
:class:`~harness_ops.sampling.ReviewSampler` both qualify.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Protocol, Sequence

from harness_core.types import Clock, Event, utcnow
from harness_memory.locking import file_lock

from ._util import to_iso
from .redaction import ContentPolicy
from .tracing import current_span

log = logging.getLogger("harness_ops.events")


class EventExporter(Protocol):
    def export_event(self, event: Event) -> None: ...


class FanoutEventSink:
    """Implements ``EventSink.emit``. Thread-safe; ``seq`` is per session and monotonic within this process."""

    def __init__(
        self,
        exporters: Sequence[EventExporter] = (),
        *,
        policy: ContentPolicy | None = None,
        clock: Clock = utcnow,
        assign_seq: bool = True,
    ) -> None:
        self._exporters: list[EventExporter] = list(exporters)
        self.policy = policy or ContentPolicy()
        self.clock = clock
        self.assign_seq = assign_seq
        self._seq: dict[str, int] = {}
        self._lock = threading.Lock()
        self.errors: dict[str, int] = {}

    def add_exporter(self, exporter: EventExporter) -> None:
        with self._lock:
            self._exporters.append(exporter)

    def emit(self, event: Event) -> Event:
        with self._lock:
            if self.assign_seq:
                if event.seq < 0:
                    event.seq = self._seq.get(event.session_id, -1) + 1
                self._seq[event.session_id] = max(self._seq.get(event.session_id, -1), event.seq)
            if not event.ts:
                event.ts = to_iso(self.clock())
            exporters = list(self._exporters)
        clean = replace(event, data=self.policy.sanitize_event_data(dict(event.data or {})))
        for exp in exporters:
            try:
                exp.export_event(clean)
            except Exception as exc:  # noqa: BLE001 - isolate exporters from each other and from the loop
                name = type(exp).__name__
                with self._lock:
                    self.errors[name] = self.errors.get(name, 0) + 1
                log.warning("event exporter %s failed: %s", name, type(exc).__name__)
        return event


class InMemoryEventExporter:
    def __init__(self) -> None:
        self._events: list[Event] = []
        self._lock = threading.Lock()

    def export_event(self, event: Event) -> None:
        with self._lock:
            self._events.append(event)

    @property
    def events(self) -> list[Event]:
        with self._lock:
            return list(self._events)


class JsonlEventExporter:
    """Append sanitised events as ``Event.to_json`` lines (cross-process safe via ``file_lock``)."""

    def __init__(self, path: str | Path, *, fsync: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fsync = fsync
        self._lock_path = self.path.with_name(f".{self.path.name}.lock")

    def export_event(self, event: Event) -> None:
        data = (event.to_json() + "\n").encode("utf-8")
        with file_lock(self._lock_path):
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view):]
                if self.fsync:
                    os.fsync(fd)
            finally:
                os.close(fd)


def read_events(path: str | Path) -> list[Event]:
    """Read a JSONL event file; a torn final line (crash mid-write) is skipped."""
    p = Path(path)
    out: list[Event] = []
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(Event.from_json(line))
        except (ValueError, KeyError, TypeError):
            continue
    return out


class SpanEventExporter:
    """Attach each event to the *current* span as a span event (type + seq + scalar fields only)."""

    def __init__(self, scalar_keys: Iterable[str] = ("tool", "name", "behavior", "reason", "stop_reason", "is_error", "error_class", "model")) -> None:
        self.scalar_keys = tuple(scalar_keys)

    def export_event(self, event: Event) -> None:
        span = current_span()
        if span is None:
            return
        attrs = {"harness.event.type": event.type.value, "harness.event.seq": event.seq, "harness.session_id": event.session_id}
        for k in self.scalar_keys:
            v = (event.data or {}).get(k)
            if isinstance(v, (str, int, float, bool)):
                attrs[f"harness.event.{k}"] = v
        span.add_event(f"harness.{event.type.value}", attrs)
