"""Audit trail: every decision and approval becomes an ``EventType.PERMISSION`` event.

Checklist §7.7 step 20: "For every decision, record the deciding layer, the rule/hook/classifier
stage, inputs (hashed where sensitive), and outcome. This is the data for tuning FPR/FNR."
Inputs are stored as a SHA-256 plus a short *redacted* preview (``harness_memory``'s
:class:`~harness_memory.security.SecretRedactor`), never verbatim.
"""

from __future__ import annotations

import threading
from typing import Any

from harness_core.types import Clock, Event, EventSink, EventType, utcnow
from harness_memory.security import SecretRedactor

from ._util import canonical_json, sha256_hex, truncate


class MemoryEventSink:
    """Thread-safe in-memory :class:`~harness_core.types.EventSink` (assigns ``seq``)."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self._lock = threading.Lock()

    def emit(self, event: Event) -> Event:
        with self._lock:
            event.seq = len(self.events)
            self.events.append(event)
        return event

    def of_phase(self, phase: str) -> list[Event]:
        with self._lock:
            return [e for e in self.events if e.data.get("phase") == phase]


class AuditTrail:
    """Builds permission events and forwards them to an optional sink. Never raises into the caller."""

    def __init__(self, sink: EventSink | None = None, *, clock: Clock = utcnow, redactor: SecretRedactor | None = None, preview_chars: int = 240) -> None:
        self.sink = sink
        self.clock = clock
        self.redactor = redactor or SecretRedactor()
        self.preview_chars = preview_chars

    def input_digest(self, tool_input: dict[str, Any]) -> dict[str, str]:
        text = canonical_json(tool_input)
        return {"input_sha256": sha256_hex(text), "input_preview": truncate(self.redactor.redact(text), self.preview_chars)}

    def emit(self, session_id: str, phase: str, **data: Any) -> Event | None:
        if self.sink is None:
            return None
        ts = self.clock().isoformat().replace("+00:00", "Z")
        event = Event(session_id=session_id, type=EventType.PERMISSION, data={"phase": phase, **data}, ts=ts)
        try:
            return self.sink.emit(event)
        except Exception:  # noqa: BLE001 - auditing must never break the decision path
            return None
