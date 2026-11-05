"""MCP supply-chain hygiene: hash approved tool descriptions and detect rug pulls.

Tool poisoning hides instructions in descriptions; a *rug pull* changes the description after
the user approved it (Invariant Labs, research §7.3 technique 23). The checklist (§7.7 step 11)
asks to "record a hash of each approved tool description; re-hash on every ``tools/list`` and
block on change" and to "key sticky approvals by server, tool *and* description hash".

:class:`ToolPinStore` keeps ``{tool name: sha256(description, input_schema, annotations, server
version)}``. The first sighting is pinned (trust on first use, like SSH host keys); a later
mismatch is reported as :attr:`PinStatus.CHANGED` and the policy engine asks a human (or blocks)
before the tool runs again. Pins can be persisted atomically to a JSON file so they survive
restarts.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from harness_core.types import ToolSpec
from harness_memory.locking import atomic_write_text, file_lock

from ._util import sha256_hex


class PinStatus(str, Enum):
    NEW = "new"
    MATCH = "match"
    CHANGED = "changed"


def description_hash(spec: ToolSpec) -> str:
    """Hash of everything the model reads about a tool (plus the pinned server version)."""
    return sha256_hex({
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.input_schema,
        "annotations": spec.metadata.get("annotations") or {},
        "server_version": spec.metadata.get("mcp_server_version"),
    })


@dataclass(frozen=True)
class PinCheck:
    status: PinStatus
    current: str
    pinned: str | None


class ToolPinStore:
    """Thread-safe (and, with ``path``, process-safe) store of approved description hashes."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._pins: dict[str, str] = {}
        self._lock = threading.RLock()
        if self.path and self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            if isinstance(data, dict):
                self._pins = {str(k): str(v) for k, v in data.get("pins", {}).items()}

    def get(self, tool: str) -> str | None:
        with self._lock:
            return self._pins.get(tool)

    def pin(self, spec: ToolSpec) -> str:
        h = description_hash(spec)
        with self._lock:
            self._pins[spec.name] = h
            self._persist()
        return h

    def check(self, spec: ToolSpec, *, pin_new: bool = True) -> PinCheck:
        current = description_hash(spec)
        with self._lock:
            pinned = self._pins.get(spec.name)
            if pinned is None:
                if pin_new:
                    self._pins[spec.name] = current
                    self._persist()
                return PinCheck(PinStatus.NEW, current, None)
            return PinCheck(PinStatus.MATCH if pinned == current else PinStatus.CHANGED, current, pinned)

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            return dict(self._pins)

    def _persist(self) -> None:
        if not self.path:
            return
        payload: dict[str, Any] = {"version": 1, "pins": dict(sorted(self._pins.items()))}
        with file_lock(self.path.with_name(self.path.name + ".lock")):
            atomic_write_text(self.path, json.dumps(payload, indent=2, sort_keys=True))
