"""Built-in span exporters: JSONL file (default, stdlib only) and in-memory (tests, sampling).

The JSONL exporter is the "works without OpenTelemetry" path: one span per line, appended under
an exclusive cross-process lock (:func:`harness_memory.locking.file_lock`) so several workers can
share one file without interleaving. Readers tolerate a torn final line (a crash mid-write),
mirroring how the session log treats its tail. Ship the file to an OTel backend later with
:class:`harness_ops.otel.OTelSpanExporter` or any log shipper.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Iterable, Sequence

from harness_memory.locking import file_lock

from .tracing import SpanData


class InMemorySpanExporter:
    """Keeps finished spans in a list. Thread-safe."""

    def __init__(self) -> None:
        self._spans: list[SpanData] = []
        self._lock = threading.Lock()

    def export(self, spans: Sequence[SpanData]) -> None:
        with self._lock:
            self._spans.extend(spans)

    def shutdown(self) -> None:
        pass

    @property
    def spans(self) -> list[SpanData]:
        with self._lock:
            return list(self._spans)

    def by_name(self, name: str) -> list[SpanData]:
        return [s for s in self.spans if s.name == name]

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()


class JsonlSpanExporter:
    """Append spans as JSON lines to ``path``.

    ``max_bytes`` (optional) rotates the file to ``<name>.1`` (``.2``... up to ``backups``) before a
    write would exceed it. ``fsync=True`` makes every batch durable at the cost of latency.
    """

    def __init__(self, path: str | Path, *, fsync: bool = False, max_bytes: int | None = None, backups: int = 3) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fsync = fsync
        self.max_bytes = max_bytes
        self.backups = max(1, backups)
        self._lock_path = self.path.with_name(f".{self.path.name}.lock")

    def export(self, spans: Sequence[SpanData]) -> None:
        if not spans:
            return
        payload = "".join(json.dumps(s.to_dict(), ensure_ascii=False, default=str, separators=(",", ":")) + "\n" for s in spans)
        data = payload.encode("utf-8")
        with file_lock(self._lock_path):
            if self.max_bytes is not None and self.path.exists() and self.path.stat().st_size + len(data) > self.max_bytes:
                self._rotate()
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view):]
                if self.fsync:
                    os.fsync(fd)
            finally:
                os.close(fd)

    def _rotate(self) -> None:
        for i in range(self.backups, 0, -1):
            src = self.path if i == 1 else self.path.with_name(f"{self.path.name}.{i - 1}")
            if src.exists():
                os.replace(src, self.path.with_name(f"{self.path.name}.{i}"))

    def shutdown(self) -> None:
        pass


def read_spans(path: str | Path) -> list[SpanData]:
    """Read spans written by :class:`JsonlSpanExporter`; a torn or corrupt line is skipped."""
    return list(iter_spans(path))


def iter_spans(path: str | Path) -> Iterable[SpanData]:
    p = Path(path)
    if not p.exists():
        return
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield SpanData.from_dict(json.loads(line))
            except (ValueError, KeyError, TypeError):
                continue


def span_tree(spans: Iterable[SpanData]) -> dict[str | None, list[SpanData]]:
    """Group spans by ``parent_span_id`` (``None`` = roots), each list ordered by start time."""
    tree: dict[str | None, list[SpanData]] = {}
    for s in spans:
        tree.setdefault(s.parent_span_id, []).append(s)
    for children in tree.values():
        children.sort(key=lambda s: s.start_ns)
    return tree
