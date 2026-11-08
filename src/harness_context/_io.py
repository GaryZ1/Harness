"""Small, dependency-free helpers: canonical JSON, hashing and atomic file writes.

Canonical serialization is the first defence against silent cache invalidation: the Anthropic
prompt cache is a byte-exact prefix match, so any dict whose key order can vary between runs
(tool schemas, MCP ``tools/list`` payloads) must be rendered with sorted keys.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace, UTF-8 preserved."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sort_keys_deep(obj: Any) -> Any:
    """Return a copy of ``obj`` whose dicts have sorted insertion order (recursively).

    SDKs serialize dicts in insertion order, so sorting the insertion order makes the bytes on the
    wire deterministic without relying on the SDK's serializer settings.
    """
    if isinstance(obj, dict):
        return {k: sort_keys_deep(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [sort_keys_deep(v) for v in obj]
    return obj


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(obj: Any) -> str:
    """SHA-256 of the canonical JSON of ``obj``."""
    return sha256_text(canonical_json(obj))


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (temp file + fsync + ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:  # pragma: no cover - best effort cleanup
                pass
