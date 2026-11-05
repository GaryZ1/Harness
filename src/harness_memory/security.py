"""Security primitives shared by every memory type.

* :class:`PathSandbox` - canonicalises paths and rejects traversal (``../``, ``..\\``, ``%2e%2e%2f``),
  absolute paths, NUL bytes and symlink escapes, following the Anthropic memory-tool guidance
  (``Path.resolve()`` + ``relative_to()``).
* :class:`SecretRedactor` - replaces credentials with ``[REDACTED_SECRET]`` before anything is
  persisted (Codex Phase-1 behaviour; Letta's "never store secrets" rule).
* :class:`TaintTracker` - session taint tracking. Any untrusted/open-world tool output marks the
  session ``polluted`` (Codex ``disable_on_external_context`` semantics).
"""

from __future__ import annotations

import re
import threading
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable, Pattern

from .errors import PathSecurityError

REDACTED = "[REDACTED_SECRET]"


class PathSandbox:
    """Resolve user/model-supplied relative paths strictly inside ``root``."""

    _TRAVERSAL = re.compile(r"(^|[\\/])\.\.([\\/]|$)")

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def normalise(self, rel: str) -> str:
        """Validate ``rel`` and return its canonical POSIX form relative to the root ('' = root)."""
        if not isinstance(rel, str):
            raise PathSecurityError("path must be a string")
        if "\x00" in rel:
            raise PathSecurityError("path contains a NUL byte")
        decoded = rel
        # Decode repeatedly so double-encoded traversal (%252e%252e) is also caught.
        for _ in range(3):
            nxt = urllib.parse.unquote(decoded)
            if nxt == decoded:
                break
            decoded = nxt
        if self._TRAVERSAL.search(decoded) or self._TRAVERSAL.search(rel):
            raise PathSecurityError(f"path traversal rejected: {rel!r}")
        candidate = decoded.replace("\\", "/")
        if candidate.startswith("/") or re.match(r"^[A-Za-z]:", candidate):
            raise PathSecurityError(f"absolute paths are not allowed: {rel!r}")
        parts = [p for p in PurePosixPath(candidate).parts if p not in ("", ".")]
        return "/".join(parts)

    def resolve(self, rel: str) -> Path:
        """Return the absolute path for ``rel``, guaranteeing it stays under the root (symlinks included)."""
        clean = self.normalise(rel)
        target = (self.root / clean).resolve() if clean else self.root
        try:
            target.relative_to(self.root)
        except ValueError as exc:  # symlink escape
            raise PathSecurityError(f"path escapes sandbox: {rel!r}") from exc
        return target

    def relative(self, absolute: Path) -> str:
        """Inverse of :meth:`resolve`."""
        return absolute.resolve().relative_to(self.root).as_posix()

    def is_root(self, rel: str) -> bool:
        return self.normalise(rel) == ""


_DEFAULT_SECRET_PATTERNS: tuple[str, ...] = (
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----",
    r"\bsk-ant-[A-Za-z0-9_\-]{16,}",
    r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}",
    r"\bgh[pousr]_[A-Za-z0-9]{30,}",
    r"\bgithub_pat_[A-Za-z0-9_]{40,}",
    r"\bxox[abprs]-[A-Za-z0-9\-]{10,}",
    r"\bAKIA[0-9A-Z]{16}\b",
    r"\bAIza[0-9A-Za-z_\-]{35}\b",
    r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}",
    r"(?i)\bbearer\s+([A-Za-z0-9_\-\.=]{20,})",
    r"(?i)(?:[a-z][a-z0-9+.\-]*://)[^\s:/@]+:([^\s@/]+)@",
)

# key=value style assignments where only the value is redacted.
_ASSIGNMENT = re.compile(
    r"(?i)\b([A-Z0-9_\-]*(?:password|passwd|secret|token|api[_\-]?key|access[_\-]?key|private[_\-]?key)[A-Z0-9_\-]*)"
    r"(\s*[:=]\s*)(['\"]?)(?![0-9]+\b)([^\s'\",;]{8,})(\3)"
)


class SecretRedactor:
    """Regex-based secret scrubber. Extend with :meth:`add_pattern` for organisation-specific tokens."""

    def __init__(self, extra_patterns: Iterable[str | Pattern[str]] = ()) -> None:
        self._patterns: list[Pattern[str]] = [re.compile(p) for p in _DEFAULT_SECRET_PATTERNS]
        for p in extra_patterns:
            self.add_pattern(p)

    def add_pattern(self, pattern: str | Pattern[str]) -> None:
        self._patterns.append(re.compile(pattern) if isinstance(pattern, str) else pattern)

    def redact(self, text: str) -> str:
        out = _ASSIGNMENT.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}{REDACTED}{m.group(5)}", text)
        for pat in self._patterns:
            if pat.groups:
                out = pat.sub(lambda m: m.group(0).replace(m.group(1), REDACTED), out)
            else:
                out = pat.sub(REDACTED, out)
        return out

    def contains_secret(self, text: str) -> bool:
        return self.redact(text) != text


@dataclass
class TaintTracker:
    """Track whether a session ingested untrusted content.

    ``trusted_tools`` are tools whose output stays inside the trust boundary (local file reads,
    the memory tool itself, test runners). Everything else - web fetch, email, third-party MCP
    servers - is open-world and taints the session.
    """

    trusted_tools: set[str] = field(default_factory=lambda: {"memory", "read", "edit", "write", "grep", "glob", "bash", "todo"})
    polluted: bool = False
    reasons: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record_tool_output(self, tool_name: str, *, untrusted: bool | None = None) -> bool:
        """Record one tool result; returns the session's polluted flag afterwards."""
        is_untrusted = untrusted if untrusted is not None else tool_name.lower() not in self.trusted_tools
        if is_untrusted:
            with self._lock:
                self.polluted = True
                self.reasons.append(tool_name)
        return self.polluted

    def mark_polluted(self, reason: str) -> None:
        with self._lock:
            self.polluted = True
            self.reasons.append(reason)

    @property
    def memory_mode(self) -> str:
        return "polluted" if self.polluted else "clean"
