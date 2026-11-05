"""Violation records and the model-facing ``<sandbox_violations>`` block.

Research basis (§6.3 technique 6, checklist §6.7 item 5): when the sandbox blocks something, tell the
model *why*. srt attributes violations by ``commandId`` and surfaces reasons in a
``<sandbox_violations>`` block; the open-source notes call this loop "worth copying directly".
Each record carries the command ID, the operation, the rule that blocked it and a hint pointing to an
in-boundary alternative or a scoped exception. Records never include file contents or environment
values; paths outside the workspace are reported as given by the rule, not expanded further.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable
from xml.sax.saxutils import escape

HINTS = {
    "network": "Network egress is limited to the allowlisted domains. Use an allowlisted host, work offline, or ask the user for a scoped exception.",
    "filesystem": "Writes are allowed only inside the workspace and session tmp; credential paths are unreadable. Choose a path inside the workspace or ask for a scoped exception.",
    "timeout": "The command exceeded its wall-clock limit. Run a narrower command, add a --fast/sample mode, or run it in the background with output to a log file.",
    "resource": "The command hit a resource limit (CPU, memory, file size or process count). Reduce the workload or ask for higher limits.",
    "policy": "The request is outside the sandbox policy. Pick an in-boundary alternative or ask for a scoped exception.",
}


@dataclass(frozen=True)
class Violation:
    kind: str            # network | filesystem | timeout | resource | policy
    operation: str       # e.g. "CONNECT evil.example:443", "write /etc/passwd"
    rule: str            # e.g. "private-ip:10.0.0.0/8", "deny_read:~/.ssh"
    command_id: str = ""
    hint: str = ""

    def __post_init__(self) -> None:
        if not self.hint:
            object.__setattr__(self, "hint", HINTS.get(self.kind, HINTS["policy"]))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def format_violations(violations: Iterable[Violation]) -> str:
    """Render violations as the stable XML-ish block srt surfaces to the model ('' if none)."""
    items = list(violations)
    if not items:
        return ""
    lines = ["<sandbox_violations>"]
    for v in items:
        lines.append(
            f'  <violation kind="{escape(v.kind)}" command_id="{escape(v.command_id)}" rule="{escape(v.rule, {chr(34): "&quot;"})}">'
            f"{escape(v.operation)} -- {escape(v.hint)}</violation>"
        )
    lines.append("</sandbox_violations>")
    return "\n".join(lines)


_FS_PATTERNS = (
    re.compile(r"(?P<path>/[^\s:'\"]+)['\"]?: Read-only file system"),
    re.compile(r"Read-only file system: ['\"](?P<path>[^'\"]+)['\"]"),
    re.compile(r"(?P<path>/[^\s:'\"]+)['\"]?: Permission denied"),
    re.compile(r"Permission denied: ['\"](?P<path>[^'\"]+)['\"]"),
    re.compile(r"Operation not permitted: ['\"](?P<path>[^'\"]+)['\"]"),
)


def detect_fs_violations(stderr: str, command_id: str, *, limit: int = 5) -> list[Violation]:
    """Best-effort: map EROFS/EACCES messages in stderr to filesystem violations."""
    out: list[Violation] = []
    seen: set[str] = set()
    for pat in _FS_PATTERNS:
        for m in pat.finditer(stderr):
            p = m.group("path")
            if p in seen:
                continue
            seen.add(p)
            out.append(Violation("filesystem", f"access {p}", "sandbox-filesystem", command_id))
            if len(out) >= limit:
                return out
    return out
