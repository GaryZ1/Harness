"""Session taint: once untrusted content enters, exfiltration-capable actions need a human.

Research §7.3 technique 22 / checklist §7.7 step 10: "Taint the session on any open-world or
untrusted input (``openWorldHint`` defaults to ``true``). While tainted: network writes,
messaging, pushes outside the session branch, new egress domains and memory writes require the
classifier at strict settings or a human; new egress domains can be approved only by a human."
Codex's ``disable_on_external_context`` (``memory_mode = "polluted"``) is the same idea.

Invariants enforced here:

* taint is monotonic - nothing in the agent loop can clear it; :meth:`SessionTaint.clear`
  requires an explicit ``confirmed_by_user=True`` ("clearing taint requires a new session or
  explicit user action");
* taint never *lowers* a restriction - callers only ever use it to add checks.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from harness_core.types import ToolOutcome

from .boundary import ActionProfile, Crossing, ToolKind
from .errors import PermissionsError


class TaintSource(str, Enum):
    WEB_FETCH = "web_fetch"
    BROWSER = "browser"
    MCP_OPEN_WORLD = "mcp:openWorld"
    EMAIL = "email"
    OUT_OF_REPO_FILES = "out_of_repo_files"
    SUBAGENT_REPORT = "subagent_report"
    MEMORY_READ = "memory_read"
    # always-on sources (not configurable): the harness declared the output untrusted
    OPEN_WORLD_TOOL = "open_world_tool"
    UNTRUSTED_OUTPUT = "untrusted_output"
    INJECTION_SUSPECTED = "injection_suspected"
    PARENT = "parent_session"
    MANUAL = "manual"


DEFAULT_SOURCES: tuple[str, ...] = ("web_fetch", "browser", "mcp:openWorld", "email", "out_of_repo_files", "subagent_report")
ALWAYS_ON = frozenset({TaintSource.OPEN_WORLD_TOOL, TaintSource.UNTRUSTED_OUTPUT, TaintSource.INJECTION_SUSPECTED, TaintSource.PARENT, TaintSource.MANUAL})
#: Schema default plus our additions (``deploy``, ``irreversible``): outward/irreversible actions ask while tainted.
DEFAULT_STRICT_ACTIONS: tuple[str, ...] = (
    "network_write", "message_send", "git_push_non_session_branch", "new_egress_domain", "memory_write", "deploy", "irreversible",
)
_EMAIL_HINT = ("mail", "inbox", "imap", "smtp", "outlook")


@dataclass(frozen=True)
class TaintEvent:
    source: TaintSource
    tool: str
    detail: str = ""


def sources_for(profile: ActionProfile, outcome: ToolOutcome | None, enabled: Iterable[str]) -> list[TaintEvent]:
    """Which taint sources a finished tool call represents."""
    on = set(enabled)
    events: list[TaintEvent] = []
    name = profile.tool
    spec = profile.spec

    def add(src: TaintSource, detail: str = "") -> None:
        if src in ALWAYS_ON or src.value in on:
            events.append(TaintEvent(src, name, detail))

    if spec is not None and spec.open_world:
        add(TaintSource.OPEN_WORLD_TOOL, "ToolSpec.open_world")
    if outcome is not None and outcome.metadata.get("untrusted"):
        add(TaintSource.UNTRUSTED_OUTPUT, "outcome.metadata.untrusted")
    kind = profile.kind
    if kind in {ToolKind.WEB_FETCH, ToolKind.WEB_SEARCH}:
        add(TaintSource.WEB_FETCH)
    elif kind is ToolKind.BROWSER:
        add(TaintSource.BROWSER)
    elif kind is ToolKind.SUBAGENT:
        add(TaintSource.SUBAGENT_REPORT)
    elif kind is ToolKind.MCP and profile.mcp is not None and profile.mcp.open_world:
        add(TaintSource.MCP_OPEN_WORLD, f"server {profile.mcp.server}")
    elif kind is ToolKind.MEMORY:
        add(TaintSource.MEMORY_READ)
    elif kind is ToolKind.SHELL and profile.has(Crossing.NETWORK_EGRESS, Crossing.NEW_EGRESS_DOMAIN, Crossing.NETWORK_WRITE):
        add(TaintSource.WEB_FETCH, "shell network output")
    if any(h in name.lower() for h in _EMAIL_HINT):
        add(TaintSource.EMAIL)
    if profile.has(Crossing.READ_OUTSIDE_WORKSPACE) and kind in {ToolKind.READ, ToolKind.SEARCH, ToolKind.SHELL}:
        add(TaintSource.OUT_OF_REPO_FILES)
    return events


class TaintClearRefused(PermissionsError):
    """Clearing taint was attempted without explicit user confirmation."""


@dataclass
class SessionTaint:
    """Thread-safe, monotonic taint flag with provenance."""

    events: list[TaintEvent] = field(default_factory=list)
    _tainted: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @property
    def tainted(self) -> bool:
        return self._tainted

    def mark(self, event: TaintEvent) -> None:
        with self._lock:
            self._tainted = True
            self.events.append(event)

    def reasons(self) -> list[str]:
        with self._lock:
            return [f"{e.source.value}:{e.tool}" + (f" ({e.detail})" if e.detail else "") for e in self.events]

    def clear(self, *, confirmed_by_user: bool) -> None:
        if confirmed_by_user is not True:
            raise TaintClearRefused("taint can only be cleared by an explicit user action (or a new session)")
        with self._lock:
            self._tainted = False
            self.events.clear()
