"""Permission modes, their permissiveness order, and the bypass guard.

Mode vocabulary follows the research checklist (§7.7 step 3): ship ``plan``, ``accept-edits``,
``auto`` (default for interactive), ``dont-ask`` (default for headless) and ``bypass``, plus
``default`` (Claude Code's manual mode: ask for anything not pre-approved). Claude Code / Agent
SDK spellings (``acceptEdits``, ``dontAsk``, ``bypassPermissions``) are accepted as aliases.

``bypass`` is only reachable when (a) admin (managed-layer) policy allows it, (b) the caller has
*explicitly* attested an external sandbox and (c) the process is not root - Claude Code refuses
``bypassPermissions`` as root and both vendors restrict bypass/yolo to "externally sandboxed
environments".
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .errors import BypassRefusedError, ModeError


class PermissionMode(str, Enum):
    PLAN = "plan"                  # reads and read-only shell only; mutations denied
    DONT_ASK = "dont-ask"          # never prompts; anything not pre-approved is denied (headless)
    DEFAULT = "default"            # manual: ask for anything not pre-approved
    ACCEPT_EDITS = "accept-edits"  # + in-workspace edits and common fs commands auto-allowed
    AUTO = "auto"                  # + model auto-reviewer for boundary crossings
    BYPASS = "bypass"              # everything except hard denies; attested sandbox only

    @classmethod
    def parse(cls, value: "str | PermissionMode") -> "PermissionMode":
        if isinstance(value, PermissionMode):
            return value
        key = str(value).strip().replace("_", "-")
        alias = _ALIASES.get(key.lower()) or _ALIASES.get(key)
        if alias is None:
            raise ModeError(f"unknown permission mode {value!r}; expected one of {[m.value for m in cls]}")
        return alias

    @property
    def rank(self) -> int:
        """Permissiveness rank: higher = more actions auto-approved."""
        return _ORDER.index(self)

    @property
    def mutations_allowed(self) -> bool:
        return self is not PermissionMode.PLAN


_ORDER = [
    PermissionMode.PLAN,
    PermissionMode.DONT_ASK,
    PermissionMode.DEFAULT,
    PermissionMode.ACCEPT_EDITS,
    PermissionMode.AUTO,
    PermissionMode.BYPASS,
]

_ALIASES: dict[str, PermissionMode] = {
    "plan": PermissionMode.PLAN,
    "read-only": PermissionMode.PLAN,
    "dont-ask": PermissionMode.DONT_ASK,
    "dontask": PermissionMode.DONT_ASK,
    "non-interactive": PermissionMode.DONT_ASK,
    "default": PermissionMode.DEFAULT,
    "manual": PermissionMode.DEFAULT,
    "accept-edits": PermissionMode.ACCEPT_EDITS,
    "acceptedits": PermissionMode.ACCEPT_EDITS,
    "auto": PermissionMode.AUTO,
    "bypass": PermissionMode.BYPASS,
    "bypasspermissions": PermissionMode.BYPASS,
    "yolo": PermissionMode.BYPASS,
}


def least_permissive(*modes: PermissionMode) -> PermissionMode:
    """The most restrictive of ``modes`` (used to clamp sub-agent modes to their parent's)."""
    return min(modes, key=lambda m: m.rank)


@dataclass(frozen=True)
class SandboxAttestation:
    """What the host asserts about the execution environment.

    ``externally_sandboxed`` must be set *explicitly* by the host (or via the
    ``HARNESS_EXTERNAL_SANDBOX=1`` environment variable read by :meth:`detect`). Container
    markers such as ``/.dockerenv`` are recorded as ``evidence`` but are never sufficient on their
    own: a dev container with the user's credentials mounted is not a disposable sandbox.
    """

    externally_sandboxed: bool = False
    kind: str = "none"                       # e.g. "container", "vm", "firecracker"
    running_as_root: bool = False
    evidence: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def detect(cls) -> "SandboxAttestation":
        evidence: list[str] = []
        for marker in ("/.dockerenv", "/run/.containerenv"):
            if Path(marker).exists():
                evidence.append(marker)
        explicit = os.environ.get("HARNESS_EXTERNAL_SANDBOX", "").strip().lower() in {"1", "true", "yes"}
        if explicit:
            evidence.append("HARNESS_EXTERNAL_SANDBOX")
        geteuid = getattr(os, "geteuid", None)
        root = bool(geteuid and geteuid() == 0)
        return cls(
            externally_sandboxed=explicit,
            kind=os.environ.get("HARNESS_SANDBOX_KIND", "container" if evidence else "none"),
            running_as_root=root,
            evidence=tuple(evidence),
        )


def require_bypass_allowed(attestation: SandboxAttestation | None, *, admin_allows: bool) -> None:
    """Raise :class:`BypassRefusedError` unless every bypass precondition holds."""
    problems: list[str] = []
    if not admin_allows:
        problems.append("managed policy does not set defaults.allow_bypass = true")
    if attestation is None or not attestation.externally_sandboxed:
        problems.append("no explicit external-sandbox attestation (SandboxAttestation(externally_sandboxed=True))")
    if attestation is not None and attestation.running_as_root:
        problems.append("bypass cannot run as root")
    if problems:
        raise BypassRefusedError("bypass mode refused: " + "; ".join(problems))
