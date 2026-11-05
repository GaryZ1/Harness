"""Markdown + YAML frontmatter documents and the shared memory metadata schema.

Every learned-memory file is Markdown with this frontmatter (an extension of Claude Code's
``type`` + ``modified`` fields; see ``reports/SOTA agentic harness reference.md`` section 4.7)::

    ---
    type: feedback            # user | feedback | project | reference | lesson | workflow | skill
    scope: project            # org | user | project | agent | local
    status: active            # active | superseded | stale
    description: One line used in the MEMORY.md index
    supersedes: null
    source: {sessions: [s_8f2c], evidence: ["rollout_summaries/s_8f2c_fix-ci.md:12-30"], epistemic: verified}
    created: 2026-10-30T12:00:00Z
    modified: 2026-11-02T09:14:00Z
    last_used: 2026-11-03T08:00:00Z
    use_count: 4
    tainted: false
    ---
    CI runs `make test-fast` first; full suite only on main.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import yaml

from ._time import parse_iso, to_iso
from .errors import ValidationError

_FENCE = "---"


class MemoryType(str, Enum):
    USER = "user"  # role, expertise, preferences (semantic)
    FEEDBACK = "feedback"  # corrections and confirmed approaches (semantic)
    PROJECT = "project"  # decisions, deadlines not derivable from code/git (semantic)
    REFERENCE = "reference"  # where to find external information (semantic)
    LESSON = "lesson"  # failure -> fix, environment gotchas (procedural/experiential)
    WORKFLOW = "workflow"  # reusable routines with placeholders (procedural)
    SKILL = "skill"  # SKILL.md procedural memory


#: Types that must carry provenance (sessions or evidence) because they are experience-derived.
PROVENANCE_REQUIRED = frozenset({MemoryType.LESSON, MemoryType.WORKFLOW})


class Scope(str, Enum):
    ORG = "org"
    USER = "user"
    PROJECT = "project"
    AGENT = "agent"
    LOCAL = "local"


class Status(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    STALE = "stale"  # also used for candidate (unpromoted) skills


class Epistemic(str, Enum):
    VERIFIED = "verified"
    CLAIMED = "claimed"


@dataclass
class Provenance:
    sessions: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    epistemic: Epistemic = Epistemic.CLAIMED

    def to_dict(self) -> dict[str, Any]:
        return {"sessions": list(self.sessions), "evidence": list(self.evidence), "epistemic": self.epistemic.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Provenance":
        data = data or {}
        if not isinstance(data, dict):
            raise ValidationError("source must be a mapping")
        return cls(
            sessions=[str(s) for s in data.get("sessions") or []],
            evidence=[str(e) for e in data.get("evidence") or []],
            epistemic=Epistemic(data.get("epistemic") or Epistemic.CLAIMED.value),
        )

    @property
    def is_empty(self) -> bool:
        return not self.sessions and not self.evidence


@dataclass
class MemoryMeta:
    """Typed view of a memory file's frontmatter. Unknown keys are preserved in ``extra``."""

    type: MemoryType
    scope: Scope = Scope.PROJECT
    status: Status = Status.ACTIVE
    description: str = ""
    supersedes: str | None = None
    superseded_by: str | None = None
    source: Provenance = field(default_factory=Provenance)
    created: datetime | None = None
    modified: datetime | None = None
    last_used: datetime | None = None
    valid_to: datetime | None = None
    use_count: int = 0
    tainted: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    _KNOWN = (
        "type",
        "scope",
        "status",
        "description",
        "supersedes",
        "superseded_by",
        "source",
        "created",
        "modified",
        "last_used",
        "valid_to",
        "use_count",
        "tainted",
    )

    def validate(self) -> None:
        if self.type in PROVENANCE_REQUIRED and self.source.is_empty:
            raise ValidationError(f"memories of type '{self.type.value}' require provenance (source.sessions or source.evidence)")
        if "\n" in self.description:
            raise ValidationError("description must be a single line")
        if self.use_count < 0:
            raise ValidationError("use_count must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "type": self.type.value,
            "scope": self.scope.value,
            "status": self.status.value,
        }
        if self.description:
            out["description"] = self.description
        out["supersedes"] = self.supersedes
        if self.superseded_by:
            out["superseded_by"] = self.superseded_by
        out["source"] = self.source.to_dict()
        out["created"] = to_iso(self.created)
        out["modified"] = to_iso(self.modified)
        out["last_used"] = to_iso(self.last_used)
        if self.valid_to:
            out["valid_to"] = to_iso(self.valid_to)
        out["use_count"] = int(self.use_count)
        out["tainted"] = bool(self.tainted)
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryMeta":
        if not isinstance(data, dict):
            raise ValidationError("frontmatter must be a mapping")
        try:
            meta = cls(
                type=MemoryType(data["type"]),
                scope=Scope(data.get("scope") or Scope.PROJECT.value),
                status=Status(data.get("status") or Status.ACTIVE.value),
                description=str(data.get("description") or ""),
                supersedes=data.get("supersedes") or None,
                superseded_by=data.get("superseded_by") or None,
                source=Provenance.from_dict(data.get("source")),
                created=parse_iso(data.get("created")),
                modified=parse_iso(data.get("modified")),
                last_used=parse_iso(data.get("last_used")),
                valid_to=parse_iso(data.get("valid_to")),
                use_count=int(data.get("use_count") or 0),
                tainted=bool(data.get("tainted", False)),
                extra={k: v for k, v in data.items() if k not in cls._KNOWN},
            )
        except KeyError as exc:
            raise ValidationError(f"frontmatter missing required key: {exc.args[0]}") from exc
        except ValueError as exc:
            raise ValidationError(f"invalid frontmatter value: {exc}") from exc
        return meta


def split_frontmatter(text: str) -> tuple[dict[str, Any] | None, str]:
    """Split ``text`` into ``(frontmatter_mapping_or_None, body)``."""
    if not text.startswith(_FENCE + "\n") and not text.startswith(_FENCE + "\r\n"):
        return None, text
    lines = text.splitlines(keepends=True)
    for i in range(1, len(lines)):
        if lines[i].rstrip("\r\n") == _FENCE:
            raw = "".join(lines[1:i])
            body = "".join(lines[i + 1 :])
            try:
                data = yaml.safe_load(raw) or {}
            except yaml.YAMLError as exc:
                raise ValidationError(f"malformed YAML frontmatter: {exc}") from exc
            if not isinstance(data, dict):
                raise ValidationError("frontmatter must be a YAML mapping")
            return _normalise(data), body.lstrip("\n")
    return None, text


def _normalise(value: Any) -> Any:
    """Convert YAML-native datetimes back to ISO strings so round-trips are stable."""
    if isinstance(value, datetime):
        return to_iso(value)
    if isinstance(value, dict):
        return {k: _normalise(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalise(v) for v in value]
    return value


def join_frontmatter(data: dict[str, Any] | None, body: str) -> str:
    if data is None:
        return body
    dumped = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False, width=1000)
    body = body if body.endswith("\n") or body == "" else body + "\n"
    return f"{_FENCE}\n{dumped}{_FENCE}\n{body}"


@dataclass
class MemoryDocument:
    """A typed memory file: metadata + Markdown body."""

    meta: MemoryMeta
    body: str
    path: str = ""

    def to_markdown(self) -> str:
        self.meta.validate()
        return join_frontmatter(self.meta.to_dict(), self.body)

    @classmethod
    def from_markdown(cls, text: str, path: str = "") -> "MemoryDocument":
        data, body = split_frontmatter(text)
        if data is None:
            raise ValidationError(f"{path or 'document'} has no frontmatter")
        return cls(meta=MemoryMeta.from_dict(data), body=body, path=path)

    @property
    def title(self) -> str:
        for line in self.body.splitlines():
            if line.startswith("#"):
                return line.lstrip("#").strip()
        return self.meta.description or self.path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
