"""Subagent definitions: Markdown + YAML frontmatter files under ``.harness/agents/<name>.md``.

Shape follows Claude Code's subagent files (reference section 8.5: ``name``, ``description``,
``tools``, ``disallowedTools``, ``model``, ``maxTurns``, ``memory``, ``effort``, ``isolation``...)
with two deliberate changes encoded as invariants:

* ``model`` names a **role** (``lead``, ``worker``, ``reviewer``...), never a model id. Ids come from
  the caller's role -> model mapping (:class:`~harness_multiagent.runner.ModelRouter`), so tiering
  (frontier lead, cheaper workers; checklist item 15) is configuration, not code.
* Least privilege by default (checklist items 3 and 11): ``write_access`` defaults to ``none``,
  which strips every tool whose :attr:`ToolSpec.read_only` is false; ``memory`` defaults to
  ``read_only``; nested delegation is off unless ``can_spawn: true``.

Example::

    ---
    name: code-explorer
    description: Read-only codebase search. Use for "where is X defined / how does Y work" questions.
    tools: [read_file, grep, glob]
    model: worker
    max_turns: 20
    max_tokens: 150000
    timeout_s: 300
    ---
    You are a fast, read-only code explorer. Cite file:line for every claim.

Scope priority mirrors Claude Code (managed > CLI > project ``.claude/agents`` > user > plugin):
:meth:`AgentRegistry.load_dirs` takes directories highest-priority first and the first definition of
a name wins.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import warnings
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Sequence

from harness_core.types import ToolSpec
from harness_memory.errors import ValidationError as _MemoryValidationError
from harness_memory.frontmatter import split_frontmatter

from .errors import DefinitionError, UnknownAgentError

_NAME = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,63}$")
#: Claude Code warns at startup when combined descriptions exceed ~15k tokens (reference 8.8).
CATALOG_TOKEN_WARNING = 15_000
MAX_DESCRIPTION_CHARS = 1_024
INHERIT_ROLE = "inherit"


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 characters per token)."""
    return (len(text) + 3) // 4


class WriteAccess(str, Enum):
    """What a subagent may mutate (delegation brief ``write_permission``, checklist item 3)."""

    NONE = "none"                  # readers and critics: no mutating tools at all
    OWN_FILES = "own_files"        # may write only files it claimed in the ownership map
    OWN_WORKTREE = "own_worktree"  # parallel writer isolated in its own git worktree
    WORKSPACE = "workspace"        # the single writer on the shared workspace (exclusive lease)

    @property
    def writes(self) -> bool:
        return self is not WriteAccess.NONE


class MemoryAccess(str, Enum):
    NONE = "none"
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"


class ContextMode(str, Enum):
    """``fresh`` (default): own prompt + brief only. ``fork``: inherits the parent's history."""

    FRESH = "fresh"
    FORK = "fork"


@dataclass(frozen=True)
class SubagentDefinition:
    """A named, routable subagent. Immutable; validated on construction."""

    name: str
    description: str
    system_prompt: str
    tools: tuple[str, ...] | None = None          # allowlist; None = parent's pool (still privilege-filtered)
    disallowed_tools: tuple[str, ...] = ()
    model_role: str = "worker"
    effort: str | None = None
    max_turns: int = 30
    max_tool_calls: int | None = None
    max_tokens: int | None = None                 # per-agent token budget; None = manager default
    max_output_tokens: int = 16_000               # per model call
    timeout_s: float | None = None                # wall clock; None = manager default
    write_access: WriteAccess = WriteAccess.NONE
    owned_files: tuple[str, ...] = ()
    memory: MemoryAccess = MemoryAccess.READ_ONLY
    context: ContextMode = ContextMode.FRESH
    can_spawn: bool = False
    source_path: str = ""
    extra: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        try:  # accept plain strings for enum fields in programmatic definitions
            object.__setattr__(self, "write_access", WriteAccess(self.write_access))
            object.__setattr__(self, "memory", MemoryAccess(self.memory))
            object.__setattr__(self, "context", ContextMode(self.context))
        except ValueError as exc:
            raise DefinitionError(f"agent {self.name!r}: {exc}") from exc
        for attr in ("tools", "disallowed_tools", "owned_files"):
            value = getattr(self, attr)
            if isinstance(value, list):
                object.__setattr__(self, attr, tuple(value))
        if not _NAME.match(self.name or ""):
            raise DefinitionError(f"invalid agent name {self.name!r} (lowercase letters, digits, '-', '_'; max 64)")
        desc = (self.description or "").strip()
        if not desc:
            raise DefinitionError(f"agent {self.name!r}: description is required (it is what routing reads)")
        if "\n" in desc or len(desc) > MAX_DESCRIPTION_CHARS:
            raise DefinitionError(f"agent {self.name!r}: description must be one line of at most {MAX_DESCRIPTION_CHARS} chars")
        if not (self.system_prompt or "").strip():
            raise DefinitionError(f"agent {self.name!r}: system prompt (Markdown body) is required")
        if not (self.model_role or "").strip():
            raise DefinitionError(f"agent {self.name!r}: model role must be non-empty")
        for label, value in (("max_turns", self.max_turns), ("max_output_tokens", self.max_output_tokens)):
            if not isinstance(value, int) or value < 1:
                raise DefinitionError(f"agent {self.name!r}: {label} must be a positive integer")
        if self.max_tool_calls is not None and (not isinstance(self.max_tool_calls, int) or self.max_tool_calls < 0):
            raise DefinitionError(f"agent {self.name!r}: max_tool_calls must be a non-negative integer")
        if self.max_tokens is not None and (not isinstance(self.max_tokens, int) or self.max_tokens < 1):
            raise DefinitionError(f"agent {self.name!r}: max_tokens must be a positive integer")
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise DefinitionError(f"agent {self.name!r}: timeout_s must be > 0")
        if self.owned_files and self.write_access is not WriteAccess.OWN_FILES:
            raise DefinitionError(f"agent {self.name!r}: owned_files requires write_access: own_files")
        if self.write_access is WriteAccess.OWN_FILES and not self.owned_files:
            raise DefinitionError(f"agent {self.name!r}: write_access own_files requires a non-empty owned_files list")
        if self.memory is MemoryAccess.READ_WRITE and not self.write_access.writes:
            raise DefinitionError(f"agent {self.name!r}: read_write memory requires a writer (write_access != none)")
        overlap = set(self.tools or ()) & set(self.disallowed_tools)
        if overlap:
            raise DefinitionError(f"agent {self.name!r}: tools both allowed and disallowed: {sorted(overlap)}")

    @property
    def read_only(self) -> bool:
        return not self.write_access.writes

    def fingerprint(self) -> str:
        """Stable hash of everything that changes behaviour (used in workflow replay keys)."""
        d = asdict(self)
        d.pop("source_path", None)
        d.pop("extra", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]

    def with_overrides(self, **changes: Any) -> "SubagentDefinition":
        return replace(self, **changes)

    # ------------------------------------------------------------------ (de)serialisation
    @classmethod
    def from_markdown(cls, text: str, path: str = "") -> "SubagentDefinition":
        try:
            meta, body = split_frontmatter(text)
        except _MemoryValidationError as exc:
            raise DefinitionError(f"{path or 'definition'}: {exc}") from exc
        if meta is None:
            raise DefinitionError(f"{path or 'definition'}: missing YAML frontmatter")
        if "name" not in meta and path:
            meta["name"] = Path(path).stem
        return cls.from_mapping(meta, body.strip(), source_path=path)

    @classmethod
    def from_mapping(cls, data: dict[str, Any], system_prompt: str | None = None, *, source_path: str = "") -> "SubagentDefinition":
        """Build from frontmatter keys (snake_case or Claude Code camelCase)."""
        d = {_ALIASES.get(k, k): v for k, v in data.items()}
        prompt = system_prompt if system_prompt is not None else d.pop("system_prompt", "")
        d.pop("system_prompt", None)
        known = {f for f in cls.__dataclass_fields__ if f not in ("extra", "source_path", "system_prompt")}
        extra = {k: v for k, v in d.items() if k not in known}
        kw = {k: v for k, v in d.items() if k in known}
        try:
            if "tools" in kw:
                kw["tools"] = None if kw["tools"] is None else _names(kw["tools"])
            if "disallowed_tools" in kw:
                kw["disallowed_tools"] = _names(kw["disallowed_tools"] or ())
            if "owned_files" in kw:
                kw["owned_files"] = _names(kw["owned_files"] or ())
            if "write_access" in kw:
                kw["write_access"] = WriteAccess(kw["write_access"] or "none")
            if extra.get("isolation") == "worktree" and "write_access" not in kw:
                kw["write_access"] = WriteAccess.OWN_WORKTREE
            if "memory" in kw:
                kw["memory"] = MemoryAccess(kw["memory"] or "read_only")
            if "context" in kw:
                kw["context"] = ContextMode(kw["context"] or "fresh")
            for k in ("max_turns", "max_tool_calls", "max_tokens", "max_output_tokens"):
                if kw.get(k) is not None:
                    kw[k] = int(kw[k])
            if kw.get("timeout_s") is not None:
                kw["timeout_s"] = float(kw["timeout_s"])
            if "can_spawn" in kw:
                kw["can_spawn"] = bool(kw["can_spawn"])
            if "description" in kw:
                kw["description"] = str(kw["description"]).strip()
            if "model_role" in kw:
                kw["model_role"] = str(kw["model_role"]).strip()
        except (TypeError, ValueError) as exc:
            raise DefinitionError(f"{source_path or data.get('name', 'definition')}: invalid value: {exc}") from exc
        if "name" not in kw or "description" not in kw:
            raise DefinitionError(f"{source_path or 'definition'}: frontmatter requires 'name' and 'description'")
        return cls(system_prompt=prompt or "", source_path=source_path, extra=extra, **kw)

    @classmethod
    def load(cls, path: str | Path) -> "SubagentDefinition":
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as exc:
            raise DefinitionError(f"cannot read {p}: {exc}") from exc
        return cls.from_markdown(text, str(p))


_ALIASES = {
    "model": "model_role",
    "role": "model_role",
    "prompt": "system_prompt",
    "maxTurns": "max_turns",
    "maxToolCalls": "max_tool_calls",
    "maxTokens": "max_tokens",
    "maxOutputTokens": "max_output_tokens",
    "timeout": "timeout_s",
    "timeoutS": "timeout_s",
    "disallowedTools": "disallowed_tools",
    "writeAccess": "write_access",
    "write_permission": "write_access",
    "ownedFiles": "owned_files",
    "canSpawn": "can_spawn",
}


def _names(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        items: Iterable[Any] = value.split(",")
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        raise TypeError(f"expected a list or comma-separated string, got {type(value).__name__}")
    out = tuple(dict.fromkeys(str(v).strip() for v in items if str(v).strip()))
    return out


# ----------------------------------------------------------------------------- tool selection


def select_tools(
    definition: SubagentDefinition,
    available: Sequence[ToolSpec],
    *,
    allow_writes: bool,
    narrow_to: Sequence[str] | None = None,
    exclude: Sequence[str] = (),
) -> list[ToolSpec]:
    """Return the tools a subagent is offered (order of ``available`` preserved).

    Invariant: when ``allow_writes`` is false (read-only definition, or a writer that could not get
    its lease), no tool with ``read_only=False`` survives, whatever the allowlist says. This is the
    "readers and critics get read-only tools" rule enforced in code (checklist item 11).
    """
    allowed = set(definition.tools) if definition.tools is not None else None
    if narrow_to is not None:
        allowed = set(narrow_to) if allowed is None else allowed & set(narrow_to)
    banned = set(definition.disallowed_tools) | set(exclude)
    out: list[ToolSpec] = []
    for spec in available:
        if allowed is not None and spec.name not in allowed:
            continue
        if spec.name in banned:
            continue
        if not spec.read_only and not (allow_writes and definition.write_access.writes):
            continue
        out.append(spec)
    return out


# ----------------------------------------------------------------------------- registry


class AgentRegistry:
    """Thread-safe name -> :class:`SubagentDefinition` map with a routing catalog."""

    def __init__(self, definitions: Iterable[SubagentDefinition] = ()) -> None:
        self._lock = threading.Lock()
        self._defs: dict[str, SubagentDefinition] = {}
        for d in definitions:
            self.register(d)

    def register(self, definition: SubagentDefinition, *, override: bool = False) -> None:
        with self._lock:
            if definition.name in self._defs and not override:
                raise DefinitionError(f"agent {definition.name!r} already registered")
            self._defs[definition.name] = definition

    def get(self, name: str) -> SubagentDefinition:
        with self._lock:
            d = self._defs.get(name)
        if d is None:
            raise UnknownAgentError(f"unknown subagent {name!r}; available: {', '.join(self.names()) or '(none)'}")
        return d

    def __contains__(self, name: object) -> bool:
        with self._lock:
            return name in self._defs

    def __len__(self) -> int:
        with self._lock:
            return len(self._defs)

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._defs)

    def definitions(self) -> list[SubagentDefinition]:
        with self._lock:
            return [self._defs[n] for n in sorted(self._defs)]

    def catalog(self) -> str:
        """One line per agent (sorted, deterministic): what the lead routes on, cache-stable."""
        lines = []
        for d in self.definitions():
            mode = "read-only" if d.read_only else f"writer:{d.write_access.value}"
            lines.append(f"- {d.name} ({mode}): {d.description}")
        return "\n".join(lines)

    def catalog_tokens(self) -> int:
        return estimate_tokens(self.catalog())

    def load_dirs(self, dirs: Sequence[str | Path], *, strict: bool = True) -> list[SubagentDefinition]:
        """Load ``*.md`` from ``dirs`` (highest priority first; first definition of a name wins).

        ``strict=False`` skips malformed files with a :class:`RuntimeWarning` instead of raising.
        Warns when the combined catalog exceeds :data:`CATALOG_TOKEN_WARNING` tokens.
        """
        loaded: list[SubagentDefinition] = []
        for directory in dirs:
            root = Path(directory).expanduser()
            if not root.is_dir():
                continue
            for path in sorted(root.glob("*.md")):
                try:
                    defn = SubagentDefinition.load(path)
                except DefinitionError as exc:
                    if strict:
                        raise
                    warnings.warn(f"skipping subagent definition: {exc}", RuntimeWarning, stacklevel=2)
                    continue
                if defn.name in self:
                    continue
                self.register(defn)
                loaded.append(defn)
        if self.catalog_tokens() > CATALOG_TOKEN_WARNING:
            warnings.warn(
                f"subagent descriptions total ~{self.catalog_tokens()} tokens (> {CATALOG_TOKEN_WARNING}); shorten them",
                RuntimeWarning,
                stacklevel=2,
            )
        return loaded

    @classmethod
    def from_project(cls, project_root: str | Path, *, user_dir: str | Path | None = "~/.harness/agents", strict: bool = True) -> "AgentRegistry":
        """Project ``.harness/agents`` overrides the user directory."""
        reg = cls()
        dirs: list[str | Path] = [Path(project_root) / ".harness" / "agents"]
        if user_dir is not None:
            dirs.append(user_dir)
        reg.load_dirs(dirs, strict=strict)
        return reg


__all__ = [
    "SubagentDefinition", "WriteAccess", "MemoryAccess", "ContextMode", "AgentRegistry", "select_tools",
    "estimate_tokens", "CATALOG_TOKEN_WARNING", "INHERIT_ROLE",
]
