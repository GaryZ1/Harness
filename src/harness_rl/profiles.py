"""Per-model harness profiles: "match the model's training harness".

Frontier models are post-trained inside specific harnesses, and mismatches cost several to 20+
points (reference §11.4 "Per-model native tool profile"; §11.7 steps 1-4, 15; §11.8). A profile
bundles tool names and schemas, edit format, system-prompt style/file, reasoning-replay mode,
tool-result role, compaction protocol and scaffold feature flags.

Rules enforced in code:

* **Loaded from config, never hardcoded.** This module contains no model ids; profiles match
  model ids by glob patterns in YAML/JSON (:meth:`ProfileRegistry.from_file`).
* **No silent generic fallback.** :meth:`ProfileRegistry.select` raises
  :class:`ProfileNotFoundError` unless the config names a ``fallback`` *and* the caller passes
  ``allow_fallback=True``; the selection then reports ``fallback=True`` so it is logged.
* **Most specific match wins**; equally specific matches from different profiles are an error.
* **Native tool-result role.** ``tool_result_role`` must be ``tool`` or ``tool_result_block``;
  ``user_text`` is rejected (tool results as user messages break native tool-role training).
* **Edit tool present.** ``edit_tool`` must name one of the profile's tools.
* **Scaffolds are flags** (todo, think, planner, evaluator gate, resets, sprints...) so every
  component can be ablated per model (:meth:`HarnessProfile.ablations`) - components go from
  load-bearing to dead weight within one model generation.
* **Mismatch detection.** :class:`ProfileMismatchMonitor` turns calls to unknown tools (e.g. an
  ``apply_patch`` attempted where it is absent) into a ``profile_mismatch`` metric and an
  actionable tool error instead of a crash.
"""

from __future__ import annotations

import copy
import difflib
import fnmatch
import hashlib
import json
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from harness_core.types import ToolOutcome, ToolSpec

from .errors import ProfileAmbiguityError, ProfileError, ProfileNotFoundError

TOOL_RESULT_ROLES = ("tool", "tool_result_block")
REASONING_REPLAY = ("none", "anthropic_thinking", "openai_encrypted_reasoning", "deepseek_reasoning_content")
DEFAULT_SCAFFOLDS = ("todo", "think", "planner", "evaluator_gate", "context_resets", "sprints", "subagents")


@dataclass(frozen=True)
class CompactionProtocol:
    mode: str = "client"               # client | provider | none
    trigger_ratio: float = 0.9         # Codex-family default: compact at 90% of the window
    prompt_file: str | None = None     # e.g. a vendor handoff-summary prompt
    reset_handoff: bool = False        # reset-plus-handoff mode for models with "context anxiety"

    def __post_init__(self) -> None:
        if self.mode not in ("client", "provider", "none"):
            raise ProfileError("compaction.mode must be client|provider|none")
        if not 0.1 <= self.trigger_ratio <= 1.0:
            raise ProfileError("compaction.trigger_ratio must be in [0.1, 1.0]")


@dataclass(frozen=True)
class HarnessProfile:
    name: str
    match: tuple[str, ...]
    tools: tuple[ToolSpec, ...]
    edit_tool: str
    family: str = ""
    edit_format: str = ""              # e.g. apply_patch_v4a | str_replace | whole_file
    prompt_style: str = ""
    prompt_file: str | None = None
    reasoning_replay: str = "none"
    tool_result_role: str = "tool_result_block"
    compaction: CompactionProtocol = field(default_factory=CompactionProtocol)
    scaffolds: Mapping[str, bool] = field(default_factory=dict)
    tool_aliases: Mapping[str, str] = field(default_factory=dict)   # foreign tool name -> our tool (for mismatch hints)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ProfileError("profile needs a name")
        if not self.match:
            raise ProfileError(f"profile {self.name}: 'match' needs at least one model-id glob")
        names = [t.name for t in self.tools]
        if len(set(names)) != len(names):
            raise ProfileError(f"profile {self.name}: duplicate tool names")
        if self.edit_tool not in names:
            raise ProfileError(f"profile {self.name}: edit_tool {self.edit_tool!r} is not one of its tools {names}")
        if self.tool_result_role not in TOOL_RESULT_ROLES:
            raise ProfileError(f"profile {self.name}: tool_result_role must be one of {TOOL_RESULT_ROLES} (never user text)")
        if self.reasoning_replay not in REASONING_REPLAY:
            raise ProfileError(f"profile {self.name}: reasoning_replay must be one of {REASONING_REPLAY}")
        for k, v in self.scaffolds.items():
            if not isinstance(v, bool):
                raise ProfileError(f"profile {self.name}: scaffold {k!r} must be true/false")

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools]

    def scaffold(self, name: str) -> bool:
        return bool(self.scaffolds.get(name, False))

    def content_hash(self) -> str:
        payload = {
            "name": self.name, "match": list(self.match), "tools": [t.to_api() for t in self.tools], "edit_tool": self.edit_tool,
            "edit_format": self.edit_format, "prompt_style": self.prompt_style, "prompt_file": self.prompt_file,
            "reasoning_replay": self.reasoning_replay, "tool_result_role": self.tool_result_role, "compaction": self.compaction.__dict__,
            "scaffolds": dict(sorted(self.scaffolds.items())), "extra": dict(self.extra),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def with_scaffold(self, name: str, on: bool) -> "HarnessProfile":
        return replace(self, scaffolds={**self.scaffolds, name: on})

    def ablations(self) -> list[tuple[str, "HarnessProfile"]]:
        """One variant per enabled scaffold with that scaffold turned off (re-run at every model upgrade)."""
        return [(k, self.with_scaffold(k, False)) for k, v in sorted(self.scaffolds.items()) if v]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "match": list(self.match), "family": self.family,
            "tools": [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in self.tools],
            "edit_tool": self.edit_tool, "edit_format": self.edit_format, "prompt_style": self.prompt_style, "prompt_file": self.prompt_file,
            "reasoning_replay": self.reasoning_replay, "tool_result_role": self.tool_result_role,
            "compaction": dict(self.compaction.__dict__), "scaffolds": dict(self.scaffolds), "tool_aliases": dict(self.tool_aliases),
            "extra": dict(self.extra),
        }


def _tool(d: Any) -> ToolSpec:
    if isinstance(d, str):
        return ToolSpec(d, "")
    if not isinstance(d, Mapping) or "name" not in d:
        raise ProfileError(f"tool entry needs a name: {d!r}")
    return ToolSpec(str(d["name"]), str(d.get("description", "")), dict(d.get("input_schema") or {"type": "object", "properties": {}}),
                    read_only=bool(d.get("read_only", False)), provider_type=d.get("provider_type"), metadata=dict(d.get("metadata") or {}))


def parse_profile(d: Mapping[str, Any]) -> HarnessProfile:
    match = d.get("match") or []
    if isinstance(match, str):
        match = [match]
    comp = d.get("compaction") or {}
    return HarnessProfile(
        name=str(d.get("name") or ""), match=tuple(str(m) for m in match), tools=tuple(_tool(t) for t in d.get("tools") or []),
        edit_tool=str(d.get("edit_tool") or ""), family=str(d.get("family") or ""), edit_format=str(d.get("edit_format") or ""),
        prompt_style=str(d.get("prompt_style") or ""), prompt_file=d.get("prompt_file"),
        reasoning_replay=str(d.get("reasoning_replay") or "none"), tool_result_role=str(d.get("tool_result_role") or "tool_result_block"),
        compaction=CompactionProtocol(**comp), scaffolds={str(k): v for k, v in (d.get("scaffolds") or {}).items()},
        tool_aliases={str(k): str(v) for k, v in (d.get("tool_aliases") or {}).items()}, extra=dict(d.get("extra") or {}),
    )


@dataclass(frozen=True)
class ProfileSelection:
    profile: HarnessProfile
    model: str
    matched_pattern: str | None
    fallback: bool


def _specificity(pattern: str) -> tuple[int, int]:
    """More literal characters and fewer wildcards = more specific."""
    wild = sum(pattern.count(c) for c in "*?[")
    return (len(pattern) - wild, -wild)


class ProfileRegistry:
    def __init__(self, profiles: Sequence[HarnessProfile], *, fallback: str | None = None) -> None:
        names = [p.name for p in profiles]
        if len(set(names)) != len(names):
            raise ProfileError("duplicate profile names")
        self.profiles = {p.name: p for p in profiles}
        if fallback is not None and fallback not in self.profiles:
            raise ProfileError(f"fallback profile {fallback!r} is not defined")
        self.fallback = fallback

    @classmethod
    def from_mapping(cls, d: Mapping[str, Any]) -> "ProfileRegistry":
        return cls([parse_profile(p) for p in d.get("profiles") or []], fallback=d.get("fallback"))

    @classmethod
    def from_file(cls, path: str | Path) -> "ProfileRegistry":
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        data = json.loads(text) if p.suffix == ".json" else yaml.safe_load(text)
        if not isinstance(data, Mapping):
            raise ProfileError(f"{p}: expected a mapping with 'profiles'")
        return cls.from_mapping(data)

    def get(self, name: str) -> HarnessProfile:
        try:
            return self.profiles[name]
        except KeyError as exc:
            raise ProfileNotFoundError(f"no profile named {name!r}") from exc

    def select(self, model: str, *, allow_fallback: bool = False) -> ProfileSelection:
        if not model:
            raise ProfileError("model id is required to select a profile")
        hits: list[tuple[tuple[int, int], str, HarnessProfile]] = []
        for prof in self.profiles.values():
            for pat in prof.match:
                if fnmatch.fnmatchcase(model, pat):
                    hits.append((_specificity(pat), pat, prof))
        if hits:
            hits.sort(key=lambda h: h[0], reverse=True)
            best = hits[0]
            rivals = {h[2].name for h in hits if h[0] == best[0]}
            if len(rivals) > 1:
                raise ProfileAmbiguityError(f"model {model!r} matches equally specific patterns in profiles {sorted(rivals)}")
            return ProfileSelection(best[2], model, best[1], False)
        if allow_fallback and self.fallback:
            return ProfileSelection(self.profiles[self.fallback], model, None, True)
        raise ProfileNotFoundError(f"no harness profile matches model {model!r}; add a profile (generic fallback is never silent)")


class ProfileMismatchMonitor:
    """Counts calls to tools the active profile does not offer and builds actionable errors."""

    def __init__(self, profile: HarnessProfile) -> None:
        self.profile = profile
        self._lock = threading.Lock()
        self.counts: dict[str, int] = {}

    def check(self, tool_name: str) -> ToolOutcome | None:
        if tool_name in self.profile.tool_names:
            return None
        with self._lock:
            self.counts[tool_name] = self.counts.get(tool_name, 0) + 1
        hint = self.profile.tool_aliases.get(tool_name)
        if hint is None:
            close = difflib.get_close_matches(tool_name, self.profile.tool_names, n=1)
            hint = close[0] if close else None
        msg = f"Tool '{tool_name}' is not available in this harness (profile '{self.profile.name}'). Available tools: {', '.join(self.profile.tool_names)}."
        if hint:
            msg += f" Use '{hint}' instead."
        return ToolOutcome(msg, is_error=True, metadata={"profile_mismatch": True, "requested_tool": tool_name, "profile": self.profile.name})

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return copy.deepcopy(self.counts)


__all__ = [
    "CompactionProtocol",
    "DEFAULT_SCAFFOLDS",
    "HarnessProfile",
    "ProfileMismatchMonitor",
    "ProfileRegistry",
    "ProfileSelection",
    "REASONING_REPLAY",
    "TOOL_RESULT_ROLES",
    "parse_profile",
]
