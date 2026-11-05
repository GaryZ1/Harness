"""The delegation brief: the *only* thing a fresh subagent receives from its parent.

"The only content you pass from parent to subagent is the Agent tool's prompt string, so include any
file paths, error messages, or decisions the subagent needs directly in that prompt" (Agent SDK
docs, reference 8.6). Anthropic found vague briefs ("research the semiconductor shortage") made
subagents duplicate work and leave gaps; every brief needs "an objective, an output format, guidance
on the tools and sources to use, and clear task boundaries". This module encodes the checklist's
brief template (reference 8.7 item 3) and validates it in code:

* ``objective`` is required and must be a single sentence-sized line;
* budgets and tool lists may only **narrow** what the definition grants, never widen it;
* the condensed return cap (``output_max_tokens``) is bounded by the manager's hard cap (2k default).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from .definitions import SubagentDefinition, WriteAccess
from .errors import BriefValidationError

MAX_OBJECTIVE_CHARS = 600
DEFAULT_OUTPUT_MAX_TOKENS = 2_000


@dataclass(frozen=True)
class DelegationBrief:
    objective: str
    why_context: str = ""
    inputs: tuple[str, ...] = ()
    scope_in: tuple[str, ...] = ()
    out_of_scope: tuple[str, ...] = ()
    tools_allowed: tuple[str, ...] | None = None
    sources_guidance: str = ""
    max_turns: int | None = None
    max_tool_calls: int | None = None
    max_tokens: int | None = None
    timeout_s: float | None = None
    effort: str | None = None
    write_permission: WriteAccess | None = None    # None = the definition's own setting
    owned_files: tuple[str, ...] = ()
    output_max_tokens: int = DEFAULT_OUTPUT_MAX_TOKENS
    metadata: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.write_permission is not None and not isinstance(self.write_permission, WriteAccess):
            try:
                object.__setattr__(self, "write_permission", WriteAccess(self.write_permission))
            except ValueError as exc:
                raise BriefValidationError(f"brief.write_permission: {exc}") from exc
        for attr in ("inputs", "scope_in", "out_of_scope", "owned_files", "tools_allowed"):
            value = getattr(self, attr)
            if isinstance(value, list):
                object.__setattr__(self, attr, tuple(value))
        obj = (self.objective or "").strip()
        if not obj:
            raise BriefValidationError("brief.objective is required: one sentence describing what done looks like")
        if "\n" in obj or len(obj) > MAX_OBJECTIVE_CHARS:
            raise BriefValidationError(f"brief.objective must be one line of at most {MAX_OBJECTIVE_CHARS} chars; put detail in why_context/inputs")
        for label, value in (("max_turns", self.max_turns), ("max_tokens", self.max_tokens), ("output_max_tokens", self.output_max_tokens)):
            if value is not None and (not isinstance(value, int) or value < 1):
                raise BriefValidationError(f"brief.{label} must be a positive integer")
        if self.max_tool_calls is not None and (not isinstance(self.max_tool_calls, int) or self.max_tool_calls < 0):
            raise BriefValidationError("brief.max_tool_calls must be a non-negative integer")
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise BriefValidationError("brief.timeout_s must be > 0")
        if self.owned_files and self.write_permission not in (None, WriteAccess.OWN_FILES):
            raise BriefValidationError("brief.owned_files requires write_permission own_files")

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DelegationBrief":
        """Build from a tool-call input or YAML mapping (tolerates the checklist's nested layout)."""
        if not isinstance(data, Mapping):
            raise BriefValidationError("brief must be a mapping")
        d = dict(data)
        scope = d.pop("scope", None) or {}
        budget = d.pop("budget", None) or d.pop("effort_budget", None) or {}
        if not isinstance(scope, Mapping) or not isinstance(budget, Mapping):
            raise BriefValidationError("brief.scope and brief.budget must be mappings")
        output = d.pop("output", None) or {}
        kw: dict[str, Any] = {
            "objective": str(d.get("objective") or ""),
            "why_context": str(d.get("why_context") or d.get("why_it_matters") or ""),
            "inputs": _strs(d.get("inputs")),
            "scope_in": _strs(d.get("scope_in", scope.get("in"))),
            "out_of_scope": _strs(d.get("out_of_scope", scope.get("out_non_goals", scope.get("out_of_scope")))),
            "sources_guidance": str(d.get("sources_guidance") or ""),
            "owned_files": _strs(d.get("owned_files")),
            "effort": d.get("effort", budget.get("effort")),
        }
        if d.get("tools_allowed") is not None:
            kw["tools_allowed"] = _strs(d["tools_allowed"])
        for key in ("max_turns", "max_tool_calls", "max_tokens"):
            v = d.get(key, budget.get(key))
            if v is not None:
                kw[key] = _int(key, v)
        t = d.get("timeout_s", budget.get("timeout_s"))
        if t is not None:
            kw["timeout_s"] = float(t)
        wp = d.get("write_permission", d.get("write_access"))
        if wp is not None:
            try:
                kw["write_permission"] = WriteAccess(wp)
            except ValueError as exc:
                raise BriefValidationError(f"brief.write_permission must be one of {[w.value for w in WriteAccess]}") from exc
        omt = d.get("output_max_tokens", output.get("max_tokens") if isinstance(output, Mapping) else None)
        if omt is not None:
            kw["output_max_tokens"] = _int("output_max_tokens", omt)
        return cls(**kw)

    # ------------------------------------------------------------------ privilege narrowing
    def check_against(self, definition: SubagentDefinition, *, hard_output_cap: int) -> None:
        """Raise :class:`BriefValidationError` if the brief would widen the definition's grants."""
        if self.write_permission is not None and self.write_permission.writes:
            if self.write_permission is not definition.write_access:
                raise BriefValidationError(
                    f"brief requests write_permission={self.write_permission.value} but agent {definition.name!r} "
                    f"is defined with write_access={definition.write_access.value}; briefs may only narrow privileges"
                )
        if self.owned_files and definition.owned_files:
            outside = [f for f in self.owned_files if not any(_within(f, o) for o in definition.owned_files)]
            if outside:
                raise BriefValidationError(f"brief.owned_files outside the definition's ownership: {outside}")
        if self.tools_allowed is not None and definition.tools is not None:
            extra = sorted(set(self.tools_allowed) - set(definition.tools))
            if extra:
                raise BriefValidationError(f"brief.tools_allowed includes tools not granted to {definition.name!r}: {extra}")
        if self.output_max_tokens > hard_output_cap:
            raise BriefValidationError(f"brief.output_max_tokens {self.output_max_tokens} exceeds the hard cap {hard_output_cap}")

    def effective_write_access(self, definition: SubagentDefinition) -> WriteAccess:
        if self.write_permission is None:
            return definition.write_access
        return self.write_permission  # check_against guarantees it is NONE or equal to the definition's

    def narrowed(self, **changes: Any) -> "DelegationBrief":
        return replace(self, **changes)

    # ------------------------------------------------------------------ rendering
    def render(self) -> str:
        """The self-contained task message (deterministic; identical briefs give identical text)."""
        parts = [f"# Objective\n{self.objective.strip()}"]
        if self.why_context.strip():
            parts.append(f"# Why this matters / decisions already made\n{self.why_context.strip()}")
        if self.inputs:
            parts.append("# Inputs\n" + "\n".join(f"- {i}" for i in self.inputs))
        if self.scope_in:
            parts.append("# In scope\n" + "\n".join(f"- {i}" for i in self.scope_in))
        if self.out_of_scope:
            parts.append("# Out of scope (owned by siblings or non-goals; do not touch)\n" + "\n".join(f"- {i}" for i in self.out_of_scope))
        if self.sources_guidance.strip():
            parts.append(f"# Tools and sources guidance\n{self.sources_guidance.strip()}")
        budget = []
        if self.max_turns is not None:
            budget.append(f"max {self.max_turns} turns")
        if self.max_tool_calls is not None:
            budget.append(f"max {self.max_tool_calls} tool calls")
        if self.max_tokens is not None:
            budget.append(f"max {self.max_tokens} tokens")
        if budget:
            parts.append("# Budget\n" + ", ".join(budget) + ". Stop early when the objective is met.")
        if self.write_permission is not None:
            perm = self.write_permission.value
            if self.owned_files:
                perm += " (" + ", ".join(self.owned_files) + ")"
            parts.append(f"# Write permission\n{perm}")
        parts.append(f"# Output\nFinish with the report JSON described in your instructions, at most ~{self.output_max_tokens} tokens. "
                     "Put large outputs in files and list their paths in `files`.")
        return "\n\n".join(parts)


def _strs(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, Sequence):
        return tuple(str(v) for v in value if str(v).strip())
    raise BriefValidationError(f"expected a list of strings, got {type(value).__name__}")


def _int(key: str, value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise BriefValidationError(f"brief.{key} must be an integer") from exc


def _within(path: str, owned: str) -> bool:
    p, o = path.strip("/"), owned.strip("/")
    return p == o or p.startswith(o + "/") or o in ("", ".")


__all__ = ["DelegationBrief", "DEFAULT_OUTPUT_MAX_TOKENS", "MAX_OBJECTIVE_CHARS"]
