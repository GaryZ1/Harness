"""Tool-design hygiene: definition budgets, description lint and description pinning.

Research basis (``reports/SOTA agentic harness reference.md`` §5.3 A, §5.7 steps 8, 10, 15 and §5.8
"Tool surface"):

* **Description hygiene** (Anthropic *Writing effective tools*): namespaced names, unambiguous
  parameters (``user_id`` not ``user``), every parameter described, descriptions written "as if
  onboarding a new hire", absolute paths, enums for modes, ``input_examples`` (1-5) for nested
  or ID-formatted parameters (complex-parameter accuracy 72% -> 90%).
* **Budgets**: tool search once there are 10+ tools or >10K definition tokens; Claude's
  selection degrades past 30-50 loaded tools. :func:`lint_tools` reports both.
* **Pin approved descriptions** (MCP hygiene, rug-pull defence): hash each approved definition
  and refuse or alert when a server later changes it (:class:`DescriptionPins`).

Lint never blocks registration on its own; run it in CI (``assert not errors(lint_tools(...))``).
"""

from __future__ import annotations

import difflib
import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from harness_core.types import ToolSpec
from harness_memory.locking import atomic_write_text

from .catalog import DEFINITION_TOKEN_THRESHOLD, TOOL_COUNT_THRESHOLD, definition_tokens
from .registry import TOOL_NAME_RE
from .schema import Target, lint_schema

Severity = Literal["error", "warning", "info"]

MIN_DESCRIPTION_CHARS = 40
MAX_DESCRIPTION_TOKENS = 1_000
MAX_LOADED_TOOLS = 30  # lower edge of Claude's 30-50 degradation range
AMBIGUOUS_PARAMS = frozenset({"user", "id", "name", "data", "value", "obj", "item", "input", "args", "params", "query_params", "info"})


@dataclass(frozen=True)
class LintFinding:
    tool: str
    severity: Severity
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.tool}: {self.message}"


def errors(findings: Iterable[LintFinding]) -> list[LintFinding]:
    return [f for f in findings if f.severity == "error"]


def _is_complex(prop: Any) -> bool:
    if not isinstance(prop, dict):
        return False
    t = prop.get("type")
    if t == "object" or (isinstance(t, list) and "object" in t):
        return True
    if t == "array" and isinstance(prop.get("items"), dict):
        return _is_complex(prop["items"]) or prop["items"].get("type") == "object"
    return bool(prop.get("pattern")) or prop.get("format") in ("date", "date-time")


def lint_tool(spec: ToolSpec, *, target: Target = "anthropic") -> list[LintFinding]:
    """Lint one tool definition."""
    out: list[LintFinding] = []
    n = spec.name

    def add(sev: Severity, msg: str) -> None:
        out.append(LintFinding(n, sev, msg))

    if not TOOL_NAME_RE.match(n or ""):
        add("error", f"name must match {TOOL_NAME_RE.pattern}")
    if spec.provider_type:
        return out
    desc = spec.description.strip()
    if not desc:
        add("error", "missing description")
    elif len(desc) < MIN_DESCRIPTION_CHARS:
        add("warning", f"description is only {len(desc)} chars; say what it does, when to use it, and its limits")
    tokens = definition_tokens([spec])
    if tokens > MAX_DESCRIPTION_TOKENS:
        add("warning", f"definition costs ~{tokens} tokens; trim it or defer the tool")
    for issue in lint_schema(spec.input_schema, target):
        add("error" if "requires" in issue or "does not support" in issue else "info", issue)
    props = (spec.input_schema or {}).get("properties") or {}
    complex_params = []
    for pname, prop in props.items():
        if not (isinstance(prop, dict) and str(prop.get("description", "")).strip()) and not (isinstance(prop, dict) and "enum" in prop):
            add("warning", f"parameter {pname!r} has no description")
        if pname.lower() in AMBIGUOUS_PARAMS:
            add("info", f"parameter {pname!r} is ambiguous; prefer a specific name such as '{pname}_id' or 'user_id'")
        if pname.lower() in ("path", "file", "file_path", "filepath", "dir", "directory") and "absolute" not in str(prop.get("description", "")).lower():
            add("info", f"parameter {pname!r}: require absolute paths and say so in the description (poka-yoke)")
        if _is_complex(prop):
            complex_params.append(pname)
    if complex_params and not spec.metadata.get("input_examples"):
        add("info", f"nested/formatted parameters {complex_params}: add 1-5 input_examples")
    return out


def lint_tools(specs: Sequence[ToolSpec], *, target: Target = "anthropic") -> list[LintFinding]:
    """Lint a whole tool array, including catalog-level budget checks."""
    out: list[LintFinding] = []
    for s in specs:
        out.extend(lint_tool(s, target=target))
    names = [s.name for s in specs]
    for dup in sorted({x for x in names if names.count(x) > 1}):
        out.append(LintFinding(dup, "error", "duplicate tool name"))
    loaded = [s for s in specs if not s.defer_loading]
    has_deferred = any(s.defer_loading for s in specs)
    total = definition_tokens(loaded)
    if not has_deferred and (len(loaded) >= TOOL_COUNT_THRESHOLD or total > DEFINITION_TOKEN_THRESHOLD):
        out.append(
            LintFinding("*", "warning", f"{len(loaded)} loaded tools (~{total:,} definition tokens): enable tool search and defer all but 3-5 hot tools")
        )
    if len(loaded) > MAX_LOADED_TOOLS:
        out.append(LintFinding("*", "warning", f"{len(loaded)} tools loaded at once; Claude's tool selection degrades past 30-50"))
    lowered = sorted({n.lower() for n in names})
    for i, a in enumerate(lowered):
        for b in lowered[i + 1 :]:
            if difflib.SequenceMatcher(None, a, b).ratio() >= 0.9:
                out.append(LintFinding(f"{a}/{b}", "info", "names are nearly identical; overlapping tools confuse selection"))
    if names != sorted(names) and len(set(names)) == len(names):
        out.append(LintFinding("*", "info", "tool array is not in sorted order; keep serialisation deterministic for prompt caching"))
    return out


# ----------------------------------------------------------------------------- pinning


def fingerprint(definition: ToolSpec | Mapping[str, Any]) -> str:
    """SHA-256 over the model-visible parts of a definition (name, description, schema, annotations)."""
    if isinstance(definition, ToolSpec):
        payload = {
            "name": definition.name,
            "description": definition.description,
            "input_schema": definition.input_schema,
            "annotations": definition.metadata.get("mcp_annotations"),
        }
    else:
        payload = {
            "name": definition.get("name"),
            "description": definition.get("description", ""),
            "input_schema": definition.get("inputSchema", definition.get("input_schema")),
            "annotations": definition.get("annotations"),
        }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


PinStatus = Literal["approved", "new", "changed"]


class DescriptionPins:
    """Approved tool fingerprints, optionally persisted as JSON (atomic writes). Thread-safe."""

    def __init__(self, pins: Mapping[str, str] | None = None, *, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._pins: dict[str, str] = dict(pins or {})
        self._lock = threading.Lock()
        if self.path and self.path.exists() and not pins:
            self._pins = dict(json.loads(self.path.read_text(encoding="utf-8")))

    def status(self, key: str, fp: str) -> PinStatus:
        with self._lock:
            known = self._pins.get(key)
        if known is None:
            return "new"
        return "approved" if known == fp else "changed"

    def approve(self, key: str, fp: str) -> None:
        with self._lock:
            self._pins[key] = fp
            if self.path:
                atomic_write_text(self.path, json.dumps(self._pins, indent=2, sort_keys=True) + "\n")

    def as_dict(self) -> dict[str, str]:
        with self._lock:
            return dict(self._pins)


__all__ = [
    "AMBIGUOUS_PARAMS",
    "DescriptionPins",
    "LintFinding",
    "MAX_LOADED_TOOLS",
    "PinStatus",
    "errors",
    "fingerprint",
    "lint_tool",
    "lint_tools",
]
