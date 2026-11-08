"""Stable, layered system-prompt assembly (request layout layers [1] and [2]).

The build checklist (reference report section 3.7, step 2) fixes one request layout for every
provider::

    [0] tools            hot tools in deterministic order             <- optional breakpoint
    [1] system           static harness prompt at the "right altitude"
    [2] project context  AGENTS.md / CLAUDE.md, skills index, memory index, repo map
                                                                      <- breakpoint at the end
    [3] messages         append-only history + dynamic tail

This module renders layers [1] and [2]. Everything it renders must be *byte-stable for the whole
session*: no timestamps, session ids, cwd listings, budgets or git status (those go through the
dynamic channel, :mod:`harness_context.dynamic`). Layer text is normalized (newlines, trailing
whitespace, block HTML comments stripped as Claude Code does) and size-budgeted *deterministically*
so the same inputs always produce the same bytes. Instruction files above 200 lines are linted
("reduce adherence"); a total project-context byte budget mirrors Codex ``project_doc_max_bytes``.

File discovery (root -> cwd, imports, overrides) is out of scope here: use
:class:`harness_memory.InstructionLoader` (or any loader) and pass the rendered text in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

from harness_core.types import TextBlock

from .errors import ConfigurationError
from .references import ContextReference, render_reference_index

HTML_BLOCK_COMMENT = re.compile(r"<!--.*?-->\s*", re.S)
RECOMMENDED_MAX_INSTRUCTION_LINES = 200
DEFAULT_PROJECT_BYTE_BUDGET = 32 * 1024          # our choice, modeled on Codex project_doc_max_bytes
MEMORY_INDEX_MAX_LINES = 200                      # Claude Code: first 200 lines / 25 KB of MEMORY.md
MEMORY_INDEX_MAX_BYTES = 25_000
SKILL_DESCRIPTION_MAX_CHARS = 400                 # ~100 tokens per skill in the index
REPO_MAP_MAX_BYTES = 4_096                        # ~1k tokens (Aider map_tokens=1024)

_NAME = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


class LayerKind(str, Enum):
    SYSTEM = "system"    # static harness prompt sections
    PROJECT = "project"  # per-session project context (instructions, indexes, repo map)


@dataclass(frozen=True)
class PromptLayer:
    """One named, stable section of the system prompt.

    :param tag: when set, the text is wrapped as ``<tag source="...">...</tag>``.
    :param max_bytes: per-layer UTF-8 byte cap (deterministic truncation from the end).
    :param max_lines: per-layer line cap (applied before ``max_bytes``).
    :param lint_lines: emit a lint warning above this many lines (instruction files: 200).
    """

    name: str
    text: str
    kind: LayerKind = LayerKind.SYSTEM
    tag: str | None = None
    source: str = ""
    max_bytes: int | None = None
    max_lines: int | None = None
    lint_lines: int | None = None
    strip_comments: bool = True

    def __post_init__(self) -> None:
        if not _NAME.match(self.name):
            raise ConfigurationError(f"invalid layer name {self.name!r}")


@dataclass(frozen=True)
class SkillEntry:
    """A skill as listed in the always-loaded index (body loads on demand)."""

    name: str
    description: str
    model_invocable: bool = True   # False = ``disable-model-invocation: true`` (kept out of context)


@dataclass(frozen=True)
class LayerLint:
    layer: str
    severity: str  # "warning" | "info"
    message: str


@dataclass
class RenderedPrompt:
    blocks: list[TextBlock]
    lints: list[LayerLint] = field(default_factory=list)
    layer_bytes: dict[str, int] = field(default_factory=dict)
    kinds: list[LayerKind] = field(default_factory=list)  # kind of each block (block 0 = base system)


def normalize_text(text: str, *, strip_comments: bool = True) -> str:
    """Normalize newlines and trailing whitespace; optionally strip block HTML comments."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if strip_comments:
        text = HTML_BLOCK_COMMENT.sub("", text)
    lines = [ln.rstrip() for ln in text.split("\n")]
    return "\n".join(lines).strip("\n")


def truncate_bytes(text: str, max_bytes: int, *, hint: str = "") -> tuple[str, int]:
    """Cut ``text`` to at most ``max_bytes`` UTF-8 bytes on a line boundary; return (text, bytes_cut)."""
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text, 0
    marker = f"\n[... truncated: over the {max_bytes}-byte budget{'; ' + hint if hint else ''}]"
    budget = max(0, max_bytes - len(marker.encode("utf-8")))
    cut = data[:budget].decode("utf-8", errors="ignore")
    if "\n" in cut:
        cut = cut[: cut.rfind("\n")]
    return cut + marker, len(data) - len(cut.encode("utf-8"))


def _render_layer(layer: PromptLayer, budget_left: int | None) -> tuple[str, list[LayerLint]]:
    lints: list[LayerLint] = []
    text = normalize_text(layer.text, strip_comments=layer.strip_comments)
    n_lines = text.count("\n") + 1 if text else 0
    if layer.lint_lines and n_lines > layer.lint_lines:
        lints.append(LayerLint(layer.name, "warning", f"{n_lines} lines; keep instruction files under {layer.lint_lines} lines"))
    if layer.max_lines is not None and n_lines > layer.max_lines:
        text = "\n".join(text.split("\n")[: layer.max_lines]) + f"\n[... {n_lines - layer.max_lines} more lines not loaded]"
        lints.append(LayerLint(layer.name, "info", f"truncated to {layer.max_lines} lines"))
    hint = f"read {layer.source} for the rest" if layer.source else ""
    caps = [c for c in (layer.max_bytes, budget_left) if c is not None]
    if caps:
        text, cut = truncate_bytes(text, min(caps), hint=hint)
        if cut:
            lints.append(LayerLint(layer.name, "warning", f"truncated {cut} bytes to fit the byte budget"))
    if layer.tag:
        src = f' source="{layer.source}"' if layer.source else ""
        text = f"<{layer.tag}{src}>\n{text}\n</{layer.tag}>"
    return text, lints


class PromptStack:
    """Ordered, immutable-by-convention set of stable layers.

    Rendering order is fixed: the caller's base system prompt, then every ``SYSTEM`` layer in
    insertion order, then every ``PROJECT`` layer in insertion order. Names are unique.
    """

    def __init__(self, layers: Iterable[PromptLayer] = (), *, project_byte_budget: int | None = DEFAULT_PROJECT_BYTE_BUDGET) -> None:
        self._layers: list[PromptLayer] = []
        self.project_byte_budget = project_byte_budget
        for layer in layers:
            self.add(layer)

    @property
    def layers(self) -> tuple[PromptLayer, ...]:
        return tuple(self._layers)

    def add(self, layer: PromptLayer) -> "PromptStack":
        if any(existing.name == layer.name for existing in self._layers):
            raise ConfigurationError(f"duplicate layer name {layer.name!r}")
        self._layers.append(layer)
        return self

    # ------------------------------------------------------------------ convenience constructors
    def add_instructions(self, text: str, *, source: str = "AGENTS.md", name: str | None = None) -> "PromptStack":
        """Root/project instruction files (already discovered and concatenated by a loader)."""
        return self.add(
            PromptLayer(
                name or f"instructions:{source}"[:64], text, LayerKind.PROJECT, tag="instructions", source=source,
                lint_lines=RECOMMENDED_MAX_INSTRUCTION_LINES,
            )
        )

    def add_skills_index(self, skills: Sequence[SkillEntry], *, name: str = "skills") -> "PromptStack":
        """Skill catalog: name + description only, sorted by name, non-invocable skills omitted."""
        lines = []
        for s in sorted(skills, key=lambda s: s.name):
            if not s.model_invocable:
                continue
            desc = " ".join(s.description.split())
            if len(desc) > SKILL_DESCRIPTION_MAX_CHARS:
                desc = desc[: SKILL_DESCRIPTION_MAX_CHARS - 3].rstrip() + "..."
            lines.append(f"- {s.name}: {desc}")
        return self.add(PromptLayer(name, "\n".join(lines), LayerKind.PROJECT, tag="skills"))

    def add_memory_index(self, text: str, *, source: str = "MEMORY.md", name: str = "memory-index") -> "PromptStack":
        return self.add(
            PromptLayer(
                name, text, LayerKind.PROJECT, tag="memory-index", source=source,
                max_lines=MEMORY_INDEX_MAX_LINES, max_bytes=MEMORY_INDEX_MAX_BYTES,
            )
        )

    def add_reference_index(self, refs: Iterable[ContextReference], *, name: str = "references") -> "PromptStack":
        """Just-in-time pointers (paths, URLs, queries) the model loads on demand; must be stable per session."""
        return self.add(PromptLayer(name, render_reference_index(refs), LayerKind.PROJECT))

    def add_repo_map(self, text: str, *, name: str = "repo-map") -> "PromptStack":
        return self.add(PromptLayer(name, text, LayerKind.PROJECT, tag="repo-map", max_bytes=REPO_MAP_MAX_BYTES))

    # ------------------------------------------------------------------ rendering
    def render(self, base_system: str) -> RenderedPrompt:
        """Render ``base_system`` plus all layers to text blocks (no cache markers; the assembler places them)."""
        out = RenderedPrompt(blocks=[])
        base = normalize_text(base_system, strip_comments=False)
        if base:
            out.blocks.append(TextBlock(base))
            out.kinds.append(LayerKind.SYSTEM)
            out.layer_bytes["system"] = len(base.encode("utf-8"))
        budget = self.project_byte_budget
        ordered = [la for la in self._layers if la.kind is LayerKind.SYSTEM] + [la for la in self._layers if la.kind is LayerKind.PROJECT]
        for layer in ordered:
            left = budget if layer.kind is LayerKind.PROJECT else None
            if left is not None and left <= 0:
                out.lints.append(LayerLint(layer.name, "warning", "dropped: project-context byte budget exhausted"))
                continue
            text, lints = _render_layer(layer, left)
            out.lints.extend(lints)
            if not text.strip():
                continue
            size = len(text.encode("utf-8"))
            if budget is not None and layer.kind is LayerKind.PROJECT:
                budget -= size
            out.blocks.append(TextBlock(text))
            out.kinds.append(layer.kind)
            out.layer_bytes[layer.name] = size
        return out

    def lint(self) -> list[LayerLint]:
        return self.render("").lints


__all__ = [
    "DEFAULT_PROJECT_BYTE_BUDGET",
    "LayerKind",
    "LayerLint",
    "PromptLayer",
    "PromptStack",
    "RenderedPrompt",
    "SkillEntry",
    "normalize_text",
    "truncate_bytes",
]
