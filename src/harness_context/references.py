"""Just-in-time references: lightweight identifiers kept in context instead of the data itself.

Anthropic's guidance is to "keep lightweight identifiers (file paths, stored queries, URLs, line
ranges) in context and load data at runtime with tools" rather than pre-loading documents; Manus
calls the same idea *restorable compression*: drop the bulky content but keep the pointer so it
can be re-fetched. This module provides:

* :func:`extract_locators` - pulls the re-fetchable identifiers (path / URL / query / command)
  out of a tool call's input, used by clearing placeholders and compaction file tracking;
* :class:`ContextReference` and :func:`render_reference_index` - a deterministic, compact index
  of things the model can load on demand;
* :func:`find_offload_path` - recognizes an offloaded-output reference written by
  :mod:`harness_context.offload` so a cleared placeholder can point at the file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

#: Input keys that identify re-fetchable content, in priority order.
LOCATOR_KEYS: tuple[str, ...] = (
    "path", "file_path", "filepath", "filename", "file", "paths", "url", "uri", "query", "pattern", "glob",
    "command", "cmd", "id",
)
#: Keys whose value names a file (used for "files touched" tracking).
PATH_KEYS: tuple[str, ...] = ("path", "file_path", "filepath", "filename", "file", "paths", "notebook_path")

_OFFLOAD_PATH = re.compile(r'<offloaded-tool-output path="([^"]+)"')


def _short(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else ",".join(str(v) for v in value) if isinstance(value, (list, tuple)) else str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def extract_locators(tool_input: Mapping[str, Any] | None, *, max_value_chars: int = 200) -> dict[str, str]:
    """Return ``{key: value}`` for locator keys present in ``tool_input`` (deterministic order)."""
    if not tool_input:
        return {}
    out: dict[str, str] = {}
    for key in LOCATOR_KEYS:
        value = tool_input.get(key)
        if value in (None, "", [], ()):
            continue
        if isinstance(value, (str, int, float, list, tuple)):
            out[key] = _short(value, max_value_chars)
    return out


def extract_paths(tool_input: Mapping[str, Any] | None) -> list[str]:
    """File paths named in ``tool_input`` (``paths`` lists are flattened)."""
    if not tool_input:
        return []
    out: list[str] = []
    for key in PATH_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            out.append(value.strip())
        elif isinstance(value, (list, tuple)):
            out.extend(str(v).strip() for v in value if isinstance(v, str) and v.strip())
    return out


def find_offload_path(text: str) -> str | None:
    m = _OFFLOAD_PATH.search(text or "")
    return m.group(1) if m else None


@dataclass(frozen=True)
class ContextReference:
    """Something the model can load on demand.

    :param kind: ``file`` | ``url`` | ``query`` | ``artifact`` | ``memory``.
    :param locator: the path, URL or query string.
    :param how: how to load it (e.g. ``read_file(path, offset, limit)``).
    """

    kind: str
    locator: str
    description: str = ""
    size_hint: str = ""
    how: str = ""

    def render(self) -> str:
        parts = [f"- [{self.kind}] {self.locator}"]
        if self.size_hint:
            parts.append(f"({self.size_hint})")
        if self.description:
            parts.append(f"- {' '.join(self.description.split())}")
        if self.how:
            parts.append(f"[load: {self.how}]")
        return " ".join(parts)


def render_reference_index(refs: Iterable[ContextReference], *, title: str = "references") -> str:
    """Deduplicated, sorted ``<references>`` block (stable bytes for identical inputs)."""
    unique = {(r.kind, r.locator): r for r in refs}
    if not unique:
        return ""
    body = "\n".join(unique[k].render() for k in sorted(unique))
    return (
        f"<{title}>\nLoad these just in time with your tools; they are not in context.\n{body}\n</{title}>"
    )


__all__ = [
    "ContextReference",
    "LOCATOR_KEYS",
    "PATH_KEYS",
    "extract_locators",
    "extract_paths",
    "find_offload_path",
    "render_reference_index",
]
