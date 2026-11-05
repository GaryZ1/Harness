"""Searchable tool catalog: deferred loading behind a ``tool_search`` tool.

Research basis (``reports/SOTA agentic harness reference.md`` §5.3 B, §5.4, §5.7 step 10, §5.8
"Tool-search misconfiguration"):

* Five MCP servers with 58 tools cost ~55K definition tokens; Claude's tool selection "degrades
  once you exceed 30-50 available tools". Deferring tools behind search cut definition tokens
  by >85% and lifted MCP-eval accuracy (Opus 4.5: 79.5% -> 88.1%).
* **When**: switch tool search on past 10 tools or ~10K definition tokens
  (:func:`should_enable_tool_search`); keep 3-5 hot tools (:func:`apply_tiering`).
* **Invariants**: never defer the search tool; keep at least one non-deferred tool; never put
  ``cache_control`` on deferred tools (enforced by :meth:`ToolRegistry.validate_tool_array` and
  :func:`apply_tiering`).
* **Two search back-ends** mirroring Anthropic's server tools: BM25 over natural language
  (queries <= 500 chars) and Python regex (patterns <= 200 chars), both over tool names,
  descriptions, argument names and argument descriptions; default 5 results.
* **Two delivery modes**: ``reference`` (Anthropic: deferred tools stay in the request with
  ``defer_loading: true`` and the client-side search returns ``tool_reference`` blocks, which the
  API expands) and ``reveal`` (any provider: deferred tools are omitted until found, then their
  full definitions are returned as text and appended to the tool array).
* :func:`server_tool_search_spec` emits the server-side tool instead
  (``tool_search_tool_bm25_20251119`` / ``tool_search_tool_regex_20251119``).
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

from harness_core.types import ToolOutcome, ToolSpec
from harness_memory.retrieval import BM25Index

from . import descriptions as D
from .errors import ToolDefinitionError, ToolInputError
from .output import estimate_tokens
from .registry import TOOL_SEARCH_NAME, Tool, ToolInvocation, ToolRegistry

SearchMode = Literal["bm25", "regex"]
DEFAULT_SEARCH_LIMIT = 5
MAX_SEARCH_LIMIT = 20
BM25_MAX_QUERY = 500
REGEX_MAX_PATTERN = 200
#: Thresholds from Anthropic's guidance ("10+ tools or >10K tokens of definitions").
TOOL_COUNT_THRESHOLD = 10
DEFINITION_TOKEN_THRESHOLD = 10_000

SERVER_TOOL_SEARCH = {
    "bm25": ("tool_search_tool_bm25_20251119", "tool_search_tool_bm25"),
    "regex": ("tool_search_tool_regex_20251119", "tool_search_tool_regex"),
}

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WORD = re.compile(r"[A-Za-z0-9]+")


def _stem(word: str) -> str:
    for suffix in ("ing", "ies", "es", "ed", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def normalise_text(text: str) -> str:
    """Split identifiers (``mcp__github__createIssue`` -> ``mcp github create issue``) and stem lightly."""
    spaced = _CAMEL.sub(" ", text).replace("_", " ").replace("-", " ").replace(".", " ")
    return " ".join(_stem(w.lower()) for w in _WORD.findall(spaced))


def searchable_text(spec: ToolSpec) -> str:
    """Name, description, argument names and argument descriptions (Anthropic's search fields)."""
    parts = [spec.name, spec.description]
    for name, prop in ((spec.input_schema or {}).get("properties") or {}).items():
        parts.append(name)
        if isinstance(prop, dict) and isinstance(prop.get("description"), str):
            parts.append(prop["description"])
    return "\n".join(parts)


def definition_tokens(specs: Iterable[ToolSpec]) -> int:
    """Estimated tokens the given tool definitions cost in the prompt."""
    return sum(estimate_tokens(json.dumps(s.to_api(), ensure_ascii=False)) for s in specs)


def should_enable_tool_search(
    specs: Sequence[ToolSpec], *, count_threshold: int = TOOL_COUNT_THRESHOLD, token_threshold: int = DEFINITION_TOKEN_THRESHOLD
) -> bool:
    """True once there are ``count_threshold``+ tools or definitions exceed ``token_threshold`` tokens."""
    return len(specs) >= count_threshold or definition_tokens(specs) > token_threshold


@dataclass(frozen=True)
class SearchHit:
    name: str
    score: float


class ToolCatalog:
    """BM25 / regex search over a registry's deferred tools. The index rebuilds when the set changes."""

    def __init__(self, registry: ToolRegistry, *, include_loaded: bool = False) -> None:
        self.registry = registry
        self.include_loaded = include_loaded
        self._lock = threading.Lock()
        self._key: str | None = None
        self._index = BM25Index()
        self._texts: dict[str, str] = {}

    def _candidates(self) -> list[Tool]:
        tools = self.registry.tools()
        return [
            t for t in tools
            if t.name != TOOL_SEARCH_NAME and t.handler is not None and (self.include_loaded or t.spec.defer_loading)
        ]

    def _refresh(self) -> dict[str, str]:
        cands = self._candidates()
        key = hashlib.sha256("\x1e".join(f"{t.name}\x1f{t.spec.description}" for t in cands).encode()).hexdigest()
        with self._lock:
            if key != self._key:
                self._index = BM25Index()
                self._texts = {}
                for t in cands:
                    text = searchable_text(t.spec)
                    self._texts[t.name] = text
                    self._index.add(t.name, normalise_text(text))
                self._key = key
            return dict(self._texts)

    def search_bm25(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[SearchHit]:
        if len(query) > BM25_MAX_QUERY:
            raise ToolInputError(f"query is {len(query)} characters; the limit is {BM25_MAX_QUERY}")
        self._refresh()
        with self._lock:
            hits = self._index.search(normalise_text(query), k=limit)
        return [SearchHit(n, s) for n, s in hits]

    def search_regex(self, pattern: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[SearchHit]:
        if len(pattern) > REGEX_MAX_PATTERN:
            raise ToolInputError(f"regex is {len(pattern)} characters; the limit is {REGEX_MAX_PATTERN}")
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise ToolInputError(f"invalid regular expression {pattern!r}: {exc}") from exc
        texts = self._refresh()
        hits: list[SearchHit] = []
        for name in sorted(texts):
            if rx.search(name):
                hits.append(SearchHit(name, 2.0))
            elif rx.search(texts[name]):
                hits.append(SearchHit(name, 1.0))
        hits.sort(key=lambda h: (-h.score, h.name))
        return hits[:limit]

    def search(self, query: str, *, mode: SearchMode = "bm25", limit: int = DEFAULT_SEARCH_LIMIT) -> list[SearchHit]:
        limit = max(1, min(int(limit), MAX_SEARCH_LIMIT))
        return self.search_regex(query, limit) if mode == "regex" else self.search_bm25(query, limit)


def _render_definition(spec: ToolSpec) -> str:
    schema = json.dumps(spec.input_schema, ensure_ascii=False, separators=(",", ":"))
    return f"## {spec.name}\n{spec.description}\ninput_schema: {schema}"


def enable_tool_search(
    registry: ToolRegistry,
    *,
    default_mode: SearchMode = "bm25",
    default_limit: int = DEFAULT_SEARCH_LIMIT,
    include_loaded: bool = False,
) -> ToolCatalog:
    """Register the client-side ``tool_search`` tool (never deferred) and return its catalog."""
    catalog = ToolCatalog(registry, include_loaded=include_loaded)

    def handler(inp: dict[str, Any], inv: ToolInvocation) -> ToolOutcome:
        query = str(inp["query"]).strip()
        if not query:
            raise ToolInputError("query is empty; describe the capability you need, e.g. 'send a slack message'")
        mode: SearchMode = inp.get("mode", default_mode)
        hits = catalog.search(query, mode=mode, limit=int(inp.get("limit", default_limit)))
        if not hits:
            return ToolOutcome(
                f"No tools matched {query!r}. Try other keywords (the service name, the object, the action)"
                + (" or mode='bm25' with a natural-language query." if mode == "regex" else " or a regex with mode='regex'."),
                metadata={"matched": []},
            )
        names = [h.name for h in hits]
        registry.discover(names)
        if registry.deferral == "reference":
            return ToolOutcome([{"type": "tool_reference", "tool_name": n} for n in names], metadata={"matched": names})
        defs = "\n\n".join(_render_definition(registry.get(n).spec) for n in names if registry.get(n) is not None)  # type: ignore[union-attr]
        return ToolOutcome(f"Loaded {len(names)} tool(s); you can call them now.\n\n{defs}", metadata={"matched": names})

    spec = ToolSpec(
        name=TOOL_SEARCH_NAME,
        description=D.TOOL_SEARCH,
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": BM25_MAX_QUERY, "description": "What you need, or a regex when mode='regex'."},
                "mode": {"type": "string", "enum": ["bm25", "regex"], "default": default_mode},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_SEARCH_LIMIT, "default": default_limit, "description": "Maximum tools to load."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        read_only=True,
    )
    registry.register(Tool(spec=spec, handler=handler, concurrency_safe=False, idempotent=True, timeout_s=10.0))
    return catalog


def server_tool_search_spec(variant: SearchMode = "bm25") -> ToolSpec:
    """The Anthropic server-side tool search tool (executed by the API, not locally)."""
    try:
        type_, name = SERVER_TOOL_SEARCH[variant]
    except KeyError as exc:
        raise ValueError(f"variant must be one of {sorted(SERVER_TOOL_SEARCH)}") from exc
    return ToolSpec(name=name, description="", provider_type=type_, read_only=True)


def server_tool_search_definition(variant: SearchMode = "bm25") -> dict[str, Any]:
    """``{"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}`` (or regex)."""
    return server_tool_search_spec(variant).to_api()


def apply_tiering(registry: ToolRegistry, hot: Iterable[str]) -> list[str]:
    """Defer every tool except ``hot`` and the search tools; returns the names that were deferred.

    Must run before :meth:`ToolRegistry.freeze`. Raises if nothing would remain loaded, or if a
    hot name is unknown. ``cache_control`` is stripped from deferred specs' metadata.
    """
    if registry.frozen:
        raise ToolDefinitionError("apply_tiering must run before the registry is frozen")
    hot_set = set(hot)
    unknown = hot_set - set(registry.names())
    if unknown:
        raise ToolDefinitionError(f"unknown hot tool(s): {sorted(unknown)}")
    deferred: list[str] = []
    for tool in registry.tools():
        spec = tool.spec
        is_search = spec.name == TOOL_SEARCH_NAME or (spec.provider_type or "").startswith("tool_search_tool_")
        if spec.name in hot_set or is_search:
            spec.defer_loading = False
            continue
        if not spec.defer_loading:
            spec.defer_loading = True
            spec.metadata.pop("cache_control", None)
            deferred.append(spec.name)
    registry.validate_tool_array()
    return deferred


__all__ = [
    "BM25_MAX_QUERY",
    "DEFAULT_SEARCH_LIMIT",
    "DEFINITION_TOKEN_THRESHOLD",
    "REGEX_MAX_PATTERN",
    "SERVER_TOOL_SEARCH",
    "SearchHit",
    "SearchMode",
    "TOOL_COUNT_THRESHOLD",
    "ToolCatalog",
    "apply_tiering",
    "definition_tokens",
    "enable_tool_search",
    "normalise_text",
    "searchable_text",
    "server_tool_search_definition",
    "server_tool_search_spec",
    "should_enable_tool_search",
]
