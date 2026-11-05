"""Prompt-cache policy, breakpoint budget, stability checking and hit-rate monitoring.

Research basis (reference report section 3.3 techniques 5-7, checklist steps 1-3, pitfalls
"Cache breakers"):

* The Anthropic cache is a prefix match in render order ``tools -> system -> messages``; one
  changed byte invalidates everything after it. Agents are input-dominated (Manus: ~100:1
  input:output), so the cache-hit rate is "the single most important metric" in production.
* At most **4** ``cache_control`` breakpoints; reads look back at most **20 blocks**, so a turn
  that adds 20+ blocks needs a second (mid-history) breakpoint.
* Silent invalidators: timestamps / UUIDs / session ids in the system prompt, unsorted JSON,
  varying tool sets, switching effort/thinking mid-session. A breakpoint on a block containing a
  timestamp means "a write every request and never a read".
* Prompts below the model's minimum cacheable length (512-4,096 tokens) silently don't cache.
* Log "expected rebuilds" (compaction, clearing, deliberate tool changes) separately from
  unexpected misses, and treat unexpected misses as regressions.

:class:`CacheStabilityChecker` implements these checks over a finished
:class:`~harness_core.types.ModelRequest`; :class:`CacheMonitor` tracks ``usage`` to verify hits.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Pattern, Sequence

from harness_core.types import Message, ModelRequest, RunContext, Usage

from ._io import fingerprint
from .errors import BreakpointLimitError, CacheStabilityError, ConfigurationError
from .messages import count_breakpoints, message_fingerprint

MAX_BREAKPOINTS = 4
LOOKBACK_BLOCKS = 20
VALID_TTLS = (None, "5m", "1h")

#: (name, pattern, severity). "error" findings raise in strict mode; "warning" never raises.
VOLATILE_PATTERNS: tuple[tuple[str, Pattern[str], str], ...] = (
    ("iso_timestamp", re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?"), "error"),
    ("uuid", re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "error"),
    ("clock_time", re.compile(r"\b\d{1,2}:\d{2}:\d{2}\b"), "error"),
    ("unix_epoch", re.compile(r"\b1[6-9]\d{8}(?:\d{3})?\b"), "error"),
    ("run_id", re.compile(r"\b(?:session|sess|req|request|run|trace|msg|evt|call|toolu)[_-][0-9A-Za-z]{8,}\b"), "error"),
    ("date", re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b"), "warning"),
    ("relative_now", re.compile(r"\b(?:today is|current (?:date|time)|right now it is|as of now)\b", re.I), "warning"),
)


@dataclass(frozen=True)
class CachePolicy:
    """Where and how long to cache.

    Defaults: TTL ``1h`` for the interactive main thread and the 5-minute vendor default for
    sub-agents (``ctx.depth > 0``) *(recommended)*; max 4 breakpoints and a 20-block lookback
    *(vendor limits)*; the tools breakpoint is off because the system breakpoint already covers
    the tools prefix.
    """

    enabled: bool = True
    main_ttl: str | None = "1h"
    subagent_ttl: str | None = None
    max_breakpoints: int = MAX_BREAKPOINTS
    lookback_blocks: int = LOOKBACK_BLOCKS
    breakpoint_on_system: bool = True
    breakpoint_on_tail: bool = True
    breakpoint_on_compaction: bool = True
    breakpoint_mid_history: bool = True
    breakpoint_on_tools: bool = False
    min_cacheable_tokens: int = 1_024

    def __post_init__(self) -> None:
        for ttl in (self.main_ttl, self.subagent_ttl):
            if ttl not in VALID_TTLS:
                raise ConfigurationError(f"invalid cache TTL {ttl!r}; expected one of {VALID_TTLS}")
        if not 1 <= self.max_breakpoints <= MAX_BREAKPOINTS:
            raise ConfigurationError(f"max_breakpoints must be between 1 and {MAX_BREAKPOINTS}")

    def cache_control(self, ctx: RunContext) -> dict[str, Any]:
        ttl = self.subagent_ttl if ctx.depth > 0 else self.main_ttl
        cc: dict[str, Any] = {"type": "ephemeral"}
        if ttl and ttl != "5m":
            cc["ttl"] = ttl
        return cc


@dataclass(frozen=True)
class CacheFinding:
    kind: str           # pattern name, "breakpoint_limit", "below_min_cacheable", "volatile_breakpoint", ...
    segment: str        # "tools" | "system" | "messages" | "request"
    location: str       # e.g. "system[1]", "tool:read_file"
    excerpt: str
    severity: str       # "error" | "warning" | "info"


@dataclass(frozen=True)
class CacheBreak:
    """The prefix diverged from the previous request at ``segment`` (and ``index`` for messages)."""

    segment: str        # "tools" | "system" | "config" | "messages"
    index: int | None
    expected: bool
    reason: str


@dataclass(frozen=True)
class PrefixSnapshot:
    tools: str
    system: str
    config: str
    messages: tuple[str, ...]

    @classmethod
    def of(cls, request: ModelRequest, history: Sequence[Message] | None = None) -> "PrefixSnapshot":
        system = request.system if isinstance(request.system, str) else [
            {k: v for k, v in b.items() if k != "cache_control"} for b in request.system
        ]
        tools = [{k: v for k, v in t.to_api().items()} for t in request.tools]
        msgs = request.messages if history is None else history
        config = {k: v for k, v in request.extra.items()}
        config["max_tokens"] = request.max_tokens
        return cls(
            tools=fingerprint(tools),
            system=fingerprint(system),
            config=fingerprint(config),
            messages=tuple(message_fingerprint(m) for m in msgs),
        )


def _excerpt(text: str, start: int, end: int, width: int = 30) -> str:
    a, b = max(0, start - width), min(len(text), end + width)
    return text[a:b].replace("\n", " ")


class CacheStabilityChecker:
    """Detect silent cache invalidators in a request and prefix divergence between requests.

    :param strict: raise :class:`CacheStabilityError` on ``error`` findings in the cached prefix.
    :param allow: substrings that are known-stable and should not be flagged (e.g. a pinned date).
    """

    def __init__(
        self,
        *,
        strict: bool = False,
        patterns: Sequence[tuple[str, Pattern[str], str]] = VOLATILE_PATTERNS,
        allow: Iterable[str] = (),
        max_breakpoints: int = MAX_BREAKPOINTS,
    ) -> None:
        self.strict = strict
        self.patterns = tuple(patterns)
        self.allow = tuple(allow)
        self.max_breakpoints = max_breakpoints

    # ------------------------------------------------------------------ static scan
    def scan_text(self, segment: str, location: str, text: str) -> list[CacheFinding]:
        findings = []
        for name, pattern, severity in self.patterns:
            for m in pattern.finditer(text):
                if any(a in text[max(0, m.start() - 40): m.end() + 40] for a in self.allow):
                    continue
                findings.append(CacheFinding(name, segment, location, _excerpt(text, m.start(), m.end()), severity))
                break  # one finding per pattern per location is enough
        return findings

    def scan_request(self, request: ModelRequest, *, prefix_tokens: int | None = None, min_cacheable_tokens: int | None = None) -> list[CacheFinding]:
        findings: list[CacheFinding] = []
        names = [t.name for t in request.tools]
        if len(set(names)) != len(names):
            findings.append(CacheFinding("duplicate_tool", "tools", "tools", ",".join(sorted(names)), "error"))
        for t in request.tools:
            findings += self.scan_text("tools", f"tool:{t.name}", t.description)
            if t.defer_loading and t.metadata.get("cache_control"):
                findings.append(CacheFinding("deferred_tool_breakpoint", "tools", f"tool:{t.name}", "deferred tools cannot carry cache_control", "error"))
        n_bp = sum(1 for t in request.tools if t.metadata.get("cache_control"))
        if isinstance(request.system, str):
            findings += self.scan_text("system", "system", request.system)
        else:
            for i, b in enumerate(request.system):
                findings += self.scan_text("system", f"system[{i}]", str(b.get("text", "")))
                if b.get("cache_control"):
                    n_bp += 1
        n_bp += count_breakpoints(request.messages)
        if n_bp > self.max_breakpoints:
            findings.append(CacheFinding("breakpoint_limit", "request", "request", f"{n_bp} breakpoints > {self.max_breakpoints}", "error"))
        for mi, m in enumerate(request.messages):
            for bi, b in enumerate(m.content):
                cc = getattr(b, "cache_control", None)
                text = getattr(b, "text", None)
                if cc and isinstance(text, str):
                    for f in self.scan_text("messages", f"messages[{mi}].content[{bi}]", text):
                        if f.severity == "error":
                            findings.append(CacheFinding("volatile_breakpoint", f.segment, f.location, f.excerpt, "error"))
        if prefix_tokens is not None and min_cacheable_tokens and prefix_tokens < min_cacheable_tokens:
            findings.append(
                CacheFinding(
                    "below_min_cacheable", "system", "system",
                    f"cached prefix ~{prefix_tokens} tokens < model minimum {min_cacheable_tokens}; it will silently not cache",
                    "info",
                )
            )
        return findings

    def enforce(self, findings: Sequence[CacheFinding]) -> None:
        """Raise on hard limits always, and on prefix ``error`` findings in strict mode."""
        for f in findings:
            if f.kind == "breakpoint_limit":
                raise BreakpointLimitError(f.excerpt)
        if self.strict:
            errors = [f for f in findings if f.severity == "error" and f.segment in ("tools", "system")]
            if errors:
                detail = "; ".join(f"{f.kind} at {f.location}: {f.excerpt!r}" for f in errors)
                raise CacheStabilityError(f"cache-unstable prefix: {detail}")

    # ------------------------------------------------------------------ diff between requests
    @staticmethod
    def compare(previous: PrefixSnapshot | None, current: PrefixSnapshot, *, expected_reason: str | None = None) -> list[CacheBreak]:
        if previous is None:
            return []
        expected = expected_reason is not None
        why = expected_reason or ""
        out: list[CacheBreak] = []
        if previous.tools != current.tools:
            out.append(CacheBreak("tools", None, expected, why or "tool definitions changed (invalidates everything)"))
        if previous.system != current.system:
            out.append(CacheBreak("system", None, expected, why or "system prompt / project context changed"))
        if previous.config != current.config:
            out.append(CacheBreak("config", None, expected, why or "request config (effort/thinking/context_management/max_tokens) changed"))
        for i, fp in enumerate(previous.messages):
            if i >= len(current.messages) or current.messages[i] != fp:
                out.append(CacheBreak("messages", i, expected, why or f"history diverged at window message {i} (an earlier turn was edited or dropped)"))
                break
        return out


@dataclass
class CacheStats:
    requests: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    uncached_input_tokens: int = 0
    expected_rebuilds: int = 0
    unexpected_misses: int = 0
    miss_reasons: list[str] = field(default_factory=list)

    @property
    def total_input_tokens(self) -> int:
        return self.cache_read_tokens + self.cache_write_tokens + self.uncached_input_tokens

    @property
    def hit_rate(self) -> float:
        """``cache_read / total_input`` (recommended target > 0.8-0.9)."""
        total = self.total_input_tokens
        return self.cache_read_tokens / total if total else 0.0


class CacheMonitor:
    """Thread-safe usage observer that separates expected rebuilds from unexpected misses."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.stats = CacheStats()

    def observe(self, usage: Usage, *, expected_rebuild: str | None = None) -> bool:
        """Record one response's usage. Returns ``True`` when it was an *unexpected* miss."""
        with self._lock:
            s = self.stats
            first = s.requests == 0
            s.requests += 1
            s.cache_read_tokens += usage.cache_read_input_tokens
            s.cache_write_tokens += usage.cache_creation_input_tokens
            s.uncached_input_tokens += usage.input_tokens
            missed = usage.cache_read_input_tokens == 0 and usage.total_input > 0
            if not missed or first:
                return False
            if expected_rebuild:
                s.expected_rebuilds += 1
                return False
            s.unexpected_misses += 1
            s.miss_reasons.append(f"request {s.requests}: no cache read ({usage.total_input} input tokens)")
            return True


__all__ = [
    "CacheBreak",
    "CacheFinding",
    "CacheMonitor",
    "CachePolicy",
    "CacheStabilityChecker",
    "CacheStats",
    "LOOKBACK_BLOCKS",
    "MAX_BREAKPOINTS",
    "PrefixSnapshot",
    "VOLATILE_PATTERNS",
]
