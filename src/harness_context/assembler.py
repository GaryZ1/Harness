"""``LayeredContextAssembler``: the durable log -> model window projection.

Implements :class:`harness_core.types.ContextAssembler`. Each :meth:`prepare` call builds a fresh,
*derived* window in the fixed layout of the build checklist (reference report 3.7 step 2)::

    tools (canonical JSON, sorted)                         [BP optional]
    system: base prompt + SYSTEM layers + PROJECT layers   [BP: end of system/project]
    messages:
      [compaction marker: summary + re-injection]          [BP: compaction]
      kept tail + later turns, verbatim (thinking kept)
      ... sticky restorable tool-result placeholders ...   [BP: mid-history if >= 20 blocks since last]
      last durable message                                 [BP: moving tail]
      dynamic tail: budget note, warnings, recitation      (turn-scoped, never cached, never logged)

and walks the compaction ladder cheapest-first: L0 ingest cap (safety net), L2 sticky tool-result
clearing (client side, or server ``clear_tool_uses`` via ``extra``), L3 compaction (client
:class:`~harness_context.compaction.Compactor`, or server ``compact_20260112``), L4 re-injection.

Invariants enforced in code:

* the durable ``messages`` list is **never mutated** except by *appending* a compaction marker or
  re-injection bundle (via ``append_to_log``, default ``messages.append``), and appends happen
  only when they cannot split a tool_use/tool_result pair;
* every derived message keeps its durable id; provider blocks (thinking, compaction) are copied
  verbatim;
* at most 4 cache breakpoints (:class:`~harness_context.errors.BreakpointLimitError`);
* compaction markers are validated against a hash of the log prefix they cover
  (:class:`~harness_context.errors.LogIntegrityError`);
* auto-compaction is disabled when the window is unknown, while one is in flight, and after
  ``max_ineffective_compactions`` consecutive compactions that did not get under the trigger
  (thrash guard).

Thread-safe: per-session state is guarded by a re-entrant lock.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from harness_core.types import (
    Event,
    EventSink,
    EventType,
    Message,
    ModelRequest,
    ModelResponse,
    NullTracer,
    Role,
    RunContext,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    Tracer,
)

from ._io import atomic_write_text, canonical_json, sort_keys_deep
from .budget import PRE_COMPACTION_WARNING, BudgetLevel, BudgetPolicy, UsageCalibration
from .cache import CacheBreak, CacheFinding, CacheMonitor, CachePolicy, CacheStabilityChecker, PrefixSnapshot
from .clearing import ClearingPolicy, ToolResultClearer
from .compaction import CompactionResult, Compactor
from .dynamic import DynamicContext, DynamicItem, safe_to_append
from .errors import (
    CompactionError,
    CompactionInProgressError,
    ConfigurationError,
    ContextError,
    ContextOverflowError,
    LogIntegrityError,
    NothingToCompactError,
)
from .guards import StuckDetector, tool_count_warning
from .layers import LayerKind, LayerLint, PromptStack
from .messages import (
    KIND_COMPACTION,
    KIND_REINJECTION,
    META_KEY,
    context_meta,
    copy_block,
    copy_message,
    has_server_compaction,
    is_compaction_marker,
    kind_of,
    log_prefix_sha256,
    merge_consecutive_user,
    with_breakpoint,
)
from .offload import cap_text
from .reinjection import ReinjectionBuilder
from .server import ServerContextManagement
from .tokens import HeuristicTokenCounter, TokenCounter, estimate_block, estimate_message


def canonical_tools(tools: Sequence[ToolSpec], *, sort: bool = True) -> list[ToolSpec]:
    """Copies with key-sorted schemas, harness cache markers stripped, sorted by name; duplicates rejected."""
    names = [t.name for t in tools]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ConfigurationError(f"duplicate tool names: {dupes}")
    out = [
        dataclasses.replace(
            t,
            input_schema=sort_keys_deep(t.input_schema),
            metadata={k: v for k, v in t.metadata.items() if k != "cache_control"},
        )
        for t in tools
    ]
    return sorted(out, key=lambda t: t.name) if sort else out


@dataclass(frozen=True)
class AssemblerConfig:
    budget: BudgetPolicy = field(default_factory=BudgetPolicy)
    cache: CachePolicy = field(default_factory=CachePolicy)
    clearing: ClearingPolicy | None = field(default_factory=ClearingPolicy)   # None = no client clearing
    server: ServerContextManagement | None = None
    auto_compact: bool = True
    max_ineffective_compactions: int = 3
    max_compaction_failures: int = 3
    max_tokens: int = 16_000
    request_extra: dict[str, Any] = field(default_factory=dict)   # pinned per session (effort, thinking...)
    sort_tools: bool = True
    merge_consecutive_user: bool = True
    budget_note: bool = True
    pre_compaction_warning: bool = True
    stuck_detection: bool = True
    max_result_tokens: int | None = 25_000
    strict_cache: bool = False
    raise_on_overflow: bool = True

    def __post_init__(self) -> None:
        if self.server and self.server.clear_tool_uses and self.clearing is not None:
            raise ConfigurationError("choose client-side clearing (clearing=...) or server clear_tool_uses, not both")
        if self.max_ineffective_compactions < 1 or self.max_compaction_failures < 1:
            raise ConfigurationError("compaction guard limits must be >= 1")


@dataclass
class ContextReport:
    """``/context``-style view of the last window (per-category token estimates)."""

    session_id: str
    categories: dict[str, int]
    estimated_tokens: int
    level: BudgetLevel
    compaction_trigger: int
    window_tokens: int | None
    breakpoints: list[str]
    cleared_tool_results: int
    compactions: int
    actions: list[str]
    findings: list[CacheFinding]
    cache_breaks: list[CacheBreak]
    lints: list[LayerLint]

    def render(self) -> str:
        total = max(1, sum(self.categories.values()))
        lines = [f"Context for {self.session_id}: ~{self.estimated_tokens:,} tokens ({self.level.value}); "
                 f"compaction at ~{self.compaction_trigger:,}"
                 + (f" of {self.window_tokens:,}" if self.window_tokens else "")]
        for name, n in self.categories.items():
            lines.append(f"  {name:<20} {n:>10,}  {100 * n / total:5.1f}%")
        lines.append(f"  breakpoints: {', '.join(self.breakpoints) or 'none'}")
        lines.append(f"  cleared tool results: {self.cleared_tool_results}; compactions: {self.compactions}")
        for a in self.actions:
            lines.append(f"  action: {a}")
        for f in self.findings:
            lines.append(f"  cache {f.severity}: {f.kind} at {f.location}: {f.excerpt}")
        for b in self.cache_breaks:
            lines.append(f"  cache break ({'expected' if b.expected else 'UNEXPECTED'}): {b.segment} - {b.reason}")
        for la in self.lints:
            lines.append(f"  lint {la.severity}: {la.layer}: {la.message}")
        return "\n".join(lines)


@dataclass
class _SessionState:
    cleared: set[str] = field(default_factory=set)
    duplicates: dict[str, str] = field(default_factory=dict)
    warned: bool = False
    ineffective: int = 0
    failures: int = 0
    auto_disabled: str | None = None
    last_tail_id: str | None = None
    seen_server: set[str] = field(default_factory=set)
    pending_rebuild: str | None = None
    last_rebuild: str | None = None
    snapshot: PrefixSnapshot | None = None
    last_raw_estimate: int = 0
    calibration: UsageCalibration = field(default_factory=UsageCalibration)
    observed: int = 0  # responses observed by this process (first one is a cache write by definition)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cleared": sorted(self.cleared), "duplicates": dict(sorted(self.duplicates.items())), "warned": self.warned,
            "ineffective": self.ineffective, "failures": self.failures, "auto_disabled": self.auto_disabled,
            "last_tail_id": self.last_tail_id, "seen_server": sorted(self.seen_server),
            "calibration_ratio": self.calibration.ratio,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "_SessionState":
        s = cls()
        s.cleared = set(d.get("cleared", []))
        s.duplicates = dict(d.get("duplicates", {}))
        s.warned = bool(d.get("warned", False))
        s.ineffective = int(d.get("ineffective", 0))
        s.failures = int(d.get("failures", 0))
        s.auto_disabled = d.get("auto_disabled")
        s.last_tail_id = d.get("last_tail_id")
        s.seen_server = set(d.get("seen_server", []))
        s.calibration = UsageCalibration(float(d.get("calibration_ratio", 1.0)))
        # A restarted process has no snapshot: the first window is treated as an expected rebuild.
        return s


class LayeredContextAssembler:
    """Production :class:`~harness_core.types.ContextAssembler`.

    :param stack: stable system/project layers rendered after the caller's base system prompt.
    :param compactor: client-side compactor (``None`` = no client compaction).
    :param reinjection: bundle rebuilt after *server-side* compaction (client compaction carries
        its own via the compactor).
    :param append_to_log: how to append a marker/bundle to the durable log
        (default: ``messages.append`` on the list passed to :meth:`prepare`).
    :param state_dir: persist per-session clearing/guard state atomically for resume.
    """

    def __init__(
        self,
        *,
        stack: PromptStack | None = None,
        config: AssemblerConfig | None = None,
        counter: TokenCounter | None = None,
        compactor: Compactor | None = None,
        reinjection: ReinjectionBuilder | None = None,
        dynamic: DynamicContext | None = None,
        events: EventSink | None = None,
        tracer: Tracer | None = None,
        append_to_log: Callable[[list[Message], Message], None] | None = None,
        state_dir: str | Path | None = None,
        checker: CacheStabilityChecker | None = None,
        stuck_detector: StuckDetector | None = None,
    ) -> None:
        self.config = config or AssemblerConfig()
        if self.config.server and self.config.server.compact and compactor is not None and self.config.auto_compact:
            raise ConfigurationError("server compaction and client auto-compaction are mutually exclusive")
        self.stack = stack or PromptStack()
        self.counter = counter or HeuristicTokenCounter()
        self.compactor = compactor
        self.reinjection = reinjection
        self.dynamic = dynamic or DynamicContext()
        self.events = events
        self.tracer = tracer or NullTracer()
        self._append = append_to_log or (lambda log, msg: log.append(msg))
        self.state_dir = Path(state_dir) if state_dir else None
        self.checker = checker or CacheStabilityChecker(strict=self.config.strict_cache, max_breakpoints=self.config.cache.max_breakpoints)
        self.stuck = stuck_detector or StuckDetector()
        self.clearer = ToolResultClearer(self.config.clearing, self.counter) if self.config.clearing else None
        self.cache_monitor = CacheMonitor()
        self._states: dict[str, _SessionState] = {}
        self._reports: dict[str, ContextReport] = {}
        self._verified_markers: set[str] = set()
        self._registry = threading.Lock()

    # ------------------------------------------------------------------ state
    def _state(self, session_id: str) -> _SessionState:
        with self._registry:
            st = self._states.get(session_id)
            if st is None:
                st = self._load_state(session_id) or _SessionState()
                self._states[session_id] = st
            return st

    def _state_path(self, session_id: str) -> Path | None:
        if self.state_dir is None:
            return None
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in session_id)
        return self.state_dir / f"{safe}.context-state.json"

    def _load_state(self, session_id: str) -> _SessionState | None:
        path = self._state_path(session_id)
        if path is None or not path.is_file():
            return None
        try:
            return _SessionState.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return None

    def _save_state(self, session_id: str, st: _SessionState) -> None:
        path = self._state_path(session_id)
        if path is not None:
            atomic_write_text(path, json.dumps(st.to_dict(), sort_keys=True, indent=1))

    def _emit(self, ctx: RunContext, type_: EventType, data: dict[str, Any]) -> None:
        if self.events is not None:
            self.events.emit(Event(ctx.session_id, type_, data))

    # ------------------------------------------------------------------ projection
    def derive_history(self, messages: Sequence[Message]) -> list[Message]:
        """``[latest compaction marker] + messages after the message it covers`` (copies, markers stripped of cache)."""
        idx = next((i for i in range(len(messages) - 1, -1, -1) if is_compaction_marker(messages[i])), None)
        if idx is None:
            return [copy_message(m) for m in messages if kind_of(m) != KIND_COMPACTION]
        marker = messages[idx]
        meta = context_meta(marker)
        covers = meta.get("covers_through")
        pos = next((i for i in range(idx - 1, -1, -1) if messages[i].id == covers), None)
        if pos is None:
            raise LogIntegrityError(f"compaction marker {marker.id} covers {covers!r}, which is not before it in the log")
        with self._registry:
            verified = marker.id in self._verified_markers
        if not verified:
            expected = meta.get("prefix_sha256")
            if expected and log_prefix_sha256(messages, pos) != expected:
                raise LogIntegrityError(f"messages covered by compaction marker {marker.id} were modified after compaction")
            with self._registry:
                self._verified_markers.add(marker.id)
        tail = [copy_message(m) for m in messages[pos + 1:] if not is_compaction_marker(m)]
        return [copy_message(marker)] + tail

    def _cap(self, history: list[Message]) -> list[Message]:
        cap = self.config.max_result_tokens
        if not cap:
            return history
        out = []
        for m in history:
            if any(estimate_block(self.counter, r) > cap for r in m.tool_results):
                blocks = []
                for b in m.content:
                    if isinstance(b, ToolResultBlock) and isinstance(b.content, str) and estimate_block(self.counter, b) > cap:
                        text, _ = cap_text(b.content, cap, self.counter)
                        b = ToolResultBlock(b.tool_use_id, text, b.is_error)
                    blocks.append(b)
                m = copy_message(m, blocks)
            out.append(m)
        return out

    def _estimate(self, system_api: list[dict[str, Any]], history: list[Message], tools: list[ToolSpec], st: _SessionState) -> tuple[int, int]:
        start = max((i for i, m in enumerate(history) if has_server_compaction(m)), default=0)
        raw = self.counter.count_request(ModelRequest(system=system_api, messages=history[start:], tools=tools))
        return raw, st.calibration.apply(raw)

    # ------------------------------------------------------------------ protocol entry point
    def prepare(self, system: str, messages: list[Message], tools: list[ToolSpec], ctx: RunContext) -> ModelRequest:
        st = self._state(ctx.session_id)
        with st.lock, self.tracer.span("context.prepare", session_id=ctx.session_id):
            before = st.to_dict()
            request = self._prepare(system, messages, tools, ctx, st)
            if st.to_dict() != before:
                self._save_state(ctx.session_id, st)
            return request

    def _prepare(self, system: str, messages: list[Message], tools: list[ToolSpec], ctx: RunContext, st: _SessionState) -> ModelRequest:
        cfg = self.config
        actions: list[str] = []
        self._handle_server_compaction(messages, ctx, st, actions)
        tools_c = canonical_tools(tools, sort=cfg.sort_tools)
        rendered = self.stack.render(system)
        system_api = [b.to_api() for b in rendered.blocks]

        history = self._cap(self.derive_history(messages))
        if self.clearer:
            history = self.clearer.apply(history, st.cleared, duplicate_of=st.duplicates)
        raw, tokens = self._estimate(system_api, history, tools_c, st)

        # L2: tool-result clearing (sticky, only when it frees >= clear_at_least)
        if self.clearer and tokens >= self.clearer.policy.trigger_tokens:
            plan = self.clearer.plan(history, st.cleared)
            if plan.applied:
                st.cleared.update(plan.cleared_ids)
                st.duplicates.update(plan.duplicate_of)
                history = self.clearer.apply(history, st.cleared, duplicate_of=st.duplicates)
                before_tokens = tokens
                raw, tokens = self._estimate(system_api, history, tools_c, st)
                st.pending_rebuild = "tool_result_clearing"
                actions.append(f"cleared {len(plan.cleared_ids)} tool results (~{plan.freed_tokens:,} tokens)")
                self._emit(ctx, EventType.COMPACTION, {
                    "kind": "tool_result_clearing", "cleared": plan.cleared_ids, "freed_tokens": plan.freed_tokens,
                    "tokens_before": before_tokens, "tokens_after": tokens,
                })

        # L3: client compaction
        if self._should_auto_compact(tokens, st):
            result = self._auto_compact(system_api, messages, history, tools_c, ctx, st, actions)
            if result is not None:
                history = self._cap(self.derive_history(messages))
                ids = {r.tool_use_id for m in history for r in m.tool_results}
                st.cleared &= ids
                st.duplicates = {k: v for k, v in st.duplicates.items() if k in st.cleared}
                if self.clearer:
                    history = self.clearer.apply(history, st.cleared, duplicate_of=st.duplicates)
                raw, tokens = self._estimate(system_api, history, tools_c, st)
                if tokens >= cfg.budget.compaction_trigger:
                    st.ineffective += 1
                    if st.ineffective >= cfg.max_ineffective_compactions:
                        st.auto_disabled = f"{st.ineffective} consecutive compactions did not get below the trigger"
                        self._emit(ctx, EventType.ERROR, {"kind": "compaction_thrash", "reason": st.auto_disabled})
                else:
                    st.ineffective = 0

        level = cfg.budget.classify(tokens, max_tokens=cfg.max_tokens)
        if level is BudgetLevel.OVERFLOW and cfg.raise_on_overflow:
            self._emit(ctx, EventType.ERROR, {"kind": "context_overflow", "tokens": tokens, "window": cfg.budget.window_tokens})
            raise ContextOverflowError(
                f"window needs ~{tokens:,} + {cfg.max_tokens:,} output tokens > effective window "
                f"{cfg.budget.effective_window:,}; compact or start a fresh context"
            )

        # Dynamic tail (turn-scoped; after the moving breakpoint)
        extra_items: list[DynamicItem] = []
        if cfg.pre_compaction_warning and level is not BudgetLevel.OK and not st.warned:
            extra_items.append(DynamicItem("pre-compaction-warning", PRE_COMPACTION_WARNING, priority=50))
            st.warned = True
        if cfg.budget_note:
            extra_items.append(DynamicItem("budget", cfg.budget.note(tokens), priority=900))
        if cfg.stuck_detection:
            note = self.stuck.check(history)
            if note:
                extra_items.append(DynamicItem("loop-check", note, priority=60))
        tail_msg = self.dynamic.render(self.dynamic.collect(ctx, extra_items))

        # Cache breakpoints (<= max_breakpoints, priority: system, tail, mid, compaction, tools)
        bp_desc: list[str] = []
        sys_blocks = list(rendered.blocks)
        tools_out = list(tools_c)
        if cfg.cache.enabled:
            cc = cfg.cache.cache_control(ctx)
            wanted = self._plan_breakpoints(history, sys_blocks, tools_out, st)
            chosen = wanted[: cfg.cache.max_breakpoints]
            for kind, target in chosen:
                if kind == "system":
                    sys_blocks[target] = TextBlock(sys_blocks[target].text, cache_control=cc)
                elif kind == "tools":
                    t = tools_out[target]
                    tools_out[target] = dataclasses.replace(t, metadata={**t.metadata, "cache_control": cc})
                else:
                    marked = with_breakpoint(history[target], cc)
                    if marked is not None:
                        history[target] = marked
                bp_desc.append(f"{kind}@{target}")
            tail_idx = next((t for k, t in chosen if k == "tail"), None)
            st.last_tail_id = history[tail_idx].id if tail_idx is not None else st.last_tail_id

        window = history + ([tail_msg] if tail_msg is not None else [])
        if cfg.merge_consecutive_user:
            window = merge_consecutive_user(window)
        extra = dict(cfg.request_extra)
        if cfg.server is not None:
            extra = cfg.server.apply(extra)
        request = ModelRequest(
            system=[b.to_api() for b in sys_blocks], messages=window, tools=tools_out, max_tokens=cfg.max_tokens, extra=extra,
        )

        # Stability checks
        prefix_tokens = self.counter.count_request(ModelRequest(system=system_api, messages=[], tools=tools_c))
        findings = self.checker.scan_request(request, prefix_tokens=prefix_tokens, min_cacheable_tokens=cfg.cache.min_cacheable_tokens)
        self.checker.enforce(findings)
        snapshot = PrefixSnapshot.of(request, history=history)
        breaks = self.checker.compare(st.snapshot, snapshot, expected_reason=st.pending_rebuild)
        for b in breaks:
            if not b.expected:
                self._emit(ctx, EventType.CUSTOM, {"kind": "cache_break", "segment": b.segment, "index": b.index, "reason": b.reason})
        st.snapshot = snapshot
        st.last_rebuild = st.pending_rebuild
        st.pending_rebuild = None
        st.last_raw_estimate = raw + (estimate_message(self.counter, tail_msg) if tail_msg is not None else 0)

        lints = list(rendered.lints)
        warn = tool_count_warning(tools_c)
        if warn:
            lints.append(LayerLint("tools", "warning", warn))
        self._reports[ctx.session_id] = ContextReport(
            session_id=ctx.session_id,
            categories=self._categories(rendered.kinds, rendered.blocks, history, tools_c, tail_msg),
            estimated_tokens=tokens,
            level=level,
            compaction_trigger=cfg.budget.compaction_trigger,
            window_tokens=cfg.budget.window_tokens,
            breakpoints=bp_desc,
            cleared_tool_results=len(st.cleared),
            compactions=sum(1 for m in messages if is_compaction_marker(m) or has_server_compaction(m)),
            actions=actions,
            findings=findings,
            cache_breaks=breaks,
            lints=lints,
        )
        return request

    # ------------------------------------------------------------------ breakpoints
    def _plan_breakpoints(self, history: list[Message], sys_blocks: list[TextBlock], tools: list[ToolSpec], st: _SessionState) -> list[tuple[str, int]]:
        pol = self.config.cache
        out: list[tuple[str, int]] = []
        if pol.breakpoint_on_system and sys_blocks:
            out.append(("system", len(sys_blocks) - 1))
        eligible = [i for i, m in enumerate(history) if with_breakpoint(m, {"type": "ephemeral"}) is not None]
        tail = eligible[-1] if eligible else None
        if pol.breakpoint_on_tail and tail is not None:
            out.append(("tail", tail))
        if pol.breakpoint_mid_history and tail is not None and st.last_tail_id:
            prev = next((i for i, m in enumerate(history) if m.id == st.last_tail_id), None)
            if prev is not None and prev < tail:
                blocks_since = sum(len(m.content) for m in history[prev + 1: tail + 1])
                target = max((i for i in eligible if i <= prev), default=None)
                if blocks_since >= pol.lookback_blocks and target is not None:
                    out.append(("mid", target))
        if pol.breakpoint_on_compaction:
            comp = max((i for i, m in enumerate(history) if (is_compaction_marker(m) or has_server_compaction(m)) and i in eligible), default=None)
            if comp is not None and comp != tail:
                out.append(("compaction", comp))
        if pol.breakpoint_on_tools:
            hot = [i for i, t in enumerate(tools) if not t.defer_loading]
            if hot:
                out.append(("tools", hot[-1]))
        seen: set[tuple[str, int]] = set()
        unique = []
        for kind, idx in out:
            key = ("msg" if kind in ("tail", "mid", "compaction") else kind, idx)
            if key not in seen:
                seen.add(key)
                unique.append((kind, idx))
        return unique

    # ------------------------------------------------------------------ compaction
    def _should_auto_compact(self, tokens: int, st: _SessionState) -> bool:
        cfg = self.config
        return bool(
            self.compactor is not None and cfg.auto_compact and cfg.budget.window_known
            and st.auto_disabled is None and tokens >= cfg.budget.compaction_trigger
        )

    def _auto_compact(self, system_api, messages, history, tools, ctx, st, actions) -> CompactionResult | None:
        if not safe_to_append(messages):
            actions.append("compaction deferred: log ends with an unanswered tool_use")
            return None
        try:
            result = self._run_compaction(system_api, messages, history, tools, ctx, trigger="auto")
        except CompactionInProgressError:
            actions.append("compaction skipped: already in progress")
            return None
        except NothingToCompactError as exc:
            actions.append(f"compaction skipped: {exc}")
            return None
        except CompactionError as exc:
            st.failures += 1
            if st.failures >= self.config.max_compaction_failures:
                st.auto_disabled = f"{st.failures} consecutive compaction failures"
            self._emit(ctx, EventType.ERROR, {"kind": "compaction_failed", "error": str(exc), "failures": st.failures})
            actions.append(f"compaction failed: {exc}")
            return None
        st.failures = 0
        actions.append(f"compacted {result.summarized_count} messages (~{result.tokens_before:,} -> ~{result.tokens_after:,} tokens)")
        return result

    def _run_compaction(self, system_api, messages, history, tools, ctx, *, trigger: str, focus: str | None = None) -> CompactionResult:
        assert self.compactor is not None
        st = self._state(ctx.session_id)
        eff = self.config.budget.effective_window
        result = self.compactor.compact(
            system=system_api, history=history, log=messages, tools=tools, ctx=ctx,
            max_request_tokens=eff, focus=focus, trigger=trigger,
        )
        self._append(messages, result.marker)
        st.pending_rebuild = "compaction"
        st.warned = False
        return result

    def compact(self, system: str, messages: list[Message], tools: list[ToolSpec], ctx: RunContext, *, focus: str | None = None) -> CompactionResult:
        """Manual (``/compact <focus>``) compaction; resets the thrash guard."""
        if self.compactor is None:
            raise ConfigurationError("no compactor configured")
        if not safe_to_append(messages):
            raise CompactionError("cannot compact while the last assistant tool_use is unanswered")
        st = self._state(ctx.session_id)
        with st.lock:
            rendered = self.stack.render(system)
            history = self._cap(self.derive_history(messages))
            if self.clearer:
                history = self.clearer.apply(history, st.cleared, duplicate_of=st.duplicates)
            result = self._run_compaction([b.to_api() for b in rendered.blocks], messages, history,
                                          canonical_tools(tools, sort=self.config.sort_tools), ctx, trigger="manual", focus=focus)
            st.auto_disabled = None
            st.ineffective = st.failures = 0
            self._save_state(ctx.session_id, st)
            return result

    def _handle_server_compaction(self, messages: list[Message], ctx: RunContext, st: _SessionState, actions: list[str]) -> None:
        idx = max((i for i, m in enumerate(messages) if m.role is Role.ASSISTANT and has_server_compaction(m)), default=None)
        if idx is None:
            return
        comp_id = messages[idx].id
        if comp_id in st.seen_server:
            return
        already = any(kind_of(m) == KIND_REINJECTION and context_meta(m).get("after") == comp_id for m in messages[idx + 1:])
        if self.reinjection is not None and not already:
            if not safe_to_append(messages):
                return  # retry on the next prepare, once the tool results are in
            bundle = self.reinjection.build(ctx)
            if bundle.text:
                self._append(messages, Message(Role.USER, [TextBlock(bundle.text)], metadata={
                    META_KEY: {"kind": KIND_REINJECTION, "after": comp_id, "sections": bundle.sections}}))
                actions.append(f"re-injected {', '.join(bundle.sections)} after server compaction")
        st.seen_server.add(comp_id)
        st.warned = False
        st.pending_rebuild = "server_compaction"
        self._emit(ctx, EventType.COMPACTION, {"kind": "server_compaction", "message_id": comp_id})

    # ------------------------------------------------------------------ feedback & helpers
    def observe_response(self, ctx: RunContext, response: ModelResponse) -> bool:
        """Feed ``response.usage`` back: calibrates estimates and tracks cache hits. Returns ``True`` on an unexpected miss."""
        st = self._state(ctx.session_id)
        with st.lock:
            if st.last_raw_estimate:
                st.calibration.observe(st.last_raw_estimate, response.usage)
            reason = st.last_rebuild or ("session_start" if st.observed == 0 else None)
            st.observed += 1
            miss = self.cache_monitor.observe(response.usage, expected_rebuild=reason)
        if miss:
            self._emit(ctx, EventType.CUSTOM, {"kind": "cache_miss", "usage": dataclasses.asdict(response.usage)})
        return miss

    def declare_rebuild(self, ctx: RunContext, reason: str) -> None:
        """Mark the next window as an expected rebuild (e.g. a deliberate tool-set change at a session boundary)."""
        st = self._state(ctx.session_id)
        with st.lock:
            st.pending_rebuild = reason

    def append_reminder(self, messages: list[Message], key: str, text: str) -> Message | None:
        """Append a *persistent* reminder (e.g. a nested AGENTS.md just loaded) to the log, de-duplicated."""
        if not safe_to_append(messages):
            raise ContextError("cannot append a reminder between a tool_use and its tool_result")
        msg = self.dynamic.persistent_message(key, text)
        if msg is not None:
            self._append(messages, msg)
        return msg

    def report(self, session_id: str) -> ContextReport | None:
        return self._reports.get(session_id)

    def _categories(self, kinds, sys_blocks, history, tools, tail_msg) -> dict[str, int]:
        c = self.counter
        cats = {"tools": 0, "system": 0, "project": 0, "compaction_summary": 0, "history": 0, "tool_results": 0, "dynamic": 0}
        count_tools = getattr(c, "count_tools", None)
        cats["tools"] = int(count_tools(tools)) if callable(count_tools) else sum(
            c.count_text(canonical_json(t.to_api())) for t in tools if not t.defer_loading
        )
        for kind, block in zip(kinds, sys_blocks, strict=False):
            cats["project" if kind is LayerKind.PROJECT else "system"] += c.count_text(block.text)
        for m in history:
            if is_compaction_marker(m) or kind_of(m) == KIND_REINJECTION:
                cats["compaction_summary"] += estimate_message(c, m)
                continue
            for b in m.content:
                cats["tool_results" if isinstance(b, ToolResultBlock) else "history"] += estimate_block(c, copy_block(b))
        if tail_msg is not None:
            cats["dynamic"] = estimate_message(c, tail_msg)
        return cats


__all__ = ["AssemblerConfig", "ContextReport", "LayeredContextAssembler", "canonical_tools"]
