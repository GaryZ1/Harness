"""Client-side compaction: structured handoff summary + kept tail + re-injection (ladder L3/L4).

Research basis (reference report 3.3 techniques 17, 19, 20; checklist steps 11-12; pitfalls
"Compaction errors"):

* **Summary schema** merges the Anthropic SDK and Codex prompts: task overview, *user preferences
  and learnings*, current state, *files touched*, *decisions* and rationale, errors and failed
  approaches, *open tasks* / next steps, critical references - with an explicit "do not call any
  tools" (otherwise the model sometimes calls a tool instead of summarizing). The schema is
  enforced in code: every section is always rendered, missing ones are recorded, and the
  harness merges its own list of touched files (from tool-call inputs) into the model's list.
* **Keep tail**: the last ``keep_recent_turns`` (2, as in the reference loop) turns stay
  verbatim, thinking included; the cut never splits a ``tool_use`` from its ``tool_result``.
* **Codex recipe** for providers without a server primitive: the user's own earlier messages
  are carried forward newest-first up to **20,000** tokens (``COMPACT_USER_MESSAGE_MAX_TOKENS``),
  and mid-conversation ``system`` instructions in the summarized range are carried forward too
  (the docs warn they otherwise stop applying).
* **Same system and tools** as the session so the summarization call reads the session cache;
  the request is checked to *fit* before it is sent ("the summarization call itself must fit"),
  shrinking tool results to placeholders first if needed.
* **Never two compactions at once** per session (:class:`CompactionInProgressError`).
* **Append-only**: the result is a *marker message* that the integrator appends to the durable
  log. The assembler then projects ``[marker] + messages after covers_through`` - earlier
  messages are never edited or deleted. The marker stores a hash of the log prefix it covers
  (Cline-style validated projection) so tampering is detected.
"""

from __future__ import annotations

import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

from harness_core.types import Event as CoreEvent
from harness_core.types import (
    EventSink,
    EventType,
    Message,
    ModelClient,
    ModelRequest,
    NullTracer,
    Role,
    RunContext,
    StopReason,
    TextBlock,
    ToolSpec,
    Tracer,
    Usage,
    new_id,
)

from .clearing import ClearingPolicy, ToolResultClearer
from .errors import CompactionError, CompactionInProgressError, ConfigurationError, LogIntegrityError, NothingToCompactError
from .messages import (
    KIND_COMPACTION,
    META_KEY,
    context_meta,
    copy_message,
    is_compaction_marker,
    is_safe_cut,
    is_user_turn_start,
    log_prefix_sha256,
    merge_consecutive_user,
)
from .references import extract_paths
from .reinjection import ReinjectionBuilder
from .tokens import HeuristicTokenCounter, TokenCounter, estimate_message

COMPACT_USER_MESSAGE_MAX_TOKENS = 20_000  # Codex

SUMMARY_PROMPT = """You are performing a context checkpoint. Write a handoff summary for another model that will resume this task without access to the conversation above. Do not call any tools while writing this summary; respond with text only.

Use exactly these Markdown headings, in this order, and keep every heading even if the section is empty (write "None." under it):
## Task overview
The user's goal, success criteria and constraints.
## User preferences and learnings
Stable user preferences, corrections, commitments, and lessons learned about the environment or codebase.
## Current state
What is done, artifacts produced, open processes or background jobs.
## Files touched
One bullet per file path created, modified or read that matters for continuing.
## Decisions and rationale
Decisions made and why; important discoveries.
## Errors and failed approaches
Errors encountered, how they were resolved, and approaches that failed (so they are not retried).
## Open tasks and next steps
Remaining work, blockers and priorities, most important first.
## Critical references
Identifiers, paths, URLs, commands and data needed to continue (prefer pointers over copied content).

Maximize recall first, then be concise and structured. Wrap the whole summary in <summary></summary>."""

SUMMARY_PREFIX = (
    "This conversation was compacted to save context. Another model instance started this task and wrote the "
    "handoff summary below; the original messages it covers are no longer in context. Build on this work, do not "
    "repeat completed steps or approaches recorded as failed, and re-read files before editing them because their "
    "contents are not in context."
)

#: (key, heading, keywords used to recognize the heading). Order = render order.
SUMMARY_SECTIONS: tuple[tuple[str, str], ...] = (
    ("task_overview", "Task overview"),
    ("user_preferences", "User preferences and learnings"),
    ("current_state", "Current state"),
    ("files_touched", "Files touched"),
    ("decisions", "Decisions and rationale"),
    ("errors", "Errors and failed approaches"),
    ("open_tasks", "Open tasks and next steps"),
    ("references", "Critical references"),
)
REQUIRED_SECTIONS = ("task_overview", "user_preferences", "current_state", "files_touched", "decisions", "open_tasks")

_SUMMARY_TAG = re.compile(r"<summary>(.*?)(?:</summary>|\Z)", re.S | re.I)
_HEADING = re.compile(r"^\s*(?:#{1,6}\s+(?P<h>.+?)|\*\*(?P<b>[^*]+)\*\*)\s*:?\s*$")
_NONE = re.compile(r"^\s*(?:none\.?|n/?a\.?|-)?\s*$", re.I)


def _section_for(title: str) -> str | None:
    t = re.sub(r"^\d+[.)]\s*", "", title.strip().lower()).rstrip(":")
    if "preference" in t or "learning" in t:
        return "user_preferences"
    if "file" in t:
        return "files_touched"
    if "decision" in t:
        return "decisions"
    if "error" in t or "failed" in t or "mistake" in t:
        return "errors"
    if "next" in t or "open task" in t or "todo" in t or "remaining" in t or "pending" in t:
        return "open_tasks"
    if "task" in t or "goal" in t or "overview" in t:
        return "task_overview"
    if "state" in t or "progress" in t:
        return "current_state"
    if "reference" in t or "critical" in t:
        return "references"
    return None


@dataclass
class CompactionSummary:
    task_overview: str = ""
    user_preferences: str = ""
    current_state: str = ""
    files_touched: str = ""
    decisions: str = ""
    errors: str = ""
    open_tasks: str = ""
    references: str = ""
    missing_sections: list[str] = field(default_factory=list)
    tagged: bool = True

    def merge_files(self, paths: Sequence[str]) -> None:
        """Append harness-observed paths the model did not mention (enforced, not prompted)."""
        existing = self.files_touched if not _NONE.match(self.files_touched) else ""
        lines = [existing] if existing else []
        for p in paths:
            if p and p not in existing:
                lines.append(f"- {p}")
                existing += "\n" + p
        self.files_touched = "\n".join(lines).strip()
        if self.files_touched and "files_touched" in self.missing_sections:
            self.missing_sections.remove("files_touched")

    def render(self) -> str:
        out = []
        for key, heading in SUMMARY_SECTIONS:
            body = getattr(self, key).strip()
            out.append(f"## {heading}\n{body if body and not _NONE.match(body) else 'None recorded.'}")
        return "\n\n".join(out)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_summary(text: str) -> CompactionSummary:
    """Parse the model's handoff summary into the fixed schema.

    Accepts ``<summary>`` tags (preferred) or a bare Markdown body with recognizable headings.
    Raises :class:`CompactionError` when neither is present.
    """
    m = _SUMMARY_TAG.search(text or "")
    tagged = m is not None
    body = (m.group(1) if m else (text or "")).strip()
    sections: dict[str, list[str]] = {}
    current: str | None = None
    preamble: list[str] = []
    for line in body.splitlines():
        hm = _HEADING.match(line)
        key = _section_for(hm.group("h") or hm.group("b")) if hm else None
        if key:
            current = key
            sections.setdefault(key, [])
            continue
        (sections[current] if current else preamble).append(line)
    if not sections:
        if not tagged or not body:
            raise CompactionError("model returned no recognizable summary (no <summary> tags and no section headings)")
        sections["current_state"] = preamble  # untagged structure: keep the text rather than lose it
        preamble = []
    summary = CompactionSummary(tagged=tagged)
    for key, lines in sections.items():
        setattr(summary, key, "\n".join(lines).strip())
    if preamble and "".join(preamble).strip():
        summary.task_overview = ("\n".join(preamble).strip() + "\n" + summary.task_overview).strip()
    summary.missing_sections = [k for k in REQUIRED_SECTIONS if _NONE.match(getattr(summary, k) or "")]
    return summary


def choose_cut(history: Sequence[Message], *, keep_turns: int = 2, min_tail_messages: int = 4) -> int:
    """Index splitting ``history`` into (summarize, keep). Prefers turn boundaries; never splits a tool pair."""
    start = 1 if history and is_compaction_marker(history[0]) else 0
    turn_starts = [i for i in range(start, len(history)) if is_user_turn_start(history[i])]
    if len(turn_starts) >= keep_turns:
        cut = turn_starts[-keep_turns]
        if cut > start and is_safe_cut(history, cut):
            return cut
    for cut in range(len(history) - min_tail_messages, start, -1):
        m = history[cut]
        if (m.role is Role.ASSISTANT or is_user_turn_start(m)) and is_safe_cut(history, cut):
            return cut
    raise NothingToCompactError("no safe cut point leaves anything to summarize")


@dataclass(frozen=True)
class CompactionConfig:
    keep_recent_turns: int = 2
    min_tail_messages: int = 4
    retain_user_message_tokens: int = COMPACT_USER_MESSAGE_MAX_TOKENS
    summary_max_tokens: int = 8_000
    instructions: str = SUMMARY_PROMPT
    send_tools: bool = True
    request_extra: dict[str, Any] = field(default_factory=dict)
    carry_operator_instructions: bool = True

    def __post_init__(self) -> None:
        if self.keep_recent_turns < 1:
            raise ConfigurationError("keep_recent_turns must be >= 1")
        if self.summary_max_tokens < 1_024:
            raise ConfigurationError("summary_max_tokens must allow several thousand tokens (>= 1024)")


@dataclass
class CompactionResult:
    marker: Message
    summary: CompactionSummary
    covers_through: str
    summarized_count: int
    tokens_before: int
    tokens_after: int
    usage: Usage
    reinjected: list[str]


def _utc_iso(clock: Callable[[], datetime]) -> str:
    return clock().astimezone(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


class Compactor:
    """Summarize older history through a :class:`~harness_core.types.ModelClient`.

    The model id lives in the client (configured by the caller); this class never names one.
    """

    def __init__(
        self,
        client: ModelClient,
        *,
        config: CompactionConfig | None = None,
        counter: TokenCounter | None = None,
        reinjection: ReinjectionBuilder | None = None,
        events: EventSink | None = None,
        tracer: Tracer | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.client = client
        self.config = config or CompactionConfig()
        self.counter = counter or HeuristicTokenCounter()
        self.reinjection = reinjection
        self.events = events
        self.tracer = tracer or NullTracer()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._locks: dict[str, threading.Lock] = {}
        self._registry = threading.Lock()

    def _lock_for(self, session_id: str) -> threading.Lock:
        with self._registry:
            return self._locks.setdefault(session_id, threading.Lock())

    def in_progress(self, session_id: str) -> bool:
        return self._lock_for(session_id).locked()

    # ------------------------------------------------------------------ main entry point
    def compact(
        self,
        *,
        system: str | list[dict[str, Any]],
        history: Sequence[Message],
        log: Sequence[Message],
        tools: Sequence[ToolSpec],
        ctx: RunContext,
        max_request_tokens: int | None = None,
        focus: str | None = None,
        trigger: str = "auto",
    ) -> CompactionResult:
        """Summarize ``history`` (the derived window, marker first if any) up to a safe cut.

        :param log: the durable log, used to validate and fingerprint the covered prefix.
        :param max_request_tokens: the summarization request (plus ``summary_max_tokens``) must fit.
        :param focus: optional user focus (``/compact <focus>``) appended to the instructions.
        """
        lock = self._lock_for(ctx.session_id)
        if not lock.acquire(blocking=False):
            raise CompactionInProgressError(f"a compaction is already running for session {ctx.session_id}")
        try:
            with self.tracer.span("context.compaction", session_id=ctx.session_id, trigger=trigger):
                return self._compact(system, list(history), list(log), list(tools), ctx, max_request_tokens, focus, trigger)
        finally:
            lock.release()

    def _compact(self, system, history, log, tools, ctx, max_request_tokens, focus, trigger) -> CompactionResult:
        cfg = self.config
        cut = choose_cut(history, keep_turns=cfg.keep_recent_turns, min_tail_messages=cfg.min_tail_messages)
        head, tail = history[:cut], history[cut:]
        prev = head[0] if head and is_compaction_marker(head[0]) else None
        real_head = [m for m in head if not is_compaction_marker(m)]
        if not real_head:
            raise NothingToCompactError("only the previous summary precedes the kept tail")
        covers = real_head[-1]
        log_index = next((i for i in range(len(log) - 1, -1, -1) if log[i].id == covers.id), None)
        if log_index is None:
            raise LogIntegrityError(f"message {covers.id} is not in the durable log; compact the derived window of this log")

        request = self._request(system, head, tools, focus)
        if max_request_tokens is not None:
            request = self._fit(request, head, tools, focus, system, max_request_tokens)
        response = self.client.create(request)
        if response.tool_uses:
            reinforced = self._request(system, head, tools, focus, reinforce=True)
            response = self.client.create(reinforced)
        usage = response.usage
        if response.tool_uses:
            raise CompactionError("the model called a tool instead of writing the summary (twice)")
        if response.stop_reason not in (StopReason.END_TURN, StopReason.STOP_SEQUENCE) or not response.text.strip():
            raise CompactionError(
                f"compaction produced no usable summary (stop_reason={response.stop_reason.value}); "
                "compact earlier so the summarization request fits"
            )
        summary = parse_summary(response.text)
        prev_meta = context_meta(prev) if prev else {}
        files: list[str] = list(prev_meta.get("files_touched", []))
        for m in real_head:
            for u in m.tool_uses:
                for p in extract_paths(u.input):
                    if p not in files:
                        files.append(p)
        summary.merge_files(files)

        user_msgs = self._retained_user_messages(prev_meta.get("user_messages", []), real_head)
        operator = list(prev_meta.get("operator_instructions", [])) if cfg.carry_operator_instructions else []
        if cfg.carry_operator_instructions:
            operator += [m.text for m in real_head if m.role is Role.SYSTEM and m.text.strip()]
        bundle = self.reinjection.build(ctx) if self.reinjection else None

        blocks = [TextBlock(f"{SUMMARY_PREFIX}\n\n<summary>\n{summary.render()}\n</summary>")]
        if user_msgs:
            rendered = "\n".join(f"<message>\n{t}\n</message>" for t in user_msgs)
            blocks.append(TextBlock(
                "<prior-user-messages>\nThe user's own earlier messages, verbatim and oldest first (budget-truncated):\n"
                f"{rendered}\n</prior-user-messages>"
            ))
        if operator:
            listing = "\n".join(f"- {' '.join(t.split())}" for t in operator)
            blocks.append(TextBlock(f"<operator-instructions>\nThese earlier operator instructions still apply:\n{listing}\n</operator-instructions>"))
        if bundle and bundle.text:
            blocks.append(TextBlock(bundle.text))

        tokens_before = sum(estimate_message(self.counter, m) for m in history)
        meta: dict[str, Any] = {
            "kind": KIND_COMPACTION,
            "compaction_id": new_id("cmp"),
            "mode": "client",
            "trigger": trigger,
            "covers_through": covers.id,
            "covers_log_index": log_index,
            "prefix_sha256": log_prefix_sha256(log, log_index),
            "previous": prev.id if prev else None,
            "summarized_count": len(real_head),
            "user_messages": user_msgs,
            "operator_instructions": operator,
            "files_touched": files,
            "missing_sections": list(summary.missing_sections),
            "reinjected": list(bundle.sections) if bundle else [],
            "created": _utc_iso(self.clock),
            "usage": asdict(usage),
        }
        marker = Message(Role.USER, blocks, metadata={META_KEY: meta})
        tokens_after = estimate_message(self.counter, marker) + sum(estimate_message(self.counter, m) for m in tail)
        meta["tokens_before"] = tokens_before
        meta["tokens_after"] = tokens_after
        if self.events is not None:
            self.events.emit(CoreEvent(ctx.session_id, EventType.COMPACTION, {
                "kind": "client_compaction", "compaction_id": meta["compaction_id"], "trigger": trigger,
                "covers_through": covers.id, "summarized_count": len(real_head), "tokens_before": tokens_before,
                "tokens_after": tokens_after, "missing_sections": meta["missing_sections"],
                "reinjected": meta["reinjected"], "usage": meta["usage"],
            }))
        return CompactionResult(marker, summary, covers.id, len(real_head), tokens_before, tokens_after, usage, meta["reinjected"])

    # ------------------------------------------------------------------ helpers
    def _request(self, system, head, tools, focus, *, reinforce: bool = False) -> ModelRequest:
        prompt = self.config.instructions
        if focus:
            prompt += f"\n\nThe user asked this summary to focus on: {focus}"
        if reinforce:
            prompt += "\n\nIMPORTANT: your previous attempt called a tool. Tools are unavailable now. Respond with the text summary only."
        msgs = [copy_message(m) for m in head] + [Message.user(prompt)]
        return ModelRequest(
            system=system,
            messages=merge_consecutive_user(msgs),
            tools=list(tools) if self.config.send_tools else [],
            max_tokens=self.config.summary_max_tokens,
            extra=dict(self.config.request_extra),
        )

    def _fit(self, request, head, tools, focus, system, limit) -> ModelRequest:
        need = self.counter.count_request(request) + self.config.summary_max_tokens
        if need <= limit:
            return request
        shrinker = ToolResultClearer(
            ClearingPolicy(keep_recent=0, clear_at_least_tokens=0, min_result_tokens=0, exclude_tools=frozenset(), dedupe=False),
            self.counter,
        )
        ids = [r.tool_use_id for m in head for r in m.tool_results]
        shrunk = shrinker.apply(head, ids)
        request = self._request(system, shrunk, tools, focus)
        need = self.counter.count_request(request) + self.config.summary_max_tokens
        if need > limit:
            raise CompactionError(
                f"summarization request needs ~{need} tokens > limit {limit} even with tool results cleared; "
                "the compaction threshold is too close to the window"
            )
        return request

    def _retained_user_messages(self, previous: Sequence[str], head: Sequence[Message]) -> list[str]:
        candidates = list(previous) + [m.text.strip() for m in head if is_user_turn_start(m) and m.text.strip()]
        budget = self.config.retain_user_message_tokens
        kept: list[str] = []
        for text in reversed(candidates):  # newest first
            cost = self.counter.count_text(text)
            if cost <= budget:
                kept.append(text)
                budget -= cost
                continue
            if budget > 50:
                chars = int(len(text) * budget / max(cost, 1))
                kept.append(text[:chars].rstrip() + " [...truncated]")
            break
        kept.reverse()
        return kept


__all__ = [
    "COMPACT_USER_MESSAGE_MAX_TOKENS",
    "CompactionConfig",
    "CompactionResult",
    "CompactionSummary",
    "Compactor",
    "REQUIRED_SECTIONS",
    "SUMMARY_PREFIX",
    "SUMMARY_PROMPT",
    "SUMMARY_SECTIONS",
    "choose_cut",
    "parse_summary",
]
