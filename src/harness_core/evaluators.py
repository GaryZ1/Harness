"""Stop evaluators: the outer check that can veto "done" and feed guidance back.

Research basis (reference report section 2.7 steps 1, 9, 10; build plan "The loop"):

* The canonical predicate is "stop only when the last response has no tool calls, no user input is
  pending, and every Stop hook or goal evaluator allows it" (Claude Agent SDK loop, Codex
  ``run_turn``). Evaluators are the third clause.
* Claude Code ``/goal`` is a prompt-based Stop hook judged by a small fast model returning **Met**,
  **Not yet met** (the reason becomes guidance) or **Impossible**; the evaluator "does not call
  tools, so it can only judge what Claude has already surfaced". :class:`GoalEvaluator`.
* Codex thread goals inject the objective as *data* with a token readout, require completion to be
  *proven*, and allow ``blocked`` only after the same blocker repeats for three consecutive turns.
  :class:`GoalEvaluator` renders that continuation and needs ``impossible_after`` (default 3)
  consecutive Impossible verdicts before ending the run.
* The Ralph loop re-feeds the same prompt until ``<promise>...</promise>`` appears.
  :class:`CompletionPromiseEvaluator`, optionally gated on an executable verifier.

Runaway protection lives in the loop, not here: a block without feedback text is ignored (Codex),
and after 8 consecutive blocks with no intervening tool call the stop is allowed (Claude Code).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable

from .prompts import GOAL_EVALUATOR_PROMPT, GOAL_EVALUATOR_SYSTEM, render_goal_continuation
from .types import Message, ModelClient, ModelRequest, RunContext, StopReason, Usage


class Verdict(str, Enum):
    MET = "met"
    NOT_MET = "not_met"
    IMPOSSIBLE = "impossible"


@dataclass
class Evaluation:
    verdict: Verdict
    reason: str = ""
    feedback: str = ""      # text injected as the continuation message (defaults to ``reason``)
    evaluator: str = ""
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def continuation(self) -> str:
        return (self.feedback or self.reason).strip()


@dataclass
class StopContext:
    """Everything an evaluator may look at when the agent tries to stop."""

    session_id: str
    messages: list[Message]
    last_assistant_text: str
    ctx: RunContext
    num_turns: int
    usage: Usage
    stop_hook_active: bool = False
    consecutive_blocks: int = 0
    total_blocks: int = 0
    token_budget: int | None = None
    tokens_used: int = 0


@runtime_checkable
class StopEvaluator(Protocol):
    name: str

    def evaluate(self, sc: StopContext) -> Evaluation: ...


class CallableEvaluator:
    """Wrap ``fn(sc) -> Evaluation | bool | str | None``.

    ``True``/``None`` mean met; ``False`` means not met without feedback (ignored by the loop);
    a string means not met, with that string as feedback.
    """

    def __init__(self, fn: Callable[[StopContext], Any], name: str = "") -> None:
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "callable")

    def evaluate(self, sc: StopContext) -> Evaluation:
        out = self.fn(sc)
        if isinstance(out, Evaluation):
            out.evaluator = out.evaluator or self.name
            return out
        if out is None or out is True:
            return Evaluation(Verdict.MET, evaluator=self.name)
        if out is False:
            return Evaluation(Verdict.NOT_MET, evaluator=self.name)
        return Evaluation(Verdict.NOT_MET, reason=str(out), evaluator=self.name)


Verifier = Callable[[], "tuple[bool, str]"]


class CompletionPromiseEvaluator:
    """Ralph-style: met only when the last assistant text contains ``<promise>PROMISE</promise>``.

    If ``verifier`` is given (strongly recommended: "use it only with an executable verifier"),
    the promise is accepted only when ``verifier()`` returns ``(True, _)``; otherwise its message
    is fed back. ``refeed`` is the continuation text (Ralph re-feeds the same prompt); it may be a
    callable receiving the :class:`StopContext`.
    """

    def __init__(self, promise: str, *, refeed: str | Callable[[StopContext], str], verifier: Verifier | None = None,
                 name: str = "completion_promise") -> None:
        if not promise.strip():
            raise ValueError("promise must be non-empty")
        self.promise = promise.strip()
        self.refeed = refeed
        self.verifier = verifier
        self.name = name
        self._pattern = re.compile(r"<promise>\s*" + re.escape(self.promise) + r"\s*</promise>", re.S)

    def _refeed(self, sc: StopContext) -> str:
        return self.refeed(sc) if callable(self.refeed) else self.refeed

    def evaluate(self, sc: StopContext) -> Evaluation:
        if not self._pattern.search(sc.last_assistant_text or ""):
            return Evaluation(Verdict.NOT_MET, reason="completion promise not found", feedback=self._refeed(sc), evaluator=self.name)
        if self.verifier is not None:
            ok, detail = self.verifier()
            if not ok:
                fb = f"You output the completion promise, but verification failed:\n{detail}\n\n{self._refeed(sc)}"
                return Evaluation(Verdict.NOT_MET, reason=f"verifier failed: {detail}", feedback=fb, evaluator=self.name)
            return Evaluation(Verdict.MET, reason=detail or "promise found and verified", evaluator=self.name)
        return Evaluation(Verdict.MET, reason="completion promise found", evaluator=self.name)


def render_transcript(messages: list[Message], *, max_chars: int = 40_000) -> str:
    """Plain-text transcript for an evaluator, keeping the most recent ``max_chars``."""
    lines: list[str] = []
    for m in messages:
        for b in m.content:
            t = getattr(b, "type", "")
            if t == "text":
                lines.append(f"[{m.role.value}] {b.text}")  # type: ignore[union-attr]
            elif t == "tool_use":
                lines.append(f"[tool_use {b.name}] {b.input}")  # type: ignore[union-attr]
            elif t == "tool_result":
                tag = "tool_error" if b.is_error else "tool_result"  # type: ignore[union-attr]
                lines.append(f"[{tag}] {b.text}")  # type: ignore[union-attr]
    text = "\n".join(lines)
    return text if len(text) <= max_chars else "...(earlier transcript omitted)...\n" + text[-max_chars:]


_VERDICT_LINE = re.compile(r"^\s*\**\s*(MET|NOT[_ ]MET|NOT YET MET|IMPOSSIBLE)\s*\**\s*[:\-]?\s*(.*)$", re.I)


def parse_verdict(text: str) -> tuple[Verdict, str] | None:
    for line in text.strip().splitlines():
        m = _VERDICT_LINE.match(line)
        if m:
            word = m.group(1).upper().replace(" ", "_")
            verdict = Verdict.MET if word == "MET" else Verdict.IMPOSSIBLE if word == "IMPOSSIBLE" else Verdict.NOT_MET
            return verdict, m.group(2).strip()
    return None


class GoalEvaluator:
    """``/goal``-style evaluator: a separate (small) model judges the condition from the transcript.

    :param model: the evaluator's :class:`ModelClient` (configure it with a small fast model id).
    :param condition: the goal condition (Claude Code caps it at 4,000 characters; so do we).
    :param impossible_after: consecutive Impossible verdicts required before the run ends
        (Codex's blocked-after-three rule); earlier ones are fed back as Not yet met.
    :param cost_estimator: optional ``(usage, model) -> usd`` so evaluator spend counts toward budgets.

    An unparseable reply or a refusal yields a Not-met verdict with no feedback, which the loop
    ignores (so a broken evaluator can never trap the agent in a loop).
    """

    MAX_CONDITION_CHARS = 4_000

    def __init__(self, model: ModelClient, condition: str, *, impossible_after: int = 3, max_tokens: int = 2_000,
                 transcript_chars: int = 40_000, name: str = "goal",
                 cost_estimator: Callable[[Usage, str], float] | None = None) -> None:
        if not condition.strip():
            raise ValueError("goal condition must be non-empty")
        if len(condition) > self.MAX_CONDITION_CHARS:
            raise ValueError(f"goal condition exceeds {self.MAX_CONDITION_CHARS} characters")
        if impossible_after < 1:
            raise ValueError("impossible_after must be >= 1")
        self.model = model
        self.condition = condition.strip()
        self.impossible_after = impossible_after
        self.max_tokens = max_tokens
        self.transcript_chars = transcript_chars
        self.name = name
        self.cost_estimator = cost_estimator
        self._impossible_streak = 0

    def evaluate(self, sc: StopContext) -> Evaluation:
        prompt = GOAL_EVALUATOR_PROMPT.substitute(condition=self.condition,
                                                  transcript=render_transcript(sc.messages, max_chars=self.transcript_chars))
        req = ModelRequest(system=GOAL_EVALUATOR_SYSTEM, messages=[Message.user(prompt)], tools=[], max_tokens=self.max_tokens)
        resp = self.model.create(req)
        cost = self.cost_estimator(resp.usage, resp.model) if self.cost_estimator else 0.0
        parsed = None if resp.stop_reason is StopReason.REFUSAL else parse_verdict(resp.text)
        if parsed is None:
            self._impossible_streak = 0
            return Evaluation(Verdict.NOT_MET, reason="", evaluator=self.name, usage=resp.usage, cost_usd=cost,
                              data={"unparseable": True, "raw": resp.text[:500]})
        verdict, reason = parsed
        if verdict is Verdict.IMPOSSIBLE:
            self._impossible_streak += 1
            if self._impossible_streak < self.impossible_after:
                fb = render_goal_continuation(
                    self.condition,
                    f"The evaluator thinks the goal may be impossible ({reason}). If the same blocker persists, explain "
                    "it with evidence; otherwise try a different approach.",
                    tokens_used=sc.tokens_used, token_budget=sc.token_budget)
                return Evaluation(Verdict.NOT_MET, reason=reason, feedback=fb, evaluator=self.name, usage=resp.usage,
                                  cost_usd=cost, data={"impossible_streak": self._impossible_streak})
            return Evaluation(Verdict.IMPOSSIBLE, reason=reason, evaluator=self.name, usage=resp.usage, cost_usd=cost)
        self._impossible_streak = 0
        if verdict is Verdict.MET:
            return Evaluation(Verdict.MET, reason=reason, evaluator=self.name, usage=resp.usage, cost_usd=cost)
        fb = render_goal_continuation(self.condition, reason, tokens_used=sc.tokens_used, token_budget=sc.token_budget)
        return Evaluation(Verdict.NOT_MET, reason=reason, feedback=fb, evaluator=self.name, usage=resp.usage, cost_usd=cost)
