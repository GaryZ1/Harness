"""Long-horizon outer loops: initializer + coding sessions, and Ralph loops.

Research basis (reference report section 2.7 steps 11, 12, 15; build plan "Outer loops"):

* Anthropic's long-running harness runs an **initializer** session that writes a JSON feature list,
  a progress file, ``init.sh`` and an initial commit; every later **coding** session starts fresh,
  runs the start-of-session ritual, implements one feature, verifies it end to end, commits and
  logs. The two differ *only in the initial user prompt* (system prompt, tools and harness are
  identical), and "compaction isn't sufficient" for multi-window work. :class:`LongRunDriver`
  runs that schedule, one fresh :class:`~harness_core.session.Session` per context window, which
  also serves as the ``new_context``-style reset for models that show "context anxiety".
* The Ralph loop re-feeds the same prompt until ``<promise>...</promise>`` appears; it requires
  ``--max-iterations`` and should only be used with an executable verifier. :class:`RalphLoop`
  enforces both (a verifier is required unless explicitly waived) and keeps a state file.

Task-state invariants (feature list is flip-only, tests are immutable) are enforced by a
``pre_tool`` hook, e.g. ``harness_memory.task_state.guard_tool_call``, not by these prompts.
These drivers must run *outside* the agent's sandbox so the agent cannot kill its own supervisor.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from ._locking import atomic_write_text
from .evaluators import CompletionPromiseEvaluator, StopContext, Verdict, Verifier
from .loop import AgentLoop
from .prompts import CODING_SESSION_PROMPT, INITIALIZER_PROMPT, PERSISTENCE_PROMPT, RALPH_REFEED
from .results import ResultSubtype, RunResult
from .session import Session, SessionStore
from .types import utcnow

#: Run outcomes after which the long-run driver stops instead of starting another session.
TERMINAL_SUBTYPES = frozenset({
    ResultSubtype.ERROR_REFUSAL, ResultSubtype.ERROR_PERMISSION_DENIALS, ResultSubtype.ERROR_GOAL_IMPOSSIBLE,
    ResultSubtype.ERROR_DURING_EXECUTION, ResultSubtype.ERROR_MAX_BUDGET_USD, ResultSubtype.ERROR_PROMPT_BLOCKED,
    ResultSubtype.HALTED_BY_HOOK, ResultSubtype.PAUSED_FOR_APPROVAL, ResultSubtype.INTERRUPTED,
})


@dataclass
class LongRunPrompts:
    """File names default to the Anthropic post's; point them at ``harness_memory`` task-state paths if used."""

    objective: str
    feature_list: str = "feature_list.json"
    progress_file: str = "claude-progress.txt"
    init_script: str = "init.sh"
    persistence: bool = False   # append PERSISTENCE_PROMPT (for models that stop early to ask)

    def _fields(self) -> dict[str, str]:
        return {"objective": self.objective.strip(), "feature_list": self.feature_list, "progress_file": self.progress_file,
                "init_script": self.init_script}

    def initializer(self) -> str:
        text = INITIALIZER_PROMPT.substitute(self._fields())
        return text + ("\n\n" + PERSISTENCE_PROMPT if self.persistence else "")

    def coding(self) -> str:
        text = CODING_SESSION_PROMPT.substitute(self._fields())
        return text + ("\n\n" + PERSISTENCE_PROMPT if self.persistence else "")


@dataclass
class SessionRecord:
    index: int
    kind: str          # "initializer" | "coding"
    session_id: str
    subtype: str
    num_turns: int
    total_cost_usd: float
    final_text: str = ""


@dataclass
class LongRunReport:
    sessions: list[SessionRecord] = field(default_factory=list)
    completed: bool = False
    reason: str = ""

    @property
    def total_cost_usd(self) -> float:
        return sum(s.total_cost_usd for s in self.sessions)


class LongRunDriver:
    """Run fresh-context sessions until the task is complete or a limit is reached.

    :param store: where each session's durable log lives.
    :param loop_factory: ``f(session) -> AgentLoop``; must build identical system prompt and tools
        for every session (only the user prompt differs between initializer and coding sessions).
    :param is_initialized: e.g. ``lambda: TaskState(repo).exists()``.
    :param is_complete: e.g. "every feature passes"; checked after each session.
    :param max_sessions: hard cap on sessions started by one :meth:`run` (required).
    :param state_path: optional JSON file recording sessions so a restarted driver continues numbering.
    """

    def __init__(self, store: SessionStore, loop_factory: Callable[[Session], AgentLoop], prompts: LongRunPrompts, *,
                 is_initialized: Callable[[], bool], is_complete: Callable[[], bool], max_sessions: int,
                 state_path: str | Path | None = None, session_prefix: str = "longrun") -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be >= 1")
        self.store = store
        self.loop_factory = loop_factory
        self.prompts = prompts
        self.is_initialized = is_initialized
        self.is_complete = is_complete
        self.max_sessions = max_sessions
        self.state_path = Path(state_path) if state_path else None
        self.session_prefix = session_prefix

    def _load(self) -> list[SessionRecord]:
        if self.state_path is None or not self.state_path.exists():
            return []
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        return [SessionRecord(**r) for r in data.get("sessions", [])]

    def _save(self, records: list[SessionRecord], report: LongRunReport) -> None:
        if self.state_path is None:
            return
        atomic_write_text(self.state_path, json.dumps({"updated_at": utcnow().isoformat(), "completed": report.completed,
                                                       "reason": report.reason, "sessions": [asdict(r) for r in records]},
                                                      indent=2) + "\n")

    def run(self) -> LongRunReport:
        history = self._load()
        report = LongRunReport()
        if self.is_initialized() and self.is_complete():
            report.completed, report.reason = True, "already complete"
            return report
        for _ in range(self.max_sessions):
            index = len(history)
            kind = "coding" if self.is_initialized() else "initializer"
            session = self.store.create(f"{self.session_prefix}-{index:04d}-{kind}", metadata={"long_run_kind": kind, "index": index})
            loop = self.loop_factory(session)
            result = loop.run(self.prompts.coding() if kind == "coding" else self.prompts.initializer())
            if result.subtype is not ResultSubtype.PAUSED_FOR_APPROVAL:
                loop.end_session(reason=result.subtype.value)
            rec = SessionRecord(index, kind, session.id, result.subtype.value, result.num_turns, result.total_cost_usd,
                                result.final_text[-2000:])
            history.append(rec)
            report.sessions.append(rec)
            if kind == "initializer" and not self.is_initialized():
                report.reason = "initializer session did not create the task state"
                break
            if self.is_complete():
                report.completed, report.reason = True, "task complete"
                break
            if result.subtype in TERMINAL_SUBTYPES:
                report.reason = f"stopped after {result.subtype.value}"
                break
            self._save(history, report)
        else:
            report.reason = f"max_sessions={self.max_sessions} reached"
        self._save(history, report)
        return report


@dataclass
class RalphReport:
    result: RunResult
    iterations: int
    completed: bool


class RalphLoop:
    """Re-feed the same prompt inside one session until the completion promise is output and verified.

    :param max_iterations: required cap (an iteration is one attempt to stop).
    :param completion_promise: text the model must emit as ``<promise>TEXT</promise>``.
    :param verifier: ``() -> (ok, detail)``; required unless ``require_verifier=False``.
    :param state_path: JSON state file (``iteration``, ``max_iterations``, ``completion_promise``,
        ``session_id``, ``status``) updated on every iteration.
    """

    def __init__(self, loop: AgentLoop, prompt: str, *, max_iterations: int, completion_promise: str,
                 verifier: Verifier | None = None, require_verifier: bool = True, state_path: str | Path | None = None) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations is required and must be >= 1")
        if verifier is None and require_verifier:
            raise ValueError("Ralph loops need an executable verifier (pass verifier=..., or require_verifier=False to waive)")
        self.loop = loop
        self.prompt = prompt
        self.max_iterations = max_iterations
        self.promise = completion_promise
        self.verifier = verifier
        self.state_path = Path(state_path) if state_path else None
        self.iteration = 1

    def _render(self, iteration: int) -> str:
        return RALPH_REFEED.substitute(prompt=self.prompt.strip(), iteration=iteration, max_iterations=self.max_iterations,
                                       promise=self.promise)

    def _write_state(self, status: str) -> None:
        if self.state_path is None:
            return
        state: dict[str, Any] = {"iteration": self.iteration, "max_iterations": self.max_iterations,
                                 "completion_promise": self.promise, "session_id": self.loop.session.id, "status": status}
        atomic_write_text(self.state_path, json.dumps(state, indent=2) + "\n")

    def _refeed(self, sc: StopContext) -> str:
        self.iteration += 1
        self._write_state("running")
        return self._render(self.iteration)

    def run(self) -> RalphReport:
        evaluator = CompletionPromiseEvaluator(self.promise, refeed=self._refeed, verifier=self.verifier, name="ralph")
        saved_config, saved_evaluators = self.loop.config, self.loop.evaluators
        self.loop.config = replace(saved_config, max_total_stop_blocks=self.max_iterations - 1,
                                   max_consecutive_stop_blocks=max(saved_config.max_consecutive_stop_blocks, self.max_iterations))
        self.loop.evaluators = [*saved_evaluators, evaluator]
        self._write_state("running")
        try:
            result = self.loop.run(self._render(1))
        finally:
            self.loop.config, self.loop.evaluators = saved_config, saved_evaluators
        ralph = [e for e in result.evaluations if e.evaluator == "ralph"]
        completed = bool(ralph) and ralph[-1].verdict is Verdict.MET and result.subtype is ResultSubtype.SUCCESS
        iterations = min(self.iteration, self.max_iterations)
        self.iteration = iterations
        self._write_state("completed" if completed else result.subtype.value if result.is_error else "max_iterations")
        return RalphReport(result, iterations, completed)
