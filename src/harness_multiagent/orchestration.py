"""Scripted fan-out: code, not the model, controls *how* agents run.

Research basis (reference 8.7 item 7, 8.5 workflow rows, 8.8 "Nondeterministic orchestration
scripts"): for large fan-outs, code-driven orchestration has displaced LLM-driven orchestration.
Claude Code's dynamic workflows ("Claude's context holds only the final answer") expose
``agent()``, ``parallel()``, ``pipeline()``, ``phase()`` and ``log()``; OpenAI recommends orchestrating
"via code" because it is "more deterministic and predictable, in terms of speed, cost and
performance". The hybrid rule is: the LLM decides *whether* to fan out, code runs *how*.

:class:`Workflow` mirrors that API in Python with the vendor defaults:

=========================  ======================  ===========================================
Behaviour                  Default                 Source
=========================  ======================  ===========================================
concurrent agents          16                      ``CLAUDE_CODE_WORKFLOW_MAX_CONCURRENT_AGENTS``
items per parallel/pipeline 4,096                  workflow limits
agents per run             1,000                   workflow limits
restarts of stalled agents 5                       ``stallMs`` auto-restart
sibling prefix stagger     5 s (cut short when     ``CLAUDE_CODE_WORKFLOW_PREFIX_STAGGER_MS``
                           the first sibling ends)
cost warning               > 25 agents / 1.5M tok  workflow cost guard
=========================  ======================  ===========================================

Invariants:

* **Deterministic ordering**: ``parallel`` / ``pipeline`` return results in input order regardless of
  completion order.
* **Partial failure**: a failed agent yields ``None`` and the run continues; a *directly awaited*
  step (``required=True``) or a hard ceiling ends the run with an exception.
* **Replay**: with a :class:`WorkflowJournal`, each successful agent result is stored under
  ``sha256(agent, definition fingerprint, rendered brief, output cap)``; re-running the same script
  returns cached reports for unchanged prompts and only re-runs changed or failed ones. Scripts must
  therefore be deterministic: pass inputs as arguments, and do not use wall-clock time or randomness.
"""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generic, Iterator, Mapping, Sequence, TypeVar

from harness_core.types import RunContext
from harness_memory.locking import file_lock

from .brief import DelegationBrief
from .errors import WorkflowLimitError, WorkflowStepFailed
from .gate import fanout_warnings
from .manager import SubagentManager
from .result import SubagentReport, SubagentResult, SubagentStatus

T = TypeVar("T")
R = TypeVar("R")

_FATAL = (WorkflowLimitError, WorkflowStepFailed)


class Verdict(str, Enum):
    """Outcome of verifying one claim. Uncheckable is ``UNVERIFIED``, never ``REFUTED``."""

    VERIFIED = "verified"
    REFUTED = "refuted"
    UNVERIFIED = "unverified"

    @classmethod
    def coerce(cls, value: Any) -> "Verdict":
        if isinstance(value, Verdict):
            return value
        if value is True:
            return cls.VERIFIED
        if value is False:
            return cls.REFUTED
        if isinstance(value, str) and value in cls._value2member_map_:
            return cls(value)
        return cls.UNVERIFIED


@dataclass(frozen=True)
class WorkflowConfig:
    max_concurrent: int = 16
    max_items: int = 4_096
    max_agents: int = 1_000
    max_restarts: int = 5
    prefix_stagger_s: float = 5.0
    warn_agents: int = 25
    warn_tokens: int = 1_500_000

    def __post_init__(self) -> None:
        if not 1 <= self.max_concurrent <= 256:
            raise ValueError("max_concurrent must be within 1..256")
        if self.max_restarts < 0 or self.prefix_stagger_s < 0:
            raise ValueError("max_restarts and prefix_stagger_s must be >= 0")


class WorkflowJournal:
    """Append-only JSONL replay cache of successful agent reports (thread- and process-safe)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn final line after a crash: ignore, the agent simply re-runs
            if isinstance(rec, dict) and isinstance(rec.get("key"), str):
                self._entries[rec["key"]] = rec

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            return self._entries.get(key)

    def put(self, key: str, record: Mapping[str, Any]) -> None:
        rec = {"key": key, **record}
        line = json.dumps(rec, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        with self._lock, file_lock(self.path.with_name(self.path.name + ".lock")):
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
            self._entries[key] = rec

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


@dataclass
class StepFailure:
    label: str
    status: str
    error: str
    attempts: int


@dataclass
class VerifiedItem(Generic[T]):
    index: int
    value: T
    verdict: Verdict


@dataclass
class MapReduceResult(Generic[T, R]):
    mapped: list[T | None]
    kept: list[VerifiedItem[T]]
    refuted: list[VerifiedItem[T]]
    failed: list[int]
    reduced: R | None


@dataclass
class WorkflowStats:
    agents_spawned: int = 0
    cache_hits: int = 0
    failures: list[StepFailure] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)


class Workflow:
    """A deterministic, resumable fan-out run on top of a :class:`SubagentManager`."""

    def __init__(
        self,
        manager: SubagentManager,
        ctx: RunContext,
        *,
        config: WorkflowConfig | None = None,
        journal: WorkflowJournal | None = None,
        name: str = "workflow",
    ) -> None:
        self.manager = manager
        self.ctx = ctx
        self.config = config or WorkflowConfig()
        self.journal = journal
        self.name = name
        self.stats = WorkflowStats()
        self.results: dict[str, list[SubagentResult]] = {}
        self._sem = threading.BoundedSemaphore(self.config.max_concurrent)
        self._lock = threading.Lock()

    @property
    def token(self):
        return self.manager.run_token

    # ------------------------------------------------------------------ observability
    def log(self, message: str) -> None:
        with self._lock:
            self.stats.log.append(message)
        self.manager._emit(self.ctx, {"phase": "workflow_log", "workflow": self.name, "message": message[:500]})

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        self.manager._emit(self.ctx, {"phase": "workflow_phase", "workflow": self.name, "name": name, "state": "start"})
        try:
            with self.manager.tracer.span(f"workflow_phase {name}", workflow=self.name):
                yield
        finally:
            self.manager._emit(self.ctx, {"phase": "workflow_phase", "workflow": self.name, "name": name, "state": "end"})

    def warn_if_large(self, n_agents: int, projected_tokens: int | None = None) -> list[str]:
        w = fanout_warnings(n_agents, projected_tokens, warn_agents=self.config.warn_agents, warn_tokens=self.config.warn_tokens)
        with self._lock:
            self.stats.warnings.extend(x for x in w if x not in self.stats.warnings)
        for x in w:
            self.log(f"warning: {x}")
        return w

    # ------------------------------------------------------------------ agent()
    def cache_key(self, agent: str, brief: DelegationBrief) -> str:
        definition = self.manager.registry.get(agent)
        payload = {
            "agent": agent, "definition": definition.fingerprint(), "brief": brief.render(),
            "output_max_tokens": brief.output_max_tokens, "tools_allowed": list(brief.tools_allowed or []),
            "write_permission": brief.write_permission.value if brief.write_permission else None,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def run_agent(self, agent: str, brief: DelegationBrief | Mapping[str, Any], *, label: str | None = None,
                  required: bool = False, max_restarts: int | None = None) -> SubagentResult | None:
        """Run one agent, restarting stalls / transient errors up to ``max_restarts`` times.

        Returns the last :class:`SubagentResult`, or ``None`` if the run was cancelled before the
        first attempt. Budget exhaustion, cancellation, refusals and structured-output failures are
        never retried. Does not consult the journal; :meth:`agent` does.
        """
        if not isinstance(brief, DelegationBrief):
            brief = DelegationBrief.from_mapping(brief)
        label = label or agent
        restarts = self.config.max_restarts if max_restarts is None else max_restarts
        last: SubagentResult | None = None
        attempts = 0
        with self._sem:
            for _ in range(restarts + 1):
                if self.token.cancelled:
                    break
                with self._lock:
                    if self.stats.agents_spawned >= self.config.max_agents:
                        raise WorkflowLimitError(f"workflow {self.name!r} reached max_agents={self.config.max_agents}")
                    self.stats.agents_spawned += 1
                attempts += 1
                last = self.manager.spawn(agent, brief, self.ctx)
                with self._lock:
                    self.results.setdefault(label, []).append(last)
                if last.ok or not last.status.retryable:
                    break
                self.log(f"{label}: attempt {attempts} ended {last.status.value}; restarting" if attempts <= restarts else f"{label}: giving up")
        if last is None or not last.ok:
            status = last.status.value if last is not None else SubagentStatus.CANCELLED.value
            error = last.error if last is not None else "workflow cancelled"
            with self._lock:
                self.stats.failures.append(StepFailure(label, status, error, attempts))
            if required:
                raise WorkflowStepFailed(f"required step {label!r} failed: {status}: {error}")
        return last

    def agent(self, agent: str, brief: DelegationBrief | Mapping[str, Any], *, label: str | None = None,
              required: bool = False, max_restarts: int | None = None) -> SubagentReport | None:
        """Claude Code ``agent()``: the validated report, or ``None`` on failure (run continues)."""
        if not isinstance(brief, DelegationBrief):
            brief = DelegationBrief.from_mapping(brief)
        key = self.cache_key(agent, brief) if self.journal is not None else None
        if key is not None:
            cached = self.journal.get(key)  # type: ignore[union-attr]
            if cached is not None:
                with self._lock:
                    self.stats.cache_hits += 1
                return SubagentReport.from_dict(cached["report"])
        result = self.run_agent(agent, brief, label=label, required=required, max_restarts=max_restarts)
        if result is None or not result.ok:
            return None
        assert result.report is not None
        if key is not None:
            self.journal.put(key, {"agent": agent, "label": label or agent, "report": result.report.to_dict()})  # type: ignore[union-attr]
        return result.report

    # ------------------------------------------------------------------ parallel / pipeline
    def parallel(self, thunks: Sequence[Callable[[], T]]) -> list[T | None]:
        """Run thunks concurrently; results in input order; a raising thunk yields ``None``.

        :class:`WorkflowLimitError` and :class:`WorkflowStepFailed` propagate (lowest index first).
        """
        n = len(thunks)
        if n > self.config.max_items:
            raise WorkflowLimitError(f"parallel() got {n} items (max {self.config.max_items})")
        if n == 0:
            return []
        if n > self.config.warn_agents:
            self.warn_if_large(n)
        results: list[T | None] = [None] * n
        errors: dict[int, BaseException] = {}

        def guarded(i: int) -> None:
            if self.token.cancelled:
                return
            try:
                results[i] = thunks[i]()
            except _FATAL as exc:
                errors[i] = exc
            except Exception as exc:  # partial failure: record and continue
                with self._lock:
                    self.stats.failures.append(StepFailure(f"parallel[{i}]", "exception", f"{type(exc).__name__}: {exc}", 1))

        with ThreadPoolExecutor(max_workers=min(n, self.config.max_concurrent), thread_name_prefix=f"{self.name}-par") as pool:
            first = pool.submit(guarded, 0)
            if n > 1 and self.config.prefix_stagger_s > 0:
                # Hold siblings briefly so they can read the first sibling's prompt-cache prefix.
                wait([first], timeout=self.config.prefix_stagger_s, return_when=FIRST_COMPLETED)
            rest = [pool.submit(guarded, i) for i in range(1, n)]
            for f in [first, *rest]:
                f.result()
        if errors:
            raise errors[min(errors)]
        return results

    def pipeline(self, items: Sequence[Any], *stages: Callable[[Any], Any]) -> list[Any | None]:
        """Push each item through ``stages`` in order (items run concurrently, stages sequentially per
        item). A stage returning ``None`` or raising stops that item; results keep input order."""
        if not stages:
            raise ValueError("pipeline() needs at least one stage")

        def run_item(item: Any) -> Any:
            value = item
            for stage in stages:
                value = stage(value)
                if value is None:
                    return None
            return value

        return self.parallel([lambda item=item: run_item(item) for item in items])

    def map_reduce(
        self,
        items: Sequence[T],
        map_fn: Callable[[T], R | None],
        reduce_fn: Callable[[list[VerifiedItem[R]]], Any],
        *,
        verify_fn: Callable[[T, R], Any] | None = None,
    ) -> MapReduceResult:
        """Map (fan-out) -> verify each result independently -> single reduce step.

        ``verify_fn`` returns a :class:`Verdict` (or bool / None). Refuted results are dropped;
        unverifiable ones are kept and labelled ``UNVERIFIED``. Never concatenates raw outputs: the
        reducer receives labelled items and is the single merge point.
        """
        mapped = self.pipeline(items, map_fn)
        failed = [i for i, m in enumerate(mapped) if m is None]
        ok = [(i, m) for i, m in enumerate(mapped) if m is not None]
        if verify_fn is not None:
            verdicts = self.parallel([lambda i=i, m=m: Verdict.coerce(verify_fn(items[i], m)) for i, m in ok])
            labelled = [VerifiedItem(i, m, v if v is not None else Verdict.UNVERIFIED) for (i, m), v in zip(ok, verdicts, strict=True)]
        else:
            labelled = [VerifiedItem(i, m, Verdict.UNVERIFIED) for i, m in ok]
        kept = [x for x in labelled if x.verdict is not Verdict.REFUTED]
        refuted = [x for x in labelled if x.verdict is Verdict.REFUTED]
        reduced = reduce_fn(kept) if kept else None
        return MapReduceResult(mapped=mapped, kept=kept, refuted=refuted, failed=failed, reduced=reduced)


__all__ = [
    "Workflow", "WorkflowConfig", "WorkflowJournal", "WorkflowStats", "StepFailure", "Verdict", "VerifiedItem", "MapReduceResult",
]
