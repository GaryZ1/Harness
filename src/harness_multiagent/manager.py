"""``SubagentManager``: the agent-as-tool spawn primitive with harness-enforced caps.

One primitive (reference 8.7 item 2): the parent passes a brief; the subagent runs in an **isolated
context** (own system prompt, brief only, scoped tools); only a **condensed, typed, untrusted report**
returns. Everything that research says must not be left to the prompt is enforced here, in this
order, before any model call (items 4, 11, 18):

========  =====================================================  ===========================
Order     Check                                                  Refusal code
========  =====================================================  ===========================
1         run cancelled / run budget spent                       ``cancelled`` / ``budget``
2         depth: ``ctx.depth + 1 <= max_depth`` (default 3)      ``depth_limit``
3         parent may delegate (``can_spawn``)                    ``nesting_disabled``
4         start hooks / spawn policy (privileged action)         ``policy``
5         agents per run (default 1,000)                         ``agent_limit``
6         concurrency slot (default 20; waits, cancellable)      ``concurrency_limit``
7         write lease / ownership / worktree for writers         ``lease_held`` / ``ownership``
========  =====================================================  ===========================

While running, each subagent has a child :class:`BudgetLedger` (per-agent tokens, rolling up into the
run ledger) and a child :class:`CancellationToken` (per-agent timeout, cancelled with the run). A run
budget hit cancels every running agent and refuses new spawns ("Budget limit reached"). Every spawn
emits ``EventType.SUBAGENT`` events (``start`` / ``stop`` / ``refused``) carrying structure, not
content (agent, run ids, depth, status, usage, turns, tool calls, wall clock), and one
``invoke_agent {name}`` tracer span.
"""

from __future__ import annotations

import hashlib
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from harness_core.types import (
    Event,
    EventSink,
    EventType,
    Message,
    NullTracer,
    RunContext,
    ToolCall,
    ToolOutcome,
    ToolSpec,
    Tracer,
    new_id,
)
from harness_memory.store import Access, FileMemoryStore

from . import cancel as _cancel
from .brief import DelegationBrief
from .budget import AgentRecord, BudgetLedger, BudgetLimits, CostFn, RunAccounting
from .cancel import CancellationToken
from .definitions import INHERIT_ROLE, AgentRegistry, ContextMode, MemoryAccess, SubagentDefinition, WriteAccess, select_tools
from .errors import (
    AgentLimitExceeded,
    BriefValidationError,
    BudgetExceeded,
    ConcurrencyLimitExceeded,
    DefinitionError,
    DepthLimitExceeded,
    LeaseHeldError,
    MultiAgentError,
    OwnershipConflict,
    PolicyDenied,
    SpawnRefused,
    UnknownAgentError,
    WorktreeError,
)
from .gate import DELEGATION_POLICY
from .leases import FileOwnership, Lease, LeaseManager
from .result import SubagentResult, SubagentStatus, condense, scan_untrusted
from .runner import AgentRunner, SubagentTask, compose_system_prompt, status_for_reason
from .tools import ScopedToolExecutor, ToolSource, as_handler
from .worktree import Worktree, WorktreeManager

SPAWN_TOOL_NAME = "spawn_subagent"

#: ``(definition, brief, parent_ctx) -> deny reason | None``: SubagentStart hooks / spawn policy.
StartHook = Callable[[SubagentDefinition, DelegationBrief, RunContext], "str | None"]
#: ``(result) -> feedback | None``: SubagentStop hooks; feedback is attached to the result.
StopHook = Callable[[SubagentResult], "str | None"]


@dataclass(frozen=True)
class ManagerConfig:
    """Harness-enforced caps. Defaults are Claude Code's shipped values unless marked otherwise."""

    max_depth: int = 3                          # CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH; 1 disables nesting
    max_concurrent: int = 20                    # CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS
    max_agents_per_run: int = 1_000             # workflow hard ceiling, applied to the whole run (ours)
    run_max_tokens: int | None = 2_000_000      # ours: mandatory run budget (vendor: no limit)
    run_max_usd: float | None = None            # needs cost_fn
    default_agent_max_tokens: int = 200_000     # ours
    default_timeout_s: float = 600.0            # ours
    result_max_tokens: int = 2_000              # condensed-return cap (~2k tokens)
    concurrency_wait_s: float | None = None     # None = wait for a slot (cancellable); 0 = refuse at once
    lease_wait_s: float = 0.0                   # 0 = refuse a second writer immediately
    join_grace_s: float = 5.0                   # how long a cancelled runner gets to return
    spawn_tool_name: str = SPAWN_TOOL_NAME
    shared_context: str = ""                    # project instructions / repo-status snapshot for fresh agents
    include_roster: bool = True                 # sibling roster in subagent system prompts

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1 (1 = subagents cannot nest)")
        if self.max_concurrent < 1 or self.max_agents_per_run < 1:
            raise ValueError("max_concurrent and max_agents_per_run must be >= 1")
        if self.result_max_tokens < 200:
            raise ValueError("result_max_tokens must be >= 200")


class SubagentManager:
    """Spawns isolated subagents; also a ``ToolExecutor`` exposing the spawn tool to a lead agent."""

    def __init__(
        self,
        registry: AgentRegistry,
        runner: AgentRunner,
        *,
        tools: ToolSource | None = None,
        config: ManagerConfig | None = None,
        leases: LeaseManager | None = None,
        ownership: FileOwnership | None = None,
        worktrees: WorktreeManager | None = None,
        memory_store: FileMemoryStore | None = None,
        event_sink: EventSink | None = None,
        tracer: Tracer | None = None,
        cost_fn: CostFn | None = None,
        start_hooks: Sequence[StartHook] = (),
        stop_hooks: Sequence[StopHook] = (),
        token: CancellationToken | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.registry = registry
        self.runner = runner
        self.config = config or ManagerConfig()
        self.leases = leases or LeaseManager()
        self.ownership = ownership or FileOwnership()
        self.worktrees = worktrees
        self.memory_store = memory_store
        self.event_sink = event_sink
        self.tracer: Tracer = tracer or NullTracer()
        self.start_hooks = list(start_hooks)
        self.stop_hooks = list(stop_hooks)
        self.monotonic = monotonic
        self.run_token = token or CancellationToken(monotonic=monotonic)
        self.ledger = BudgetLedger(
            "run", BudgetLimits(self.config.run_max_tokens, self.config.run_max_usd), cost_fn=cost_fn, on_exceeded=self._on_run_budget,
        )
        self.accounting = RunAccounting()
        self._tool_source = tools
        self._handler = as_handler(tools)
        self._slots = threading.BoundedSemaphore(self.config.max_concurrent)
        self._lock = threading.Lock()
        self._admitted = 0
        self._running: dict[str, tuple[CancellationToken, BudgetLedger]] = {}
        self._worktrees: dict[str, Worktree] = {}

    # ------------------------------------------------------------------ run-level controls
    def _on_run_budget(self, reason: str) -> None:
        self.run_token.cancel(_cancel.BUDGET, reason)

    def cancel_all(self, reason: str = _cancel.CANCELLED, detail: str = "cancelled by caller") -> None:
        """Cancel the run: running subagents stop at their next check; new spawns are refused."""
        self.run_token.cancel(reason, detail)

    @property
    def running(self) -> list[str]:
        with self._lock:
            return sorted(self._running)

    @property
    def admitted(self) -> int:
        with self._lock:
            return self._admitted

    def worktree(self, run_id: str) -> Worktree | None:
        """The worktree an ``own_worktree`` writer used (kept for the integrator to merge/remove)."""
        with self._lock:
            return self._worktrees.get(run_id)

    def _parent_specs(self) -> list[ToolSpec]:
        specs = getattr(self._tool_source, "specs", None)
        if callable(specs):
            return [s for s in specs() if s.name != self.config.spawn_tool_name]
        return []

    # ------------------------------------------------------------------ spawn
    def spawn(
        self,
        agent: str,
        brief: DelegationBrief | Mapping[str, Any],
        ctx: RunContext,
        *,
        history: Sequence[Message] | None = None,
        tool_use_id: str | None = None,
    ) -> SubagentResult:
        """Run one subagent to completion and return its condensed result.

        Raises :class:`UnknownAgentError` / :class:`BriefValidationError` for caller mistakes. All
        runtime outcomes, including refusals by caps, come back as a typed :class:`SubagentResult`.
        """
        definition = self.registry.get(agent)
        if not isinstance(brief, DelegationBrief):
            brief = DelegationBrief.from_mapping(brief)
        brief.check_against(definition, hard_output_cap=self.config.result_max_tokens)
        if definition.context is ContextMode.FORK and history is None:
            raise BriefValidationError(f"agent {agent!r} uses context: fork and needs the parent history")
        run_id = new_id("sub")
        parent_run_id = ctx.metadata.get("subagent_run_id")
        depth = ctx.depth + 1
        try:
            self._admit(definition, brief, ctx, depth)
        except SpawnRefused as exc:
            return self._refused(definition, ctx, run_id, parent_run_id, depth, exc, tool_use_id)
        try:
            self._acquire_slot()
        except SpawnRefused as exc:
            return self._refused(definition, ctx, run_id, parent_run_id, depth, exc, tool_use_id)
        try:
            return self._run_admitted(definition, brief, ctx, run_id, parent_run_id, depth, history, tool_use_id)
        finally:
            self._slots.release()

    def _admit(self, definition: SubagentDefinition, brief: DelegationBrief, ctx: RunContext, depth: int) -> None:
        parent_run_id = ctx.metadata.get("subagent_run_id")
        parent_token, parent_ledger = self._parent_controls(parent_run_id)
        if parent_token.cancelled:
            if parent_token.reason == _cancel.BUDGET:
                raise BudgetExceeded(parent_token.detail or "Budget limit reached")
            raise SpawnRefused(f"run cancelled ({parent_token.reason}); no new subagents", code="cancelled")
        reason = parent_ledger.exceeded()
        if reason:
            raise BudgetExceeded(reason)
        if depth > self.config.max_depth:
            raise DepthLimitExceeded(f"subagent depth {depth} exceeds max_depth={self.config.max_depth}; do the work yourself")
        if ctx.depth > 0:
            parent_name = ctx.metadata.get("subagent_definition", ctx.agent_name)
            parent_def = self.registry.get(parent_name) if parent_name in self.registry else None
            if parent_def is None or not parent_def.can_spawn:
                raise SpawnRefused(f"subagent {parent_name!r} is not allowed to delegate (can_spawn: false)", code="nesting_disabled")
        for hook in self.start_hooks:
            deny = hook(definition, brief, ctx)
            if deny:
                raise PolicyDenied(f"spawn of {definition.name!r} blocked: {deny}")
        with self._lock:
            if self._admitted >= self.config.max_agents_per_run:
                raise AgentLimitExceeded(f"run already spawned {self._admitted} agents (max_agents_per_run={self.config.max_agents_per_run})")
            self._admitted += 1

    def _parent_controls(self, parent_run_id: str | None) -> tuple[CancellationToken, BudgetLedger]:
        with self._lock:
            found = self._running.get(parent_run_id) if parent_run_id else None
        return found if found is not None else (self.run_token, self.ledger)

    def _acquire_slot(self) -> None:
        wait = self.config.concurrency_wait_s
        if wait == 0:
            if not self._slots.acquire(blocking=False):
                raise ConcurrencyLimitExceeded(f"{self.config.max_concurrent} subagents already running")
            return
        deadline = None if wait is None else self.monotonic() + wait
        while True:
            if self._slots.acquire(timeout=0.02):
                return
            if self.run_token.cancelled:
                raise SpawnRefused(f"run cancelled ({self.run_token.reason}) while waiting for a slot", code="cancelled")
            if deadline is not None and self.monotonic() >= deadline:
                raise ConcurrencyLimitExceeded(f"no free subagent slot within {wait}s ({self.config.max_concurrent} running)")

    def _setup_writes(
        self, definition: SubagentDefinition, brief: DelegationBrief, write_access: WriteAccess, workspace: Path, run_id: str, token: CancellationToken,
    ) -> tuple[list[Lease], Path, Worktree | None, bool]:
        """Acquire what this agent's write access needs. Returns (leases, workspace, worktree, claimed)."""
        wait = self.config.lease_wait_s
        leases: list[Lease] = []
        try:
            if write_access is WriteAccess.WORKSPACE:
                leases.append(self.leases.acquire_write(workspace, run_id, timeout=wait, token=token))
                return leases, workspace, None, False
            if write_access is WriteAccess.OWN_FILES:
                owned = brief.owned_files or definition.owned_files
                self.ownership.claim(run_id, owned)
                try:
                    for rel in owned:
                        leases.append(self.leases.acquire_write(workspace / rel, run_id, timeout=wait, token=token))
                except BaseException:
                    self.ownership.release(run_id)
                    raise
                return leases, workspace, None, True
            if write_access is WriteAccess.OWN_WORKTREE:
                if self.worktrees is None:
                    raise SpawnRefused(f"agent {definition.name!r} needs a worktree but no WorktreeManager is configured", code="no_worktree")
                wt = self.worktrees.create(definition.name, owned_files=brief.owned_files or definition.owned_files)
                with self._lock:
                    self._worktrees[run_id] = wt
                # Keyed by a non-path resource so it does not overlap the main checkout's lease.
                leases.append(self.leases.acquire_write(f"worktree:{wt.path.as_posix()}", run_id, timeout=wait, token=token))
                return leases, wt.path, wt, False
            leases.append(self.leases.acquire_read(workspace, run_id))
            return leases, workspace, None, False
        except BaseException:
            for lease in leases:
                lease.release()
            raise

    def _run_admitted(
        self,
        definition: SubagentDefinition,
        brief: DelegationBrief,
        ctx: RunContext,
        run_id: str,
        parent_run_id: str | None,
        depth: int,
        history: Sequence[Message] | None,
        tool_use_id: str | None,
    ) -> SubagentResult:
        parent_token, parent_ledger = self._parent_controls(parent_run_id)
        timeout_s = _min_opt(brief.timeout_s, definition.timeout_s) or self.config.default_timeout_s
        token = parent_token.child(timeout_s)
        write_access = brief.effective_write_access(definition)
        try:
            leases, workspace, worktree, claimed = self._setup_writes(definition, brief, write_access, Path(ctx.cwd), run_id, token)
        except (LeaseHeldError, OwnershipConflict, WorktreeError, SpawnRefused) as exc:
            code = "lease_held" if isinstance(exc, LeaseHeldError) else "ownership" if isinstance(exc, OwnershipConflict) else getattr(exc, "code", "worktree")
            return self._refused(definition, ctx, run_id, parent_run_id, depth, SpawnRefused(str(exc), code=code), tool_use_id)

        agent_max = _min_opt(brief.max_tokens, definition.max_tokens) or self.config.default_agent_max_tokens
        ledger = parent_ledger.child(f"{definition.name}:{run_id}", BudgetLimits(max_tokens=agent_max),
                                     on_exceeded=lambda reason: token.cancel(_cancel.BUDGET, reason))
        with self._lock:
            self._running[run_id] = (token, ledger)
        started = self.monotonic()
        result: SubagentResult | None = None
        try:
            model_role = definition.model_role
            if model_role == INHERIT_ROLE:
                model_role = str(ctx.metadata.get("model_role") or "")
                if not model_role:
                    raise DefinitionError(f"agent {definition.name!r} inherits the model role but the parent context has none")
            nested = definition.can_spawn and depth < self.config.max_depth
            specs = select_tools(definition, self._parent_specs(), allow_writes=write_access.writes, narrow_to=brief.tools_allowed)
            routes = {}
            if nested:
                specs.append(self.spawn_tool_spec())
                routes[self.config.spawn_tool_name] = self.execute
            scoped = ScopedToolExecutor(
                specs, self._handler, write_leases=[x for x in leases if x.info.mode == "write"],
                ownership=self.ownership if claimed else None, owner=run_id, workspace=workspace, routes=routes,
            )
            system_prompt = compose_system_prompt(
                definition, write_access=write_access, can_spawn=nested,
                roster=self.registry.catalog() if self.config.include_roster else "", shared_context=self.config.shared_context,
            )
            max_tool_calls = _min_opt(brief.max_tool_calls, definition.max_tool_calls)
            task = SubagentTask(
                run_id=run_id, definition=definition, brief=brief, system_prompt=system_prompt, task_message=brief.render(),
                tools=scoped, model_role=model_role, ledger=ledger, token=token,
                max_turns=min(brief.max_turns or definition.max_turns, definition.max_turns),
                max_tool_calls=max_tool_calls, max_output_tokens=definition.max_output_tokens,
                result_max_tokens=brief.output_max_tokens, write_access=write_access, effort=brief.effort or definition.effort,
                history=list(history or []) if definition.context is ContextMode.FORK else [],
                memory=self._memory_view(definition), parent_run_id=parent_run_id, workspace=str(workspace),
            )
            child_ctx = RunContext(
                session_id=ctx.session_id, cwd=str(workspace), agent_name=definition.name, depth=depth, tainted=ctx.tainted,
                metadata={
                    "subagent_run_id": run_id, "parent_run_id": parent_run_id, "subagent_definition": definition.name,
                    "model_role": model_role, "write_access": write_access.value, "memory_access": definition.memory.value,
                    "parent_tool_use_id": tool_use_id,
                },
            )
            self._emit(ctx, {
                "phase": "start", "agent": definition.name, "run_id": run_id, "parent_run_id": parent_run_id, "depth": depth,
                "parent_tool_use_id": tool_use_id, "model_role": model_role, "write_access": write_access.value,
                "tools": scoped.names, "objective_sha": hashlib.sha256(brief.objective.encode("utf-8")).hexdigest()[:12],
                "workspace": str(workspace) if worktree else None,
            })
            with self.tracer.span(f"invoke_agent {definition.name}", agent=definition.name, run_id=run_id, depth=depth,
                                  parent_run_id=parent_run_id, parent_tool_use_id=tool_use_id):
                result = self._execute(task, child_ctx, token, ledger)
        except MultiAgentError as exc:
            result = SubagentResult(definition.name, run_id, SubagentStatus.ERROR_DURING_EXECUTION, depth=depth,
                                    parent_run_id=parent_run_id, error=str(exc))
        finally:
            for lease in leases:
                lease.release()
            if claimed:
                self.ownership.release(run_id)
            with self._lock:
                self._running.pop(run_id, None)
        assert result is not None
        result = self._finalize(result, definition, brief, run_id, parent_run_id, depth, ledger, started, worktree)
        self.accounting.add(AgentRecord(
            run_id=run_id, agent=definition.name, parent_run_id=parent_run_id, depth=depth, status=result.status.value,
            model=result.model, usage=result.usage, usd=ledger.usd, turns=result.turns, tool_calls=result.tool_calls,
            wall_clock_s=result.wall_clock_s,
        ))
        for hook in self.stop_hooks:
            feedback = hook(result)
            if feedback:
                result.warnings.append(f"stop hook: {feedback}")
                result.metadata.setdefault("stop_hook_feedback", []).append(feedback)
        self._emit(ctx, {
            "phase": "stop", "agent": definition.name, "run_id": run_id, "parent_run_id": parent_run_id, "depth": depth,
            "parent_tool_use_id": tool_use_id, "status": result.status.value, "model": result.model,
            "usage": asdict(result.usage), "usd": round(ledger.usd, 6), "turns": result.turns, "tool_calls": result.tool_calls,
            "wall_clock_s": round(result.wall_clock_s, 3), "warnings": len(result.warnings), "error": result.error[:200],
        })
        return result

    def _execute(self, task: SubagentTask, ctx: RunContext, token: CancellationToken, ledger: BudgetLedger) -> SubagentResult:
        """Run the runner on a worker thread so a hung model call cannot outlive the deadline."""
        box: dict[str, Any] = {}

        def target() -> None:
            try:
                box["result"] = self.runner.run(task, ctx)
            except BaseException as exc:  # surfaced as a typed failure below
                box["error"] = exc

        th = threading.Thread(target=target, name=f"subagent-{task.agent}-{task.run_id}", daemon=True)
        th.start()
        while th.is_alive():
            th.join(0.02)
            if token.cancelled and th.is_alive():
                th.join(self.config.join_grace_s)
                break
        if "result" in box:
            return box["result"]
        if "error" in box:
            exc = box["error"]
            return SubagentResult(task.agent, task.run_id, SubagentStatus.ERROR_DURING_EXECUTION, usage=ledger.usage,
                                  error=f"runner raised {type(exc).__name__}: {exc}")
        status = status_for_reason(token.reason or _cancel.CANCELLED)
        return SubagentResult(task.agent, task.run_id, status, usage=ledger.usage,
                              error=f"{token.detail or status.value}; runner did not stop within {self.config.join_grace_s}s and was abandoned")

    def _finalize(
        self, result: SubagentResult, definition: SubagentDefinition, brief: DelegationBrief, run_id: str, parent_run_id: str | None,
        depth: int, ledger: BudgetLedger, started: float, worktree: Worktree | None,
    ) -> SubagentResult:
        result.agent, result.run_id, result.depth, result.parent_run_id = definition.name, run_id, depth, parent_run_id
        if ledger.tokens == 0 and result.usage.total_input + result.usage.output_tokens > 0:
            # The runner did not charge as it went; charge now so run totals stay right.
            ledger.charge(result.usage, model=result.model)
        if result.wall_clock_s <= 0:
            result.wall_clock_s = self.monotonic() - started
        if result.status is SubagentStatus.SUCCESS and result.report is None:
            result.status, result.error = SubagentStatus.ERROR_DURING_EXECUTION, "runner returned success without a report"
        if result.report is not None:
            text = " ".join([result.report.summary, *(f.claim + " " + f.evidence for f in result.report.findings)])
            for w in scan_untrusted(text):
                if w not in result.warnings:
                    result.warnings.append(w)
            result.report = condense(result.report, brief.output_max_tokens)
        if worktree is not None:
            result.workspace = str(worktree.path)
            result.metadata.update({"worktree_branch": worktree.branch, "worktree_base": worktree.base})
        return result

    def _memory_view(self, definition: SubagentDefinition) -> FileMemoryStore | None:
        """Subagents never get the main agent's writable memory unless explicitly granted."""
        store = self.memory_store
        if store is None or definition.memory is MemoryAccess.NONE:
            return None
        if definition.memory is MemoryAccess.READ_WRITE:
            return store
        return FileMemoryStore(store.root, scope=store.scope, access=Access.READ_ONLY, limits=store.limits,
                               redactor=store.redactor, clock=store.clock, create=False)

    def _refused(
        self, definition: SubagentDefinition, ctx: RunContext, run_id: str, parent_run_id: str | None, depth: int,
        exc: SpawnRefused, tool_use_id: str | None,
    ) -> SubagentResult:
        result = SubagentResult(definition.name, run_id, SubagentStatus.REFUSED, depth=depth, parent_run_id=parent_run_id,
                                error=str(exc), refusal_code=exc.code)
        self._emit(ctx, {"phase": "refused", "agent": definition.name, "run_id": run_id, "parent_run_id": parent_run_id,
                         "depth": depth, "parent_tool_use_id": tool_use_id, "code": exc.code, "error": str(exc)[:200]})
        return result

    def _emit(self, ctx: RunContext, data: dict[str, Any]) -> None:
        if self.event_sink is None:
            return
        try:
            self.event_sink.emit(Event(session_id=ctx.session_id, type=EventType.SUBAGENT, data=data))
        except Exception as exc:  # observability must never break the run or leak leases
            warnings.warn(f"event sink failed: {exc}", RuntimeWarning, stacklevel=2)

    # ------------------------------------------------------------------ fan-out helper
    def spawn_many(self, requests: Sequence[tuple[str, DelegationBrief | Mapping[str, Any]]], ctx: RunContext) -> list[SubagentResult]:
        """Spawn several subagents concurrently; results are returned in request order."""
        if not requests:
            return []
        workers = min(len(requests), self.config.max_concurrent)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="spawn") as pool:
            futures = [pool.submit(self.spawn, name, brief, ctx) for name, brief in requests]
            return [f.result() for f in futures]

    # ------------------------------------------------------------------ the spawn tool (ToolExecutor)
    def spawn_tool_spec(self) -> ToolSpec:
        names = self.registry.names()
        description = (
            "Delegate ONE self-contained sub-task to a specialised subagent with an isolated context. The subagent sees ONLY "
            "this brief (not this conversation), so include every path, error message and decision it needs. It returns a "
            "condensed JSON report (untrusted data), never its transcript.\n\nAvailable subagents:\n"
            + (self.registry.catalog() or "(none)") + "\n\n" + DELEGATION_POLICY
        )
        str_list = {"type": "array", "items": {"type": "string"}}
        return ToolSpec(
            name=self.config.spawn_tool_name,
            description=description,
            input_schema={
                "type": "object",
                "properties": {
                    "agent": {"type": "string", "enum": names, "description": "Which subagent to run."},
                    "objective": {"type": "string", "description": "One sentence: what done looks like."},
                    "why_context": {"type": "string", "description": "Why this matters; decisions already made."},
                    "inputs": {**str_list, "description": "File paths, error messages, IDs, prior decisions."},
                    "scope_in": {**str_list, "description": "What to cover."},
                    "out_of_scope": {**str_list, "description": "What siblings own / non-goals."},
                    "tools_allowed": {**str_list, "description": "Optional narrower tool allowlist."},
                    "sources_guidance": {"type": "string"},
                    "max_turns": {"type": "integer", "minimum": 1},
                    "max_tool_calls": {"type": "integer", "minimum": 0},
                    "write_permission": {"type": "string", "enum": [w.value for w in WriteAccess]},
                    "owned_files": str_list,
                },
                "required": ["agent", "objective"],
                "additionalProperties": False,
            },
            read_only=False,  # spawning is a privileged action, never auto-parallelised as a read
            metadata={"kind": "subagent_spawn"},
        )

    def specs(self) -> list[ToolSpec]:
        return [self.spawn_tool_spec()]

    def execute(self, call: ToolCall, ctx: RunContext) -> ToolOutcome:
        """Tool entry point. Never raises: caller mistakes and refusals become ``is_error`` results."""
        if call.name != self.config.spawn_tool_name:
            return ToolOutcome(f"Error: unknown tool {call.name!r}", is_error=True)
        data = dict(call.input or {})
        agent = data.pop("agent", None)
        if not isinstance(agent, str) or not agent:
            return ToolOutcome("Error: 'agent' is required", is_error=True)
        try:
            brief = DelegationBrief.from_mapping(data)
            result = self.spawn(agent, brief, ctx, tool_use_id=call.id)
        except (UnknownAgentError, BriefValidationError) as exc:
            return ToolOutcome(f"Error: {exc}", is_error=True)
        return result.to_tool_outcome(max_tokens=brief.output_max_tokens)


def _min_opt(*values: Any) -> Any:
    present = [v for v in values if v is not None]
    return min(present) if present else None


__all__ = ["SubagentManager", "ManagerConfig", "SPAWN_TOOL_NAME", "StartHook", "StopHook"]
