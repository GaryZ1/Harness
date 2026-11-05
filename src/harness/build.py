"""``build_harness``: construct every package and connect them into one working harness.

Wiring (see ``docs/integration.md`` for why each seam is resolved the way it is)::

    model client ── OverflowGuardClient ─────────────────────────────▶ AgentLoop (harness_core)
    SessionStore session (durable JSONL) ◀── every message / event ───┤
    LayeredContextAssembler ◀── SessionBoundAssembler (canonical log, ┤ assembler; post_model hook →
        PromptStack ← MemorySystem.session_start_context              │   observe_response
        Compactor (harness_context) ◀── ContextCompactorAdapter ──────┤ compactor (overflow path)
    PolicyEngine ◀── EnginePolicy (DENY text = denial_outcome) ───────┤ policy; ASK → approver/ApprovalBroker
        post_tool hook → record_result; pre_model/pre_tool → must_terminate
    task-state guard_tool_call ──────────────────────────────────────┤ pre_tool hook
    OffloadingToolExecutor(ToolRegistry: kernel(Workspace, sandbox runner), memory tool,
                           spawn_subagent (SubagentManager → LoopAgentRunner), tool_search, MCP) ─┤ executor
    OpsTracer(JsonlSpanExporter) ────────────────────────────────────┤ tracer
    OpsEventAdapter → FanoutEventSink(HarnessMetrics, JsonlEventExporter) ┘ extra sink
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from harness_context import (
    AssemblerConfig,
    BudgetPolicy,
    FileActivityTracker,
    LayeredContextAssembler,
    LayerKind,
    OffloadingToolExecutor,
    PromptLayer,
    PromptStack,
    RecentFilesSource,
    ReinjectionBuilder,
    TextSource,
    ToolOutputOffloader,
)
from harness_context import Compactor as ContextCompactor
from harness_context.budget import DEFAULT_COMPACTION_THRESHOLD
from harness_core.budgets import Budgets
from harness_core.hooks import HookEvent, HookInput, HookRegistry, HookResult
from harness_core.loop import AgentLoop, LoopConfig
from harness_core.permissions import Approver as CoreApprover
from harness_core.results import RunResult
from harness_core.session import Session, SessionStore
from harness_core.types import (
    EventSink,
    Message,
    ModelClient,
    PermissionBehavior,
    PermissionDecision,
    RunContext,
    ToolCall,
    ToolExecutor,
    ToolOutcome,
    ToolSpec,
    new_id,
)
from harness_memory import MemorySystem
from harness_memory.errors import ValidationError as MemoryValidationError
from harness_memory.task_state import guard_tool_call
from harness_multiagent import DELEGATION_POLICY, AgentRegistry, ManagerConfig, SubagentManager
from harness_multiagent.manager import SPAWN_TOOL_NAME
from harness_ops import FanoutEventSink, HarnessMetrics, JsonlEventExporter, JsonlSpanExporter, OpsTracer, PriceTable
from harness_permissions import ApprovalBroker, ApprovalScope, PendingApproval, PolicyEngine, denial_outcome, load_layers
from harness_permissions import Approver as HumanApprover
from harness_permissions import Workspace as PermissionWorkspace
from harness_sandbox import SandboxPolicy, create_runner, describe_sandbox, load_policy
from harness_tools import (
    MCPServerConfig,
    MemoryCheckpointer,
    Tool,
    ToolRegistry,
    Workspace,
    connect_mcp_servers,
    enable_tool_search,
    register_kernel,
)

from .adapters import (
    ContextCompactorAdapter,
    ExcludingExecutor,
    LoopAgentRunner,
    OpsEventAdapter,
    OverflowGuardClient,
    SessionBoundAssembler,
    SessionEventSink,
    StepIndexSink,
    ToolCallCap,
)
from .config import HarnessConfig, HarnessConfigError

MEMORY_TOOL_DESCRIPTION = (
    "Persistent memory mounted at /memories (view, create, str_replace, insert, delete, rename). "
    "Save durable learnings; never store secrets."
)

#: ``approval_ui(pending) -> True | "once" | "session" | False | None`` - a stand-in for a human UI.
#: ``None`` leaves the item pending (another thread/process answers through ``harness.broker``).
ApprovalUI = Callable[[PendingApproval], Any]


class EnginePolicy:
    """The loop's :class:`~harness_core.types.PermissionPolicy` over :class:`harness_permissions.PolicyEngine`.

    ALLOW / ASK pass through (ASK goes to the loop's approver). For DENY the reason is rewritten so
    the loop's tool result (``"Permission denied: " + reason``) is exactly
    :func:`harness_permissions.denial_outcome` for that decision.
    """

    def __init__(self, engine: PolicyEngine) -> None:
        self.engine = engine

    def check(self, call: ToolCall, spec: ToolSpec | None, ctx: RunContext) -> PermissionDecision:
        decision = self.engine.check(call, spec, ctx)
        if decision.behavior is PermissionBehavior.DENY:
            text = denial_outcome(decision).content
            assert isinstance(text, str)
            return PermissionDecision(PermissionBehavior.DENY, text.removeprefix("Permission denied: "), decision.updated_input)
        return decision


def make_approver(engine: PolicyEngine, broker: ApprovalBroker, *, approver: CoreApprover | None, approval_ui: ApprovalUI | None,
                  user_id: str, wait_for_broker: bool, timeout_s: float | None = None) -> CoreApprover | None:
    """Route the loop's ASK decisions: ``approval_ui`` (through the broker) > ``approver`` > blocking broker wait."""
    if approval_ui is not None:
        def via_broker(call: ToolCall, spec: ToolSpec | None, ctx: RunContext, decision: PermissionDecision) -> PermissionDecision:
            item = broker.submit(call, spec, ctx, decision, timeout_s=timeout_s)
            answer = approval_ui(item)
            human = HumanApprover.human(user_id)
            if answer is True or answer in ("once", "session"):
                scope = ApprovalScope.SESSION if answer == "session" else ApprovalScope.ONCE
                broker.approve(item.id, human, scope=scope)
            elif answer is False:
                broker.deny(item.id, human, reason="denied by the user")
            resolved = broker.wait(item.id, timeout_s)
            if resolved.allowed and decision.updated_input is not None:
                resolved.updated_input = decision.updated_input
            return resolved
        return via_broker
    if approver is not None:
        def direct(call: ToolCall, spec: ToolSpec | None, ctx: RunContext, decision: PermissionDecision) -> bool | PermissionDecision:
            t0 = time.monotonic()
            answer = approver(call, spec, ctx, decision)
            ok = answer.allowed if isinstance(answer, PermissionDecision) else bool(answer)
            try:
                engine.record_human_decision(ctx.session_id, ok, time.monotonic() - t0)
            except Exception:  # noqa: BLE001 - bookkeeping only
                pass
            return answer
        return direct
    if wait_for_broker:
        def blocking(call: ToolCall, spec: ToolSpec | None, ctx: RunContext, decision: PermissionDecision) -> PermissionDecision:
            return broker.request(call, spec, ctx, decision, timeout_s=timeout_s)
        return blocking
    return None


@dataclass
class Harness:
    """A wired harness. ``run``/``steer``/``end_session``/``metrics`` plus every underlying component."""

    config: HarnessConfig
    loop: AgentLoop
    session: Session
    store: SessionStore | None
    model_client: ModelClient
    executor: ToolExecutor
    registry: ToolRegistry | None
    kernel: Any
    workspace: Workspace | None
    runner: Any
    sandbox_policy: SandboxPolicy
    engine: PolicyEngine
    broker: ApprovalBroker
    assembler: LayeredContextAssembler
    context_compactor: ContextCompactor
    memory: MemorySystem | None
    memory_session: Any
    manager: SubagentManager | None
    agent_runner: LoopAgentRunner | None
    tracer: OpsTracer
    ops_metrics: HarnessMetrics
    event_sink: FanoutEventSink
    prices: PriceTable | None = None
    mcp_clients: list[Any] = field(default_factory=list)
    file_tracker: FileActivityTracker | None = None
    last_reply: str = ""
    last_raw_reply: str = ""
    _sessions: dict[str, Session] = field(default_factory=dict)
    _ended: bool = False

    # ------------------------------------------------------------------ lifecycle
    def run(self, prompt: str | Message | None = None, **kwargs: Any) -> RunResult:
        """Run one turn. The returned ``final_text`` has the memory-citation trailer removed (citations are recorded)."""
        if self.memory_session is not None and prompt is not None:
            self.memory_session.record("user", prompt if isinstance(prompt, str) else prompt.text)
        result = self.loop.run(prompt, **kwargs)
        self.last_raw_reply = result.final_text
        if self.memory_session is not None and result.final_text:
            self.last_reply = self.memory_session.finish_reply(result.final_text)
            result = replace(result, final_text=self.last_reply)
        else:
            self.last_reply = result.final_text
        self.write_metrics()
        return result

    def steer(self, text: str) -> None:
        self.loop.steer(text)

    def interrupt(self) -> None:
        self.loop.interrupt()

    def compact(self, focus: str | None = None) -> Any:
        """Manual ``/compact [focus]``: appends a compaction marker to the durable log."""
        return self.assembler.compact(self.loop.system, self.session.messages(), list(self.executor.specs()), self.loop.ctx, focus=focus)

    def end_session(self, reason: str = "", *, progress_summary: str | None = None) -> None:
        """Fire session_end, persist memory taint/commit (``MemorySystem.end_session``), release the sandbox."""
        if self._ended:
            return
        self._ended = True
        try:
            self.loop.end_session(reason)
        finally:
            if self.memory is not None and self.memory_session is not None:
                try:
                    self.memory.end_session(self.memory_session, progress_summary=progress_summary)
                except Exception:  # noqa: BLE001 - never lose the rest of the shutdown
                    pass
            for c in self.mcp_clients:
                try:
                    c.close()
                except Exception:  # noqa: BLE001
                    pass
            close = getattr(self.runner, "close", None)
            if callable(close):
                close()
            self.write_metrics()

    def __enter__(self) -> "Harness":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.end_session("closed")

    # ------------------------------------------------------------------ observability
    def metrics(self) -> dict[str, Any]:
        """The harness_ops dashboard (cache hit rate, cost, tool error rate, permission rates, stop reasons...)."""
        return self.ops_metrics.dashboard()

    def write_metrics(self) -> Path:
        path = self.config.telemetry_path / "metrics.prom"
        try:
            path.write_text(self.ops_metrics.render_prometheus(), encoding="utf-8")
        except OSError:
            pass
        return path

    def context_report(self) -> Any:
        return self.assembler.report(self.session.id)

    @property
    def telemetry_dir(self) -> Path:
        return self.config.telemetry_path


# ============================================================================= builder


def _load_sandbox_policy(config: HarnessConfig) -> SandboxPolicy:
    policy = load_policy(config.sandbox.policy_file) if config.sandbox.policy_file else SandboxPolicy()
    if config.sandbox.mode:
        policy = policy.with_mode(config.sandbox.mode)
    return policy


def _open_session(config: HarnessConfig, session_id: str | None) -> tuple[Session, SessionStore | None]:
    sid = session_id or new_id("sess")
    if not config.durable_sessions:
        return Session.in_memory(sid), None
    store = SessionStore(config.state_path / "sessions")
    session = store.wake(sid) if store.exists(sid) else store.create(sid)
    return session, store


def _start_memory_session(ms: MemorySystem, session_id: str) -> Any:
    candidate, n = session_id, 0
    while True:
        try:
            return ms.start_session(candidate)
        except MemoryValidationError:
            n += 1  # resumed core session: memory transcripts are per process run
            candidate = f"{session_id}-r{n}"
            if n > 1000:
                raise


def build_harness(
    config: HarnessConfig,
    *,
    model_client: ModelClient | None = None,
    approver: CoreApprover | None = None,
    approval_ui: ApprovalUI | None = None,
    session_id: str | None = None,
    tool_executor: ToolExecutor | None = None,
    agents: AgentRegistry | None = None,
    hooks: HookRegistry | None = None,
    evaluators: Sequence[Any] = (),
    sinks: Sequence[EventSink] = (),
    bash_env: Mapping[str, str] | None = None,
    rng_seed: int | None = None,
    max_tool_calls: int | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
    loop_config: LoopConfig | None = None,
) -> Harness:
    """Construct and connect every package (see the module docstring for the wiring).

    :param model_client: inject a client (tests: ``harness_core.testing.ScriptedModelClient``);
        default ``AnthropicModelClient(config.model, effort=config.effort)`` (needs ``pip install anthropic``).
    :param approver: core-style approver ``(call, spec, ctx, decision) -> bool | PermissionDecision`` for ASK.
    :param approval_ui: route ASK through the :class:`ApprovalBroker`; called with the pending item.
    :param session_id: resume (``SessionStore.wake``) when it exists, else create it.
    :param tool_executor: replace the tool registry entirely (RL: the rollout-scoped backend).
    :param agents: sub-agent definitions (default: ``<workspace>/.harness/agents`` + ``config.agents_dirs``).
    """
    ws = config.workspace_path
    if not ws.is_dir():
        raise HarnessConfigError(f"workspace does not exist: {ws}")
    state = config.state_path
    state.mkdir(parents=True, exist_ok=True)
    tel = config.telemetry_path
    tel.mkdir(parents=True, exist_ok=True)

    # ---- model ------------------------------------------------------------------------------
    if model_client is None:
        from harness_core.anthropic_adapter import AnthropicModelClient

        model_client = AnthropicModelClient(config.model, effort=config.effort)
    guarded = OverflowGuardClient(model_client)

    # ---- observability (harness_ops) ---------------------------------------------------------
    prices = PriceTable.load(config.pricing_file, strict=False) if config.pricing_file else None
    ops_metrics = HarnessMetrics(prices=prices)
    fanout = FanoutEventSink([ops_metrics, JsonlEventExporter(tel / "events.jsonl")])
    ops_sink = OpsEventAdapter(fanout)
    audit_sink = FanoutEventSink([JsonlEventExporter(tel / "audit.jsonl")])
    tracer = OpsTracer([JsonlSpanExporter(tel / "spans.jsonl")], resource={"service.name": "harness"})
    cost_estimator = None
    if prices is not None:
        def cost_estimator(usage: Any, model: str) -> float:
            return float(prices.cost(model, usage).total_usd)

    # ---- session log -----------------------------------------------------------------------
    session, store = _open_session(config, session_id)
    forward = [ops_sink, *sinks]
    session_sink = SessionEventSink(session, forward)

    # ---- sandbox (harness_sandbox) -----------------------------------------------------------
    sandbox_policy = _load_sandbox_policy(config)
    runner = create_runner(sandbox_policy, str(ws), require_isolation=config.sandbox.require_isolation, session_id=session.id,
                           event_sink=session_sink)
    choice = runner.choice

    # ---- memory (harness_memory) -----------------------------------------------------------
    ms: MemorySystem | None = None
    msession: Any = None
    if config.features.memory:
        ms = MemorySystem(base=config.base_path, user_id=config.user_id, repo_root=ws, git=config.features.memory_git,
                          context_window_tokens=config.context_window_tokens)
        msession = _start_memory_session(ms, session.id)

    # ---- permissions (harness_permissions) ------------------------------------------------------
    session_flags: dict[str, Any] = {}
    if not choice.auto_allow_in_boundary:
        # No real boundary: shell commands are not boundary-auto-allowed (docs/sandbox: auto-allow
        # in-boundary actions only when BackendChoice.auto_allow_in_boundary is true).
        session_flags = {"defaults": {"sandbox": "none"}}
    perm_policy = load_layers(managed=config.permissions.managed_policy_files, project=config.permissions.policy_files,
                              user=config.permissions.user_policy_files, session=session_flags or None)
    sessions: dict[str, Session] = {session.id: session}

    def transcript(ctx: RunContext) -> list[Message]:
        s = sessions.get(ctx.session_id)
        return s.messages() if s is not None else list(ctx.metadata.get("messages") or [])

    pws = PermissionWorkspace(roots=list(perm_policy["workspace.roots"]) or [str(ws)],
                              read_roots=list(perm_policy["workspace.read_roots"]), temp_roots=list(perm_policy["workspace.temp_roots"]))
    engine = PolicyEngine(perm_policy, workspace=pws, cwd=str(ws), sink=audit_sink, interactive=config.permissions.interactive,
                          transcript_provider=transcript)
    broker = ApprovalBroker(engine, state_path=state / "approvals.json", default_timeout_s=config.permissions.approval_timeout_s)
    loop_policy = EnginePolicy(engine)
    loop_approver = make_approver(engine, broker, approver=approver, approval_ui=approval_ui, user_id=config.user_id,
                                  wait_for_broker=config.permissions.wait_for_broker, timeout_s=config.permissions.approval_timeout_s)

    # ---- tools (harness_tools + memory tool + spawn tool) -------------------------------------
    registry: ToolRegistry | None = None
    kernel = None
    tws: Workspace | None = None
    manager: SubagentManager | None = None
    agent_runner: LoopAgentRunner | None = None
    mcp_clients: list[Any] = []
    if tool_executor is not None:
        inner: ToolExecutor = tool_executor
    else:
        registry = ToolRegistry(tracer=tracer)
        tws = Workspace(ws, checkpointer=MemoryCheckpointer())
        kernel = register_kernel(registry, tws, runner=runner, profile=config.tool_profile,  # type: ignore[arg-type]
                                 bash_env=dict(bash_env) if bash_env else None)
        if msession is not None:
            mem_handler = msession.memory_tool

            def memory_tool(inp: dict[str, Any], inv: Any) -> ToolOutcome:
                r = mem_handler.handle(inp)
                return ToolOutcome(r.content, is_error=r.is_error)

            registry.register(Tool(
                spec=ToolSpec("memory", MEMORY_TOOL_DESCRIPTION, provider_type="memory_20250818", metadata={"permission_kind": "memory"}),
                handler=memory_tool, call_style="raw",
            ))
        if config.features.mcp_servers:
            mcp_clients = connect_mcp_servers(registry, [
                MCPServerConfig(s.name, s.command, tuple(s.args), env=dict(s.env) or None, cwd=str(ws), trusted=s.trusted)
                for s in config.features.mcp_servers
            ])
        if config.features.subagents:
            agent_registry = agents if agents is not None else AgentRegistry.from_project(ws, user_dir=None, strict=False)
            if agents is None and config.agents_dirs:
                agent_registry.load_dirs(config.agents_dirs, strict=False)
            if len(agent_registry):
                def on_child_start(child: Session, ctx: RunContext) -> None:
                    sessions[child.id] = child

                def on_child_finish(child: Session, result: Any) -> None:
                    sessions.pop(child.id, None)
                    try:
                        engine.subagent_finished(child.id)
                    except Exception:  # noqa: BLE001
                        pass

                agent_runner = LoopAgentRunner(
                    guarded, model_for_role=config.model_for_role, policy=loop_policy, approver=loop_approver, tracer=tracer,
                    sinks=[ops_sink], cost_estimator=cost_estimator, on_session_start=on_child_start, on_finish=on_child_finish,
                )
                manager = SubagentManager(agent_registry, agent_runner, tools=ExcludingExecutor(registry, {"memory", SPAWN_TOOL_NAME}),
                                          config=ManagerConfig(), event_sink=session_sink, tracer=tracer)
                spawn_spec = manager.spawn_tool_spec()
                spawn_spec.metadata["permission_kind"] = "subagent"
                mgr = manager

                def spawn(inp: dict[str, Any], inv: Any) -> ToolOutcome:
                    return mgr.execute(ToolCall(inv.call.id, inv.call.name, dict(inp)), inv.ctx)

                registry.register(Tool(spec=spawn_spec, handler=spawn, call_style="raw", timeout_s=None))
        if config.features.tool_search or registry.deferred_tools():
            enable_tool_search(registry)
            ts = registry.get("tool_search")
            if ts is not None:
                ts.spec.metadata.setdefault("permission_kind", "internal")
        registry.freeze()
        inner = registry
    executor: ToolExecutor = inner
    if max_tool_calls is not None:
        executor = ToolCallCap(executor, max_tool_calls)
    if config.features.offload:
        executor = OffloadingToolExecutor(executor, ToolOutputOffloader(ws / ".harness" / "tool-outputs", display_path=lambda p: p.as_posix()))

    # ---- context (harness_context) -------------------------------------------------------------
    stack = PromptStack()
    stack.add(PromptLayer("sandbox", describe_sandbox(sandbox_policy, workspace=str(ws), choice=choice), LayerKind.SYSTEM))
    if manager is not None:
        stack.add(PromptLayer("delegation-policy", DELEGATION_POLICY, LayerKind.SYSTEM))
    tracker = FileActivityTracker()
    sources: list[Any] = [RecentFilesSource(tracker, ws)]
    if ms is not None:
        stack.add(PromptLayer("memory", ms.session_start_context(msession, include_task_state=False), LayerKind.PROJECT))
        if ms.task.initialized:
            stack.add(PromptLayer("task-state", ms.task.orient().render(), LayerKind.PROJECT))
        mem, msess = ms, msession
        sources.insert(0, TextSource("memory", lambda: mem.post_compaction_context(msess)))
    reinjection = ReinjectionBuilder(sources)
    context_compactor = ContextCompactor(guarded, reinjection=reinjection, events=session_sink, tracer=tracer)
    budget = BudgetPolicy(window_tokens=config.context_window_tokens,
                          compaction_threshold_tokens=config.compaction_threshold_tokens or DEFAULT_COMPACTION_THRESHOLD)
    bound = SessionBoundAssembler(session, sinks=forward)
    assembler = LayeredContextAssembler(
        stack=stack, config=AssemblerConfig(budget=budget, max_tokens=config.max_tokens), compactor=context_compactor,
        events=session_sink, tracer=tracer, append_to_log=bound.append_to_log, state_dir=state / "context",
    )
    bound.inner = assembler
    core_compactor = ContextCompactorAdapter(assembler, session, system=config.system_prompt, tools=executor.specs)

    # ---- hooks ---------------------------------------------------------------------------------
    hooks = hooks or HookRegistry()
    cfg_loop = replace(loop_config or LoopConfig(), max_tokens=config.max_tokens, model_label=config.model,
                       context_window_tokens=None, compaction_threshold_tokens=None)
    b = config.budgets
    loop = AgentLoop(
        guarded, executor, system=config.system_prompt, config=cfg_loop,
        budgets=Budgets(max_turns=b.max_turns, max_total_tokens=b.max_total_tokens, max_wall_clock_s=b.max_wall_clock_s,
                        max_budget_usd=b.max_budget_usd),
        assembler=bound, policy=loop_policy, approver=loop_approver, hooks=hooks, evaluators=evaluators, compactor=core_compactor,
        session=session, sinks=forward, tracer=tracer, cost_estimator=cost_estimator, cwd=str(ws),
        rng=random.Random(rng_seed) if rng_seed is not None else None,
    )
    loop.sinks.insert(0, StepIndexSink(loop.ctx))
    md = loop.ctx.metadata
    md["interactive"] = config.permissions.interactive
    if config.permissions.mode:
        md["permission_mode"] = config.permissions.mode
    md.update(dict(extra_metadata or {}))
    _register_hooks(hooks, loop=loop, engine=engine, assembler=bound, msession=msession, workspace=ws, tracker=tracker)

    return Harness(
        config=config, loop=loop, session=session, store=store, model_client=model_client, executor=executor, registry=registry,
        kernel=kernel, workspace=tws, runner=runner, sandbox_policy=sandbox_policy, engine=engine, broker=broker, assembler=assembler,
        context_compactor=context_compactor, memory=ms, memory_session=msession, manager=manager, agent_runner=agent_runner,
        tracer=tracer, ops_metrics=ops_metrics, event_sink=fanout, prices=prices, mcp_clients=mcp_clients, file_tracker=tracker,
        _sessions=sessions,
    )


def _register_hooks(hooks: HookRegistry, *, loop: AgentLoop, engine: PolicyEngine, assembler: SessionBoundAssembler, msession: Any,
                    workspace: Path, tracker: FileActivityTracker) -> None:
    def terminated(sid: str) -> HookResult | None:
        if engine.must_terminate(sid):
            st = engine.session(sid)
            return HookResult(continue_=False, stop_reason=f"permission engine: headless session must terminate ({st.terminate_reason})")
        return None

    def pre_model(inp: HookInput) -> HookResult | None:
        return terminated(inp.session_id)

    def post_model(inp: HookInput) -> HookResult | None:
        if inp.response is not None:
            assembler.observe_response(inp.ctx, inp.response)
        return None

    def pre_tool(inp: HookInput) -> HookResult | None:
        halt = terminated(inp.session_id)
        if halt is not None:
            return halt
        call = inp.tool_call
        if call is None:
            return None
        tracker.observe_call(call)
        decision = guard_tool_call(call.name, call.input, repo_root=workspace)
        if not decision.allow:
            return HookResult(permission=PermissionBehavior.DENY, reason=f"task-state guard: {decision.reason}")
        return None

    def post_tool(inp: HookInput) -> HookResult | None:
        call, outcome = inp.tool_call, inp.outcome
        if call is None or outcome is None:
            return None
        # A copy: the loop itself flips ctx.tainted from the outcome's ``untrusted`` flag and logs the taint event.
        updated = engine.record_result(call, inp.tool_spec, replace(inp.ctx), outcome)
        violations = updated.metadata.get("sandbox_violations")
        if violations:
            engine.record_sandbox_violation(inp.ctx, "; ".join(str(v) for v in violations)[:500])
        if msession is not None and call.name != "memory":
            text = updated.content if isinstance(updated.content, str) else str(updated.content)
            msession.record_tool_output(call.name, text, untrusted=bool(updated.metadata.get("untrusted")))
        return HookResult(updated_outcome=updated)

    hooks.on(HookEvent.PRE_MODEL, pre_model, name="harness.must_terminate")
    hooks.on(HookEvent.POST_MODEL, post_model, name="harness.observe_response")
    hooks.on(HookEvent.PRE_TOOL, pre_tool, name="harness.pre_tool")
    hooks.on(HookEvent.POST_TOOL, post_tool, name="harness.record_result")
    hooks.on(HookEvent.POST_TOOL_FAILURE, post_tool, name="harness.record_result")


__all__ = ["ApprovalUI", "EnginePolicy", "Harness", "build_harness", "make_approver"]
