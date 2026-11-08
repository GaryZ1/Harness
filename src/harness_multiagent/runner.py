"""The subagent execution seam: :class:`AgentRunner` protocol, :class:`SubagentTask`, built-in runner.

The real agent loop lives in ``harness_core`` (built concurrently by another package), so this
package depends only on a protocol::

    class AgentRunner(Protocol):
        def run(self, task: SubagentTask, ctx: RunContext) -> SubagentResult: ...

A :class:`SubagentTask` is fully prepared by :class:`~harness_multiagent.manager.SubagentManager`:
composed system prompt (identical across siblings so they share a prompt-cache prefix; reference 8.7
item 14), the rendered brief as the *only* user message (fresh context; item 2), the scoped tool
executor (allowlist + write lease + ownership; item 11), the per-agent budget ledger and the
cancellation token. A runner's obligations are small and are what :class:`SimpleAgentRunner` does:

1. call the model with ``task.system_prompt``, ``task.initial_messages()``, ``task.tools.specs()``
   and ``task.request_extra(model_id)``; resolve the model through ``task.model_role`` (never a
   hard-coded id);
2. after **every** response call ``task.charge(usage, model)`` then stop if ``task.stop_reason()``
   is not ``None`` (cancellation, timeout, budget);
3. execute tool calls only through ``task.tools.execute`` (never the raw executor);
4. stop after ``task.max_turns`` turns / ``task.max_tool_calls`` tool calls;
5. parse the final message with :func:`~harness_multiagent.result.parse_report`, re-prompting up to
   ``task.max_structured_retries`` times, then return a typed failure;
6. return only the report, never the transcript.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from harness_core.types import (
    Message,
    ModelClient,
    ModelRequest,
    RunContext,
    StopReason,
    ToolCall,
    ToolResultBlock,
    Usage,
)
from harness_memory.store import FileMemoryStore

from . import cancel as _cancel
from .brief import DelegationBrief
from .budget import BudgetLedger
from .definitions import SubagentDefinition, WriteAccess
from .errors import Cancelled, DefinitionError, StructuredOutputError
from .result import REPORT_INSTRUCTIONS, SubagentResult, SubagentStatus, parse_report, scan_untrusted
from .tools import ScopedToolExecutor

#: Claude Code workflows validate schema output with up to 5 retries (``MAX_STRUCTURED_OUTPUT_RETRIES``).
MAX_STRUCTURED_OUTPUT_RETRIES = 5


@dataclass
class SubagentTask:
    """Everything a runner needs to execute one isolated subagent run."""

    run_id: str
    definition: SubagentDefinition
    brief: DelegationBrief
    system_prompt: str
    task_message: str
    tools: ScopedToolExecutor
    model_role: str
    ledger: BudgetLedger
    token: _cancel.CancellationToken
    max_turns: int
    max_tool_calls: int | None
    max_output_tokens: int
    result_max_tokens: int
    write_access: WriteAccess = WriteAccess.NONE
    effort: str | None = None
    history: list[Message] = field(default_factory=list)   # only for context: fork
    memory: FileMemoryStore | None = None                  # read-only view unless definition says read_write
    parent_run_id: str | None = None
    workspace: str = "."
    max_structured_retries: int = MAX_STRUCTURED_OUTPUT_RETRIES

    @property
    def agent(self) -> str:
        return self.definition.name

    def initial_messages(self) -> list[Message]:
        """Fresh: the brief only. Fork: the parent's history followed by the brief."""
        return [*self.history, Message.user(self.task_message, subagent_run_id=self.run_id)]

    def request_extra(self, model_id: str) -> dict[str, Any]:
        """Provider params. Convention: the resolved model id travels as ``extra['model']``."""
        extra: dict[str, Any] = {"model": model_id}
        if self.effort:
            extra["effort"] = self.effort
        return extra

    def charge(self, usage: Usage, model: str = "") -> None:
        self.ledger.charge(usage, model=model)

    def stop_reason(self) -> SubagentStatus | None:
        """Typed reason to stop now (cancellation, timeout or budget), else ``None``."""
        reason = self.token.reason
        if reason is not None:
            return status_for_reason(reason)
        if self.ledger.exceeded():
            return SubagentStatus.ERROR_MAX_BUDGET
        return None


def status_for_reason(reason: str) -> SubagentStatus:
    if reason == _cancel.TIMEOUT:
        return SubagentStatus.ERROR_TIMEOUT
    if reason == _cancel.BUDGET:
        return SubagentStatus.ERROR_MAX_BUDGET
    return SubagentStatus.CANCELLED


@runtime_checkable
class AgentRunner(Protocol):
    """Implemented by the integrator on top of the real ``harness_core`` agent loop."""

    def run(self, task: SubagentTask, ctx: RunContext) -> SubagentResult: ...


class ModelRouter:
    """Role -> model id mapping plus how to obtain a client. Model ids only ever come from here.

    ``client`` is shared for all roles (the id is passed as ``request.extra['model']``); ``factory``
    builds/caches a client per model id instead.
    """

    def __init__(
        self,
        roles: Mapping[str, str],
        client: ModelClient | None = None,
        *,
        factory: Callable[[str], ModelClient] | None = None,
    ) -> None:
        if client is None and factory is None:
            raise DefinitionError("ModelRouter needs a client or a factory")
        bad = [r for r, m in roles.items() if not isinstance(m, str) or not m.strip()]
        if bad:
            raise DefinitionError(f"model ids must be non-empty strings for roles {bad}")
        self.roles = dict(roles)
        self._client = client
        self._factory = factory
        self._cache: dict[str, ModelClient] = {}

    def model_for(self, role: str) -> str:
        try:
            return self.roles[role]
        except KeyError:
            raise DefinitionError(f"no model configured for role {role!r}; configured roles: {sorted(self.roles)}") from None

    def resolve(self, role: str) -> tuple[str, ModelClient]:
        model_id = self.model_for(role)
        if self._factory is None:
            assert self._client is not None
            return model_id, self._client
        client = self._cache.get(model_id)
        if client is None:
            client = self._cache[model_id] = self._factory(model_id)
        return model_id, client


class SimpleAgentRunner:
    """A minimal, complete tool-use loop over a :class:`ModelClient` (standalone use and tests).

    Read-only tool calls in one response run concurrently (up to ``read_parallelism``); mutating
    calls run sequentially in order (reference 8.7 item 11, last sentence).
    """

    def __init__(
        self,
        router: ModelRouter,
        *,
        read_parallelism: int = 8,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.router = router
        self.read_parallelism = max(1, read_parallelism)
        self.monotonic = monotonic

    def run(self, task: SubagentTask, ctx: RunContext) -> SubagentResult:
        started = self.monotonic()
        usage = Usage()
        turns = 0
        tool_calls = 0
        repairs = 0
        model_id = ""
        warnings: list[str] = []

        def result(status: SubagentStatus, report=None, error: str = "") -> SubagentResult:
            return SubagentResult(
                agent=task.agent, run_id=task.run_id, status=status, report=report, usage=usage, turns=turns,
                tool_calls=tool_calls, wall_clock_s=self.monotonic() - started, model=model_id, depth=ctx.depth,
                parent_run_id=task.parent_run_id, error=error, warnings=warnings,
            )

        try:
            model_id, client = self.router.resolve(task.model_role)
        except DefinitionError as exc:
            return result(SubagentStatus.ERROR_DURING_EXECUTION, error=str(exc))
        messages = task.initial_messages()
        specs = task.tools.specs()
        read_only = {s.name for s in specs if s.read_only}
        try:
            while True:
                task.token.raise_if_cancelled()
                if turns >= task.max_turns:
                    return result(SubagentStatus.ERROR_MAX_TURNS, error=f"reached max_turns={task.max_turns}")
                request = ModelRequest(
                    system=task.system_prompt, messages=list(messages), tools=specs,
                    max_tokens=task.max_output_tokens, extra=task.request_extra(model_id),
                )
                response = client.create(request)
                turns += 1
                usage = usage + response.usage
                task.charge(response.usage, response.model or model_id)
                stop = task.stop_reason()
                if stop is not None:
                    return result(stop, error=task.token.detail or task.ledger.exceeded() or stop.value)
                messages.append(Message.assistant(response.content))
                if response.stop_reason is StopReason.REFUSAL:
                    return result(SubagentStatus.ERROR_DURING_EXECUTION, error="model refused")
                uses = response.tool_uses
                if uses:
                    blocks, tool_calls = self._run_tools(task, ctx, [ToolCall.from_block(u) for u in uses], tool_calls, read_only)
                    messages.append(Message.user(blocks))
                    continue
                if response.stop_reason is StopReason.MAX_TOKENS:
                    messages.append(Message.user("Your reply was cut off. Reply with ONLY the compact report JSON."))
                    continue
                text = response.text
                warnings.extend(w for w in scan_untrusted(text) if w not in warnings)
                try:
                    report = parse_report(text)
                except StructuredOutputError as exc:
                    if repairs >= task.max_structured_retries:
                        return result(SubagentStatus.ERROR_MAX_STRUCTURED_OUTPUT_RETRIES, error=str(exc))
                    repairs += 1
                    messages.append(Message.user(f"Your final message was not a valid report ({exc}). Reply with ONLY the report JSON object."))
                    continue
                return result(SubagentStatus.SUCCESS, report=report)
        except Cancelled as exc:
            return result(status_for_reason(exc.reason), error=exc.detail or exc.reason)
        except Exception as exc:  # API errors are never delivered as a subagent's result
            return result(SubagentStatus.ERROR_DURING_EXECUTION, error=f"{type(exc).__name__}: {exc}")


    def _run_tools(
        self, task: SubagentTask, ctx: RunContext, calls: list[ToolCall], used: int, read_only: set[str],
    ) -> tuple[list[ToolResultBlock], int]:
        """Execute one response's tool calls: consecutive read-only calls concurrently, mutating
        calls sequentially; results keep the model's call order."""
        blocks: list[ToolResultBlock | None] = [None] * len(calls)
        pending: list[tuple[int, ToolCall]] = []

        def flush() -> None:
            if len(pending) == 1:
                i, c = pending[0]
                blocks[i] = task.tools.execute(c, ctx).to_block(c.id)
            elif pending:
                with ThreadPoolExecutor(max_workers=min(self.read_parallelism, len(pending))) as pool:
                    futures = [(i, c, pool.submit(task.tools.execute, c, ctx)) for i, c in pending]
                    for i, c, f in futures:
                        blocks[i] = f.result().to_block(c.id)
            pending.clear()

        for i, call in enumerate(calls):
            if task.max_tool_calls is not None and used >= task.max_tool_calls:
                blocks[i] = ToolResultBlock(call.id, f"Error: tool-call budget ({task.max_tool_calls}) exhausted; finish with your report now.", True)
                continue
            used += 1
            if call.name in read_only:
                pending.append((i, call))
            else:
                flush()
                task.token.raise_if_cancelled()
                blocks[i] = task.tools.execute(call, ctx).to_block(call.id)
        flush()
        return [b for b in blocks if b is not None], used


# ----------------------------------------------------------------------------- prompt composition

SUBAGENT_PREAMBLE = """You are the subagent "{name}", working for a lead agent on ONE delegated task.
- Your context is isolated: you know only the task message and what your tools show you. There is no user in this conversation and nobody can grant you approvals.
- {write_line}
- Stay inside the brief's scope; out-of-scope items belong to siblings or are non-goals.
- {spawn_line}
- Be economical: stop as soon as the objective is met."""


def compose_system_prompt(
    definition: SubagentDefinition,
    *,
    write_access: WriteAccess,
    can_spawn: bool,
    roster: str = "",
    shared_context: str = "",
) -> str:
    """Deterministic system prompt for a subagent (no run ids, timestamps or brief text, so siblings
    of the same definition get byte-identical prefixes and share the prompt cache)."""
    if write_access is WriteAccess.NONE:
        write_line = "You are READ-ONLY: do not try to modify files or external state; describe proposed changes in your report instead."
    elif write_access is WriteAccess.OWN_FILES:
        write_line = "You may modify ONLY the files your write permission lists; all other writes are rejected."
    elif write_access is WriteAccess.OWN_WORKTREE:
        write_line = "You work in your own git worktree; an integrator verifies and merges your changes. Commit nothing outside it."
    else:
        write_line = "You are the single writer for this workspace; keep changes minimal and within scope."
    spawn_line = ("You may delegate to other subagents only when a sub-task is independent and read-heavy."
                  if can_spawn else "Do not delegate: you cannot spawn subagents.")
    parts = [definition.system_prompt.strip(), SUBAGENT_PREAMBLE.format(name=definition.name, write_line=write_line, spawn_line=spawn_line)]
    if shared_context.strip():
        parts.append("<project_context>\n" + shared_context.strip() + "\n</project_context>")
    if roster.strip():
        parts.append("Subagents available in this harness (for awareness):\n" + roster.strip())
    parts.append(REPORT_INSTRUCTIONS)
    return "\n\n".join(parts)


__all__ = [
    "AgentRunner", "SubagentTask", "ModelRouter", "SimpleAgentRunner", "compose_system_prompt", "status_for_reason",
    "MAX_STRUCTURED_OUTPUT_RETRIES", "SUBAGENT_PREAMBLE",
]
