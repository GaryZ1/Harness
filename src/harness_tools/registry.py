"""Tool registry: the :class:`~harness_core.types.ToolExecutor` every agent loop calls.

Research basis (``reports/SOTA agentic harness reference.md`` §5.7 steps 5, 6, 10-12, 24 and §5.8):

* **Errors are tool results, never transport errors.** Unknown tools, schema-validation failures,
  permission denials, timeouts and handler exceptions all come back as ``is_error: true``
  results with a specific message and, where possible, a corrected-input example (MCP SEP-1303;
  OpenAI Agents SDK ``"return_error_to_model"``). :meth:`ToolRegistry.execute` never raises.
* **Escalate repeated failure.** Consecutive and total error/denial counters flag escalation after
  3 consecutive or 20 total (Claude Code auto-mode precedent).
* **Validate before running.** Inputs must be JSON objects; strings are parsed as JSON, never
  string-matched. They are validated against the tool's JSON Schema before the handler runs.
* **Output budgets on every tool.** 25,000-token cap per result with offload + preview
  (:mod:`harness_tools.output`).
* **Concurrency classes.** Read-only (or explicitly ``concurrency_safe``) tools run in parallel;
  mutating tools run alone, in order (Claude Agent SDK rule); concurrency is capped
  (``max_function_tool_concurrency``-style) and results are returned in call order so the loop
  can answer every ``tool_use`` in ONE user message.
* **Cache-stable tool array.** :meth:`ToolRegistry.specs` is deterministic (sorted by name).
  After :meth:`ToolRegistry.freeze` only deferred tools may be added, because mutating the tool
  array mid-session invalidates the prompt cache from the tools onward.
* **Taint.** Results of ``open_world`` tools are marked ``metadata["untrusted"] = True`` and set
  ``RunContext.tainted``; an optional :class:`~harness_memory.security.TaintTracker` is fed too.
* **Telemetry per call.** Tool name, call id, args hash, result size, truncated flag, error class,
  duration and idempotency key are recorded (:class:`ToolCallRecord`).
"""

from __future__ import annotations

import contextvars
import dataclasses
import difflib
import hashlib
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

from harness_core.types import (
    Message,
    ModelResponse,
    NullTracer,
    PermissionBehavior,
    PermissionPolicy,
    RunContext,
    ToolCall,
    ToolOutcome,
    ToolSpec,
    ToolUseBlock,
    Tracer,
)

from .errors import SchemaError, ToolDefinitionError, ToolInputError, ToolTimeoutError
from .output import OutputPolicy, apply_budget, estimate_tokens
from .schema import apply_defaults, check_schema, example_from_schema, format_issues, schema_from_function, validate

log = logging.getLogger(__name__)

#: Anthropic tool-name rule (also safe for OpenAI and MCP SEP-986 guidance).
TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_MAX_CONCURRENCY = 8
#: Name of the client-side search tool (see :mod:`harness_tools.catalog`).
TOOL_SEARCH_NAME = "tool_search"
_CONTENT_BLOCK_TYPES = frozenset({"text", "image", "document", "tool_reference", "search_result"})

Deferral = Literal["reference", "reveal"]


# ----------------------------------------------------------------------------- tool model


@dataclass
class ToolInvocation:
    """What a raw-style handler receives besides the input."""

    call: ToolCall
    ctx: RunContext
    tool: "Tool"
    cancelled: threading.Event = field(default_factory=threading.Event)
    deadline: float | None = None  # time.monotonic() deadline, None = no timeout

    def remaining(self) -> float | None:
        return None if self.deadline is None else max(self.deadline - time.monotonic(), 0.0)


@dataclass
class Tool:
    """A registered tool: a :class:`ToolSpec` plus the handler and execution policy.

    ``handler`` styles:

    * ``"kwargs"`` (decorated functions): ``handler(**input)`` (plus ``ctx=RunContext`` when the
      function declares a ``ctx`` parameter);
    * ``"raw"``: ``handler(input: dict, invocation: ToolInvocation)``.

    ``handler=None`` registers a spec-only tool executed by the provider (server tools such as
    ``tool_search_tool_bm25_20251119``); calling it locally returns an error result.
    """

    spec: ToolSpec
    handler: Callable[..., Any] | None
    call_style: Literal["kwargs", "raw"] = "raw"
    wants_ctx: bool = False
    timeout_s: float | None = DEFAULT_TIMEOUT_S
    max_output_tokens: int | None = None
    concurrency_safe: bool | None = None  # None = derive from spec.read_only
    idempotent: bool = False
    destructive: bool = False
    input_examples: list[dict[str, Any]] = field(default_factory=list)
    validate_input: bool = True
    truncation_hint: str = "Narrow the request (smaller range, more specific pattern, or a filter) to see less at once."
    source: str = "local"  # "local" | "provider" | "mcp:<server>"

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def parallel_safe(self) -> bool:
        return self.spec.read_only if self.concurrency_safe is None else self.concurrency_safe

    def describe_flags(self) -> dict[str, Any]:
        return {
            "read_only": self.spec.read_only,
            "open_world": self.spec.open_world,
            "concurrency_safe": self.parallel_safe,
            "idempotent": self.idempotent,
            "destructive": self.destructive,
            "source": self.source,
        }


def function_tool(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    input_schema: dict[str, Any] | None = None,
    read_only: bool = False,
    open_world: bool = False,
    defer_loading: bool = False,
    strict: bool = False,
    timeout_s: float | None = DEFAULT_TIMEOUT_S,
    max_output_tokens: int | None = None,
    concurrency_safe: bool | None = None,
    idempotent: bool | None = None,
    destructive: bool = False,
    input_examples: Sequence[dict[str, Any]] = (),
    metadata: Mapping[str, Any] | None = None,
) -> Tool:
    """Build a :class:`Tool` from a plain Python function (schema derived from its signature)."""
    doc_desc, derived, wants_ctx = schema_from_function(fn)
    spec = ToolSpec(
        name=name or fn.__name__,
        description=(description or doc_desc).strip(),
        input_schema=input_schema or derived,
        read_only=read_only,
        open_world=open_world,
        defer_loading=defer_loading,
        strict=strict,
        metadata=dict(metadata or {}),
    )
    return Tool(
        spec=spec,
        handler=fn,
        call_style="kwargs",
        wants_ctx=wants_ctx,
        timeout_s=timeout_s,
        max_output_tokens=max_output_tokens,
        concurrency_safe=concurrency_safe,
        idempotent=read_only if idempotent is None else idempotent,
        destructive=destructive,
        input_examples=[dict(e) for e in input_examples],
    )


# ----------------------------------------------------------------------------- failure tracking & telemetry


@dataclass
class FailureTracker:
    """Counts consecutive and total failures; :meth:`record` returns True when escalation is due."""

    consecutive_limit: int = 3
    total_limit: int = 20
    consecutive: int = 0
    total: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, failed: bool) -> bool:
        with self._lock:
            if failed:
                self.consecutive += 1
                self.total += 1
            else:
                self.consecutive = 0
            return self._escalated()

    def _escalated(self) -> bool:
        return self.consecutive >= self.consecutive_limit or self.total >= self.total_limit

    @property
    def escalated(self) -> bool:
        with self._lock:
            return self._escalated()

    def reset(self) -> None:
        with self._lock:
            self.consecutive = self.total = 0


@dataclass(frozen=True)
class ToolCallRecord:
    """Telemetry for one call (checklist step 24)."""

    call_id: str
    tool: str
    args_sha256: str
    duration_ms: float
    result_chars: int
    result_tokens: int
    truncated: bool
    is_error: bool
    error_class: str | None
    untrusted: bool
    idempotency_key: str
    offload_path: str | None = None
    session_id: str = ""


def canonical_args(args: Any) -> str:
    return json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def args_hash(args: Any) -> str:
    return hashlib.sha256(canonical_args(args).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- registry


class _ErrorResult(Exception):
    """Internal: short-circuit execution with an error result of a given class."""

    def __init__(self, message: str, error_class: str, *, denied: bool = False) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.denied = denied


Approver = Callable[[ToolCall, ToolSpec, RunContext, str], bool]


class ToolRegistry:
    """Register tools, publish a cache-stable tool array, and execute calls safely.

    Thread-safe. Use one registry per session when relying on ``deferral="reveal"`` (the set of
    discovered tools is registry state).

    :param output_policy: per-result budget (25k tokens by default; set ``offload_dir`` to offload).
    :param permission_policy: optional :class:`PermissionPolicy` consulted before every call.
    :param approver: called for ``ASK`` decisions; without one, ``ASK`` is treated as a denial.
    :param taint_tracker: anything with ``record_tool_output(name, untrusted=bool)``.
    :param on_record: callback receiving a :class:`ToolCallRecord` after every call.
    :param deferral: ``"reference"`` (Anthropic: deferred tools stay in the array with
        ``defer_loading: true`` and search returns ``tool_reference`` blocks) or ``"reveal"``
        (other providers: deferred tools are omitted until ``tool_search`` discovers them).
    """

    def __init__(
        self,
        *,
        output_policy: OutputPolicy | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        permission_policy: PermissionPolicy | None = None,
        approver: Approver | None = None,
        taint_tracker: Any = None,
        tracer: Tracer | None = None,
        on_record: Callable[[ToolCallRecord], None] | None = None,
        deferral: Deferral = "reference",
        error_limits: tuple[int, int] = (3, 20),
        denial_limits: tuple[int, int] = (3, 20),
    ) -> None:
        if deferral not in ("reference", "reveal"):
            raise ValueError("deferral must be 'reference' or 'reveal'")
        self.output_policy = output_policy or OutputPolicy()
        self.max_concurrency = max(1, int(max_concurrency))
        self.permission_policy = permission_policy
        self.approver = approver
        self.taint_tracker = taint_tracker
        self.tracer: Tracer = tracer or NullTracer()
        self.on_record = on_record
        self.deferral: Deferral = deferral
        self.errors = FailureTracker(*error_limits)
        self.denials = FailureTracker(*denial_limits)
        self._tools: dict[str, Tool] = {}
        self._discovered: list[str] = []
        self._frozen = False
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ registration
    def register(self, tool: Tool, *, replace: bool = False) -> Tool:
        """Register ``tool``; validates name, description and schema (fails loudly, at registration)."""
        spec = tool.spec
        if not TOOL_NAME_RE.match(spec.name or ""):
            raise ToolDefinitionError(f"invalid tool name {spec.name!r}: must match {TOOL_NAME_RE.pattern}")
        if spec.provider_type is None:
            if not spec.description.strip():
                raise ToolDefinitionError(f"tool {spec.name!r} needs a description (write it as if onboarding a new hire)")
            try:
                check_schema(spec.input_schema)
            except SchemaError as exc:
                raise ToolDefinitionError(f"tool {spec.name!r} has an invalid input_schema: {exc}") from exc
            if spec.input_schema.get("type") not in ("object", None):
                raise ToolDefinitionError(f"tool {spec.name!r}: input_schema must describe an object")
            if spec.strict and spec.input_schema.get("additionalProperties") is not False:
                raise ToolDefinitionError(f"tool {spec.name!r}: strict tools need additionalProperties: false")
        if spec.name == TOOL_SEARCH_NAME and spec.defer_loading:
            raise ToolDefinitionError("the tool search tool must never be deferred")
        for ex in tool.input_examples:
            issues = validate(ex, spec.input_schema)
            if issues:
                raise ToolDefinitionError(f"tool {spec.name!r}: input_example does not match its schema:\n{format_issues(issues)}")
        spec.metadata.update(tool.describe_flags())
        if tool.input_examples:
            spec.metadata["input_examples"] = [dict(e) for e in tool.input_examples]
        with self._lock:
            if spec.name in self._tools and not replace:
                raise ToolDefinitionError(f"duplicate tool name {spec.name!r}")
            if self._frozen and not spec.defer_loading:
                raise ToolDefinitionError(
                    f"registry is frozen: adding non-deferred tool {spec.name!r} mid-session would invalidate the prompt cache; "
                    "register it with defer_loading=True instead"
                )
            self._tools[spec.name] = tool
        return tool

    def register_function(self, fn: Callable[..., Any], **kwargs: Any) -> Tool:
        return self.register(function_tool(fn, **kwargs))

    def tool(self, fn: Callable[..., Any] | None = None, **kwargs: Any) -> Any:
        """Decorator: ``@registry.tool`` or ``@registry.tool(read_only=True, ...)``. Returns the function unchanged."""

        def deco(f: Callable[..., Any]) -> Callable[..., Any]:
            self.register_function(f, **kwargs)
            return f

        return deco(fn) if fn is not None else deco

    def register_spec(self, spec: ToolSpec) -> Tool:
        """Register a provider-executed tool (server tool / provider-defined type) with no local handler."""
        return self.register(Tool(spec=spec, handler=None, source="provider", concurrency_safe=True))

    def unregister(self, name: str) -> None:
        with self._lock:
            if self._frozen and name in self._tools and not self._tools[name].spec.defer_loading:
                raise ToolDefinitionError("registry is frozen: removing a loaded tool would invalidate the prompt cache")
            self._tools.pop(name, None)
            if name in self._discovered:
                self._discovered.remove(name)

    def freeze(self) -> None:
        """Lock the hot tool set (call at session start). Deferred tools can still be added."""
        with self._lock:
            self.validate_tool_array()
            self._frozen = True

    @property
    def frozen(self) -> bool:
        return self._frozen

    # ------------------------------------------------------------------ lookup
    def get(self, name: str) -> Tool | None:
        with self._lock:
            return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.get(name) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._tools)

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._tools)

    def tools(self) -> list[Tool]:
        with self._lock:
            return [self._tools[n] for n in sorted(self._tools)]

    def deferred_tools(self) -> list[Tool]:
        return [t for t in self.tools() if t.spec.defer_loading]

    def is_discovered(self, name: str) -> bool:
        with self._lock:
            return name in self._discovered

    def discover(self, names: Iterable[str]) -> list[str]:
        """Mark deferred tools as discovered (``reveal`` mode exposes them from now on). Returns newly discovered names."""
        new: list[str] = []
        with self._lock:
            for n in names:
                t = self._tools.get(n)
                if t is not None and t.spec.defer_loading and n not in self._discovered:
                    self._discovered.append(n)
                    new.append(n)
        return new

    # ------------------------------------------------------------------ tool array
    def validate_tool_array(self) -> None:
        """Enforce tool-search invariants: never defer every tool; if anything is deferred, a
        non-deferred search tool must exist (client ``tool_search`` or a server ``tool_search_tool_*``)."""
        tools = self.tools()
        deferred = [t for t in tools if t.spec.defer_loading]
        if not deferred:
            return
        hot = [t for t in tools if not t.spec.defer_loading]
        if not hot:
            raise ToolDefinitionError("every tool is deferred; keep at least one non-deferred tool (the API rejects this with 400)")
        has_search = any(
            t.spec.name == TOOL_SEARCH_NAME or (t.spec.provider_type or "").startswith("tool_search_tool_") for t in hot
        )
        if not has_search:
            raise ToolDefinitionError(
                f"{len(deferred)} tool(s) are deferred but no non-deferred tool search tool is registered; "
                "call harness_tools.catalog.enable_tool_search(registry) or register the server-side tool"
            )

    def specs(self) -> list[ToolSpec]:
        """The tool array to send, in deterministic order.

        ``reference``: every tool sorted by name (deferred ones carry ``defer_loading``).
        ``reveal``: non-deferred tools sorted by name, then discovered tools in discovery order
        (appended, so earlier entries never move).
        """
        with self._lock:
            ordered = [self._tools[n] for n in sorted(self._tools)]
            if self.deferral == "reference":
                return [t.spec for t in ordered]
            hot = [t.spec for t in ordered if not t.spec.defer_loading]
            found = [dataclasses.replace(self._tools[n].spec, defer_loading=False) for n in self._discovered if n in self._tools]
            return hot + found

    def api_tools(self, *, include_input_examples: bool = False) -> list[dict[str, Any]]:
        """Anthropic ``tools=[...]`` payload. ``input_examples`` is opt-in (beta field)."""
        out: list[dict[str, Any]] = []
        for spec in self.specs():
            d = spec.to_api()
            ex = spec.metadata.get("input_examples")
            if include_input_examples and ex and not spec.provider_type:
                d["input_examples"] = ex
            out.append(d)
        return out

    # ------------------------------------------------------------------ execution
    def execute(self, call: ToolCall, ctx: RunContext) -> ToolOutcome:
        """Execute one call. Never raises: every failure becomes an ``is_error`` result."""
        t0 = time.monotonic()
        tool = self.get(call.name)
        error_class: str | None = None
        denied = False
        try:
            with self.tracer.span("tool.execute", tool=call.name, call_id=call.id, session_id=ctx.session_id):
                try:
                    outcome = self._execute(tool, call, ctx)
                except _ErrorResult as er:
                    error_class, denied = er.error_class, er.denied
                    outcome = ToolOutcome(f"Error: {er}", is_error=True)
        except Exception as exc:  # defensive: tracer failures etc.
            log.exception("tool %s failed outside handler", call.name)
            error_class = type(exc).__name__
            outcome = ToolOutcome(f"Error: internal error while running {call.name!r}: {exc}", is_error=True)
        if outcome.is_error and error_class is None:
            error_class = str(outcome.metadata.get("error_class") or "ToolError")
        return self._finish(tool, call, ctx, outcome, t0, error_class, denied)

    def _execute(self, tool: Tool | None, call: ToolCall, ctx: RunContext) -> ToolOutcome:
        if tool is None:
            raise _ErrorResult(self._unknown_tool_message(call.name), "ToolNotFound")
        if tool.handler is None:
            raise _ErrorResult(
                f"{tool.name!r} is a provider-executed tool ({tool.spec.provider_type or 'server tool'}); it cannot be run locally.",
                "ProviderTool",
            )
        tool_input = self._coerce_input(tool, call.input)
        tool_input = self._validated(tool, tool_input)
        tool_input = self._check_permission(tool, ToolCall(call.id, call.name, tool_input), ctx)
        inv = ToolInvocation(call=ToolCall(call.id, call.name, tool_input), ctx=ctx, tool=tool)
        timeout = tool.timeout_s
        if timeout is not None:
            inv.deadline = time.monotonic() + timeout
        try:
            raw = self._run(tool, tool_input, inv, timeout)
        except ToolTimeoutError as exc:
            raise _ErrorResult(str(exc), "ToolTimeout") from exc
        except ToolInputError as exc:
            msg = str(exc)
            if exc.example is not None:
                msg += f"\nExample of a valid input: {json.dumps(exc.example, ensure_ascii=False)}"
            raise _ErrorResult(msg, type(exc).__name__) from exc
        except _ErrorResult:
            raise
        except Exception as exc:
            log.debug("tool %s raised", tool.name, exc_info=True)
            raise _ErrorResult(f"{tool.name} failed: {type(exc).__name__}: {exc}", type(exc).__name__) from exc
        return self._normalise(raw)

    def _unknown_tool_message(self, name: str) -> str:
        names = self.names()
        close = difflib.get_close_matches(name, names, n=3, cutoff=0.5)
        msg = f"Unknown tool {name!r}."
        if close:
            msg += f" Did you mean: {', '.join(close)}?"
        hot = [n for n in names if not self._tools[n].spec.defer_loading]
        msg += f" Available tools: {', '.join(hot) or '(none)'}."
        if any(self._tools[n].spec.defer_loading for n in names):
            msg += f" More tools are available on demand: call {TOOL_SEARCH_NAME} with a description of what you need."
        return msg

    @staticmethod
    def _coerce_input(tool: Tool, raw: Any) -> dict[str, Any]:
        value = raw
        if isinstance(raw, (str, bytes)):
            try:
                value = json.loads(raw) if raw.strip() else {}
            except (ValueError, TypeError) as exc:
                raise _ErrorResult(f"tool input for {tool.name!r} is not valid JSON: {exc}", "InvalidInput") from exc
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise _ErrorResult(f"tool input for {tool.name!r} must be a JSON object, got {type(value).__name__}", "InvalidInput")
        return dict(value)

    def _validated(self, tool: Tool, tool_input: dict[str, Any]) -> dict[str, Any]:
        schema = tool.spec.input_schema
        if tool.validate_input and schema:
            issues = validate(tool_input, schema)
            if issues:
                example = tool.input_examples[0] if tool.input_examples else example_from_schema(schema)
                raise _ErrorResult(
                    f"invalid input for {tool.name!r}:\n{format_issues(issues)}\n"
                    f"Example of a valid input: {json.dumps(example, ensure_ascii=False)}",
                    "ValidationError",
                )
        return apply_defaults(tool_input, schema) if schema else tool_input

    def _check_permission(self, tool: Tool, call: ToolCall, ctx: RunContext) -> dict[str, Any]:
        policy = self.permission_policy
        if policy is None:
            return call.input
        decision = policy.check(call, tool.spec, ctx)
        if decision.behavior is PermissionBehavior.ASK:
            if self.approver is not None and self.approver(call, tool.spec, ctx, decision.reason):
                decision = dataclasses.replace(decision, behavior=PermissionBehavior.ALLOW)
            else:
                reason = decision.reason or "this action requires user approval"
                raise _ErrorResult(
                    f"Permission required for {tool.name!r} and it was not granted: {reason}. "
                    "Do not retry the same call; choose a different approach or ask the user.",
                    "PermissionDenied",
                    denied=True,
                )
        if decision.behavior is PermissionBehavior.DENY:
            raise _ErrorResult(
                f"Permission denied for {tool.name!r}: {decision.reason or 'blocked by policy'}. "
                "Do not retry the same call; choose a different approach.",
                "PermissionDenied",
                denied=True,
            )
        self.denials.record(False)
        if decision.updated_input is not None:
            return self._validated(tool, self._coerce_input(tool, decision.updated_input))
        return call.input

    @staticmethod
    def _invoke(tool: Tool, tool_input: dict[str, Any], inv: ToolInvocation) -> Any:
        assert tool.handler is not None
        if tool.call_style == "kwargs":
            kwargs = dict(tool_input)
            if tool.wants_ctx:
                kwargs["ctx"] = inv.ctx
            return tool.handler(**kwargs)
        return tool.handler(tool_input, inv)

    def _run(self, tool: Tool, tool_input: dict[str, Any], inv: ToolInvocation, timeout: float | None) -> Any:
        if timeout is None:
            return self._invoke(tool, tool_input, inv)
        box: dict[str, Any] = {}
        run_ctx = contextvars.copy_context()

        def target() -> None:
            try:
                box["value"] = run_ctx.run(self._invoke, tool, tool_input, inv)
            except BaseException as exc:  # noqa: BLE001 - re-raised in the caller thread
                box["error"] = exc

        worker = threading.Thread(target=target, name=f"tool-{tool.name}-{inv.call.id}", daemon=True)
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            inv.cancelled.set()  # cooperative cancellation; the daemon thread is abandoned
            raise ToolTimeoutError(
                f"{tool.name} timed out after {timeout:g}s and was cancelled. "
                "Try a smaller request, or run long jobs in the background and poll for the result."
            )
        if "error" in box:
            raise box["error"]
        return box.get("value")

    @staticmethod
    def _normalise(raw: Any) -> ToolOutcome:
        if isinstance(raw, ToolOutcome):
            return raw
        if raw is None:
            return ToolOutcome("(no output)")
        if isinstance(raw, str):
            return ToolOutcome(raw if raw else "(no output)")
        if isinstance(raw, list) and raw and all(isinstance(b, dict) and b.get("type") in _CONTENT_BLOCK_TYPES for b in raw):
            return ToolOutcome([dict(b) for b in raw])
        if isinstance(raw, (dict, list, tuple, int, float, bool)):
            return ToolOutcome(json.dumps(raw, ensure_ascii=False, default=str, indent=2))
        return ToolOutcome(str(raw))

    def _budget(self, tool: Tool | None, call: ToolCall, outcome: ToolOutcome) -> tuple[ToolOutcome, bool, str | None]:
        policy = self.output_policy.with_max_tokens(tool.max_output_tokens if tool else None)
        hint = tool.truncation_hint if tool else "Narrow the request."
        if isinstance(outcome.content, str):
            b = apply_budget(outcome.content, policy, call_id=call.id, hint=hint)
            if b.truncated:
                outcome = ToolOutcome(b.text, outcome.is_error, dict(outcome.metadata))
            return outcome, b.truncated, b.offload_path
        texts = [i for i, blk in enumerate(outcome.content) if isinstance(blk, dict) and blk.get("type") == "text"]
        if not texts:
            return outcome, False, None
        share = OutputPolicy(
            max(policy.max_tokens // len(texts), 1), policy.chars_per_token, policy.preview_tokens, policy.offload_dir, policy.head_fraction
        )
        blocks = [dict(b) for b in outcome.content]
        truncated, offload = False, None
        for n, i in enumerate(texts):
            b = apply_budget(str(blocks[i].get("text", "")), share, call_id=f"{call.id}-{n}", hint=hint)
            if b.truncated:
                blocks[i]["text"] = b.text
                truncated = True
                offload = offload or b.offload_path
        return ToolOutcome(blocks, outcome.is_error, dict(outcome.metadata)), truncated, offload

    def _finish(
        self, tool: Tool | None, call: ToolCall, ctx: RunContext, outcome: ToolOutcome, t0: float, error_class: str | None, denied: bool
    ) -> ToolOutcome:
        try:
            outcome, truncated, offload = self._budget(tool, call, outcome)
        except OSError as exc:  # offload dir unwritable: fall back to in-place truncation
            log.warning("offload failed for %s: %s", call.id, exc)
            fallback = dataclasses.replace(self.output_policy, offload_dir=None)
            b = apply_budget(outcome.content if isinstance(outcome.content, str) else json.dumps(outcome.content), fallback)
            outcome, truncated, offload = ToolOutcome(b.text, outcome.is_error, dict(outcome.metadata)), b.truncated, None
        # Successful output of an open-world tool is untrusted; handlers (e.g. MCP) may also flag
        # error text from third parties explicitly. Our own validation/permission errors are not.
        untrusted = bool(outcome.metadata.get("untrusted")) or bool(tool and tool.spec.open_world and not outcome.is_error)
        if untrusted:
            ctx.tainted = True
            if self.taint_tracker is not None:
                self.taint_tracker.record_tool_output(call.name, untrusted=True)
        text_len = len(outcome.content) if isinstance(outcome.content, str) else len(json.dumps(outcome.content, default=str))
        sha = args_hash(call.input)
        record = ToolCallRecord(
            call_id=call.id,
            tool=call.name,
            args_sha256=sha,
            duration_ms=round((time.monotonic() - t0) * 1000, 3),
            result_chars=text_len,
            result_tokens=estimate_tokens(outcome.content, self.output_policy.chars_per_token),
            truncated=truncated,
            is_error=outcome.is_error,
            error_class=error_class if outcome.is_error else None,
            untrusted=untrusted,
            idempotency_key=f"{call.id}:{sha[:16]}",
            offload_path=offload,
            session_id=ctx.session_id,
        )
        meta = dict(outcome.metadata)
        meta.update(
            tool=call.name,
            duration_ms=record.duration_ms,
            bytes=record.result_chars,
            truncated=truncated,
            untrusted=untrusted,
            idempotency_key=record.idempotency_key,
        )
        if offload:
            meta["offload_path"] = offload
        if outcome.is_error:
            meta["error_class"] = record.error_class
        escalate_errors = self.errors.record(outcome.is_error)
        escalate_denials = self.denials.record(True) if denied else self.denials.escalated
        if escalate_errors or escalate_denials:
            reason = "denials" if escalate_denials else "errors"
            tracker = self.denials if escalate_denials else self.errors
            meta["escalate"] = f"{tracker.consecutive} consecutive / {tracker.total} total tool {reason}; hand control to a human"
        if self.on_record is not None:
            try:
                self.on_record(record)
            except Exception:  # telemetry must never break the loop
                log.exception("on_record callback failed")
        return ToolOutcome(outcome.content, outcome.is_error, meta)

    # ------------------------------------------------------------------ batches
    def execute_batch(self, calls: Sequence[ToolCall], ctx: RunContext, *, max_concurrency: int | None = None) -> list[ToolOutcome]:
        """Execute calls from one assistant turn; returns outcomes in call order.

        Consecutive concurrency-safe calls run in parallel (capped); any mutating call runs alone,
        after everything before it finished and before anything after it starts.
        """
        limit = max(1, max_concurrency or self.max_concurrency)
        results: list[ToolOutcome | None] = [None] * len(calls)
        batch: list[int] = []

        def flush() -> None:
            if not batch:
                return
            if len(batch) == 1 or limit == 1:
                for i in batch:
                    results[i] = self.execute(calls[i], ctx)
            else:
                with ThreadPoolExecutor(max_workers=min(limit, len(batch)), thread_name_prefix="tool-batch") as pool:
                    futures = {i: pool.submit(contextvars.copy_context().run, self.execute, calls[i], ctx) for i in batch}
                    for i, fut in futures.items():
                        results[i] = fut.result()
            batch.clear()

        for i, call in enumerate(calls):
            tool = self.get(call.name)
            if tool is None or tool.parallel_safe:
                batch.append(i)
            else:
                flush()
                results[i] = self.execute(call, ctx)
        flush()
        return [r for r in results if r is not None]

    def run_tool_uses(self, source: ModelResponse | Message | Sequence[ToolUseBlock], ctx: RunContext) -> Message:
        """Execute every ``tool_use`` in an assistant turn and return ONE user message with all results."""
        blocks = source.tool_uses if isinstance(source, (ModelResponse, Message)) else list(source)
        calls = [ToolCall.from_block(b) for b in blocks]
        return tool_results_message(calls, self.execute_batch(calls, ctx))


def tool_results_message(calls: Sequence[ToolCall], outcomes: Sequence[ToolOutcome]) -> Message:
    """Pair calls with outcomes into a single user message (every ``tool_use`` answered, in order)."""
    if len(calls) != len(outcomes):
        raise ValueError(f"{len(calls)} calls but {len(outcomes)} outcomes; every tool_use needs exactly one tool_result")
    ids = [c.id for c in calls]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate tool_use ids in one turn")
    return Message.user([o.to_block(c.id) for c, o in zip(calls, outcomes, strict=True)])


__all__ = [
    "Approver",
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_TIMEOUT_S",
    "Deferral",
    "FailureTracker",
    "TOOL_NAME_RE",
    "TOOL_SEARCH_NAME",
    "Tool",
    "ToolCallRecord",
    "ToolInvocation",
    "ToolRegistry",
    "args_hash",
    "canonical_args",
    "function_tool",
    "tool_results_message",
]
