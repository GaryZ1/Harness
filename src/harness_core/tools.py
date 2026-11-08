"""Tool executors: the "hands" behind ``execute(name, input) -> result``.

Research basis (reference report section 1.7 step 3, section 2.7 step 5; build plan "Architecture"):

* Managed Agents dispatches tools through ``execute(name, input) -> string``, provisions sandboxes
  lazily ("only when a tool call needs one", which cut p50 time-to-first-token by about 60%), and
  turns a dead sandbox into a tool error the model can retry. The session is never lost.
  :class:`LazyExecutor` implements exactly that behind the shared
  :class:`~harness_core.types.ToolExecutor` protocol, so the harness never imports a sandbox.
* The Agents SDK option ``tool_not_found_behavior="return_error_to_model"`` makes a hallucinated
  tool name a correctable error. Every executor here returns ``is_error`` outcomes instead of
  raising, and the loop adds a second guard.
* Read-only metadata (:attr:`ToolSpec.read_only`) is the *trusted* concurrency signal; it is set by
  whoever registers the tool, never parsed from an untrusted server's hints.

Thread safety: the loop may call :meth:`execute` concurrently for read-only tools, so executors
must not share unguarded mutable state between calls.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .errors import SandboxError
from .types import RunContext, ToolCall, ToolExecutor, ToolOutcome, ToolSpec

ToolFn = Callable[[dict[str, Any], RunContext], Any]


def to_outcome(value: Any) -> ToolOutcome:
    """Normalise a tool function's return value (``ToolOutcome`` | ``str`` | JSON-able) to an outcome."""
    if isinstance(value, ToolOutcome):
        return value
    if isinstance(value, str):
        return ToolOutcome(value)
    if isinstance(value, list) and all(isinstance(v, dict) and "type" in v for v in value):
        return ToolOutcome(value)
    return ToolOutcome(json.dumps(value, ensure_ascii=False, default=str))


def error_outcome(message: str, **metadata: Any) -> ToolOutcome:
    return ToolOutcome(message, is_error=True, metadata=dict(metadata))


@dataclass
class FunctionTool:
    spec: ToolSpec
    fn: ToolFn


class FunctionToolExecutor:
    """Executor over plain Python callables ``fn(input, ctx) -> ToolOutcome | str | JSON-able``.

    Exceptions raised by a tool become ``is_error`` outcomes carrying the exception type and
    message, so the model sees the failure and can adapt.
    """

    def __init__(self, tools: Iterable[FunctionTool] = ()) -> None:
        self._tools: dict[str, FunctionTool] = {}
        self._lock = threading.Lock()
        for t in tools:
            self.add(t.spec, t.fn)

    def add(self, spec: ToolSpec, fn: ToolFn) -> None:
        with self._lock:
            if spec.name in self._tools:
                raise ValueError(f"duplicate tool name: {spec.name}")
            self._tools[spec.name] = FunctionTool(spec, fn)

    def tool(self, name: str, description: str, input_schema: dict[str, Any] | None = None, *,
             read_only: bool = False, open_world: bool = False, **spec_kwargs: Any) -> Callable[[ToolFn], ToolFn]:
        """Decorator form of :meth:`add`."""
        def deco(fn: ToolFn) -> ToolFn:
            spec = ToolSpec(name, description, input_schema or {"type": "object", "properties": {}},
                            read_only=read_only, open_world=open_world, **spec_kwargs)
            self.add(spec, fn)
            return fn
        return deco

    def specs(self) -> list[ToolSpec]:
        with self._lock:
            return [t.spec for t in self._tools.values()]

    def execute(self, call: ToolCall, ctx: RunContext) -> ToolOutcome:
        tool = self._tools.get(call.name)
        if tool is None:
            return error_outcome(f"Error: unknown tool {call.name!r}", reason="unknown_tool")
        try:
            return to_outcome(tool.fn(dict(call.input), ctx))
        except SandboxError:
            raise
        except Exception as exc:  # noqa: BLE001 - tool failures are returned to the model
            return error_outcome(f"Error: {type(exc).__name__}: {exc}", exception=type(exc).__name__)


class CompositeExecutor:
    """Route calls to several executors by tool name (e.g. local tools + a remote sandbox)."""

    def __init__(self, *executors: ToolExecutor) -> None:
        self.executors = list(executors)
        self._routes: dict[str, ToolExecutor] = {}
        self._specs: list[ToolSpec] = []
        for ex in self.executors:
            for spec in ex.specs():
                if spec.name in self._routes:
                    raise ValueError(f"duplicate tool name across executors: {spec.name}")
                self._routes[spec.name] = ex
                self._specs.append(spec)

    def specs(self) -> list[ToolSpec]:
        return list(self._specs)

    def execute(self, call: ToolCall, ctx: RunContext) -> ToolOutcome:
        ex = self._routes.get(call.name)
        if ex is None:
            return error_outcome(f"Error: unknown tool {call.name!r}", reason="unknown_tool")
        return ex.execute(call, ctx)

    def cancel(self) -> None:
        for ex in self.executors:
            cancel = getattr(ex, "cancel", None)
            if callable(cancel):
                cancel()


class LazyExecutor:
    """Provision the real executor (sandbox) on the first tool call; re-provision after it dies.

    ``specs`` are declared up front so listing tools never provisions anything. When the inner
    executor raises :class:`~harness_core.errors.SandboxError` (or ``factory`` itself fails), the
    call returns an ``is_error`` outcome, the instance is dropped (``on_dispose`` is called), and
    the next call provisions a fresh one.
    """

    def __init__(self, specs: Iterable[ToolSpec], factory: Callable[[], ToolExecutor], *,
                 on_dispose: Callable[[ToolExecutor], None] | None = None) -> None:
        self._specs = list(specs)
        self.factory = factory
        self.on_dispose = on_dispose
        self._inner: ToolExecutor | None = None
        self._lock = threading.Lock()
        self.provision_count = 0

    @property
    def provisioned(self) -> bool:
        return self._inner is not None

    def specs(self) -> list[ToolSpec]:
        return list(self._specs)

    def _get(self) -> ToolExecutor:
        with self._lock:
            if self._inner is None:
                self._inner = self.factory()
                self.provision_count += 1
            return self._inner

    def _dispose(self, inner: ToolExecutor) -> None:
        with self._lock:
            if self._inner is inner:
                self._inner = None
        if self.on_dispose is not None:
            try:
                self.on_dispose(inner)
            except Exception:  # noqa: BLE001 - disposal is best effort
                pass

    def execute(self, call: ToolCall, ctx: RunContext) -> ToolOutcome:
        try:
            inner = self._get()
        except Exception as exc:  # noqa: BLE001
            return error_outcome(f"Error: sandbox could not be provisioned ({type(exc).__name__}: {exc}); retry later",
                                 reason="sandbox_unavailable")
        try:
            return inner.execute(call, ctx)
        except SandboxError as exc:
            self._dispose(inner)
            return error_outcome(f"Error: the sandbox died while running {call.name} ({exc}). A fresh sandbox will be "
                                 "provisioned on the next call; state outside the workspace may be lost.", reason="sandbox_died")

    def cancel(self) -> None:
        inner = self._inner
        cancel = getattr(inner, "cancel", None)
        if callable(cancel):
            cancel()
