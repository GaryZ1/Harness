"""Per-subagent tool scoping: allowlist, read-only enforcement, lease and ownership checks.

A subagent is *offered* only the tools :func:`~harness_multiagent.definitions.select_tools` keeps,
but models hallucinate tool names and a buggy runner might forward anything, so the same rules are
enforced again at *execution* time by :class:`ScopedToolExecutor` (defence in depth):

1. a call to a tool outside the subagent's set is rejected;
2. a call to a mutating tool (``ToolSpec.read_only`` false) is rejected unless every one of the
   agent's :class:`~harness_multiagent.leases.WriteLease` objects is still held;
3. for ``own_files`` writers, every path-like argument must be inside the agent's claimed files.

Rejections are returned as ``is_error`` tool results (the model can recover), never raised.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, Union

from harness_core.types import RunContext, ToolCall, ToolExecutor, ToolOutcome, ToolSpec

from .errors import LeaseError
from .leases import FileOwnership, WriteLease

ToolHandler = Callable[[ToolCall, RunContext], ToolOutcome]
ToolSource = Union[ToolExecutor, ToolHandler]

#: Input keys treated as file paths for ownership checks.
PATH_KEYS = ("path", "file_path", "filename", "file", "target", "destination", "old_path", "new_path", "paths", "files")


def as_handler(source: ToolSource | None) -> ToolHandler:
    """Adapt a :class:`ToolExecutor` or a plain callable to a :data:`ToolHandler`."""
    if source is None:
        def _none(call: ToolCall, ctx: RunContext) -> ToolOutcome:
            return ToolOutcome(f"Error: no tool executor configured for {call.name!r}", is_error=True)
        return _none
    execute = getattr(source, "execute", None)
    if callable(execute):
        return execute  # type: ignore[return-value]
    if callable(source):
        return source  # type: ignore[return-value]
    raise TypeError("tool source must be a ToolExecutor or a callable(call, ctx) -> ToolOutcome")


def _paths_in(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple)):
        for v in value:
            if isinstance(v, str):
                yield v


class ScopedToolExecutor:
    """The tool surface one subagent sees. Implements the ``ToolExecutor`` protocol."""

    def __init__(
        self,
        specs: Iterable[ToolSpec],
        handler: ToolHandler,
        *,
        write_leases: Sequence[WriteLease] = (),
        ownership: FileOwnership | None = None,
        owner: str = "",
        workspace: str | os.PathLike[str] = ".",
        routes: dict[str, ToolHandler] | None = None,
    ) -> None:
        self._specs = list(specs)
        self._by_name = {s.name: s for s in self._specs}
        self._handler = handler
        self._leases = tuple(write_leases)
        self._ownership = ownership
        self._owner = owner
        self._workspace = Path(workspace).expanduser().resolve()
        self._routes = dict(routes or {})

    def specs(self) -> list[ToolSpec]:
        return list(self._specs)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self._specs]

    def check(self, call: ToolCall) -> str | None:
        """Return a denial message, or ``None`` when the call may run."""
        spec = self._by_name.get(call.name)
        if spec is None:
            return f"Error: tool {call.name!r} is not available to this subagent (available: {', '.join(self.names) or 'none'})"
        if spec.read_only or call.name in self._routes:
            # Routed tools (the nested spawn tool) are not workspace writes; the manager applies its
            # own depth / budget / policy checks to them.
            return None
        if not self._leases or not all(lease.held for lease in self._leases):
            return (f"Error: write denied for {call.name!r}: this agent holds no write lease "
                    "(many readers, one writer). Report the change you would make instead.")
        if self._ownership is not None:
            for key in PATH_KEYS:
                for raw in _paths_in((call.input or {}).get(key)):
                    try:
                        rel = self._relative(raw)
                        self._ownership.check(self._owner, rel)
                    except LeaseError as exc:
                        return f"Error: write denied for {call.name!r}: {exc}"
        return None

    def _relative(self, raw: str) -> str:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = self._workspace / p
        p = Path(os.path.normpath(p))
        try:
            return p.relative_to(self._workspace).as_posix()
        except ValueError:
            return raw  # absolute path outside the workspace -> ownership check rejects it

    def execute(self, call: ToolCall, ctx: RunContext) -> ToolOutcome:
        denial = self.check(call)
        if denial is not None:
            return ToolOutcome(denial, is_error=True, metadata={"denied": True})
        handler = self._routes.get(call.name, self._handler)
        try:
            return handler(call, ctx)
        except Exception as exc:  # tool bugs must not kill the agent loop
            return ToolOutcome(f"Error: tool {call.name!r} raised {type(exc).__name__}: {exc}", is_error=True)

    __call__ = execute


__all__ = ["ToolHandler", "ToolSource", "as_handler", "ScopedToolExecutor", "PATH_KEYS"]
