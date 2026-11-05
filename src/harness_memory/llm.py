"""Model access for the background pipeline, plus an optional Anthropic adapter.

The pipeline only needs :class:`LLM` - a ``complete(system, prompt) -> str`` callable - so it can
run against any provider or a deterministic fake in tests. :class:`AnthropicLLM` and
:func:`run_memory_agent` use the official ``anthropic`` SDK (``pip install harness-memory[anthropic]``).
The model id is always supplied by the caller (e.g. from configuration).
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from .errors import MemoryError_
from .tool import TOOL_DEFINITION, MemoryToolHandler


class LLM(Protocol):
    def complete(self, *, system: str, prompt: str, max_tokens: int = 16_000) -> str: ...


class LLMRefusal(MemoryError_):
    """The model declined the request (``stop_reason == "refusal"``)."""


class CallableLLM:
    """Wrap a plain function ``f(system, prompt) -> str`` as an :class:`LLM`."""

    def __init__(self, fn: Callable[[str, str], str]) -> None:
        self.fn = fn

    def complete(self, *, system: str, prompt: str, max_tokens: int = 16_000) -> str:
        return self.fn(system, prompt)


def _text_of(response: Any) -> str:
    return "".join(getattr(b, "text", "") for b in response.content if getattr(b, "type", "") == "text")


class AnthropicLLM:
    """Single-shot completion via ``client.messages.create`` with adaptive thinking.

    :param model: model id to call (required; supply it from configuration).
    :param effort: ``output_config.effort`` - ``low`` | ``medium`` | ``high`` | ``xhigh`` | ``max``.
    :param extra: additional ``messages.create`` keyword arguments (e.g. refusal ``fallbacks``).
    """

    def __init__(self, model: str, *, client: Any | None = None, effort: str = "high", extra: dict[str, Any] | None = None) -> None:
        if not model:
            raise ValueError("model is required")
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("install the 'anthropic' extra: pip install harness-memory[anthropic]") from exc
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.extra = extra or {}

    def complete(self, *, system: str, prompt: str, max_tokens: int = 16_000) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": prompt}],
            **self.extra,
        )
        if response.stop_reason == "refusal":
            raise LLMRefusal("model refused the memory-pipeline request")
        return _text_of(response)


def run_memory_agent(
    client: Any,
    *,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    handler: MemoryToolHandler,
    max_turns: int = 25,
    max_tokens: int = 16_000,
    effort: str = "high",
    extra_tools: list[dict[str, Any]] | None = None,
    on_other_tool: Callable[[Any], dict[str, Any]] | None = None,
    **create_kwargs: Any,
) -> Any:
    """Manual agent loop with the client-side ``memory_20250818`` tool (write channel A).

    Appends every assistant ``response.content`` and one user message containing *all*
    ``tool_result`` blocks per turn (keeps parallel tool use intact). Returns the final response.
    ``messages`` is mutated in place so the caller keeps the full, append-only history.
    """
    tools = [dict(TOOL_DEFINITION), *(extra_tools or [])]
    response = None
    for _ in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            tools=tools,
            messages=messages,
            **create_kwargs,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "refusal":
            raise LLMRefusal("model refused")
        if response.stop_reason != "tool_use":
            return response
        results = []
        for block in response.content:
            if getattr(block, "type", "") != "tool_use":
                continue
            if block.name == TOOL_DEFINITION["name"]:
                results.append(handler.handle(dict(block.input)).to_block(block.id))
            elif on_other_tool is not None:
                results.append(on_other_tool(block))
            else:
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": f"Error: unknown tool {block.name}", "is_error": True}
                )
        messages.append({"role": "user", "content": results})
    return response
