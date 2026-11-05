"""Token counting behind a protocol, with an offline heuristic default.

The research is explicit that token counts for Claude must come from the provider
(``client.messages.count_tokens``), never from a third-party tokenizer such as tiktoken. Offline
and in tests we still need *some* estimate, so :class:`HeuristicTokenCounter` approximates
~4 characters per token plus small per-message/per-block overheads, and every component in this
package depends only on the :class:`TokenCounter` protocol so a real counter can be plugged in.

Budget decisions additionally *calibrate* the heuristic against real ``usage`` (see
:class:`~harness_context.budget.UsageCalibration`): total input is
``input_tokens + cache_read_input_tokens + cache_creation_input_tokens``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

from harness_core.types import ContentBlock, Message, ModelRequest, ToolResultBlock, ToolSpec, messages_to_api

from ._io import canonical_json
from .messages import block_text


@runtime_checkable
class TokenCounter(Protocol):
    """Anything that can estimate tokens for raw text and for a whole request."""

    def count_text(self, text: str) -> int: ...

    def count_request(self, request: ModelRequest) -> int: ...


@dataclass(frozen=True)
class HeuristicTokenCounter:
    """~``chars_per_token`` characters per token, plus structural overheads.

    Deliberately conservative (rounds up, counts JSON punctuation) so budget decisions err toward
    compacting slightly early rather than overflowing.
    """

    chars_per_token: float = 4.0
    message_overhead: int = 4
    block_overhead: int = 3
    tool_overhead: int = 12
    image_tokens: int = 1_600

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return int(math.ceil(len(text) / self.chars_per_token))

    def count_block(self, block: ContentBlock) -> int:
        if isinstance(block, ToolResultBlock) and isinstance(block.content, list):
            n = self.block_overhead
            for item in block.content:
                if isinstance(item, dict) and item.get("type") in ("image", "document"):
                    n += self.image_tokens
                else:
                    n += self.count_text(canonical_json(item))
            return n
        return self.block_overhead + self.count_text(block_text(block))

    def count_message(self, message: Message) -> int:
        return self.message_overhead + sum(self.count_block(b) for b in message.content)

    def count_messages(self, messages: Sequence[Message]) -> int:
        return sum(self.count_message(m) for m in messages)

    def count_tools(self, tools: Sequence[ToolSpec]) -> int:
        return sum(self.tool_overhead + self.count_text(canonical_json(t.to_api())) for t in tools if not t.defer_loading)

    def count_system(self, system: str | list[dict[str, Any]]) -> int:
        if isinstance(system, str):
            return self.count_text(system)
        return sum(self.block_overhead + self.count_text(str(b.get("text", ""))) for b in system)

    def count_request(self, request: ModelRequest) -> int:
        return self.count_system(request.system) + self.count_messages(request.messages) + self.count_tools(request.tools)


def estimate_block(counter: TokenCounter, block: ContentBlock) -> int:
    """Per-block estimate using the counter's own heuristic when it has one, else ``count_text``."""
    fn = getattr(counter, "count_block", None)
    if callable(fn):
        return int(fn(block))
    return counter.count_text(block_text(block))


def estimate_message(counter: TokenCounter, message: Message) -> int:
    fn = getattr(counter, "count_message", None)
    if callable(fn):
        return int(fn(message))
    return sum(estimate_block(counter, b) for b in message.content)


class AnthropicTokenCounter:
    """Exact request counts via ``client.messages.count_tokens``; text counts fall back to a heuristic.

    :param model: model id to count for (required; supplied by the caller/config).
    :param client: an ``anthropic.Anthropic`` instance (created lazily if omitted).
    :param fallback: counter used for :meth:`count_text` and when the API call fails.
    :param strict: re-raise API errors instead of falling back.
    """

    def __init__(self, model: str, *, client: Any | None = None, fallback: TokenCounter | None = None, strict: bool = False) -> None:
        if not model:
            raise ValueError("model is required")
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("AnthropicTokenCounter needs the 'anthropic' package: pip install anthropic") from exc
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.fallback = fallback or HeuristicTokenCounter()
        self.strict = strict

    def count_text(self, text: str) -> int:
        return self.fallback.count_text(text)

    def count_block(self, block: ContentBlock) -> int:
        return estimate_block(self.fallback, block)

    def count_request(self, request: ModelRequest) -> int:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages_to_api(request.messages)}
        if request.system:
            kwargs["system"] = request.system
        tools = [t.to_api() for t in request.tools]
        if tools:
            kwargs["tools"] = tools
        try:
            result = self.client.messages.count_tokens(**kwargs)
        except Exception:
            if self.strict:
                raise
            return self.fallback.count_request(request)
        value = result.get("input_tokens") if isinstance(result, dict) else getattr(result, "input_tokens", None)
        return int(value) if value is not None else self.fallback.count_request(request)


__all__ = ["AnthropicTokenCounter", "HeuristicTokenCounter", "TokenCounter", "estimate_block", "estimate_message"]
