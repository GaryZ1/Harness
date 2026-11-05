"""Pure helpers over :class:`~harness_core.types.Message` lists: copying, tagging and cut safety.

Everything here returns *new* objects. The durable log handed to the assembler is never mutated:
the research is explicit that history must be append-only (editing an earlier turn breaks the
prompt-cache prefix and, on the newest Claude models, invalidates thinking-block signatures), so
every reduction is performed on a derived copy and recorded as an explicit, logged event.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Iterable, Sequence

from harness_core.types import (
    ContentBlock,
    Message,
    OpaqueBlock,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

from ._io import canonical_json

#: Metadata namespace this package writes into ``Message.metadata``.
META_KEY = "harness_context"

KIND_COMPACTION = "compaction"            # client-side compaction marker (summary + re-injection)
KIND_REINJECTION = "reinjection"          # post-server-compaction re-injection bundle
KIND_DYNAMIC = "dynamic"                  # turn-scoped dynamic tail (never stored in the log)
KIND_REMINDER = "reminder"                # persistent appended reminder (stored in the log)

#: Provider block types that carry reasoning state and must never be edited or re-ordered.
REASONING_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})
SERVER_COMPACTION_BLOCK_TYPE = "compaction"


def context_meta(message: Message) -> dict[str, Any]:
    meta = message.metadata.get(META_KEY)
    return meta if isinstance(meta, dict) else {}


def kind_of(message: Message) -> str | None:
    kind = context_meta(message).get("kind")
    return str(kind) if kind else None


def is_compaction_marker(message: Message) -> bool:
    return kind_of(message) == KIND_COMPACTION


def has_server_compaction(message: Message) -> bool:
    return any(isinstance(b, OpaqueBlock) and b.type == SERVER_COMPACTION_BLOCK_TYPE for b in message.content)


def has_reasoning(message: Message) -> bool:
    return any(isinstance(b, OpaqueBlock) and b.type in REASONING_BLOCK_TYPES for b in message.content)


def is_user_turn_start(message: Message) -> bool:
    """A *real* user turn: a user message with no tool results that this package did not author."""
    return message.role is Role.USER and not message.tool_results and kind_of(message) is None


# ----------------------------------------------------------------------------- copying


def copy_block(block: ContentBlock, *, cache_control: dict[str, Any] | None = None) -> ContentBlock:
    """Shallow-copy ``block`` with ``cache_control`` replaced (``None`` strips it).

    Thinking / redacted-thinking / server blocks are copied verbatim; only a server ``compaction``
    block may carry a breakpoint (the docs recommend ``cache_control`` on it).
    """
    if isinstance(block, (TextBlock, ToolResultBlock)):
        return dataclasses.replace(block, cache_control=cache_control)
    if isinstance(block, ToolUseBlock):
        return ToolUseBlock(id=block.id, name=block.name, input=dict(block.input))
    raw = dict(block.raw)
    raw.pop("cache_control", None)
    if cache_control and raw.get("type") == SERVER_COMPACTION_BLOCK_TYPE:
        raw["cache_control"] = dict(cache_control)
    return OpaqueBlock(raw=raw)


def accepts_cache_control(block: ContentBlock) -> bool:
    if isinstance(block, (TextBlock, ToolResultBlock)):
        return True
    return isinstance(block, OpaqueBlock) and block.type == SERVER_COMPACTION_BLOCK_TYPE


def copy_message(message: Message, content: Sequence[ContentBlock] | None = None) -> Message:
    """Copy ``message`` keeping its id (so derived windows can be mapped back onto the log)."""
    blocks = [copy_block(b) for b in message.content] if content is None else list(content)
    return Message(role=message.role, content=blocks, id=message.id, metadata=dict(message.metadata))


def with_breakpoint(message: Message, cache_control: dict[str, Any]) -> Message | None:
    """Return a copy with ``cache_control`` on the last eligible block, or ``None`` if none is eligible."""
    for i in range(len(message.content) - 1, -1, -1):
        if accepts_cache_control(message.content[i]):
            blocks = list(message.content)
            blocks[i] = copy_block(blocks[i], cache_control=cache_control)
            return copy_message(message, blocks)
    return None


def count_breakpoints(messages: Iterable[Message]) -> int:
    n = 0
    for m in messages:
        for b in m.content:
            if isinstance(b, (TextBlock, ToolResultBlock)) and b.cache_control:
                n += 1
            elif isinstance(b, OpaqueBlock) and b.raw.get("cache_control"):
                n += 1
    return n


# ----------------------------------------------------------------------------- fingerprints


def message_fingerprint(message: Message) -> str:
    """Hash of what the provider sees for ``message`` (cache markers excluded)."""
    stripped = [copy_block(b).to_api() for b in message.content]
    return hashlib.sha256(canonical_json({"role": message.role.value, "content": stripped}).encode("utf-8")).hexdigest()


def log_prefix_sha256(log: Sequence[Message], through_index: int) -> str:
    """Hash chain over ``log[: through_index + 1]`` (ids and content), used to validate compaction markers."""
    h = hashlib.sha256()
    for m in log[: through_index + 1]:
        h.update(m.id.encode("utf-8"))
        h.update(b"\x00")
        h.update(message_fingerprint(m).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


# ----------------------------------------------------------------------------- tool pairing


def tool_use_map(messages: Iterable[Message]) -> dict[str, ToolUseBlock]:
    return {b.id: b for m in messages for b in m.content if isinstance(b, ToolUseBlock)}


def is_safe_cut(messages: Sequence[Message], cut: int) -> bool:
    """True when splitting at ``cut`` keeps every ``tool_use`` with its ``tool_result``.

    Invariant from the compaction docs: "the cut must not split a tool call from its result".
    """
    if cut <= 0 or cut >= len(messages):
        return False
    head_uses = {b.id for m in messages[:cut] for b in m.content if isinstance(b, ToolUseBlock)}
    head_results = {b.tool_use_id for m in messages[:cut] for b in m.content if isinstance(b, ToolResultBlock)}
    if head_uses - head_results:
        return False
    tail_results = {b.tool_use_id for m in messages[cut:] for b in m.content if isinstance(b, ToolResultBlock)}
    return not (tail_results & head_uses)


def merge_consecutive_user(messages: Sequence[Message]) -> list[Message]:
    """Merge runs of user messages into one (derived window only).

    ``tool_result`` blocks are moved to the front of the merged content, which the Messages API
    requires for the user turn that answers a ``tool_use``. The first message's id is kept and the
    merged ids are listed in metadata.
    """
    out: list[Message] = []
    for m in messages:
        if out and m.role is Role.USER and out[-1].role is Role.USER:
            prev = out[-1]
            blocks = list(prev.content) + list(m.content)
            results = [b for b in blocks if isinstance(b, ToolResultBlock)]
            others = [b for b in blocks if not isinstance(b, ToolResultBlock)]
            meta = dict(prev.metadata)
            meta["merged_ids"] = [*meta.get("merged_ids", [prev.id]), m.id]
            out[-1] = Message(role=Role.USER, content=results + others, id=prev.id, metadata=meta)
        else:
            out.append(m)
    return out


def block_text(block: ContentBlock) -> str:
    """Serialized text used for token estimation and size accounting."""
    if isinstance(block, TextBlock):
        return block.text
    if isinstance(block, ToolUseBlock):
        return block.name + canonical_json(block.input)
    if isinstance(block, ToolResultBlock):
        if isinstance(block.content, str):
            return block.content
        return canonical_json(block.content)
    return canonical_json(block.raw)


def result_text(block: ToolResultBlock) -> str:
    return block.text if isinstance(block.content, str) or all(
        isinstance(c, dict) and c.get("type") == "text" for c in block.content
    ) else block_text(block)
