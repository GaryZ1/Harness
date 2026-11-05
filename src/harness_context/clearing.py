"""Client-side tool-result clearing with restorable placeholders (compaction ladder rung L2).

Mirrors Anthropic's ``clear_tool_uses_20250919`` semantics for providers (or deployments) where
the server-side edit is not used:

* trigger at **100,000** input tokens and **keep the last 3** tool results *(vendor defaults)*;
* clear **oldest first**; skip the pass unless it frees at least ``clear_at_least`` tokens, so a
  cache write is only paid for a meaningful saving (pitfall: "tool-result clearing without
  ``clear_at_least`` pays a cache write for trivial savings; client-side pruning on every turn is
  the same mistake");
* ``exclude_tools`` - memory / todo / plan tools are never cleared;
* ``clear_tool_inputs`` off by default, and never applied to an assistant turn that carries
  thinking blocks (editing those breaks signatures on the newest models).

Placeholders are *restorable*: they keep the tool name and the locators from the call input
(path, URL, query, command) or the offload path, so the model can re-fetch the content. Error
results keep a short excerpt of the error ("keep errors in, drop verbose traces"). An earlier
result that is byte-identical to a later one is cleared even inside the keep window
(near-duplicate pruning against context rot).

Clearing is *sticky*: the set of cleared ``tool_use_id`` s only grows at clearing passes and is
re-applied identically on every later window, so the derived prefix stays byte-stable between
passes. The durable log is never touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Collection, Sequence

from harness_core.types import Message, ToolResultBlock, ToolUseBlock

from ._io import sha256_text
from .errors import ConfigurationError
from .messages import copy_message, has_reasoning, result_text, tool_use_map
from .references import extract_locators, find_offload_path
from .tokens import HeuristicTokenCounter, TokenCounter, estimate_block

DEFAULT_EXCLUDED_TOOLS = frozenset({"memory", "todo", "todo_write", "TodoWrite", "update_plan", "plan"})


@dataclass(frozen=True)
class ClearingPolicy:
    trigger_tokens: int = 100_000          # vendor default
    keep_recent: int = 3                   # vendor default
    clear_at_least_tokens: int = 5_000     # our choice: amortize one cache write
    exclude_tools: frozenset[str] = field(default_factory=lambda: DEFAULT_EXCLUDED_TOOLS)
    clear_tool_inputs: bool = False        # vendor default
    error_excerpt_chars: int = 300
    min_result_tokens: int = 50            # results smaller than this are not worth a placeholder
    dedupe: bool = True

    def __post_init__(self) -> None:
        if self.keep_recent < 0 or self.trigger_tokens <= 0 or self.clear_at_least_tokens < 0:
            raise ConfigurationError("invalid clearing policy values")


@dataclass
class ClearingPlan:
    cleared_ids: list[str]
    freed_tokens: int
    skipped_reason: str | None = None
    duplicate_of: dict[str, str] = field(default_factory=dict)

    @property
    def applied(self) -> bool:
        return bool(self.cleared_ids) and self.skipped_reason is None


class ToolResultClearer:
    def __init__(self, policy: ClearingPolicy | None = None, counter: TokenCounter | None = None) -> None:
        self.policy = policy or ClearingPolicy()
        self.counter = counter or HeuristicTokenCounter()

    # ------------------------------------------------------------------ planning
    def plan(self, window: Sequence[Message], already_cleared: Collection[str] = ()) -> ClearingPlan:
        """Choose which results to clear now (does not modify anything)."""
        p = self.policy
        uses = tool_use_map(window)
        results: list[ToolResultBlock] = [b for m in window for b in m.tool_results]
        eligible = [r for r in results if r.tool_use_id not in already_cleared and self._clearable(r, uses)]
        keep = {r.tool_use_id for r in results[-p.keep_recent:]} if p.keep_recent else set()
        chosen: list[str] = []
        duplicates: dict[str, str] = {}
        if p.dedupe:
            last_by_hash: dict[str, str] = {}
            for r in results:
                last_by_hash[sha256_text(result_text(r))] = r.tool_use_id
            for r in eligible:
                later = last_by_hash.get(sha256_text(result_text(r)))
                if later and later != r.tool_use_id:
                    duplicates[r.tool_use_id] = later
        for r in eligible:  # oldest first
            if r.tool_use_id in keep and r.tool_use_id not in duplicates:
                continue
            chosen.append(r.tool_use_id)
        chosen_set = set(chosen)
        freed = sum(self._savings(r, uses.get(r.tool_use_id)) for r in eligible if r.tool_use_id in chosen_set)
        if not chosen:
            return ClearingPlan([], 0, "nothing eligible to clear")
        if freed < p.clear_at_least_tokens:
            return ClearingPlan([], freed, f"would free ~{freed} tokens < clear_at_least {p.clear_at_least_tokens}")
        return ClearingPlan(chosen, freed, None, {k: v for k, v in duplicates.items() if k in chosen})

    def _clearable(self, r: ToolResultBlock, uses: dict[str, ToolUseBlock]) -> bool:
        use = uses.get(r.tool_use_id)
        if use is not None and use.name in self.policy.exclude_tools:
            return False
        return estimate_block(self.counter, r) >= self.policy.min_result_tokens

    def _savings(self, r: ToolResultBlock, use: ToolUseBlock | None) -> int:
        placeholder = self.placeholder(r, use)
        return max(0, estimate_block(self.counter, r) - self.counter.count_text(placeholder))

    # ------------------------------------------------------------------ rendering
    def placeholder(self, result: ToolResultBlock, use: ToolUseBlock | None, *, duplicate_of: str | None = None) -> str:
        """Deterministic, restorable stand-in for a cleared result (no timestamps)."""
        text = result_text(result)
        tokens = estimate_block(self.counter, result)
        name = use.name if use else "unknown_tool"
        attrs = [f'tool="{name}"', f'tokens="~{tokens}"']
        if result.is_error:
            attrs.append('error="true"')
        lines = [f"<cleared-tool-result {' '.join(attrs)}>"]
        if duplicate_of:
            lines.append(f"Identical to the later result of tool call {duplicate_of}; cleared to avoid duplication.")
        else:
            lines.append("This tool output was cleared from context to save space.")
        offloaded = find_offload_path(text)
        locators = extract_locators(use.input if use else None)
        if offloaded:
            lines.append(f"Full output is saved at: {offloaded} (read it with a range or grep to restore).")
        elif locators:
            loc = ", ".join(f"{k}={v}" for k, v in locators.items())
            lines.append(f"Locators: {loc}. To restore, re-run {name} with the same input or open the locator directly.")
        else:
            lines.append(f"To restore, re-run {name} with the same input.")
        if result.is_error:
            excerpt = " ".join(text.split())[: self.policy.error_excerpt_chars]
            lines.append(f"Error excerpt (kept so the failure is not retried blindly): {excerpt}")
        lines.append("</cleared-tool-result>")
        return "\n".join(lines)

    def apply(
        self,
        window: Sequence[Message],
        cleared_ids: Collection[str],
        *,
        duplicate_of: dict[str, str] | None = None,
    ) -> list[Message]:
        """Return a derived window with ``cleared_ids`` replaced by placeholders (input untouched)."""
        if not cleared_ids:
            return list(window)
        cleared = set(cleared_ids)
        dup = duplicate_of or {}
        uses = tool_use_map(window)
        out: list[Message] = []
        for m in window:
            if m.tool_results and any(r.tool_use_id in cleared for r in m.tool_results):
                blocks = []
                for b in m.content:
                    if isinstance(b, ToolResultBlock) and b.tool_use_id in cleared:
                        text = self.placeholder(b, uses.get(b.tool_use_id), duplicate_of=dup.get(b.tool_use_id))
                        blocks.append(ToolResultBlock(tool_use_id=b.tool_use_id, content=text, is_error=b.is_error))
                    else:
                        blocks.append(b)
                out.append(copy_message(m, blocks))
            elif self.policy.clear_tool_inputs and m.tool_uses and not has_reasoning(m) and any(u.id in cleared for u in m.tool_uses):
                blocks = [
                    ToolUseBlock(id=b.id, name=b.name, input={}) if isinstance(b, ToolUseBlock) and b.id in cleared else b
                    for b in m.content
                ]
                out.append(copy_message(m, blocks))
            else:
                out.append(m)
        return out


__all__ = ["ClearingPlan", "ClearingPolicy", "DEFAULT_EXCLUDED_TOOLS", "ToolResultClearer"]
