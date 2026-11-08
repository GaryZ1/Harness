"""Context-rot guards that are cheap to run on every window (reference report 3.3 techniques 26, 33).

* **Few-shot rut / loops** - OpenHands ``StuckDetector`` flags repetitive patterns (e.g. the same
  action 4 times). :class:`StuckDetector` looks at the tail of the window for repeated identical
  tool calls and repeated identical errors, and returns a short corrective note for the dynamic
  channel ("keep errors in" so the model updates its beliefs, but break the loop).
* **Context confusion** - :func:`tool_count_warning` flags hot tool sets above the 30-50 range
  where Claude's selection accuracy degrades (use deferred loading + tool search instead).

Other mitigations live where they act: quality-oriented thresholds (:mod:`.budget`), restorable
clearing with duplicate pruning (:mod:`.clearing`), recitation at the end (:mod:`.dynamic`),
sub-agent quarantine (:mod:`.subagent`), instruction-file linting (:mod:`.layers`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from harness_core.types import Message, ToolSpec

from ._io import canonical_json
from .messages import result_text

HOT_TOOL_SOFT_LIMIT = 10     # checklist: hot tier <= 5-10 tools
HOT_TOOL_HARD_LIMIT = 30     # Claude selection accuracy degrades past 30-50 tools


@dataclass(frozen=True)
class StuckDetector:
    repeat_threshold: int = 4
    error_repeat_threshold: int = 3

    def check(self, window: Sequence[Message]) -> str | None:
        calls = [(u.name, canonical_json(u.input)) for m in window for u in m.tool_uses]
        n = self.repeat_threshold
        if len(calls) >= n and len(set(calls[-n:])) == 1:
            name = calls[-1][0]
            return (
                f"Loop check: you have called {name} with identical input {n} times in a row. The result will not change; "
                "step back, re-read the last result, and try a different approach or ask for help."
            )
        errors = [result_text(r).strip() for m in window for r in m.tool_results if r.is_error]
        k = self.error_repeat_threshold
        results = [r for m in window for r in m.tool_results]
        if len(errors) >= k and len(set(errors[-k:])) == 1 and results[-k:] and all(r.is_error for r in results[-k:]):
            return (
                f"Loop check: the last {k} tool calls failed with the same error. Do not retry the same thing; "
                "diagnose the cause or change approach."
            )
        return None


def tool_count_warning(tools: Sequence[ToolSpec]) -> str | None:
    hot = [t for t in tools if not t.defer_loading]
    if len(hot) > HOT_TOOL_HARD_LIMIT:
        return f"{len(hot)} non-deferred tools: selection accuracy degrades past 30-50; defer most tools behind tool search"
    if len(hot) > HOT_TOOL_SOFT_LIMIT:
        return f"{len(hot)} non-deferred tools: keep the hot tier to 5-10 tools and defer the rest"
    return None


__all__ = ["HOT_TOOL_HARD_LIMIT", "HOT_TOOL_SOFT_LIMIT", "StuckDetector", "tool_count_warning"]
