"""Doom-loop (stuck) detection over the stream of tool calls.

Research basis (reference report section 2.7 step 14, section 2.3 "Loop, stuck and no-progress
detection"):

* OpenHands ``StuckDetector`` flags repetitive event patterns, e.g. **4 repeats x 2 events per
  cycle**, the same action repeated, and the same action producing the same error.
* LangChain's loop-detection middleware counts edits per file and prompts the agent to reconsider
  after repeated edits (part of its +13.7-point Terminal-Bench 2.0 gain, secondary source).

The detector only *reports*; the loop decides: the first detections inject a turn-scoped nudge, and
a detection after ``max_nudges`` nudges terminates the run with ``error_doom_loop``. Edit-count
signals are soft and only ever nudge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Sequence

from .types import ToolCall, ToolOutcome


@dataclass
class StuckConfig:
    """Thresholds (``None`` disables a rule).

    ``identical_call_threshold=4`` and ``pattern_repeats=4`` follow OpenHands' 4-repeat patterns;
    ``identical_error_threshold=3`` follows its action-error rule; the rest are our choices.
    """

    identical_call_threshold: int | None = 4
    pattern_repeats: int | None = 4
    max_cycle_len: int = 3
    identical_error_threshold: int | None = 3
    max_edits_per_file: int | None = None
    edit_tools: frozenset[str] = frozenset({"edit", "write", "str_replace", "str_replace_based_edit_tool", "apply_patch",
                                            "Edit", "Write", "MultiEdit"})
    path_keys: tuple[str, ...] = ("path", "file_path", "filename")
    max_nudges: int = 2
    window: int = 64


@dataclass
class StuckSignal:
    kind: str    # identical_call | repeating_pattern | identical_error | edit_count
    detail: str

    @property
    def hard(self) -> bool:
        return self.kind != "edit_count"


def call_signature(call: ToolCall) -> str:
    return call.name + ":" + json.dumps(call.input, sort_keys=True, ensure_ascii=False, default=str)


@dataclass
class StuckDetector:
    config: StuckConfig = field(default_factory=StuckConfig)
    _calls: list[str] = field(default_factory=list)
    _errors: list[tuple[str, str] | None] = field(default_factory=list)
    _edits: dict[str, int] = field(default_factory=dict)

    def reset(self) -> None:
        """Forget repetition history (called after a nudge, so firing again needs fresh repeats)."""
        self._calls.clear()
        self._errors.clear()

    def observe_calls(self, calls: Sequence[ToolCall]) -> StuckSignal | None:
        c = self.config
        signal: StuckSignal | None = None
        for call in calls:
            self._calls.append(call_signature(call))
            if call.name in c.edit_tools and c.max_edits_per_file:
                path = next((str(call.input[k]) for k in c.path_keys if k in call.input), None)
                if path is not None:
                    n = self._edits[path] = self._edits.get(path, 0) + 1
                    if n % c.max_edits_per_file == 0 and signal is None:
                        signal = StuckSignal("edit_count", f"{path} has been edited {n} times")
        del self._calls[:-c.window]
        hard = self._check_calls()
        return hard or signal

    def _check_calls(self) -> StuckSignal | None:
        c, h = self.config, self._calls
        n = c.identical_call_threshold
        if n and len(h) >= n and len(set(h[-n:])) == 1:
            return StuckSignal("identical_call", f"the same tool call ({h[-1][:120]}) was made {n} times in a row")
        r = c.pattern_repeats
        if r:
            for length in range(2, c.max_cycle_len + 1):
                span = length * r
                if len(h) < span:
                    break
                block = h[-length:]
                if len(set(block)) > 1 and all(h[-span + i] == block[i % length] for i in range(span)):
                    names = ", ".join(s.split(":", 1)[0] for s in block)
                    return StuckSignal("repeating_pattern", f"a cycle of {length} calls ({names}) repeated {r} times")
        return None

    def observe_results(self, calls: Sequence[ToolCall], outcomes: Sequence[ToolOutcome]) -> StuckSignal | None:
        n = self.config.identical_error_threshold
        for call, out in zip(calls, outcomes, strict=True):
            text = out.content if isinstance(out.content, str) else json.dumps(out.content, default=str)
            self._errors.append((call_signature(call), text[:500]) if out.is_error else None)
        del self._errors[:-self.config.window]
        if n and len(self._errors) >= n:
            tail = self._errors[-n:]
            if all(t is not None for t in tail) and len(set(tail)) == 1:
                return StuckSignal("identical_error", f"the same call failed {n} times in a row with the same error")
        return None
