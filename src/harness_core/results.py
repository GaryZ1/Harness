"""Typed run results.

Research basis (reference report section 2.7 step 1): emit a typed result and carry the model
``stop_reason``, cost, usage, turn count and ``session_id``. The first five subtypes are the Claude
Agent SDK's ``ResultMessage.subtype`` values verbatim (``error_max_structured_output_retries`` is
reserved for a structured-output layer); the rest name the other termination reasons this loop
enforces in code so callers never have to parse error strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .types import Message, StopReason, Usage


class ResultSubtype(str, Enum):
    SUCCESS = "success"
    ERROR_MAX_TURNS = "error_max_turns"
    ERROR_MAX_BUDGET_USD = "error_max_budget_usd"
    ERROR_DURING_EXECUTION = "error_during_execution"
    ERROR_MAX_STRUCTURED_OUTPUT_RETRIES = "error_max_structured_output_retries"
    ERROR_MAX_TOKEN_BUDGET = "error_max_token_budget"
    ERROR_MAX_WALL_CLOCK = "error_max_wall_clock"
    ERROR_REFUSAL = "error_refusal"
    ERROR_CONTEXT_WINDOW = "error_context_window"
    ERROR_DOOM_LOOP = "error_doom_loop"
    ERROR_MAX_CONSECUTIVE_ERRORS = "error_max_consecutive_errors"
    ERROR_PERMISSION_DENIALS = "error_permission_denials"
    ERROR_GOAL_IMPOSSIBLE = "error_goal_impossible"
    ERROR_PROMPT_BLOCKED = "error_prompt_blocked"
    HALTED_BY_HOOK = "halted_by_hook"
    PAUSED_FOR_APPROVAL = "paused_for_approval"
    INTERRUPTED = "interrupted"

    @property
    def is_error(self) -> bool:
        return self.value.startswith("error_")

    @property
    def is_budget(self) -> bool:
        return self in _BUDGET


_BUDGET = {ResultSubtype.ERROR_MAX_TURNS, ResultSubtype.ERROR_MAX_BUDGET_USD, ResultSubtype.ERROR_MAX_TOKEN_BUDGET,
           ResultSubtype.ERROR_MAX_WALL_CLOCK}


@dataclass
class PendingApproval:
    tool_use_id: str
    tool_name: str
    input: dict[str, Any]
    reason: str = ""


@dataclass
class RunResult:
    """Outcome of one :meth:`AgentLoop.run` call."""

    subtype: ResultSubtype
    session_id: str
    stop_reason: StopReason | None = None   # last model stop_reason (None if no model call completed)
    final_text: str = ""
    num_turns: int = 0                      # model calls in this run (retries and wrap-up excluded)
    usage: Usage = field(default_factory=Usage)
    total_cost_usd: float = 0.0
    duration_s: float = 0.0
    error: str | None = None
    reason: str = ""                        # human-readable termination detail
    messages: list[Message] = field(default_factory=list)  # canonical history after the run
    pending_approvals: list[PendingApproval] = field(default_factory=list)
    unprocessed_input: list[str] = field(default_factory=list)
    evaluations: list[Any] = field(default_factory=list)
    wrapped_up: bool = False
    stop_override: bool = False             # stop allowed by the 8-consecutive-blocks guard

    @property
    def is_error(self) -> bool:
        return self.subtype.is_error

    @property
    def ok(self) -> bool:
        return self.subtype is ResultSubtype.SUCCESS
