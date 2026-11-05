"""Layered budgets and cost estimation.

Research basis (reference report section 2.7 step 2, section 2.5, section 2.8):

* Claude Agent SDK ``max_turns`` counts each tool-calling turn plus the final text-only response;
  ``max_budget_usd`` caps estimated spend and is *checked after each response*, so it can overshoot
  by one response. Both default to no limit, and the docs say "Setting a budget is a good default
  for production agents" (so :class:`Budgets` makes every limit explicit; pick values per deployment).
* Offer a harness-counted token budget, time-budget warnings (part of LangChain's +13.7-point
  Terminal-Bench gain), and a Codex-style wrap-up turn when a budget trips.
* Claude Code auto mode escalates after **3 consecutive or 20 total denials** and terminates
  headless runs at that point; those are the denial defaults here.

Prices are never hardcoded: :class:`PricingTable` is filled by the caller per model id.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Literal

from .results import ResultSubtype
from .types import Usage

#: ``(usage_of_one_call, model_id) -> estimated USD``.
CostEstimator = Callable[[Usage, str], float]


@dataclass
class Budgets:
    """Per-run (or per-session, ``scope="session"``) limits. ``None`` disables a limit.

    =============================  =======  ===============================================
    field                          default  basis
    =============================  =======  ===============================================
    ``max_turns``                  None     Agent SDK default (no limit); set per run
    ``max_total_tokens``           None     harness-counted (input incl. cache + output)
    ``max_wall_clock_s``           None     wall clock, checked post-response and post-tools
    ``max_budget_usd``             None     Agent SDK default (no limit); checked post-response
    ``max_consecutive_error_turns``  5      our choice: turns where every tool call failed
    ``max_consecutive_denials``    3        Claude Code auto-mode escalation
    ``max_total_denials``          20       Claude Code auto-mode escalation
    ``warning_fraction``           0.8      our choice: one turn-scoped warning per budget
    =============================  =======  ===============================================
    """

    max_turns: int | None = None
    max_total_tokens: int | None = None
    max_wall_clock_s: float | None = None
    max_budget_usd: float | None = None
    max_consecutive_error_turns: int | None = 5
    max_consecutive_denials: int | None = 3
    max_total_denials: int | None = 20
    warning_fraction: float | None = 0.8
    scope: Literal["run", "session"] = "run"

    def __post_init__(self) -> None:
        for name in ("max_turns", "max_total_tokens", "max_consecutive_error_turns", "max_consecutive_denials", "max_total_denials"):
            v = getattr(self, name)
            if v is not None and v < 1:
                raise ValueError(f"{name} must be >= 1 or None")
        for name in ("max_wall_clock_s", "max_budget_usd"):
            v = getattr(self, name)
            if v is not None and v <= 0:
                raise ValueError(f"{name} must be > 0 or None")
        if self.warning_fraction is not None and not 0 < self.warning_fraction < 1:
            raise ValueError("warning_fraction must be in (0, 1)")


@dataclass(frozen=True)
class Price:
    """USD per million tokens. Cache prices default to Anthropic's multipliers on the input price
    (cache reads 0.1x, 5-minute cache writes 1.25x) when not given."""

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None

    def cost(self, u: Usage) -> float:
        cr = self.cache_read_per_mtok if self.cache_read_per_mtok is not None else self.input_per_mtok * 0.1
        cw = self.cache_write_per_mtok if self.cache_write_per_mtok is not None else self.input_per_mtok * 1.25
        return (u.input_tokens * self.input_per_mtok + u.output_tokens * self.output_per_mtok
                + u.cache_read_input_tokens * cr + u.cache_creation_input_tokens * cw) / 1_000_000


class PricingTable:
    """A :data:`CostEstimator` over caller-supplied prices keyed by model id.

    Unknown models use ``default`` if given, else cost 0 and are recorded in :attr:`unpriced`
    (check it: a dollar budget cannot trip for a model with no price).
    """

    def __init__(self, prices: dict[str, Price], *, default: Price | None = None) -> None:
        self.prices = dict(prices)
        self.default = default
        self.unpriced: set[str] = set()

    def __call__(self, usage: Usage, model: str) -> float:
        price = self.prices.get(model, self.default)
        if price is None:
            self.unpriced.add(model)
            return 0.0
        return price.cost(usage)


@dataclass
class BudgetTrip:
    subtype: ResultSubtype
    reason: str


@dataclass
class BudgetTracker:
    """Pure accounting for one run; the loop asks it whether to continue."""

    budgets: Budgets
    monotonic: Callable[[], float] = time.monotonic
    turns: int = 0
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    consecutive_error_turns: int = 0
    consecutive_denials: int = 0
    total_denials: int = 0
    _start: float = 0.0
    _warned: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self._start = self.monotonic()

    # ------------------------------------------------------------------ recording
    def record_model_call(self, usage: Usage, cost_usd: float = 0.0, *, count_turn: bool = True) -> None:
        self.usage = self.usage + usage
        self.cost_usd += cost_usd
        if count_turn:
            self.turns += 1

    def record_tool_turn(self, *, all_failed: bool) -> None:
        self.consecutive_error_turns = self.consecutive_error_turns + 1 if all_failed else 0

    def record_denial(self) -> None:
        self.consecutive_denials += 1
        self.total_denials += 1

    def record_permitted(self) -> None:
        self.consecutive_denials = 0

    # ------------------------------------------------------------------ queries
    @property
    def tokens_used(self) -> int:
        return self.usage.total_input + self.usage.output_tokens

    @property
    def elapsed_s(self) -> float:
        return self.monotonic() - self._start

    def exceeded(self, *, need_another_call: bool) -> BudgetTrip | None:
        """Hard budgets. ``max_turns`` only trips when the loop would need another model call."""
        b = self.budgets
        if b.max_budget_usd is not None and self.cost_usd >= b.max_budget_usd:
            return BudgetTrip(ResultSubtype.ERROR_MAX_BUDGET_USD, f"estimated spend ${self.cost_usd:.4f} reached max_budget_usd ${b.max_budget_usd:.4f}")
        if b.max_total_tokens is not None and self.tokens_used >= b.max_total_tokens:
            return BudgetTrip(ResultSubtype.ERROR_MAX_TOKEN_BUDGET, f"{self.tokens_used} tokens used of {b.max_total_tokens}")
        if b.max_wall_clock_s is not None and self.elapsed_s >= b.max_wall_clock_s:
            return BudgetTrip(ResultSubtype.ERROR_MAX_WALL_CLOCK, f"{self.elapsed_s:.1f}s elapsed of {b.max_wall_clock_s:.1f}s")
        if need_another_call and b.max_turns is not None and self.turns >= b.max_turns:
            return BudgetTrip(ResultSubtype.ERROR_MAX_TURNS, f"{self.turns} turns used of max_turns={b.max_turns}")
        return None

    def error_limit(self) -> BudgetTrip | None:
        b = self.budgets
        if b.max_consecutive_denials is not None and self.consecutive_denials >= b.max_consecutive_denials:
            return BudgetTrip(ResultSubtype.ERROR_PERMISSION_DENIALS, f"{self.consecutive_denials} consecutive permission denials")
        if b.max_total_denials is not None and self.total_denials >= b.max_total_denials:
            return BudgetTrip(ResultSubtype.ERROR_PERMISSION_DENIALS, f"{self.total_denials} total permission denials")
        if b.max_consecutive_error_turns is not None and self.consecutive_error_turns >= b.max_consecutive_error_turns:
            return BudgetTrip(ResultSubtype.ERROR_MAX_CONSECUTIVE_ERRORS,
                              f"{self.consecutive_error_turns} consecutive turns in which every tool call failed")
        return None

    def new_warnings(self) -> list[str]:
        """Budgets that crossed ``warning_fraction`` since the last call (each reported once)."""
        b, frac = self.budgets, self.budgets.warning_fraction
        if frac is None:
            return []
        checks = [
            ("turns", self.turns, b.max_turns, "turns"),
            ("tokens", self.tokens_used, b.max_total_tokens, "tokens"),
            ("usd", self.cost_usd, b.max_budget_usd, "USD"),
            ("wall_clock", self.elapsed_s, b.max_wall_clock_s, "seconds"),
        ]
        out = []
        for key, used, limit, unit in checks:
            if limit is None or key in self._warned or used < frac * limit:
                continue
            self._warned.add(key)
            remaining = max(0.0, limit - used)
            fmt = (lambda v: f"{v:.2f}") if isinstance(used, float) else str
            out.append(f"{fmt(used)} of {fmt(limit)} {unit} used ({fmt(remaining)} remaining)")
        return out
