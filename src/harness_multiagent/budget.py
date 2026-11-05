"""Hierarchical budgets and per-agent cost accounting.

Research basis (reference 8.7 items 4 and 19, 8.8 "Treating multi-agent as free quality", "No hard
caps"): multi-agent gains track token spend (80% of BrowseComp variance) and multi-agent systems used
~15x chat tokens, so a **run-level budget that counts subagent spend is mandatory**. Budgets are
checked *after* each model response (as in the Claude Agent SDK), so spend can overshoot by one
response plus whatever concurrently running agents spend; :class:`BudgetLedger` makes that overshoot
bounded by firing ``on_exceeded`` the moment any charge crosses the cap, which the manager wires to
cancel every running agent and refuse new spawns ("Budget limit reached").

Tokens are the primary unit (``input + cache_read + cache_creation + output``). A caller-supplied
``cost_fn(model_id, usage) -> usd`` adds a dollar cap without this package knowing any prices.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable

from harness_core.types import Usage

CostFn = Callable[[str, Usage], float]


def usage_tokens(u: Usage) -> int:
    """Tokens charged against budgets: every input category plus output."""
    return u.total_input + u.output_tokens


@dataclass(frozen=True)
class BudgetLimits:
    max_tokens: int | None = None
    max_usd: float | None = None

    def __post_init__(self) -> None:
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.max_usd is not None and self.max_usd <= 0:
            raise ValueError("max_usd must be > 0")


class BudgetLedger:
    """Thread-safe spend counter; charges propagate to every ancestor ledger."""

    def __init__(
        self,
        name: str,
        limits: BudgetLimits | None = None,
        *,
        parent: "BudgetLedger | None" = None,
        cost_fn: CostFn | None = None,
        on_exceeded: Callable[[str], None] | None = None,
    ) -> None:
        self.name = name
        self.limits = limits or BudgetLimits()
        self.parent = parent
        self.cost_fn = cost_fn if cost_fn is not None else (parent.cost_fn if parent else None)
        self.on_exceeded = on_exceeded
        self._lock = threading.Lock()
        self._usage = Usage()
        self._usd = 0.0
        self._fired = False

    def child(self, name: str, limits: BudgetLimits | None = None, *, on_exceeded: Callable[[str], None] | None = None) -> "BudgetLedger":
        return BudgetLedger(name, limits, parent=self, cost_fn=self.cost_fn, on_exceeded=on_exceeded)

    # ------------------------------------------------------------------ charging
    def charge(self, usage: Usage, *, model: str = "") -> None:
        usd = float(self.cost_fn(model, usage)) if self.cost_fn else 0.0
        self._charge(usage, usd)

    def _charge(self, usage: Usage, usd: float) -> None:
        with self._lock:
            self._usage = self._usage + usage
            self._usd += usd
            reason = self._own_exceeded_locked()
            fire = reason is not None and not self._fired
            if fire:
                self._fired = True
        if fire and self.on_exceeded is not None:
            self.on_exceeded(reason or "")
        if self.parent is not None:
            self.parent._charge(usage, usd)

    # ------------------------------------------------------------------ queries
    def _own_exceeded_locked(self) -> str | None:
        lim = self.limits
        if lim.max_tokens is not None and usage_tokens(self._usage) >= lim.max_tokens:
            return f"Budget limit reached: {self.name} used {usage_tokens(self._usage)} of {lim.max_tokens} tokens"
        if lim.max_usd is not None and self._usd >= lim.max_usd:
            return f"Budget limit reached: {self.name} spent ${self._usd:.4f} of ${lim.max_usd:.2f}"
        return None

    def exceeded(self) -> str | None:
        """Reason string if this ledger or any ancestor is at or over its cap, else ``None``."""
        node: BudgetLedger | None = self
        while node is not None:
            with node._lock:
                reason = node._own_exceeded_locked()
            if reason:
                return reason
            node = node.parent
        return None

    @property
    def usage(self) -> Usage:
        with self._lock:
            return Usage(**vars(self._usage))

    @property
    def tokens(self) -> int:
        with self._lock:
            return usage_tokens(self._usage)

    @property
    def usd(self) -> float:
        with self._lock:
            return self._usd

    def remaining_tokens(self) -> int | None:
        """Smallest remaining token headroom across this ledger and its ancestors."""
        best: int | None = None
        node: BudgetLedger | None = self
        while node is not None:
            if node.limits.max_tokens is not None:
                rem = max(0, node.limits.max_tokens - node.tokens)
                best = rem if best is None else min(best, rem)
            node = node.parent
        return best


# ----------------------------------------------------------------------------- accounting


@dataclass
class AgentRecord:
    """One finished subagent run (the ``invoke_agent`` span's attributes)."""

    run_id: str
    agent: str
    parent_run_id: str | None
    depth: int
    status: str
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    usd: float = 0.0
    turns: int = 0
    tool_calls: int = 0
    wall_clock_s: float = 0.0


@dataclass
class AgentTotals:
    runs: int = 0
    usage: Usage = field(default_factory=Usage)
    usd: float = 0.0
    turns: int = 0
    tool_calls: int = 0
    wall_clock_s: float = 0.0
    failures: int = 0

    @property
    def tokens(self) -> int:
        return usage_tokens(self.usage)


@dataclass(frozen=True)
class CriticalPath:
    """Longest dependent chain of subagent runs, measured in model turns ("critical steps")."""

    run_ids: tuple[str, ...]
    steps: int
    total_steps: int

    @property
    def parallel_speedup(self) -> float:
        """``total_steps / steps``; ~1.0 means the fan-out did not cut latency."""
        return self.total_steps / self.steps if self.steps else 1.0


class RunAccounting:
    """Thread-safe collection of :class:`AgentRecord` with per-agent aggregation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[AgentRecord] = []

    def add(self, record: AgentRecord) -> None:
        with self._lock:
            self._records.append(record)

    def records(self) -> list[AgentRecord]:
        with self._lock:
            return list(self._records)

    def by_agent(self) -> dict[str, AgentTotals]:
        out: dict[str, AgentTotals] = {}
        for r in self.records():
            t = out.setdefault(r.agent, AgentTotals())
            t.runs += 1
            t.usage = t.usage + r.usage
            t.usd += r.usd
            t.turns += r.turns
            t.tool_calls += r.tool_calls
            t.wall_clock_s += r.wall_clock_s
            t.failures += 0 if r.status == "success" else 1
        return dict(sorted(out.items()))

    def total_usage(self) -> Usage:
        total = Usage()
        for r in self.records():
            total = total + r.usage
        return total

    def critical_path(self) -> CriticalPath:
        recs = self.records()
        by_id = {r.run_id: r for r in recs}
        children: dict[str | None, list[AgentRecord]] = {}
        for r in recs:
            parent = r.parent_run_id if r.parent_run_id in by_id else None
            children.setdefault(parent, []).append(r)

        def longest(node: AgentRecord) -> tuple[int, tuple[str, ...]]:
            best: tuple[int, tuple[str, ...]] = (0, ())
            for c in sorted(children.get(node.run_id, []), key=lambda x: x.run_id):
                cand = longest(c)
                if cand[0] > best[0]:
                    best = cand
            return node.turns + best[0], (node.run_id,) + best[1]

        best: tuple[int, tuple[str, ...]] = (0, ())
        for root in sorted(children.get(None, []), key=lambda x: x.run_id):
            cand = longest(root)
            if cand[0] > best[0]:
                best = cand
        return CriticalPath(run_ids=best[1], steps=best[0], total_steps=sum(r.turns for r in recs))

    def summary(self) -> dict:
        cp = self.critical_path()
        return {
            "agents": {
                name: {"runs": t.runs, "tokens": t.tokens, "usd": round(t.usd, 6), "turns": t.turns, "tool_calls": t.tool_calls,
                       "wall_clock_s": round(t.wall_clock_s, 3), "failures": t.failures}
                for name, t in self.by_agent().items()
            },
            "total_tokens": usage_tokens(self.total_usage()),
            "critical_path_steps": cp.steps,
            "total_steps": cp.total_steps,
        }


__all__ = [
    "BudgetLimits", "BudgetLedger", "CostFn", "usage_tokens", "AgentRecord", "AgentTotals", "CriticalPath", "RunAccounting",
]
