"""Cost accounting from :class:`~harness_core.types.Usage` and a caller-supplied price table.

Research basis (reference §10.3 "Prompt caching as the main cost lever", §10.4 key numbers,
checklist item 3, pitfalls "Under-counting compaction cost" and "Budget checks that assume a hard cap"):

* Cost is dominated by input tokens (~100:1 input:output). Total input on Anthropic is
  ``cache_read_input_tokens + cache_creation_input_tokens + input_tokens``; each is priced
  differently: reads at a multiplier of base input (0.1x on most Claude models, 0.05x / 0.025x on
  the newest), 5-minute writes 1.25x, 1-hour writes 2x. Multipliers differ per model, which is why
  **no price or multiplier is baked in here**: everything comes from the caller's table.
* Compaction adds a sampling iteration that top-level usage excludes: :meth:`PriceTable.cost`
  accepts ``iterations=`` and bills their sum instead.
* Budgets are checked after each response, so spend can overshoot by one response plus running
  subagents; :class:`CostLedger` reports that headroom explicitly.

Table schema (YAML or JSON; see :data:`PRICING_SCHEMA`)::

    version: 1
    currency: USD
    defaults:                      # inherited by every model unless overridden
      cache_read_multiplier: 0.1
      cache_write_multiplier: 1.25     # 5-minute TTL writes
      cache_write_1h_multiplier: 2.0   # 1-hour TTL writes
    models:
      test-model:
        input_per_mtok: 3.0
        output_per_mtok: 15.0
        tiers:                     # optional long-prompt pricing (first tier whose threshold is exceeded wins)
          - above_input_tokens: 200000
            input_per_mtok: 6.0
            output_per_mtok: 22.5
      "test-model-*":              # glob patterns match dated/aliased ids, in file order
        input_per_mtok: 1.0
        output_per_mtok: 5.0
        cache_read_per_mtok: 0.05  # absolute prices override multipliers
"""

from __future__ import annotations

import fnmatch
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from harness_core.types import Usage

from .errors import ConfigError, PricingError

MTOK = 1_000_000

#: JSON Schema of the price-table document (for validating config in CI or other languages).
PRICING_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "harness_ops price table",
    "type": "object",
    "required": ["version", "models"],
    "additionalProperties": False,
    "properties": {
        "version": {"const": 1},
        "currency": {"type": "string", "default": "USD"},
        "defaults": {"$ref": "#/$defs/rates"},
        "models": {"type": "object", "minProperties": 1, "additionalProperties": {"$ref": "#/$defs/model"}},
    },
    "$defs": {
        "rate": {"type": "number", "minimum": 0},
        "rates": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "input_per_mtok": {"$ref": "#/$defs/rate"},
                "output_per_mtok": {"$ref": "#/$defs/rate"},
                "cache_read_per_mtok": {"$ref": "#/$defs/rate"},
                "cache_write_per_mtok": {"$ref": "#/$defs/rate"},
                "cache_write_1h_per_mtok": {"$ref": "#/$defs/rate"},
                "cache_read_multiplier": {"$ref": "#/$defs/rate"},
                "cache_write_multiplier": {"$ref": "#/$defs/rate"},
                "cache_write_1h_multiplier": {"$ref": "#/$defs/rate"},
                "batch_multiplier": {"$ref": "#/$defs/rate"},
            },
        },
        "model": {
            "allOf": [{"$ref": "#/$defs/rates"}],
            "required": ["input_per_mtok", "output_per_mtok"],
            "properties": {
                "tiers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["above_input_tokens", "input_per_mtok", "output_per_mtok"],
                        "properties": {
                            "above_input_tokens": {"type": "integer", "minimum": 0},
                            "input_per_mtok": {"$ref": "#/$defs/rate"},
                            "output_per_mtok": {"$ref": "#/$defs/rate"},
                        },
                    },
                }
            },
        },
    },
}

_RATE_KEYS = (
    "input_per_mtok",
    "output_per_mtok",
    "cache_read_per_mtok",
    "cache_write_per_mtok",
    "cache_write_1h_per_mtok",
    "cache_read_multiplier",
    "cache_write_multiplier",
    "cache_write_1h_multiplier",
    "batch_multiplier",
)


def _rate(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("must be a number", path)
    if value < 0:
        raise ConfigError("must be >= 0", path)
    return float(value)


@dataclass(frozen=True)
class PriceTier:
    above_input_tokens: int
    input_per_mtok: float
    output_per_mtok: float


@dataclass(frozen=True)
class ModelPrice:
    """Resolved per-model rates in currency per million tokens."""

    pattern: str
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float
    cache_write_per_mtok: float
    cache_write_1h_per_mtok: float
    batch_multiplier: float = 1.0
    tiers: tuple[PriceTier, ...] = ()

    def for_prompt(self, total_input_tokens: int) -> "ModelPrice":
        """Apply the long-prompt tier (if any) - cache rates scale with the tier's input ratio."""
        chosen = None
        for t in self.tiers:
            if total_input_tokens > t.above_input_tokens and (chosen is None or t.above_input_tokens > chosen.above_input_tokens):
                chosen = t
        if chosen is None or self.input_per_mtok == 0:
            return self
        scale = chosen.input_per_mtok / self.input_per_mtok
        return ModelPrice(
            self.pattern,
            chosen.input_per_mtok,
            chosen.output_per_mtok,
            self.cache_read_per_mtok * scale,
            self.cache_write_per_mtok * scale,
            self.cache_write_1h_per_mtok * scale,
            self.batch_multiplier,
            (),
        )


@dataclass(frozen=True)
class CostBreakdown:
    model: str
    input_usd: float = 0.0
    output_usd: float = 0.0
    cache_read_usd: float = 0.0
    cache_write_usd: float = 0.0
    priced: bool = True
    currency: str = "USD"

    @property
    def total_usd(self) -> float:
        return round(self.input_usd + self.output_usd + self.cache_read_usd + self.cache_write_usd, 10)

    def __add__(self, other: "CostBreakdown") -> "CostBreakdown":
        return CostBreakdown(
            self.model if self.model == other.model else "*",
            self.input_usd + other.input_usd,
            self.output_usd + other.output_usd,
            self.cache_read_usd + other.cache_read_usd,
            self.cache_write_usd + other.cache_write_usd,
            self.priced and other.priced,
            self.currency,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "input_usd": self.input_usd,
            "output_usd": self.output_usd,
            "cache_read_usd": self.cache_read_usd,
            "cache_write_usd": self.cache_write_usd,
            "total_usd": self.total_usd,
            "priced": self.priced,
        }


class PriceTable:
    """Validated, immutable price table. ``strict=True`` raises on unknown models; otherwise cost is 0 and ``priced=False``."""

    def __init__(self, models: Sequence[ModelPrice], *, currency: str = "USD", strict: bool = True) -> None:
        if not models:
            raise ConfigError("price table needs at least one model", "models")
        self._models = tuple(models)
        self.currency = currency
        self.strict = strict

    # ------------------------------------------------------------------ loading

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any], *, strict: bool = True) -> "PriceTable":
        if not isinstance(doc, Mapping):
            raise ConfigError("price table must be a mapping")
        unknown = set(doc) - {"version", "currency", "defaults", "models"}
        if unknown:
            raise ConfigError(f"unknown keys {sorted(unknown)}")
        if doc.get("version") != 1:
            raise ConfigError("unsupported or missing version (expected 1)", "version")
        currency = doc.get("currency", "USD")
        if not isinstance(currency, str) or not currency:
            raise ConfigError("must be a non-empty string", "currency")
        defaults = doc.get("defaults") or {}
        if not isinstance(defaults, Mapping):
            raise ConfigError("must be a mapping", "defaults")
        for k in defaults:
            if k not in _RATE_KEYS:
                raise ConfigError(f"unknown key {k!r}", "defaults")
        models_doc = doc.get("models")
        if not isinstance(models_doc, Mapping) or not models_doc:
            raise ConfigError("must be a non-empty mapping of model id/pattern -> rates", "models")
        models = [cls._parse_model(str(name), spec, defaults) for name, spec in models_doc.items()]
        return cls(models, currency=currency, strict=strict)

    @staticmethod
    def _parse_model(name: str, spec: Any, defaults: Mapping[str, Any]) -> ModelPrice:
        path = f"models.{name}"
        if not isinstance(spec, Mapping):
            raise ConfigError("must be a mapping", path)
        for k in spec:
            if k not in _RATE_KEYS and k != "tiers":
                raise ConfigError(f"unknown key {k!r}", path)
        merged = {**defaults, **{k: v for k, v in spec.items() if k != "tiers"}}
        for req in ("input_per_mtok", "output_per_mtok"):
            if req not in merged:
                raise ConfigError("is required", f"{path}.{req}")
        r = {k: _rate(v, f"{path}.{k}") for k, v in merged.items()}
        base = r["input_per_mtok"]

        def resolve(absolute: str, multiplier: str) -> float:
            # A model-level value always beats an inherited default; absolute beats multiplier at the same level.
            for source in (spec, defaults):
                if absolute in source:
                    return r[absolute] if source is spec else _rate(source[absolute], f"defaults.{absolute}")
                if multiplier in source:
                    m = r[multiplier] if source is spec else _rate(source[multiplier], f"defaults.{multiplier}")
                    return base * m
            raise ConfigError(f"needs {absolute} or {multiplier} (directly or under defaults)", path)

        tiers_doc = spec.get("tiers") or []
        if not isinstance(tiers_doc, list):
            raise ConfigError("must be a list", f"{path}.tiers")
        tiers = []
        for i, t in enumerate(tiers_doc):
            tp = f"{path}.tiers[{i}]"
            if not isinstance(t, Mapping) or not {"above_input_tokens", "input_per_mtok", "output_per_mtok"} <= set(t):
                raise ConfigError("needs above_input_tokens, input_per_mtok, output_per_mtok", tp)
            above = t["above_input_tokens"]
            if isinstance(above, bool) or not isinstance(above, int) or above < 0:
                raise ConfigError("must be a non-negative integer", f"{tp}.above_input_tokens")
            tiers.append(PriceTier(above, _rate(t["input_per_mtok"], f"{tp}.input_per_mtok"), _rate(t["output_per_mtok"], f"{tp}.output_per_mtok")))
        return ModelPrice(
            pattern=name,
            input_per_mtok=base,
            output_per_mtok=r["output_per_mtok"],
            cache_read_per_mtok=resolve("cache_read_per_mtok", "cache_read_multiplier"),
            cache_write_per_mtok=resolve("cache_write_per_mtok", "cache_write_multiplier"),
            cache_write_1h_per_mtok=resolve("cache_write_1h_per_mtok", "cache_write_1h_multiplier"),
            batch_multiplier=r.get("batch_multiplier", 1.0),
            tiers=tuple(tiers),
        )

    @classmethod
    def from_yaml(cls, text: str, *, strict: bool = True) -> "PriceTable":
        try:
            doc = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML: {exc}") from exc
        return cls.from_dict(doc, strict=strict)

    @classmethod
    def load(cls, path: str | Path, *, strict: bool = True) -> "PriceTable":
        """Load ``.yaml``/``.yml``/``.json`` from disk."""
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() == ".json":
            try:
                return cls.from_dict(json.loads(text), strict=strict)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"invalid JSON: {exc}") from exc
        return cls.from_yaml(text, strict=strict)

    # ------------------------------------------------------------------ lookup and cost

    def lookup(self, model: str) -> ModelPrice | None:
        """Exact id first, then glob patterns in table order."""
        for m in self._models:
            if m.pattern == model:
                return m
        for m in self._models:
            if any(ch in m.pattern for ch in "*?[") and fnmatch.fnmatchcase(model, m.pattern):
                return m
        return None

    @property
    def models(self) -> tuple[ModelPrice, ...]:
        return self._models

    def cost(
        self,
        model: str,
        usage: Usage | None = None,
        *,
        iterations: Iterable[Usage] | None = None,
        cache_write_ttl: str = "5m",
        batch: bool = False,
    ) -> CostBreakdown:
        """Price one model response.

        ``iterations`` (e.g. Anthropic ``usage.iterations`` including compaction) replaces ``usage``
        with the sum of all iterations, so the compaction sampling pass is billed.
        """
        if cache_write_ttl not in ("5m", "1h"):
            raise ValueError("cache_write_ttl must be '5m' or '1h'")
        if iterations is not None:
            total = Usage()
            for it in iterations:
                total = total + it
            usage = total
        usage = usage or Usage()
        price = self.lookup(model)
        if price is None:
            if self.strict:
                raise PricingError(f"no price configured for model {model!r}")
            return CostBreakdown(model, priced=False, currency=self.currency)
        p = price.for_prompt(usage.total_input)
        mult = p.batch_multiplier if batch else 1.0
        write_rate = p.cache_write_1h_per_mtok if cache_write_ttl == "1h" else p.cache_write_per_mtok
        return CostBreakdown(
            model,
            input_usd=usage.input_tokens * p.input_per_mtok / MTOK * mult,
            output_usd=usage.output_tokens * p.output_per_mtok / MTOK * mult,
            cache_read_usd=usage.cache_read_input_tokens * p.cache_read_per_mtok / MTOK * mult,
            cache_write_usd=usage.cache_creation_input_tokens * write_rate / MTOK * mult,
            currency=self.currency,
        )


@dataclass
class BudgetStatus:
    spent_usd: float
    limit_usd: float | None
    exceeded: bool
    remaining_usd: float | None
    overshoot_usd: float = 0.0


@dataclass
class CostLedger:
    """Thread-safe running spend per session (and its subagents), checked after each response.

    ``max_budget_usd`` mirrors the Agent SDK option (default "No limit"). The check is post-response,
    so ``overshoot_usd`` reports how far past the cap the last response (plus subagents) took us.
    """

    max_budget_usd: float | None = None
    _spent: dict[str, float] = field(default_factory=dict)
    _by_model: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, cost: CostBreakdown | float, *, scope: str = "main") -> BudgetStatus:
        amount = cost.total_usd if isinstance(cost, CostBreakdown) else float(cost)
        if amount < 0:
            raise ValueError("cost must be >= 0")
        with self._lock:
            self._spent[scope] = self._spent.get(scope, 0.0) + amount
            if isinstance(cost, CostBreakdown):
                self._by_model[cost.model] = self._by_model.get(cost.model, 0.0) + amount
        return self.status()

    @property
    def total_usd(self) -> float:
        with self._lock:
            return round(sum(self._spent.values()), 10)

    def by_scope(self) -> dict[str, float]:
        with self._lock:
            return dict(self._spent)

    def by_model(self) -> dict[str, float]:
        with self._lock:
            return dict(self._by_model)

    def status(self) -> BudgetStatus:
        spent = self.total_usd
        if self.max_budget_usd is None:
            return BudgetStatus(spent, None, False, None)
        exceeded = spent >= self.max_budget_usd
        return BudgetStatus(
            spent, self.max_budget_usd, exceeded, max(0.0, self.max_budget_usd - spent), max(0.0, spent - self.max_budget_usd)
        )
