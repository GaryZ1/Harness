"""Price table loading/validation and cost computation (made-up prices for test models)."""

from __future__ import annotations

import json

import pytest

from harness_core.types import Usage
from harness_ops import PRICING_SCHEMA, ConfigError, CostLedger, PriceTable, PricingError

YAML = """
version: 1
currency: USD
defaults:
  cache_read_multiplier: 0.1
  cache_write_multiplier: 1.25
  cache_write_1h_multiplier: 2.0
models:
  test-model:
    input_per_mtok: 4.0
    output_per_mtok: 20.0
    cache_read_multiplier: 0.05
    tiers:
      - above_input_tokens: 100000
        input_per_mtok: 8.0
        output_per_mtok: 30.0
  "test-model-*":
    input_per_mtok: 1.0
    output_per_mtok: 5.0
    cache_read_per_mtok: 0.02
    batch_multiplier: 0.5
"""


@pytest.fixture
def table() -> PriceTable:
    return PriceTable.from_yaml(YAML)


def test_cost_uses_per_model_multipliers_and_all_four_token_types(table):
    usage = Usage(input_tokens=1_000, output_tokens=2_000, cache_read_input_tokens=50_000, cache_creation_input_tokens=10_000)
    c = table.cost("test-model", usage)
    assert c.input_usd == pytest.approx(0.004) and c.output_usd == pytest.approx(0.04)
    assert c.cache_read_usd == pytest.approx(50_000 * 4.0 * 0.05 / 1e6)  # model override 0.05x
    assert c.cache_write_usd == pytest.approx(10_000 * 5.0 / 1e6)  # 1.25x default
    assert c.total_usd == pytest.approx(0.004 + 0.04 + 0.01 + 0.05)
    one_hour = table.cost("test-model", usage, cache_write_ttl="1h")
    assert one_hour.cache_write_usd == pytest.approx(10_000 * 8.0 / 1e6)  # 2x


def test_glob_patterns_absolute_overrides_batch_and_tiers(table):
    small = table.cost("test-model-2026-11-01", Usage(1_000_000, 0, 1_000_000, 0))
    assert small.input_usd == pytest.approx(1.0) and small.cache_read_usd == pytest.approx(0.02)
    assert table.cost("test-model-2026-11-01", Usage(1_000_000, 0), batch=True).total_usd == pytest.approx(0.5)
    long = table.cost("test-model", Usage(150_000, 1_000, 0, 0))  # total input > 100k -> tier
    assert long.input_usd == pytest.approx(150_000 * 8.0 / 1e6) and long.output_usd == pytest.approx(1_000 * 30.0 / 1e6)
    assert table.lookup("test-model").pattern == "test-model"  # exact beats the glob


def test_compaction_iterations_are_billed(table):
    top_level = Usage(1_000, 100)
    iterations = [Usage(1_000, 100), Usage(30_000, 2_000)]  # the compaction pass excluded from top-level usage
    assert table.cost("test-model", top_level, iterations=iterations).total_usd > table.cost("test-model", top_level).total_usd
    assert table.cost("test-model", iterations=iterations).input_usd == pytest.approx(31_000 * 4.0 / 1e6)


def test_unknown_model_strict_and_lenient():
    doc = {"version": 1, "models": {"test-model": {"input_per_mtok": 1, "output_per_mtok": 1, "cache_read_multiplier": 0.1, "cache_write_multiplier": 1.25, "cache_write_1h_multiplier": 2}}}
    with pytest.raises(PricingError):
        PriceTable.from_dict(doc).cost("other-model", Usage(1, 1))
    lenient = PriceTable.from_dict(doc, strict=False).cost("other-model", Usage(1, 1))
    assert not lenient.priced and lenient.total_usd == 0


@pytest.mark.parametrize(
    "doc, path",
    [
        ({"version": 2, "models": {}}, "version"),
        ({"version": 1, "models": {}}, "models"),
        ({"version": 1, "models": {"m": {"output_per_mtok": 1}}}, "models.m.input_per_mtok"),
        ({"version": 1, "models": {"m": {"input_per_mtok": -1, "output_per_mtok": 1}}}, "models.m.input_per_mtok"),
        ({"version": 1, "models": {"m": {"input_per_mtok": "3", "output_per_mtok": 1}}}, "models.m.input_per_mtok"),
        ({"version": 1, "models": {"m": {"input_per_mtok": 1, "output_per_mtok": 1, "typo": 1}}}, "models.m"),
        ({"version": 1, "models": {"m": {"input_per_mtok": 1, "output_per_mtok": 1}}}, "models.m"),  # no cache pricing anywhere: no baked-in default
        ({"version": 1, "extra": 1, "models": {"m": {}}}, ""),
    ],
)
def test_validation_errors_name_the_field(doc, path):
    with pytest.raises(ConfigError) as ei:
        PriceTable.from_dict(doc)
    assert ei.value.path == path


def test_load_json_file_and_schema_is_json(tmp_path):
    p = tmp_path / "prices.json"
    p.write_text(json.dumps({"version": 1, "defaults": {"cache_read_multiplier": 0.1, "cache_write_multiplier": 1.25, "cache_write_1h_multiplier": 2}, "models": {"test-model": {"input_per_mtok": 3, "output_per_mtok": 15}}}))
    assert PriceTable.load(p).cost("test-model", Usage(1_000_000, 0)).total_usd == pytest.approx(3.0)
    assert json.loads(json.dumps(PRICING_SCHEMA))["required"] == ["version", "models"]
    with pytest.raises(ConfigError):
        PriceTable.from_yaml("version: [1")


def test_cost_ledger_reports_post_response_overshoot():
    ledger = CostLedger(max_budget_usd=1.0)
    assert not ledger.add(0.6).exceeded
    st = ledger.add(0.7, scope="subagent")  # checked after the response: overshoot is visible, not prevented
    assert st.exceeded and st.overshoot_usd == pytest.approx(0.3) and st.remaining_usd == 0
    assert ledger.by_scope() == {"main": 0.6, "subagent": 0.7}
    assert CostLedger().add(5).limit_usd is None
