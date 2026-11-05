from __future__ import annotations

import math

import pytest

from harness_evals import (
    EfficiencySummary,
    bootstrap_ci,
    cluster_bootstrap_ci,
    mean_se,
    paired_compare,
    pass_at_k,
    pass_hat_k,
    percentile,
    sign_test,
)


def test_pass_at_k_unbiased_estimator():
    assert pass_at_k(5, 0, 1) == 0.0
    assert pass_at_k(5, 5, 3) == 1.0
    assert pass_at_k(10, 3, 1) == pytest.approx(0.3)
    assert pass_at_k(5, 1, 2) == pytest.approx(1 - math.comb(4, 2) / math.comb(5, 2))  # 0.4
    assert pass_at_k(4, 2, 3) == 1.0  # n - c < k
    with pytest.raises(ValueError):
        pass_at_k(3, 1, 5)


def test_pass_hat_k_matches_p_power_k_in_expectation():
    assert pass_hat_k(8, 6, 3) == pytest.approx(math.comb(6, 3) / math.comb(8, 3))
    assert pass_hat_k(5, 5, 5) == 1.0 and pass_hat_k(5, 4, 5) == 0.0
    # Sierra example: p = 0.75 -> pass^3 ~ 0.42 in the large-n limit
    assert pass_hat_k(4000, 3000, 3) == pytest.approx(0.75**3, abs=1e-3)
    assert pass_hat_k(10, 7, 1) == pytest.approx(0.7)


def test_mean_se_and_percentile():
    m, se = mean_se([0.0, 1.0, 1.0, 0.0])
    assert m == 0.5 and se == pytest.approx(math.sqrt(1 / 3) / 2)
    assert mean_se([0.7]) == (0.7, 0.0)
    assert percentile([1, 2, 3, 4], 50) == 2.5 and percentile([5], 95) == 5
    with pytest.raises(ValueError):
        percentile([], 50)


def test_bootstrap_is_seeded_and_brackets_mean():
    vals = [0, 1] * 25
    a = bootstrap_ci(vals, seed=7, n_resamples=2000)
    assert a == bootstrap_ci(vals, seed=7, n_resamples=2000)
    assert a[0] < 0.5 < a[1] and a[1] - a[0] < 0.35
    lo, hi = cluster_bootstrap_ci({"t1": [1, 1, 1], "t2": [0, 0, 0], "t3": [1, 0, 1]}, n_resamples=2000)
    assert 0 <= lo <= 2 / 3 <= hi <= 1


def test_sign_test_exact_values():
    assert sign_test(0, 0) == 1.0
    assert sign_test(10, 0) == pytest.approx(2 / 1024)
    assert sign_test(5, 5) == 1.0
    assert sign_test(8, 2) == pytest.approx(2 * sum(math.comb(10, i) for i in range(3)) / 1024)


def test_paired_compare_detects_real_gain_and_noise():
    base = {f"t{i}": 0.4 for i in range(40)}
    better = {f"t{i}": (0.4 + 0.2 if i % 4 else 0.4) for i in range(40)}
    pc = paired_compare(base, better, n_resamples=2000, seed=1)
    assert pc.significant and pc.verdict == "improved" and pc.wins == 30 and pc.ties == 10 and pc.p_sign < 1e-6
    tiny = {f"t{i}": 0.4 + (0.02 if i % 2 else -0.01) for i in range(40)}
    pn = paired_compare(base, tiny, n_resamples=2000, seed=1)
    assert not pn.significant and pn.verdict == "within_noise"
    worse = {k: v - 0.2 for k, v in base.items()}
    assert paired_compare(base, worse, n_resamples=500).verdict == "regressed"
    with pytest.raises(ValueError):
        paired_compare(base, {"other": 1.0})


def test_efficiency_summary():
    trials = [
        {"passed": True, "cost_usd": 1.0, "wall_clock_s": 10, "turns": 4, "tool_calls": 6, "compaction_events": 1,
         "usage": {"input_tokens": 100, "cache_read_input_tokens": 900, "output_tokens": 50}, "tool_errors_by_tool": {"bash": 1}},
        {"passed": False, "cost_usd": 3.0, "wall_clock_s": 30, "turns": 8, "tool_calls": 10,
         "usage": {"input_tokens": 100, "cache_read_input_tokens": 900}, "tool_errors_by_tool": {"bash": 2, "edit": 1}},
    ]
    e = EfficiencySummary.from_trials(trials)
    assert e.cost_per_success_usd == 4.0 and e.wall_p50_s == 20 and e.cache_read_share == pytest.approx(0.9)
    assert dict(e.tool_errors_by_tool) == {"bash": 3, "edit": 1} and e.compaction_events == 1 and e.mean_turns == 6
