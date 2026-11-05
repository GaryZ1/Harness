from __future__ import annotations

import json
import threading

import pytest

from harness_evals import ConfoundedComparisonError, ResultStore, ResultStoreError, compare_runs, summarize


def header(run_id="r", model="test-model", harness="h@1", policy="failure", suite_hash="abc", judge=None):
    return {"run_id": run_id, "suite": {"name": "s", "version": "1", "hash": suite_hash, "n_tasks": 2},
            "conditions": {"harness_version": harness, "model": model, "effort": "high", "thinking": None, "judge_model": None},
            "config": {"error_policy": policy}, "judge": judge}


def rec(task, i, passed, tier="capability", status="completed", cost=1.0, wall=10.0, partial=None):
    return {"trial_id": f"{task}.t{i}", "task_id": task, "trial_index": i, "passed": passed, "strict_pass": passed, "status": status,
            "partial": partial if partial is not None else float(passed), "suite_tier": tier, "cost_usd": cost, "wall_clock_s": wall,
            "turns": 3, "tool_calls": 2, "usage": {"input_tokens": 10}, "grades": [], "hack_signals": [], "infra_retries": 0}


def records(rates, tier="capability", n=5):
    out = []
    for t, r in rates.items():
        out += [rec(t, i, i < round(r * n), tier) for i in range(n)]
    return out


def test_summarize_reports_tiers_separately_with_pass_hat_k():
    rs = records({"a": 1.0, "b": 0.6}) + records({"r1": 1.0}, tier="regression", n=3)
    s = summarize(header(), rs, n_resamples=500)
    cap, reg = s.tiers["capability"], s.tiers["regression"]
    assert cap.pass1 == pytest.approx(0.8) and reg.pass1 == 1.0
    assert cap.pass_hat_k[3] == pytest.approx((1.0 + 0.1) / 2) and 8 not in cap.pass_hat_k  # only k <= trials
    assert 5 in cap.pass_at_k and 5 not in reg.pass_hat_k
    assert "a" in cap.saturation.graduate
    md = s.to_markdown()
    for needle in ("test-model", "effort: high", "se_unit: across tasks", "pass^k", "errors_counted_as: failure", "| capability |", "| regression |"):
        assert needle in md
    assert json.loads(s.to_json())["tiers"]["capability"]["pass@1_se_across_tasks"] >= 0


def test_error_policy_exclude_vs_failure_and_warnings():
    rs = records({"a": 1.0}) + [rec("b", i, False, status="infra_error") for i in range(5)]
    s_fail = summarize(header(), rs, n_resamples=200)
    s_excl = summarize(header(policy="exclude"), rs, n_resamples=200)
    assert s_fail.tiers["capability"].pass1 == 0.5 and s_excl.tiers["capability"].pass1 == 1.0 and s_excl.excluded == 5
    assert any("broken task" in w for w in s_fail.warnings)
    one = summarize(header(), records({"a": 1.0}, n=1), n_resamples=100)
    assert any("single-run" in w for w in one.warnings)


def test_compare_runs_refuses_confounded_and_detects_gain():
    base_r = records({f"t{i}": 0.4 for i in range(30)})
    cand_r = records({f"t{i}": (0.8 if i % 3 else 0.4) for i in range(30)})
    with pytest.raises(ConfoundedComparisonError):
        compare_runs(header(), base_r, header(model="other-model"), cand_r)
    with pytest.raises(ConfoundedComparisonError):
        compare_runs(header(), base_r, header(suite_hash="zzz"), cand_r)
    rep = compare_runs(header("base"), base_r, header("cand", harness="h@2"), cand_r, n_resamples=1000)
    pc = rep.tiers["capability"]
    assert pc.verdict == "improved" and rep.differs_in["harness_version"] == ("h@1", "h@2")
    assert "improved" in rep.to_markdown()


def test_store_append_is_thread_safe_and_tolerates_torn_line(tmp_path):
    st = ResultStore(tmp_path)
    st.write_header("r1", header("r1"))

    def w(k):
        for i in range(25):
            st.append("r1", {"task_id": f"t{k}", "trial_index": i, "x": "y" * 100})

    ths = [threading.Thread(target=w, args=(k,)) for k in range(4)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    with open(tmp_path / "runs" / "r1" / "trials.jsonl", "a") as fh:
        fh.write('{"torn": ')
    assert len(st.load_records("r1")) == 100 and st.torn_lines["r1"] == 1
    assert st.list_runs() == ["r1"]


def test_store_rejects_bad_ids_and_conflicting_headers(tmp_path):
    st = ResultStore(tmp_path)
    with pytest.raises(ResultStoreError):
        st.run_dir("../evil")
    st.write_header("r", header())
    st.write_header("r", header())  # identical: fine (resume)
    with pytest.raises(ResultStoreError):
        st.write_header("r", header(model="other"))
    p = st.write_report("r", "report.md", "# hi")
    assert p.read_text() == "# hi"
