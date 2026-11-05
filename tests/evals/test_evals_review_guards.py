from __future__ import annotations

import pytest
from test_evals_helpers import tool_conversation

from harness_evals import (
    ContaminationError,
    ExposureLedger,
    ReviewLog,
    SplitManager,
    TaxonomyError,
    Trajectory,
    canary_leaked,
    failure_clusters,
    gold_patch_similarity,
    render_transcript_markdown,
    sample_for_review,
    saturation_report,
)


def recs():
    out = []
    for t in ("a", "b", "c"):
        for i in range(5):
            out.append({"trial_id": f"{t}.t{i}", "task_id": t, "passed": t == "a" or i == 0, "status": "completed",
                        "grades": [{"kind": "unit_tests", "status": "pass" if t == "a" else "fail"}], "hack_signals": [], "transcript_path": None})
    out[7]["hack_signals"] = [{"kind": "protected_deleted", "severity": "tamper"}]
    return out


def test_sampling_is_stratified_seeded_and_includes_flagged():
    items = sample_for_review(recs(), n_failures=4, n_passes=2, seed=3)
    assert items == sample_for_review(recs(), n_failures=4, n_passes=2, seed=3)
    assert items[0].reason == "integrity_signal" and items[0].trial_id == "b.t2"
    fails = [i for i in items if i.reason == "failure"]
    assert len(fails) == 4 and {i.task_id for i in fails} == {"b", "c"}
    assert len([i for i in items if i.reason == "pass"]) == 2
    assert len({i.trial_id for i in items}) == len(items)


def test_failure_clusters_group_by_signature():
    cl = failure_clusters(recs())
    assert list(cl)[0] == "completed|unit_tests" and sum(len(v) for v in cl.values()) == 8
    assert any("tamper" in k for k in cl)


def test_review_log_taxonomy_broken_rate_and_agreement(tmp_path):
    log = ReviewLog(tmp_path / "review.jsonl")
    with pytest.raises(TaxonomyError):
        log.tag(trial_id="x", task_id="a", category="made.up", reviewer="r")
    log.tag(trial_id="b.t1", task_id="b", category="grader.false_negative", reviewer="ann")
    log.tag(trial_id="b.t1", task_id="b", category="grader.false_negative", reviewer="bob")
    log.tag(trial_id="c.t1", task_id="c", category="verification.premature_stop", reviewer="ann")
    log.tag(trial_id="a.t0", task_id="a", category="pass.fair", reviewer="ann", passed=True)
    assert log.broken_task_rate() == 0.5
    assert log.tasks_needing_fix() == ["b"]
    assert log.reviewer_agreement() == 1.0 and log.counts()["grader.false_negative"] == 2


def test_render_transcript_redacts_secrets():
    traj = Trajectory.from_messages(tool_conversation("bash", {"command": "export API_KEY=sk-ant-abcdefghijklmnopqrstuv"}, "ok"))
    md = render_transcript_markdown(traj.to_dict())
    assert "sk-ant-abcdefghijklmnopqrstuv" not in md and "REDACTED" in md and "tool_use bash" in md


def test_saturation_and_broken_suspects():
    s = saturation_report({"easy": (5, 5), "dead": (6, 0), "mid": (5, 2), "few": (2, 0)})
    assert s.graduate == ("easy",) and s.broken_suspects == ("dead",) and not s.suite_saturated
    assert saturation_report({"x": (5, 5), "y": (5, 5)}).suite_saturated


def test_split_manager_is_deterministic_and_rotates():
    ids = [f"t{i}" for i in range(100)]
    sm = SplitManager(heldout_fraction=0.3, salt="v1")
    a = sm.split(ids)
    assert a == SplitManager(heldout_fraction=0.3, salt="v1").split(reversed(ids))
    assert 15 < len(a["heldout"]) < 45 and not set(a["train"]) & set(a["heldout"])
    assert sm.rotated("v2").split(ids)["heldout"] != a["heldout"]
    with pytest.raises(ValueError):
        sm.rotated("v1")


def test_exposure_ledger_blocks_contaminated_heldout(tmp_path):
    led = ExposureLedger(tmp_path / "exposure.jsonl")
    led.record(optimizer="gepa", purpose="proposal", task_ids=["t1", "t2"])
    led.record(optimizer="gepa", purpose="acceptance", task_ids=["t9"])
    led.assert_clean_heldout(["t9", "t10"])
    with pytest.raises(ContaminationError):
        led.assert_clean_heldout(["t2", "t9"])
    assert led.acceptance_uses() == 1


def test_contamination_probes():
    gold = "def parse(s):\n    return datetime.fromisoformat(s.replace('Z', '+00:00'))\n"
    assert gold_patch_similarity(gold, gold) == 1.0
    assert gold_patch_similarity("x = completely_different_code(1, 2, 3, 4, 5)", gold) < 0.1
    assert canary_leaked("CANARY-7f3a", "output mentions CANARY-7f3a") and not canary_leaked(None, "x")
