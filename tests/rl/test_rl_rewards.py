from __future__ import annotations

import pytest

from harness_evals import GradeResult, HackSignal, TaskScore, Trajectory, parse_task, score_task
from harness_rl import (
    CompositeReward,
    EfficiencyPenalty,
    EpisodeOutcome,
    HackPenalty,
    RewardConfigError,
    group_advantages,
    outcome_reward,
    rl_readiness,
    rubric_reward,
    verifiable_reward,
)

TASK = parse_task({"id": "t", "instruction": "x", "graders": [{"kind": "unit_tests", "fail_to_pass": ["tests/test_a.py::test_a"]}],
                   "hidden_files": {"tests/test_a.py": "def test_a(): pass\n"}})


def ep(grades, signals=(), status="completed", traj=None):
    return EpisodeOutcome(TASK, traj or Trajectory(), grades, score_task(grades) if grades else None, list(signals), status)


UT_PASS = GradeResult("unit_tests", "unit_tests", True, 1.0)
UT_HALF = GradeResult("unit_tests", "unit_tests", False, 0.5)
JUDGE = GradeResult("quality", "llm_rubric", True, 0.8)
JUDGE_UNK = GradeResult("quality", "llm_rubric", None, None)


def test_outcome_and_verifiable_rewards():
    assert outcome_reward(ep([UT_PASS])) == 1.0 and outcome_reward(ep([UT_HALF])) == 0.0
    assert outcome_reward(ep([UT_PASS], status="timeout")) == 0.0
    assert verifiable_reward(ep([UT_HALF, JUDGE])) == 0.5  # judges excluded from verifiable rewards
    assert rubric_reward(ep([JUDGE])) == 0.8 and rubric_reward(ep([JUDGE_UNK])) == 0.0


def test_tamper_overrides_reward_instead_of_adding():
    r = CompositeReward()
    clean = r(ep([UT_PASS]))
    hacked = r(ep([UT_PASS], [HackSignal("protected_deleted", "tamper", "tests/test_a.py")]))
    assert clean.total == 1.0 and not clean.hack_detected
    assert hacked.total == -1.0 and hacked.hack_detected and "tamper:protected_deleted" in hacked.flags
    sus = CompositeReward(hack=HackPenalty(suspicious_penalty=-0.25))(ep([UT_PASS], [HackSignal("early_exit", "suspicious", "a.py")]))
    assert sus.total == 0.75 and not sus.hack_detected


def test_judge_share_is_capped_and_composite_clips():
    with pytest.raises(RewardConfigError):
        CompositeReward([("outcome", outcome_reward, 0.5), ("rubric", rubric_reward, 0.5)])
    r = CompositeReward([("outcome", outcome_reward, 1.0), ("rubric", rubric_reward, 0.25), ("eff", EfficiencyPenalty(per_turn=0.05), 1.0)])
    out = r(ep([UT_PASS, JUDGE], traj=Trajectory(turns=2)))
    assert out.total == 1.0 and out.components["eff"] == pytest.approx(-0.1)  # 1 + 0.2 - 0.1 clipped to 1.0


def test_rl_readiness_rejects_hackable_tasks():
    assert rl_readiness(TASK) == []
    judge_only = parse_task({"id": "j", "instruction": "x", "graders": [{"kind": "llm_rubric"}]})
    assert any("deterministic" in p for p in rl_readiness(judge_only))
    visible = parse_task({"id": "v", "instruction": "x", "graders": [{"kind": "unit_tests", "pass_to_pass": ["tests/"]}]})
    assert any("hidden tests" in p for p in rl_readiness(visible))


def test_group_advantages():
    adv, ok = group_advantages([1.0, 0.0, 1.0, 0.0])
    assert ok and adv[0] == pytest.approx(1.0, abs=1e-5) and sum(adv) == pytest.approx(0.0)
    assert group_advantages([1.0, 1.0, 1.0]) == ([0.0, 0.0, 0.0], False)


def test_reward_breakdown_serialises():
    out = CompositeReward()(EpisodeOutcome(TASK, Trajectory(), [UT_PASS], TaskScore(True, True, 1.0, "binary"), []))
    assert out.to_dict() == {"total": 1.0, "components": {"outcome": 1.0}, "hack_detected": False, "flags": []}
