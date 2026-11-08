from __future__ import annotations

import hashlib
import json
import random

import pytest

from harness_evals import CallableAgent, ContaminationError, ExposureLedger, ResultStore, RunConditions, RunConfig, Suite, parse_task
from harness_rl import (
    AblationProposer,
    AuditLog,
    CallableEvaluator,
    CallableProposer,
    ConfigChange,
    ConfigStore,
    ConfigStoreError,
    EvalSuiteEvaluator,
    GateConfig,
    HarnessConfig,
    HeldOutExhaustedError,
    ImprovementLoop,
    SplitResult,
    check_change,
)

TRAIN = [f"tr{i}" for i in range(20)]
HELD = [f"ho{i}" for i in range(20)]
REG = [f"rg{i}" for i in range(10)]
BASE = {"prompts": {"boost": 0.0, "train_only_boost": 0.0}, "profile": {"scaffolds": {"todo": True, "planner": False}},
        "cost": 1.0, "regression_penalty": 0.0, "hack_rate": 0.0, "graders": {"strict": True}}


def synthetic(config: HarnessConfig, ids, *, split, trials, seed):
    """Seeded simulator: pass prob = 0.4 + boosts; common random numbers across configs."""
    per = {}
    for t in ids:
        p = 0.4 + float(config.get("prompts.boost", 0.0))
        if t.startswith("tr"):
            p += float(config.get("prompts.train_only_boost", 0.0))
        if t.startswith("rg"):
            p = 0.9 - float(config.get("regression_penalty", 0.0))
        outs = []
        for k in range(trials):
            u = random.Random(int(hashlib.sha256(f"{seed}:{t}:{k}".encode()).hexdigest()[:8], 16)).random()
            outs.append(1.0 if u < p else 0.0)
        per[t] = outs
    fails = [{"task_id": t, "summary": "failed"} for t, v in per.items() if not all(v)]
    return SplitResult(per, {"cost_per_success_usd": float(config.get("cost", 1.0)), "p95_wall_s": 10.0, "hack_rate": float(config.get("hack_rate", 0.0))}, fails)


def make_loop(tmp_path, changes, *, gate=None, exposure=True, mutable=None, seen=None):
    store = ConfigStore(tmp_path / "cfg")
    store.init(BASE)

    def propose(cfg, evidence):
        if seen is not None:
            seen.extend(f["task_id"] for f in evidence.failures)
        return changes

    return ImprovementLoop(
        store=store, evaluator=CallableEvaluator(synthetic), proposer=CallableProposer(propose),
        splits={"train": TRAIN, "heldout": HELD, "regression": REG}, audit=AuditLog(tmp_path / "audit.jsonl"),
        gate=gate or GateConfig(trials=5, n_resamples=2000), exposure=ExposureLedger(tmp_path / "exposure.jsonl") if exposure else None, mutable=mutable,
    )


def test_real_improvement_is_accepted_and_audited(tmp_path):
    seen: list[str] = []
    loop = make_loop(tmp_path, [ConfigChange("better prompt", {"prompts.boost": 0.35}, kind="prompt")], seen=seen)
    rep = loop.step()
    assert rep.accepted is not None and rep.accepted_version == 2
    assert loop.store.current().get("prompts.boost") == 0.35 and loop.store.current().parent == 1
    assert rep.accepted.heldout.significant and rep.accepted.heldout.delta >= 0.05
    assert seen and not set(seen) & set(HELD)  # proposer only ever saw train failures
    assert loop.audit.verify() and loop.audit.count("accepted") == 1
    assert loop.exposure.exposed() == set(TRAIN) and loop.exposure.acceptance_uses() == 1


def test_overfit_to_train_is_rejected_on_heldout(tmp_path):
    loop = make_loop(tmp_path, [ConfigChange("memorise train", {"prompts.train_only_boost": 0.5})])
    rep = loop.step()
    d = rep.decisions[0]
    assert rep.accepted is None and d.train_delta > 0 and "not significant" in d.reasons[0]
    assert loop.store.current().version == 1


def test_protected_and_code_changes_never_evaluated(tmp_path):
    calls = []

    def ev(cfg, ids, **kw):
        calls.append(cfg.hash)
        return synthetic(cfg, ids, **kw)

    loop = make_loop(tmp_path, [ConfigChange("relax graders", {"graders.strict": False}),
                                ConfigChange("rewrite harness code", {"prompts.boost": 0.9}, kind="code")])
    loop.evaluator = CallableEvaluator(ev)
    rep = loop.step()
    assert [d.status for d in rep.decisions] == ["rejected", "pending_review"]
    assert "protected" in rep.decisions[0].reasons[0]
    assert len(calls) == 1  # only the baseline train evaluation ran
    assert check_change(ConfigChange("x", {"tools.desc": "y"}), mutable=["prompts"]) and not check_change(ConfigChange("x", {"prompts.a": 1}))


@pytest.mark.parametrize("bad, reason", [
    ({"prompts.boost": 0.35, "regression_penalty": 0.4}, "regression suite dropped"),
    ({"prompts.boost": 0.35, "cost": 1.5}, "cost per success over budget"),
    ({"prompts.boost": 0.35, "hack_rate": 0.2}, "hack-signal rate increased"),
])
def test_guard_metrics_block_acceptance(tmp_path, bad, reason):
    rep = make_loop(tmp_path, [ConfigChange("gain with side effects", bad)]).step()
    assert rep.accepted is None and any(reason in r for r in rep.decisions[0].reasons)


def test_best_candidate_wins_with_bonferroni(tmp_path):
    rep = make_loop(tmp_path, [ConfigChange("good", {"prompts.boost": 0.25}), ConfigChange("better", {"prompts.boost": 0.45}),
                               ConfigChange("noop", {"prompts.boost": 0.0})]).step()
    st = {d.change.description: d for d in rep.decisions}
    assert st["better"].status == "accepted" and st["good"].status == "rejected" and "superseded" in st["good"].reasons[0]
    assert "no-op" in st["noop"].reasons[0]
    assert st["better"].heldout.confidence == pytest.approx(1 - 0.05 / 2)


def test_heldout_budget_and_contamination(tmp_path):
    loop = make_loop(tmp_path, [ConfigChange("good", {"prompts.boost": 0.3})], gate=GateConfig(trials=3, max_heldout_uses=1, n_resamples=500))
    loop.step()
    with pytest.raises(HeldOutExhaustedError):
        loop.step()
    led = ExposureLedger(tmp_path / "leak.jsonl")
    led.record(optimizer="other", purpose="proposal", task_ids=["ho3"])
    with pytest.raises(ContaminationError):
        ImprovementLoop(store=loop.store, evaluator=loop.evaluator, proposer=loop.proposer, splits={"train": TRAIN, "heldout": HELD},
                        audit=loop.audit, exposure=led)
    with pytest.raises(ValueError):
        ImprovementLoop(store=loop.store, evaluator=loop.evaluator, proposer=loop.proposer, splits={"train": TRAIN, "heldout": TRAIN[:3]}, audit=loop.audit)


def test_rollback_and_tamper_evident_audit(tmp_path):
    loop = make_loop(tmp_path, [ConfigChange("good", {"prompts.boost": 0.35})])
    loop.step()
    assert loop.store.current().version == 2
    cfg = loop.revert_last(reason="prod regression")
    assert cfg.version == 1 and loop.store.current().get("prompts.boost") == 0.0
    assert loop.audit.entries()[-1]["event"] == "rollback"
    lines = loop.audit.path.read_text().splitlines()
    tampered = json.loads(lines[1])
    assert tampered["event"] == "proposed"
    tampered["change"]["set"] = {"graders.strict": False}
    lines[1] = json.dumps(tampered)
    loop.audit.path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ConfigStoreError):
        loop.audit.verify()


def test_config_store_integrity(tmp_path):
    st = ConfigStore(tmp_path)
    st.init({"a": 1})
    st.commit({"a": 2}, parent=1)
    with pytest.raises(ConfigStoreError):
        st.commit({"a": 3}, parent=1)  # HEAD moved
    p = tmp_path / "versions" / "000002.json"
    d = json.loads(p.read_text())
    d["data"]["a"] = 99
    p.write_text(json.dumps(d))
    with pytest.raises(ConfigStoreError):
        st.get(2)


def test_change_apply_and_ablation_proposer():
    ch = ConfigChange("x", {"profile.scaffolds.todo": False, "new.deep.key": 1}, unset=("cost",))
    out = ch.apply(BASE)
    assert out["profile"]["scaffolds"]["todo"] is False and out["new"]["deep"]["key"] == 1 and "cost" not in out
    assert BASE["profile"]["scaffolds"]["todo"] is True  # original untouched
    props = AblationProposer().propose(HarnessConfig(BASE, 1), None)
    assert [c.set for c in props] == [{"profile.scaffolds.todo": False}]


def test_end_to_end_with_real_eval_runner(tmp_path):
    tasks = [parse_task({"id": f"{p}{i}", "instruction": "write answer.txt", "graders": [{"kind": "file_regex", "path": "answer.txt", "pattern": "^42$"}]})
             for p in ("tr", "ho") for i in range(6)]
    suite = Suite("loop", "1", tuple(tasks))

    def factory(cfg):
        def agent(task, ws, ctx):
            (ws / "answer.txt").write_text("42" if cfg.get("prompts.mode") == "careful" else "41")
        return CallableAgent(agent)

    ev = EvalSuiteEvaluator(suite, factory, store=ResultStore(tmp_path / "results"), conditions=RunConditions("h@1", "test-model", effort="high"),
                            run_config=RunConfig(concurrency=4, workspace_root=tmp_path / "ws"))
    store = ConfigStore(tmp_path / "cfg")
    store.init({"prompts": {"mode": "fast"}})
    loop = ImprovementLoop(store=store, evaluator=ev, proposer=CallableProposer(lambda c, e: [ConfigChange("be careful", {"prompts.mode": "careful"})]),
                           splits={"train": [f"tr{i}" for i in range(6)], "heldout": [f"ho{i}" for i in range(6)]},
                           audit=AuditLog(tmp_path / "audit.jsonl"), gate=GateConfig(trials=3, n_resamples=500))
    rep = loop.step()
    assert rep.accepted is not None and store.current().get("prompts.mode") == "careful"
    assert rep.accepted.heldout.delta == 1.0
    assert any(r.startswith("heldout-") for r in ResultStore(tmp_path / "results").list_runs())
