from __future__ import annotations

import json
import random
import threading
import time

import pytest
from test_multiagent_support import ListSink, ScriptedModel, agent_of, always, ctx, defn, make_manager, report, text

from harness_multiagent import (
    DelegationBrief,
    Finding,
    ManagerConfig,
    OrchestratorWorker,
    SubagentReport,
    TaskProfile,
    Verdict,
    Workflow,
    WorkflowConfig,
    WorkflowJournal,
    WorkflowLimitError,
    WorkflowStepFailed,
    agent_verifier,
    best_of_n,
    dedup_findings,
    effort_tier,
    fanout_warnings,
    generate_evaluate,
    iterate_until_converged,
    planner_from_agent,
    rank_findings,
    should_delegate,
    vote,
    vote_on_claims,
)
from harness_multiagent.gate import Recommendation

FAST = WorkflowConfig(prefix_stagger_s=0.0)


def wf(model, *defs, config=FAST, journal=None, sink=None, mconfig=None):
    return Workflow(make_manager(model, *defs, sink=sink, config=mconfig), ctx(), config=config, journal=journal)


def echo_model(delay_jitter: bool = False) -> ScriptedModel:
    rng = random.Random(7)  # seeded: test determinism does not depend on scheduling

    def script(req, n):
        if delay_jitter:
            time.sleep(rng.random() * 0.03)
        obj = req.messages[0].text.split("\n")[1]
        return text(report(f"did {obj}"))

    return ScriptedModel(script)


def test_parallel_results_keep_input_order_and_failures_become_none():
    w = wf(echo_model(delay_jitter=True), defn())

    def boom():
        raise RuntimeError("worker bug")

    thunks = [lambda i=i: w.agent("explorer", DelegationBrief(f"item {i}")).summary for i in range(8)]
    thunks.insert(3, boom)
    out = w.parallel(thunks)
    assert out[3] is None and [o for o in out if o] == [f"did item {i}" for i in range(8)]
    assert any("worker bug" in f.error for f in w.stats.failures)


def test_parallel_respects_workflow_concurrency_cap():
    lock, state = threading.Lock(), {"now": 0, "peak": 0}

    def script(req, n):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.02)
        with lock:
            state["now"] -= 1
        return text(report())

    w = wf(ScriptedModel(script), defn(), config=WorkflowConfig(max_concurrent=4, prefix_stagger_s=0))
    out = w.pipeline(list(range(12)), lambda i: w.agent("explorer", DelegationBrief(f"p {i}")))
    assert all(o is not None for o in out) and state["peak"] <= 4


def test_prefix_stagger_holds_siblings_until_first_finishes():
    starts: list[tuple[str, float]] = []

    def script(req, n):
        starts.append((req.messages[0].text, time.monotonic()))
        time.sleep(0.05)
        return text(report())

    w = wf(ScriptedModel(script), defn(), config=WorkflowConfig(prefix_stagger_s=2.0))
    w.parallel([lambda i=i: w.agent("explorer", DelegationBrief(f"s {i}")) for i in range(3)])
    first = next(t for txt, t in starts if "s 0" in txt)
    assert all(t >= first + 0.04 for txt, t in starts if "s 0" not in txt)


def test_retries_stalls_but_not_budget_and_required_step_ends_run():
    def script(req, n):
        if "flaky" in req.messages[0].text and n < 2:
            raise ConnectionError("transient")
        return text(report("eventually ok"))

    w = wf(ScriptedModel(script), defn())
    assert w.agent("explorer", DelegationBrief("flaky step")).summary == "eventually ok"
    assert len(w.results["explorer"]) == 3

    junk = wf(always("not json"), defn(), config=WorkflowConfig(prefix_stagger_s=0, max_restarts=5))
    assert junk.agent("explorer", DelegationBrief("x")) is None
    assert len(junk.results["explorer"]) == 1  # structured-output failure is not retried
    with pytest.raises(WorkflowStepFailed, match="required step"):
        junk.agent("explorer", DelegationBrief("y"), required=True)


def test_hard_ceilings_items_and_agents():
    w = wf(always(report()), defn(), config=WorkflowConfig(max_items=3, max_agents=2, prefix_stagger_s=0))
    with pytest.raises(WorkflowLimitError, match="items"):
        w.parallel([lambda: 1] * 4)
    w.agent("explorer", DelegationBrief("a"))
    w.agent("explorer", DelegationBrief("b"))
    with pytest.raises(WorkflowLimitError, match="max_agents"):
        w.parallel([lambda: w.agent("explorer", DelegationBrief("c"))])


def test_journal_replays_unchanged_prompts_and_reruns_changed_ones(tmp_path):
    journal_path = tmp_path / "run.jsonl"
    model = echo_model()
    w1 = wf(model, defn(), journal=WorkflowJournal(journal_path))
    w1.pipeline(["a", "b"], lambda x: w1.agent("explorer", DelegationBrief(f"file {x}")))
    assert len(model.requests) == 2
    w2 = wf(model, defn(), journal=WorkflowJournal(journal_path))
    out = w2.pipeline(["a", "b", "c"], lambda x: w2.agent("explorer", DelegationBrief(f"file {x}")))
    assert [o.summary for o in out] == ["did file a", "did file b", "did file c"]
    assert w2.stats.cache_hits == 2 and len(model.requests) == 3
    with open(journal_path, "a") as fh:
        fh.write('{"key": "torn')  # torn trailing line from a crash is ignored
    assert len(WorkflowJournal(journal_path)) == 3


def test_map_reduce_verifies_each_item_and_drops_only_refuted():
    w = wf(echo_model(), defn())
    verdicts = {"a": True, "b": False, "c": None}
    res = w.map_reduce(
        ["a", "b", "c", "d"],
        lambda x: None if x == "d" else w.agent("explorer", DelegationBrief(f"map {x}")),
        lambda kept: [(k.index, k.verdict.value) for k in kept],
        verify_fn=lambda item, rep: verdicts[item],
    )
    assert res.failed == [3]
    assert res.reduced == [(0, "verified"), (2, "unverified")]  # uncheckable kept as unverified
    assert [r.index for r in res.refuted] == [1]


def test_orchestrator_worker_plan_dispatch_verify_synthesize_with_events():
    def script(req, n):
        name = agent_of(req)
        brief = req.messages[0].text
        if name == "planner":
            return text(report("plan", [{"claim": "Audit auth module", "evidence": "src/auth"},
                                        {"claim": "Audit billing module", "evidence": "src/billing"}]))
        if name == "checker":
            return text(report("checked", confidence=0.1 if "billing" in brief else 0.9))
        return text(report(f"worker: {brief.splitlines()[1]}", [f"issue in {brief.splitlines()[1]}"]))

    sink = ListSink()
    w = wf(ScriptedModel(script), defn("planner"), defn("worker"), defn("checker"), sink=sink)
    ow = OrchestratorWorker(
        workflow=w, worker="worker", plan=planner_from_agent(w, "planner"),
        verify=agent_verifier(w, "checker"),
        synthesize=lambda task, kept: [k.value.summary for k in kept],
    )
    res = ow.run("Security audit of the repo")
    assert [b.objective for b in res.plan] == ["Audit auth module", "Audit billing module"]
    assert res.plan[0].out_of_scope == ("Audit billing module",)  # siblings' work is out of scope
    assert res.final == ["worker: Audit auth module"] and [r.index for r in res.refuted] == [1]
    phases = [e.data.get("name") for e in sink.events if e.data["phase"] == "workflow_phase" and e.data["state"] == "start"]
    assert phases == ["plan", "dispatch", "verify", "synthesize"]


# ----------------------------------------------------------------------------- patterns


def test_vote_labels_uncheckable_as_unverified_never_refuted():
    assert vote([True, True, False]).verdict is Verdict.VERIFIED
    assert vote([False, False, None]).verdict is Verdict.REFUTED
    assert vote([None, None]).verdict is Verdict.UNVERIFIED
    assert vote([True, False]).verdict is Verdict.UNVERIFIED  # tie
    assert vote([True], min_decisive=2).verdict is Verdict.UNVERIFIED
    claims = [Finding("2+2=4"), Finding("the moon is cheese")]

    def crashy(f):
        raise RuntimeError

    votes = vote_on_claims(claims, [lambda f: "4" in f.claim, lambda f: "cheese" not in f.claim, crashy])
    assert [v.consensus.verdict for v in votes] == [Verdict.VERIFIED, Verdict.REFUTED]
    assert votes[0].consensus.abstained == 1 and votes[0].consensus.agreement == 1.0


def test_dedup_and_rank_findings_are_deterministic():
    fs = [Finding("SQL injection in login!", "a", "x.py", 0.6), Finding("sql injection in login", "longer evidence", "y.py", 0.9),
          Finding("Weak hash", confidence=None), Finding("XSS", confidence=0.7)]
    d = dedup_findings(fs)
    assert len(d) == 3 and d[0].confidence == 0.9 and d[0].source == "x.py; y.py" and d[0].evidence == "longer evidence"
    assert [f.claim for f in rank_findings(d)] == ["SQL injection in login!", "XSS", "Weak hash"]


def test_best_of_n_verifier_first_judge_breaks_ties():
    cands = ["patch-a", "patch-b", "patch-c"]
    sel = best_of_n(cands, lambda c: c != "patch-a")
    assert sel.index == 1 and sel.method == "first_of_tied"
    sel = best_of_n(cands, lambda c: c != "patch-a", judge=lambda tied: tied.index("patch-c"))
    assert sel.candidate == "patch-c" and sel.method == "judge"
    sel = best_of_n(cands, lambda c: {"patch-a": 0.2, "patch-b": 0.9, "patch-c": 0.5}[c])
    assert sel.candidate == "patch-b" and sel.method == "verifier"
    assert best_of_n(cands, lambda c: False).index is None


def test_iterate_until_two_quiet_rounds_and_generator_evaluator_gate():
    finds = {1: ["a", "b"], 2: ["b", "c"], 3: ["c"], 4: [], 5: ["late"]}
    res = iterate_until_converged(lambda r, so_far: finds.get(r, []), max_rounds=10, patience=2)
    assert res.converged and res.rounds == 4 and res.items == ["a", "b", "c"]
    attempts = []
    out = generate_evaluate(lambda r, fb: attempts.append(fb) or f"v{r}", lambda v: (v == "v2", f"{v} failed tests"), max_rounds=3)
    assert out.passed and out.output == "v2" and attempts == ["", "v1 failed tests"]


# ----------------------------------------------------------------------------- gate


def test_delegation_gate_encodes_when_not_to_go_multi_agent():
    seq = should_delegate(TaskProfile(subtasks_independent=True, sequential_planning=True, produces_code_change=True))
    assert not seq.go_multi_agent and seq.recommendation is Recommendation.SINGLE_AGENT_WITH_REVIEWER
    skills = should_delegate(TaskProfile(subtasks_independent=True, agents_differ_only_in_instructions=True))
    assert skills.recommendation is Recommendation.SINGLE_AGENT_WITH_SKILLS
    strong = should_delegate(TaskProfile(subtasks_independent=True, subtask_count=4, single_agent_baseline=0.7, value_justifies_cost=True))
    assert not strong.go_multi_agent and any("45%" in r for r in strong.reasons)
    go = should_delegate(TaskProfile(subtasks_independent=True, subtask_count=3, single_agent_baseline=0.3, value_justifies_cost=True))
    assert go.go_multi_agent and go.suggested_agents == 3 and json.dumps(go.to_dict())
    assert effort_tier("comparison").max_agents == 4 and effort_tier("simple").tool_calls_per_agent == (3, 10)
    assert len(fanout_warnings(30, 2_000_000)) == 2 and fanout_warnings(5, 10) == []


def test_large_fanout_logs_cost_warning():
    w = wf(always(report()), defn(), config=WorkflowConfig(prefix_stagger_s=0, warn_agents=3), mconfig=ManagerConfig())
    w.parallel([lambda: 1] * 5)
    assert w.stats.warnings and "large fan-out" in w.stats.warnings[0]


def test_report_roundtrip_via_journal_preserves_findings(tmp_path):
    rep = SubagentReport("s", (Finding("c", "e", "src", 0.5),), ("f.py",), ("q?",), 0.7)
    assert SubagentReport.from_dict(json.loads(json.dumps(rep.to_dict()))) == rep
