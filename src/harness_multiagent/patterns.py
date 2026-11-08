"""Orchestration patterns and verification / consensus helpers.

Research basis (reference 8.3-8.4, 8.7 items 8-10 and 20, 8.8 "No central verifier",
"Homogeneous debate", "Best-of-N without a good selector"):

* **Orchestrator-worker** (Anthropic research system): lead plans, 3-5 workers run with explicit
  briefs, results are verified, then synthesised once. :class:`OrchestratorWorker` runs
  ``plan -> dispatch -> verify -> synthesize`` on a :class:`~harness_multiagent.orchestration.Workflow`.
* **Always end fan-out in verification, then merge.** Independent agents without a checker amplified
  errors **17.2x**; never concatenate worker outputs into the answer. Claims are voted on per claim;
  uncheckable claims are labelled ``unverified``, not ``refuted`` (:func:`vote`, :func:`vote_on_claims`).
* **Best-of-N is only as good as its selector**: executable verifier first, LLM judge only as a
  tie-breaker (:func:`best_of_n`).
* **Iterate until convergence**: "stop once two rounds in a row find nothing new"
  (:func:`iterate_until_converged`).
* **Generator-evaluator gate** with a separate, fresh-context evaluator and hard thresholds
  (:func:`generate_evaluate`). Self-evaluation is a known failure mode, so the evaluator is a
  different callable (typically a different subagent) from the generator.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Hashable, Iterable, Sequence, TypeVar

from .brief import DelegationBrief
from .orchestration import Verdict, VerifiedItem, Workflow
from .result import Finding, SubagentReport

T = TypeVar("T")
R = TypeVar("R")


# ----------------------------------------------------------------------------- consensus


@dataclass(frozen=True)
class Consensus:
    verdict: Verdict
    verified: int
    refuted: int
    abstained: int

    @property
    def agreement(self) -> float:
        """Share of decisive votes agreeing with the verdict (0.0 when no decisive votes)."""
        decisive = self.verified + self.refuted
        if not decisive or self.verdict is Verdict.UNVERIFIED:
            return 0.0
        return (self.verified if self.verdict is Verdict.VERIFIED else self.refuted) / decisive


def vote(votes: Iterable[Any], *, quorum: float = 0.5, min_decisive: int = 1) -> Consensus:
    """Majority vote over verdicts (bools / :class:`Verdict` / None).

    ``None`` and ``UNVERIFIED`` abstain. A side wins only with strictly more than ``quorum`` of the
    decisive votes and at least ``min_decisive`` decisive votes; ties and thin evidence give
    ``UNVERIFIED`` (never ``REFUTED``).
    """
    if not 0.0 <= quorum < 1.0:
        raise ValueError("quorum must be within [0, 1)")
    v = r = a = 0
    for x in votes:
        verdict = Verdict.coerce(x)
        if verdict is Verdict.VERIFIED:
            v += 1
        elif verdict is Verdict.REFUTED:
            r += 1
        else:
            a += 1
    decisive = v + r
    if decisive < max(1, min_decisive):
        return Consensus(Verdict.UNVERIFIED, v, r, a)
    if v / decisive > quorum and v > r:
        return Consensus(Verdict.VERIFIED, v, r, a)
    if r / decisive > quorum and r > v:
        return Consensus(Verdict.REFUTED, v, r, a)
    return Consensus(Verdict.UNVERIFIED, v, r, a)


@dataclass(frozen=True)
class ClaimVote:
    finding: Finding
    consensus: Consensus


def vote_on_claims(
    findings: Sequence[Finding],
    verifiers: Sequence[Callable[[Finding], Any]],
    *,
    quorum: float = 0.5,
    min_decisive: int = 1,
) -> list[ClaimVote]:
    """Per-claim voting (``/deep-research`` style). A raising verifier abstains. Order preserved."""
    out = []
    for f in findings:
        ballots = []
        for check in verifiers:
            try:
                ballots.append(check(f))
            except Exception:
                ballots.append(None)
        out.append(ClaimVote(f, vote(ballots, quorum=quorum, min_decisive=min_decisive)))
    return out


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def claim_key(claim: str) -> str:
    return _WS.sub(" ", _PUNCT.sub("", claim.lower())).strip()


def dedup_findings(findings: Iterable[Finding]) -> list[Finding]:
    """Merge findings with the same normalised claim: keep first position, highest confidence,
    union of sources/evidence (deterministic)."""
    order: list[str] = []
    merged: dict[str, Finding] = {}
    for f in findings:
        k = claim_key(f.claim)
        if not k:
            continue
        if k not in merged:
            order.append(k)
            merged[k] = f
            continue
        cur = merged[k]
        conf = max((c for c in (cur.confidence, f.confidence) if c is not None), default=None)
        sources = "; ".join(dict.fromkeys(s for s in (cur.source, f.source) if s))
        evidence = cur.evidence if len(cur.evidence) >= len(f.evidence) else f.evidence
        merged[k] = Finding(cur.claim, evidence, sources, conf)
    return [merged[k] for k in order]


def rank_findings(findings: Iterable[Finding]) -> list[Finding]:
    """Highest confidence first; unknown confidence last; stable."""
    return sorted(findings, key=lambda f: -(f.confidence if f.confidence is not None else -1.0))


# ----------------------------------------------------------------------------- best-of-N


@dataclass(frozen=True)
class Selection(Generic[T]):
    index: int | None
    candidate: T | None
    scores: tuple[float, ...]
    method: str  # "verifier" | "judge" | "first_of_tied" | "none_passed"


def best_of_n(
    candidates: Sequence[T],
    verifier: Callable[[T], bool | float],
    *,
    judge: Callable[[Sequence[T]], int] | None = None,
    min_score: float = 0.0,
) -> Selection[T]:
    """Select among N independent rollouts: executable verifier first, judge only to break ties.

    Scores ``<= min_score`` (or ``False``) fail. Without a judge, ties go to the lowest index.
    """
    scores: list[float] = []
    for c in candidates:
        try:
            s = verifier(c)
        except Exception:
            s = False
        scores.append(float(s) if not isinstance(s, bool) else (1.0 if s else 0.0))
    passing = [i for i, s in enumerate(scores) if s > min_score]
    if not passing:
        return Selection(None, None, tuple(scores), "none_passed")
    top = max(scores[i] for i in passing)
    tied = [i for i in passing if scores[i] == top]
    if len(tied) == 1:
        return Selection(tied[0], candidates[tied[0]], tuple(scores), "verifier")
    if judge is not None:
        pick = judge([candidates[i] for i in tied])
        if not 0 <= pick < len(tied):
            raise ValueError(f"judge returned index {pick} outside 0..{len(tied) - 1}")
        return Selection(tied[pick], candidates[tied[pick]], tuple(scores), "judge")
    return Selection(tied[0], candidates[tied[0]], tuple(scores), "first_of_tied")


# ----------------------------------------------------------------------------- loops


@dataclass
class ConvergenceResult(Generic[T]):
    items: list[T]
    rounds: int
    converged: bool


def iterate_until_converged(
    step: Callable[[int, list[T]], Iterable[T]],
    *,
    max_rounds: int = 10,
    patience: int = 2,
    key: Callable[[T], Hashable] = lambda x: x,  # type: ignore[assignment,return-value]
) -> ConvergenceResult[T]:
    """Call ``step(round, items_so_far)`` until ``patience`` consecutive rounds add nothing new."""
    seen: set[Hashable] = set()
    items: list[T] = []
    quiet = 0
    for rnd in range(1, max_rounds + 1):
        new = 0
        for item in step(rnd, list(items)):
            k = key(item)
            if k not in seen:
                seen.add(k)
                items.append(item)
                new += 1
        quiet = quiet + 1 if new == 0 else 0
        if quiet >= patience:
            return ConvergenceResult(items, rnd, True)
    return ConvergenceResult(items, max_rounds, False)


@dataclass
class GenerateEvaluateResult(Generic[T]):
    output: T | None
    passed: bool
    rounds: int
    feedback: list[str] = field(default_factory=list)


def generate_evaluate(
    generate: Callable[[int, str], T | None],
    evaluate: Callable[[T], tuple[bool, str]],
    *,
    max_rounds: int = 3,
) -> GenerateEvaluateResult[T]:
    """Generator-evaluator loop: a separate evaluator gates the generator with feedback."""
    feedback = ""
    history: list[str] = []
    out: T | None = None
    for rnd in range(1, max_rounds + 1):
        out = generate(rnd, feedback)
        if out is None:
            history.append("generator produced nothing")
            continue
        passed, feedback = evaluate(out)
        history.append(feedback)
        if passed:
            return GenerateEvaluateResult(out, True, rnd, history)
    return GenerateEvaluateResult(out, False, max_rounds, history)


# ----------------------------------------------------------------------------- orchestrator-worker


def report_verdict(report: SubagentReport | None, *, accept: float = 0.7, reject: float = 0.3) -> Verdict:
    """Map a verifier subagent's report confidence ("the claims hold") onto a :class:`Verdict`."""
    if report is None or report.confidence is None:
        return Verdict.UNVERIFIED
    if report.confidence >= accept:
        return Verdict.VERIFIED
    if report.confidence <= reject:
        return Verdict.REFUTED
    return Verdict.UNVERIFIED


def agent_verifier(
    workflow: Workflow,
    agent: str,
    *,
    accept: float = 0.7,
    reject: float = 0.3,
) -> Callable[[DelegationBrief, SubagentReport], Verdict]:
    """Verifier backed by a fresh-context (typically read-only) subagent that adversarially checks
    a worker's report. Its brief carries the worker's claims and evidence, nothing else."""

    def check(original: DelegationBrief, report: SubagentReport) -> Verdict:
        claims = [f"{f.claim} | evidence: {f.evidence or '-'} | source: {f.source or '-'}" for f in report.findings]
        brief = DelegationBrief(
            objective="Adversarially verify the claims below; try to refute each one using your tools.",
            why_context=f"A worker reported these results for the objective: {original.objective}",
            inputs=(f"worker summary: {report.summary}", *claims),
            out_of_scope=("Doing the original task again", "Editing anything"),
            write_permission=None,
            output_max_tokens=min(1_000, original.output_max_tokens),
        )
        verdict_report = workflow.agent(agent, brief, label=f"verify:{original.objective[:40]}")
        return report_verdict(verdict_report, accept=accept, reject=reject)

    return check


@dataclass
class OrchestrationResult(Generic[R]):
    plan: list[DelegationBrief]
    reports: list[SubagentReport | None]
    kept: list[VerifiedItem[SubagentReport]]
    refuted: list[VerifiedItem[SubagentReport]]
    failed: list[int]
    final: R | None


@dataclass
class OrchestratorWorker(Generic[R]):
    """``plan -> dispatch -> verify -> synthesize`` with code-controlled fan-out.

    ``plan`` may be plain code or wrap a planner subagent (see :func:`planner_from_agent`);
    ``verify`` defaults to none (items labelled ``unverified``) but should normally be
    :func:`agent_verifier` or an executable check; ``synthesize`` is the single merge step.
    """

    workflow: Workflow
    worker: str
    plan: Callable[[str], Sequence[DelegationBrief]]
    synthesize: Callable[[str, list[VerifiedItem[SubagentReport]]], R]
    verify: Callable[[DelegationBrief, SubagentReport], Any] | None = None
    max_workers: int | None = None

    def run(self, task: str) -> OrchestrationResult[R]:
        with self.workflow.phase("plan"):
            briefs = list(self.plan(task))
        if self.max_workers is not None and len(briefs) > self.max_workers:
            self.workflow.log(f"plan had {len(briefs)} sub-tasks; truncated to max_workers={self.max_workers}")
            briefs = briefs[: self.max_workers]
        self.workflow.warn_if_large(len(briefs) * (2 if self.verify else 1))
        with self.workflow.phase("dispatch"):
            reports = self.workflow.parallel([lambda b=b, i=i: self.workflow.agent(self.worker, b, label=f"worker[{i}]") for i, b in enumerate(briefs)])
        failed = [i for i, r in enumerate(reports) if r is None]
        ok = [(i, r) for i, r in enumerate(reports) if r is not None]
        with self.workflow.phase("verify"):
            if self.verify is not None:
                verify = self.verify
                verdicts = self.workflow.parallel([lambda i=i, r=r: Verdict.coerce(verify(briefs[i], r)) for i, r in ok])
            else:
                verdicts = [Verdict.UNVERIFIED] * len(ok)
        labelled = [VerifiedItem(i, r, v or Verdict.UNVERIFIED) for (i, r), v in zip(ok, verdicts, strict=True)]
        kept = [x for x in labelled if x.verdict is not Verdict.REFUTED]
        refuted = [x for x in labelled if x.verdict is Verdict.REFUTED]
        with self.workflow.phase("synthesize"):
            final = self.synthesize(task, kept) if kept else None
        return OrchestrationResult(plan=briefs, reports=list(reports), kept=kept, refuted=refuted, failed=failed, final=final)


def planner_from_agent(workflow: Workflow, agent: str, *, max_subtasks: int = 5) -> Callable[[str], list[DelegationBrief]]:
    """Use a planner subagent: each of its findings' ``claim`` becomes one worker objective and its
    ``evidence`` the worker's ``why_context``. Siblings' objectives are listed as out of scope."""

    def plan(task: str) -> list[DelegationBrief]:
        report = workflow.agent(agent, DelegationBrief(
            objective=f"Split the task into at most {max_subtasks} independent sub-tasks, one finding per sub-task.",
            why_context="Each finding.claim must be a one-sentence objective; finding.evidence explains what the worker needs.",
            inputs=(task,),
        ), label="plan", required=True)
        assert report is not None
        objectives = [f for f in report.findings if f.claim.strip()][:max_subtasks]
        return [
            DelegationBrief(objective=f.claim.splitlines()[0][:500], why_context=f.evidence, inputs=(task,),
                            out_of_scope=tuple(o.claim for o in objectives if o is not f))
            for f in objectives
        ]

    return plan


__all__ = [
    "Consensus", "vote", "ClaimVote", "vote_on_claims", "claim_key", "dedup_findings", "rank_findings", "Selection", "best_of_n",
    "ConvergenceResult", "iterate_until_converged", "GenerateEvaluateResult", "generate_evaluate", "report_verdict",
    "agent_verifier", "OrchestrationResult", "OrchestratorWorker", "planner_from_agent",
]
