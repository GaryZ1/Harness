"""Transcript review: sampling for humans, failure taxonomy tagging, grader QA.

Weekly review is part of the method, not an extra (reference §9.7 steps 13-14):

* sample **passes as well as failures** ("failures should seem fair"), stratified across tasks so
  one flaky task cannot dominate the sample (:func:`sample_for_review`);
* tag each reviewed trial with a category from a **failure taxonomy**; MAST's specification /
  inter-agent / verification split is the starting point, extended with harness-level classes
  (tools, context, budgets, environment) and, crucially, *task* and *grader* defects;
* track the **broken-task rate** among audited failures (public suites show 30-60%) and list the
  tasks that need fixing before their scores mean anything;
* render transcripts for humans with secrets redacted (:class:`harness_memory.SecretRedactor`).
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from harness_memory import SecretRedactor
from harness_memory.locking import file_lock

from .agent import Trajectory
from .errors import TaxonomyError


@dataclass(frozen=True)
class FailureCategory:
    code: str
    group: str
    description: str
    agent_fault: bool = True     # False: the task, grader or infrastructure is at fault


DEFAULT_TAXONOMY: tuple[FailureCategory, ...] = (
    FailureCategory("spec.misread_task", "specification", "Misunderstood or ignored part of the instruction"),
    FailureCategory("spec.wrong_assumption", "specification", "Acted on an unstated, wrong assumption"),
    FailureCategory("inter_agent.handoff_loss", "inter_agent", "Information lost or distorted between agents"),
    FailureCategory("inter_agent.conflict", "inter_agent", "Sub-agents duplicated or contradicted work"),
    FailureCategory("verification.premature_stop", "verification", "Declared done without verifying the end state"),
    FailureCategory("verification.wrong_check", "verification", "Verified the wrong thing or misread results"),
    FailureCategory("tool.misuse", "tools", "Wrong tool, wrong arguments or wrong format"),
    FailureCategory("tool.error_unrecovered", "tools", "Tool error not recovered from"),
    FailureCategory("tool.profile_mismatch", "tools", "Called a tool the harness profile does not offer"),
    FailureCategory("context.lost_state", "context", "Lost track of state (compaction, long horizon)"),
    FailureCategory("budget.exhausted", "budget", "Ran out of turns, tokens, time or money"),
    FailureCategory("safety.policy_violation", "safety", "Violated a permission or safety policy"),
    FailureCategory("reward_hack", "integrity", "Gamed the grader (test edits, early exit, claim-only)"),
    FailureCategory("env.infra", "environment", "Sandbox or provider failure", agent_fault=False),
    FailureCategory("task.ambiguous", "task_defect", "Two experts would disagree on success", agent_fault=False),
    FailureCategory("task.unsolvable", "task_defect", "Task cannot be solved as specified", agent_fault=False),
    FailureCategory("grader.false_negative", "grader_defect", "Correct work graded as failure", agent_fault=False),
    FailureCategory("grader.false_positive", "grader_defect", "Wrong work graded as success", agent_fault=False),
    FailureCategory("pass.fair", "pass_review", "Pass reviewed and confirmed legitimate", agent_fault=False),
)

DEFECT_GROUPS = ("task_defect", "grader_defect")


@dataclass(frozen=True)
class ReviewItem:
    trial_id: str
    task_id: str
    passed: bool
    status: str
    transcript_path: str | None
    reason: str          # why it was sampled


def sample_for_review(
    records: Sequence[Mapping[str, Any]], *, n_failures: int = 10, n_passes: int = 5, seed: int = 0, include_flagged: bool = True
) -> list[ReviewItem]:
    """Stratified, seeded sample: round-robin over tasks for failures and passes, plus every trial with
    tamper/suspicious signals (they always deserve human eyes)."""
    rng = random.Random(seed)
    items: list[ReviewItem] = []
    seen: set[str] = set()

    def take(pool: list[Mapping[str, Any]], n: int, reason: str) -> None:
        by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for r in pool:
            by_task[r["task_id"]].append(r)
        for lst in by_task.values():
            rng.shuffle(lst)
        order = sorted(by_task)
        rng.shuffle(order)
        taken = 0
        while taken < n and any(by_task.values()):
            for t in order:
                if taken >= n:
                    break
                if by_task[t]:
                    r = by_task[t].pop()
                    if r["trial_id"] in seen:
                        continue
                    seen.add(r["trial_id"])
                    items.append(ReviewItem(r["trial_id"], r["task_id"], bool(r.get("passed")), r["status"], r.get("transcript_path"), reason))
                    taken += 1

    if include_flagged:
        for r in records:
            if r.get("hack_signals") and r["trial_id"] not in seen:
                seen.add(r["trial_id"])
                items.append(ReviewItem(r["trial_id"], r["task_id"], bool(r.get("passed")), r["status"], r.get("transcript_path"), "integrity_signal"))
    take([r for r in records if not r.get("passed")], n_failures, "failure")
    take([r for r in records if r.get("passed")], n_passes, "pass")
    return items


def failure_signature(record: Mapping[str, Any]) -> str:
    """Deterministic cluster key: status + failing grader kinds (+ tamper)."""
    failing = sorted({g["kind"] for g in record.get("grades") or [] if g.get("status") != "pass"})
    tamper = any(s.get("severity") == "tamper" for s in record.get("hack_signals") or [])
    return f"{record['status']}|{','.join(failing) or '-'}{'|tamper' if tamper else ''}"


def failure_clusters(records: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Group failed trials by :func:`failure_signature` (largest cluster first)."""
    out: dict[str, list[str]] = defaultdict(list)
    for r in records:
        if not r.get("passed"):
            out[failure_signature(r)].append(r["trial_id"])
    return dict(sorted(out.items(), key=lambda kv: (-len(kv[1]), kv[0])))


def render_transcript_markdown(transcript: Mapping[str, Any] | Trajectory, *, redactor: SecretRedactor | None = None, max_block_chars: int = 2_000) -> str:
    traj = transcript if isinstance(transcript, Trajectory) else Trajectory.from_dict(transcript)
    red = redactor or SecretRedactor()
    body = red.redact(traj.render_text(max_block_chars=max_block_chars))
    head = f"turns: {traj.turns}  tool calls: {traj.n_tool_calls}  tool errors: {traj.tool_errors_by_tool() or '-'}  stop: {traj.stop_reason or '-'}"
    if traj.error:
        head += f"\nerror: {red.redact(traj.error)}"
    return f"{head}\n\n```text\n{body}\n```\n"


class ReviewLog:
    """Append-only JSONL of review tags, validated against a taxonomy."""

    def __init__(self, path: str | Path, *, taxonomy: Sequence[FailureCategory] = DEFAULT_TAXONOMY) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.taxonomy = {c.code: c for c in taxonomy}

    def tag(self, *, trial_id: str, task_id: str, category: str, reviewer: str, note: str = "", passed: bool | None = None) -> None:
        if category not in self.taxonomy:
            raise TaxonomyError(f"unknown category {category!r}; known: {sorted(self.taxonomy)}")
        if not reviewer:
            raise TaxonomyError("reviewer is required")
        line = json.dumps({"trial_id": trial_id, "task_id": task_id, "category": category, "reviewer": reviewer, "note": note, "passed": passed}, sort_keys=True)
        with file_lock(self.path.with_suffix(self.path.suffix + ".lock")):
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def counts(self) -> dict[str, int]:
        return dict(Counter(e["category"] for e in self.entries()))

    def broken_task_rate(self) -> float | None:
        """Share of audited *failures* whose cause is a task or grader defect (one vote per trial: majority)."""
        votes: dict[str, Counter] = defaultdict(Counter)
        for e in self.entries():
            if e.get("passed") is True or self.taxonomy[e["category"]].group == "pass_review":
                continue
            votes[e["trial_id"]][self.taxonomy[e["category"]].group in DEFECT_GROUPS] += 1
        if not votes:
            return None
        defective = sum(1 for v in votes.values() if v[True] > v[False])
        return defective / len(votes)

    def tasks_needing_fix(self) -> list[str]:
        return sorted({e["task_id"] for e in self.entries() if self.taxonomy[e["category"]].group in DEFECT_GROUPS})

    def reviewer_agreement(self) -> float | None:
        """Fraction of multiply-reviewed trials where all reviewers chose the same category."""
        by_trial: dict[str, set[str]] = defaultdict(set)
        reviewers: dict[str, set[str]] = defaultdict(set)
        for e in self.entries():
            by_trial[e["trial_id"]].add(e["category"])
            reviewers[e["trial_id"]].add(e["reviewer"])
        multi = [t for t, rs in reviewers.items() if len(rs) > 1]
        if not multi:
            return None
        return sum(1 for t in multi if len(by_trial[t]) == 1) / len(multi)


__all__ = [
    "DEFAULT_TAXONOMY",
    "FailureCategory",
    "ReviewItem",
    "ReviewLog",
    "failure_clusters",
    "failure_signature",
    "render_transcript_markdown",
    "sample_for_review",
]
