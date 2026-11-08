"""Gated self-improving harness loop: propose -> screen on train -> accept only on held-out evidence.

Composition from reference §11.4 ("Gated nightly self-improvement loop") and §11.7 steps 9-11,
with the pitfalls of §9.8 / §11.8 enforced in code:

1. Evaluate the baseline config on the **train** split. Only train-split failures are shown to the
   proposer (GEPA/ACE-style reflection on traces); held-out ids are filtered out and every
   exposure is written to an :class:`harness_evals.ExposureLedger`.
2. The proposer returns :class:`ConfigChange` deltas. Changes that touch **protected keys**
   (graders, eval/test/reward config - "editing graders" is reward hacking) or fall outside the
   mutable allow-list are rejected unevaluated. ``kind="code"`` changes (DGM-style
   self-modification) are never auto-accepted: they are parked as ``pending_review``.
3. Each surviving candidate is **screened** on train (must beat the baseline there).
4. Screened candidates and the baseline are run on the **held-out** split with the same seeds
   (common random numbers) and >= ``min_trials`` trials. Acceptance needs a paired-bootstrap CI
   above zero **and** an effect of at least the noise band (default 5 points; single runs swing
   2-6), with Bonferroni correction across candidates in the same iteration.
5. **Guards**: no loss on the regression split beyond ``regression_tolerance`` (default 0), cost
   per success and p95 latency within budget (default +10%, plus 1 s absolute slack on latency so
   sub-second jitter cannot block a change), and no increase in the hack-signal rate.
6. The best eligible candidate is committed as a new config version; every proposal, evaluation,
   decision and rollback is appended to a **hash-chained audit log**; :meth:`ImprovementLoop.rollback`
   restores any earlier version.
7. The held-out split has a **reuse budget** (``max_heldout_uses``); once spent the loop refuses to
   run until the split is rotated (adaptive overfitting to a reused held-out set).
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from harness_evals import (
    AgentUnderTest,
    EfficiencySummary,
    EvalRunner,
    ExposureLedger,
    PairedComparison,
    ResultStore,
    RunConditions,
    RunConfig,
    Suite,
    paired_compare,
)
from harness_evals.metrics import NOISE_BAND
from harness_memory.locking import atomic_write_text, file_lock

from .errors import ConfigStoreError, HeldOutExhaustedError, ProtectedChangeError

#: Config prefixes no optimizer may touch (graders and evaluation are the reward; editing them is hacking).
DEFAULT_PROTECTED = ("graders", "eval", "evals", "tests", "suite", "rewards", "reward", "verifier", "judge")
_KEY = re.compile(r"^[A-Za-z0-9_\-]+(\.[A-Za-z0-9_\-]+)*$")


def _canon(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class HarnessConfig:
    data: Mapping[str, Any]
    version: int
    parent: int | None = None
    note: str = ""

    @property
    def hash(self) -> str:
        return hashlib.sha256(_canon(self.data).encode()).hexdigest()[:16]

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.data
        for part in dotted.split("."):
            if not isinstance(cur, Mapping) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def to_dict(self) -> dict[str, Any]:
        return {"data": copy.deepcopy(dict(self.data)), "version": self.version, "parent": self.parent, "note": self.note, "hash": self.hash}


@dataclass(frozen=True)
class ConfigChange:
    """A delta over the config: ``set`` dotted keys to values, ``unset`` dotted keys."""

    description: str
    set: Mapping[str, Any] = field(default_factory=dict)
    unset: tuple[str, ...] = ()
    kind: str = "config"            # config | prompt | tool_description | playbook | profile | code
    rationale: str = ""
    proposer: str = ""

    def keys(self) -> list[str]:
        return sorted(set(self.set) | set(self.unset))

    def apply(self, data: Mapping[str, Any]) -> dict[str, Any]:
        out = copy.deepcopy(dict(data))
        for key, value in self.set.items():
            parts = key.split(".")
            cur = out
            for p in parts[:-1]:
                nxt = cur.get(p)
                if not isinstance(nxt, dict):
                    nxt = cur[p] = {}
                cur = nxt
            cur[parts[-1]] = copy.deepcopy(value)
        for key in self.unset:
            parts = key.split(".")
            cur = out
            for p in parts[:-1]:
                cur = cur.get(p) if isinstance(cur.get(p), dict) else {}
            cur.pop(parts[-1], None)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"description": self.description, "set": dict(self.set), "unset": list(self.unset), "kind": self.kind,
                "rationale": self.rationale, "proposer": self.proposer}


def check_change(change: ConfigChange, *, protected: Sequence[str] = DEFAULT_PROTECTED, mutable: Sequence[str] | None = None) -> list[str]:
    """Reasons a change may not even be evaluated (empty = allowed)."""
    reasons = []
    if not change.keys():
        reasons.append("empty change")
    for k in change.keys():
        if not _KEY.match(k):
            reasons.append(f"invalid key {k!r}")
            continue
        head = k.split(".")
        if any(k == p or k.startswith(p + ".") or head[0] == p for p in protected):
            reasons.append(f"touches protected key {k!r} (graders/eval/reward config are off-limits to optimizers)")
        if mutable is not None and not any(k == m or k.startswith(m + ".") for m in mutable):
            reasons.append(f"key {k!r} is outside the mutable allow-list {list(mutable)}")
    return reasons


class ConfigStore:
    """Versioned harness configs: ``versions/<n>.json`` (immutable) + ``HEAD`` (atomic pointer)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        (self.root / "versions").mkdir(parents=True, exist_ok=True)
        self._lock = self.root / ".lock"

    def _path(self, v: int) -> Path:
        return self.root / "versions" / f"{int(v):06d}.json"

    def versions(self) -> list[int]:
        return sorted(int(p.stem) for p in (self.root / "versions").glob("*.json"))

    def init(self, data: Mapping[str, Any], *, note: str = "baseline") -> HarnessConfig:
        with file_lock(self._lock):
            if self.versions():
                return self.current()
            cfg = HarnessConfig(copy.deepcopy(dict(data)), 1, None, note)
            atomic_write_text(self._path(1), json.dumps(cfg.to_dict(), indent=2, sort_keys=True))
            atomic_write_text(self.root / "HEAD", "1\n")
            return cfg

    def get(self, v: int) -> HarnessConfig:
        p = self._path(v)
        if not p.is_file():
            raise ConfigStoreError(f"no config version {v}")
        d = json.loads(p.read_text(encoding="utf-8"))
        cfg = HarnessConfig(d["data"], int(d["version"]), d.get("parent"), d.get("note", ""))
        if cfg.hash != d.get("hash"):
            raise ConfigStoreError(f"config version {v} is corrupt (hash mismatch)")
        return cfg

    def current(self) -> HarnessConfig:
        head = self.root / "HEAD"
        if not head.is_file():
            raise ConfigStoreError("config store is not initialised")
        return self.get(int(head.read_text().strip()))

    def commit(self, data: Mapping[str, Any], *, parent: int, note: str = "") -> HarnessConfig:
        with file_lock(self._lock):
            if self.current().version != parent:
                raise ConfigStoreError(f"HEAD moved (expected parent {parent}); re-run the iteration")
            v = (self.versions() or [0])[-1] + 1
            cfg = HarnessConfig(copy.deepcopy(dict(data)), v, parent, note)
            atomic_write_text(self._path(v), json.dumps(cfg.to_dict(), indent=2, sort_keys=True))
            atomic_write_text(self.root / "HEAD", f"{v}\n")
            return cfg

    def set_head(self, v: int) -> HarnessConfig:
        with file_lock(self._lock):
            cfg = self.get(v)
            atomic_write_text(self.root / "HEAD", f"{v}\n")
            return cfg


class AuditLog:
    """Append-only, hash-chained JSONL; :meth:`verify` detects edits, deletions and reordering."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def append(self, event: str, **fields: Any) -> dict[str, Any]:
        with file_lock(self.path.with_suffix(self.path.suffix + ".lock")):
            prev = self.entries()
            entry = {"seq": len(prev), "event": event, "prev": prev[-1]["hash"] if prev else "", **fields}
            entry["hash"] = hashlib.sha256(_canon({k: v for k, v in entry.items() if k != "hash"}).encode()).hexdigest()
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(_canon(entry) + "\n")
            return entry

    def verify(self) -> bool:
        prev = ""
        for i, e in enumerate(self.entries()):
            body = {k: v for k, v in e.items() if k != "hash"}
            if e.get("seq") != i or e.get("prev") != prev or hashlib.sha256(_canon(body).encode()).hexdigest() != e.get("hash"):
                raise ConfigStoreError(f"audit log broken at entry {i}")
            prev = e["hash"]
        return True

    def count(self, event: str) -> int:
        return sum(1 for e in self.entries() if e["event"] == event)


@dataclass(frozen=True)
class SplitResult:
    """Per-task trial outcomes on one split plus guard metrics and (train-only) failure evidence."""

    per_task: Mapping[str, Sequence[float]]
    guards: Mapping[str, float | None] = field(default_factory=dict)   # cost_per_success_usd, p95_wall_s, hack_rate
    failures: Sequence[Mapping[str, Any]] = ()

    @property
    def means(self) -> dict[str, float]:
        return {t: sum(v) / len(v) for t, v in self.per_task.items() if v}

    @property
    def mean(self) -> float:
        m = self.means
        return sum(m.values()) / len(m) if m else 0.0

    @property
    def min_trials(self) -> int:
        return min((len(v) for v in self.per_task.values()), default=0)


class SplitEvaluator(Protocol):
    def evaluate(self, config: HarnessConfig, task_ids: Sequence[str], *, split: str, trials: int, seed: int) -> SplitResult: ...


class CallableEvaluator:
    def __init__(self, fn: Callable[..., SplitResult]) -> None:
        self.fn = fn

    def evaluate(self, config: HarnessConfig, task_ids: Sequence[str], *, split: str, trials: int, seed: int) -> SplitResult:
        return self.fn(config, task_ids, split=split, trials=trials, seed=seed)


class EvalSuiteEvaluator:
    """Adapter over :class:`harness_evals.EvalRunner`: one eval run per (config, split, seed).

    ``agent_factory(config)`` builds the agent under test for a config. The harness version in the
    run conditions is suffixed with the config hash so every run is attributable.
    """

    def __init__(self, suite: Suite, agent_factory: Callable[[HarnessConfig], AgentUnderTest], *, store: ResultStore,
                 conditions: RunConditions, run_config: RunConfig | None = None, judge_client: Any = None) -> None:
        self.suite, self.agent_factory, self.store = suite, agent_factory, store
        self.conditions, self.run_config, self.judge_client = conditions, run_config or RunConfig(), judge_client

    def evaluate(self, config: HarnessConfig, task_ids: Sequence[str], *, split: str, trials: int, seed: int) -> SplitResult:
        cond = replace(self.conditions, harness_version=f"{self.conditions.harness_version}+cfg.{config.hash[:12]}")
        runner = EvalRunner(self.agent_factory(config), store=self.store, conditions=cond,
                            config=replace(self.run_config, trials_per_task=trials, seed=seed), judge_client=self.judge_client)
        res = runner.run(self.suite.select(ids=task_ids), run_id=f"{split}-{config.hash[:12]}-s{seed}-k{trials}")
        per_task: dict[str, list[float]] = {}
        for r in res.records:
            per_task.setdefault(r["task_id"], []).append(1.0 if r["passed"] else 0.0)
        eff = EfficiencySummary.from_trials(res.records)
        hack_rate = sum(1 for r in res.records if r.get("hack_signals")) / max(1, len(res.records))
        failures = [{"task_id": r["task_id"], "trial_id": r["trial_id"], "status": r["status"],
                     "failing": [g["grader"] for g in r.get("grades") or [] if g.get("status") != "pass"], "error": r.get("error"),
                     "transcript_path": r.get("transcript_path")} for r in res.records if not r["passed"]]
        return SplitResult(per_task, {"cost_per_success_usd": eff.cost_per_success_usd, "p95_wall_s": eff.wall_p95_s, "hack_rate": hack_rate}, failures)


@dataclass(frozen=True)
class ProposalEvidence:
    """What the proposer may see: the baseline config and TRAIN-split results only."""

    config: HarnessConfig
    train_mean: float
    train_per_task: Mapping[str, float]
    failures: Sequence[Mapping[str, Any]]


class Proposer(Protocol):
    def propose(self, config: HarnessConfig, evidence: ProposalEvidence) -> Sequence[ConfigChange]: ...


class CallableProposer:
    def __init__(self, fn: Callable[[HarnessConfig, ProposalEvidence], Sequence[ConfigChange]], name: str = "callable") -> None:
        self.fn, self.name = fn, name

    def propose(self, config: HarnessConfig, evidence: ProposalEvidence) -> Sequence[ConfigChange]:
        return list(self.fn(config, evidence))


class AblationProposer:
    """Propose turning each enabled scaffold flag off (per-model ablation at every model upgrade)."""

    def __init__(self, key: str = "profile.scaffolds") -> None:
        self.key = key

    def propose(self, config: HarnessConfig, evidence: ProposalEvidence) -> Sequence[ConfigChange]:
        flags = config.get(self.key) or {}
        return [ConfigChange(f"ablate scaffold {k}", {f"{self.key}.{k}": False}, kind="profile", proposer="ablation",
                             rationale="components go from load-bearing to dead weight within a model generation")
                for k, v in sorted(flags.items()) if v is True]


@dataclass(frozen=True)
class GateConfig:
    trials: int = 3                     # trials per task in every evaluation (>= min_trials)
    min_trials: int = 3                 # checklist: >= 3-5 trials; never single runs
    min_effect: float = NOISE_BAND      # 5 points: the run-to-run noise band
    confidence: float = 0.95
    bonferroni: bool = True
    require_train_gain: bool = True
    regression_tolerance: float = 0.0
    max_cost_increase: float = 0.10
    max_latency_increase: float = 0.10
    latency_abs_slack_s: float = 1.0    # ignore sub-second p95 jitter (relative budgets alone flake on fast tasks)
    cost_abs_slack_usd: float = 0.0
    max_hack_rate_increase: float = 0.0
    max_heldout_uses: int = 20
    max_candidates: int = 4
    n_resamples: int = 5_000

    def __post_init__(self) -> None:
        if self.trials < self.min_trials:
            raise ValueError("GateConfig.trials must be >= min_trials")
        if self.min_trials < 1 or self.max_candidates < 1:
            raise ValueError("min_trials and max_candidates must be >= 1")


@dataclass
class CandidateDecision:
    change: ConfigChange
    status: str                          # accepted | rejected | pending_review
    reasons: list[str] = field(default_factory=list)
    config_hash: str | None = None
    train_delta: float | None = None
    heldout: PairedComparison | None = None
    regression: PairedComparison | None = None
    guards: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"change": self.change.to_dict(), "status": self.status, "reasons": self.reasons, "config_hash": self.config_hash,
                "train_delta": self.train_delta, "heldout": self.heldout.to_dict() if self.heldout else None,
                "regression": self.regression.to_dict() if self.regression else None, "guards": self.guards}


@dataclass
class IterationReport:
    baseline_version: int
    baseline_hash: str
    decisions: list[CandidateDecision]
    accepted_version: int | None = None

    @property
    def accepted(self) -> CandidateDecision | None:
        return next((d for d in self.decisions if d.status == "accepted"), None)


def _ratio_exceeds(cand: float | None, base: float | None, limit: float, abs_slack: float = 0.0) -> bool:
    return cand is not None and base is not None and base > 0 and cand > base * (1 + limit) + abs_slack + 1e-12


class ImprovementLoop:
    def __init__(
        self,
        *,
        store: ConfigStore,
        evaluator: SplitEvaluator,
        proposer: Proposer,
        splits: Mapping[str, Sequence[str]],
        audit: AuditLog,
        gate: GateConfig | None = None,
        exposure: ExposureLedger | None = None,
        protected: Sequence[str] = DEFAULT_PROTECTED,
        mutable: Sequence[str] | None = None,
        seed: int = 0,
        optimizer_name: str = "harness-optimizer",
    ) -> None:
        self.train = sorted(set(splits.get("train") or ()))
        self.heldout = sorted(set(splits.get("heldout") or ()))
        self.regression = sorted(set(splits.get("regression") or ()))
        if not self.train or not self.heldout:
            raise ValueError("splits need non-empty 'train' and 'heldout'")
        if set(self.train) & set(self.heldout):
            raise ValueError(f"train and held-out overlap: {sorted(set(self.train) & set(self.heldout))[:5]}")
        if set(self.regression) & set(self.heldout):
            raise ValueError("regression and held-out splits overlap")
        self.store, self.evaluator, self.proposer, self.audit = store, evaluator, proposer, audit
        self.gate = gate or GateConfig()
        self.exposure, self.protected, self.mutable = exposure, tuple(protected), (tuple(mutable) if mutable is not None else None)
        self.seed, self.optimizer_name = seed, optimizer_name
        if exposure is not None:
            exposure.assert_clean_heldout(self.heldout)

    def heldout_uses(self) -> int:
        return self.exposure.acceptance_uses() if self.exposure is not None else self.audit.count("heldout_evaluated")

    def _eval(self, cfg: HarnessConfig, ids: Sequence[str], split: str, seed: int) -> SplitResult:
        res = self.evaluator.evaluate(cfg, ids, split=split, trials=self.gate.trials, seed=seed)
        if set(res.per_task) != set(ids):
            raise ConfigStoreError(f"evaluator returned tasks {sorted(res.per_task)[:5]} for split {split}; expected exactly the requested ids")
        return res

    def step(self) -> IterationReport:
        g = self.gate
        base = self.store.current()
        if self.heldout_uses() >= g.max_heldout_uses:
            raise HeldOutExhaustedError(f"held-out split used {self.heldout_uses()} times (max {g.max_heldout_uses}); rotate it before more acceptance decisions")
        self.audit.append("iteration_start", baseline_version=base.version, baseline_hash=base.hash)
        it_seed = self.seed + self.audit.count("iteration_start")
        base_train = self._eval(base, self.train, "train", it_seed)
        heldout_set = set(self.heldout)
        failures = [f for f in base_train.failures if f.get("task_id") not in heldout_set]
        if len(failures) != len(base_train.failures):
            self.audit.append("warning", detail="evaluator returned held-out failures; they were withheld from the proposer")
        evidence = ProposalEvidence(base, base_train.mean, base_train.means, failures)
        if self.exposure is not None:
            self.exposure.record(optimizer=self.optimizer_name, purpose="proposal", task_ids=self.train)
        changes = list(self.proposer.propose(base, evidence))
        report = IterationReport(base.version, base.hash, [])
        survivors: list[tuple[CandidateDecision, HarnessConfig]] = []
        seen_hashes = {base.hash}
        for ch in changes[: g.max_candidates]:
            dec = CandidateDecision(ch, "rejected")
            report.decisions.append(dec)
            self.audit.append("proposed", baseline_version=base.version, change=ch.to_dict())
            if ch.kind == "code":
                dec.status, dec.reasons = "pending_review", ["code changes require human review and offline sandboxed evaluation (DGM safeguard)"]
                continue
            dec.reasons = check_change(ch, protected=self.protected, mutable=self.mutable)
            if dec.reasons:
                continue
            data = ch.apply(base.data)
            cand = HarnessConfig(data, -1, base.version)
            dec.config_hash, dec.data = cand.hash, data
            if cand.hash in seen_hashes:
                dec.reasons = ["no-op or duplicate change"]
                continue
            seen_hashes.add(cand.hash)
            ctrain = self._eval(cand, self.train, "train", it_seed)
            if self.exposure is not None:
                self.exposure.record(optimizer=self.optimizer_name, purpose="screen", task_ids=self.train)
            dec.train_delta = ctrain.mean - base_train.mean
            if g.require_train_gain and dec.train_delta <= 0:
                dec.reasons = [f"no gain on train split ({dec.train_delta:+.3f})"]
                continue
            survivors.append((dec, cand))
        for c in changes[g.max_candidates:]:
            report.decisions.append(CandidateDecision(c, "rejected", [f"over max_candidates={g.max_candidates}"]))
        if survivors:
            alpha = 1 - g.confidence
            conf = 1 - (alpha / len(survivors) if g.bonferroni and len(survivors) > 1 else alpha)
            held_seed = it_seed + 10_000
            base_held = self._eval(base, self.heldout, "heldout", held_seed)
            base_reg = self._eval(base, self.regression, "regression", held_seed) if self.regression else None
            self.audit.append("heldout_evaluated", baseline_version=base.version, n_candidates=len(survivors), confidence=conf)
            if self.exposure is not None:
                self.exposure.record(optimizer=self.optimizer_name, purpose="acceptance", task_ids=self.heldout)
            for dec, cand in survivors:
                held = self._eval(cand, self.heldout, "heldout", held_seed)
                reasons: list[str] = []
                if min(held.min_trials, base_held.min_trials) < g.min_trials:
                    reasons.append(f"fewer than {g.min_trials} trials per task")
                dec.heldout = paired_compare(base_held.means, held.means, confidence=conf, min_effect=g.min_effect, seed=held_seed, n_resamples=g.n_resamples)
                if not dec.heldout.significant:
                    reasons.append(f"held-out gain not significant: delta {dec.heldout.delta:+.3f}, CI [{dec.heldout.ci_low:+.3f}, {dec.heldout.ci_high:+.3f}] "
                                   f"at {conf:.3f}, min effect {g.min_effect}")
                if base_reg is not None:
                    reg = self._eval(cand, self.regression, "regression", held_seed)
                    dec.regression = paired_compare(base_reg.means, reg.means, confidence=conf, min_effect=g.min_effect, seed=held_seed, n_resamples=g.n_resamples)
                    if reg.mean < base_reg.mean - g.regression_tolerance - 1e-12:
                        reasons.append(f"regression suite dropped {base_reg.mean:.3f} -> {reg.mean:.3f}")
                bg, cg = base_held.guards, held.guards
                dec.guards = {"baseline": dict(bg), "candidate": dict(cg)}
                if _ratio_exceeds(cg.get("cost_per_success_usd"), bg.get("cost_per_success_usd"), g.max_cost_increase, g.cost_abs_slack_usd):
                    reasons.append("cost per success over budget")
                if _ratio_exceeds(cg.get("p95_wall_s"), bg.get("p95_wall_s"), g.max_latency_increase, g.latency_abs_slack_s):
                    reasons.append("p95 latency over budget")
                if (cg.get("hack_rate") or 0.0) > (bg.get("hack_rate") or 0.0) + g.max_hack_rate_increase + 1e-12:
                    reasons.append("hack-signal rate increased (possible reward hacking)")
                dec.reasons = reasons
                dec.status = "eligible" if not reasons else "rejected"
            eligible = sorted((d for d, _ in survivors if d.status == "eligible"), key=lambda d: (-(d.heldout.delta if d.heldout else 0), d.config_hash or ""))
            for i, d in enumerate(eligible):
                if i == 0:
                    assert d.data is not None
                    new = self.store.commit(d.data, parent=base.version, note=d.change.description)
                    d.status, report.accepted_version = "accepted", new.version
                else:
                    d.status, d.reasons = "rejected", [f"superseded by a better candidate in the same iteration ({eligible[0].change.description})"]
        for d in report.decisions:
            self.audit.append(d.status, baseline_version=base.version, decision=d.to_dict(), accepted_version=report.accepted_version if d.status == "accepted" else None)
        self.audit.append("iteration_end", baseline_version=base.version, accepted_version=report.accepted_version)
        return report

    def rollback(self, version: int, *, reason: str) -> HarnessConfig:
        if not reason:
            raise ValueError("rollback needs a reason for the audit log")
        before = self.store.current()
        cfg = self.store.set_head(version)
        self.audit.append("rollback", from_version=before.version, to_version=version, reason=reason)
        return cfg

    def revert_last(self, *, reason: str) -> HarnessConfig:
        cur = self.store.current()
        if cur.parent is None:
            raise ConfigStoreError("current config has no parent to revert to")
        return self.rollback(cur.parent, reason=reason)


def require_unprotected(change: ConfigChange, **kw: Any) -> None:
    reasons = check_change(change, **kw)
    if reasons:
        raise ProtectedChangeError("; ".join(reasons))


__all__ = [
    "AblationProposer",
    "AuditLog",
    "CallableEvaluator",
    "CallableProposer",
    "CandidateDecision",
    "ConfigChange",
    "ConfigStore",
    "DEFAULT_PROTECTED",
    "EvalSuiteEvaluator",
    "GateConfig",
    "HarnessConfig",
    "ImprovementLoop",
    "IterationReport",
    "ProposalEvidence",
    "Proposer",
    "SplitEvaluator",
    "SplitResult",
    "check_change",
    "require_unprotected",
]
