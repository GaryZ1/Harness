"""Saturation, broken-task, noise and contamination guards.

* **Saturation** (§9.7 step 15): tasks at ~100% graduate to the regression suite; a capability
  suite whose mean is near the ceiling "tracks regressions but provides no signal for
  improvement" (Anthropic) and should be refreshed with harder tasks from fresh failures.
* **Broken-task suspects** (§9.7 step 3): "0% pass@100 is most often a signal of a broken task".
  We flag any task with zero successes over at least ``min_trials`` for audit.
* **Noise** (§9.7 step 7): fewer than 3 trials, or deltas under the ~5-point band, are not evidence.
* **Held-out splits and exposure** (§9.7 step 16, §9.8 "Optimizing against the test set"):
  :class:`SplitManager` makes a deterministic, salted train / held-out split that can be rotated;
  :class:`ExposureLedger` records every time an optimizer (GEPA/ACE-style prompt evolution,
  tool refactoring, DGM-style edits) saw a task, and refuses to use exposed tasks as held-out.
* **Contamination** probes: canary-string leaks and gold-patch reproduction (every frontier model
  OpenAI tested could reproduce some SWE-bench Verified gold patches).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from harness_memory.locking import file_lock

from .errors import ContaminationError
from .metrics import NOISE_BAND

#: Per-task pass rate at or above which a capability task graduates to regression ("~100%").
GRADUATE_AT = 0.999
#: Capability-suite mean at or above which the suite is considered saturated (our choice).
SUITE_SATURATED_AT = 0.95
#: Minimum trials before a 0% task is flagged as a broken-task suspect.
BROKEN_MIN_TRIALS = 5


@dataclass(frozen=True)
class SaturationReport:
    graduate: tuple[str, ...]
    broken_suspects: tuple[str, ...]
    suite_mean: float | None
    suite_saturated: bool

    def to_dict(self) -> dict[str, Any]:
        return {"graduate": list(self.graduate), "broken_suspects": list(self.broken_suspects), "suite_mean": self.suite_mean, "suite_saturated": self.suite_saturated}


def saturation_report(
    per_task: Mapping[str, tuple[int, int]], *, graduate_at: float = GRADUATE_AT, min_trials_graduate: int = 5,
    broken_min_trials: int = BROKEN_MIN_TRIALS, suite_saturated_at: float = SUITE_SATURATED_AT,
) -> SaturationReport:
    """``per_task``: task id -> (n trials, c successes)."""
    grad = tuple(sorted(t for t, (n, c) in per_task.items() if n >= min_trials_graduate and c / n >= graduate_at))
    broken = tuple(sorted(t for t, (n, c) in per_task.items() if n >= broken_min_trials and c == 0))
    rates = [c / n for n, c in per_task.values() if n]
    mean = sum(rates) / len(rates) if rates else None
    return SaturationReport(grad, broken, mean, bool(mean is not None and mean >= suite_saturated_at))


def noise_warnings(*, trials_per_task: int, delta: float | None = None, n_tasks: int | None = None, band: float = NOISE_BAND) -> list[str]:
    out = []
    if trials_per_task < 3:
        out.append(f"only {trials_per_task} trial(s) per task: single-run pass@1 moves 2.2-6.0 points run to run; use >= 3 (PR) or 5 (capability)")
    if delta is not None and abs(delta) < band:
        out.append(f"delta {delta:+.3f} is inside the ~{band:.0%} noise band; add trials or tasks before deciding")
    if n_tasks is not None and n_tasks < 20:
        out.append(f"suite has {n_tasks} tasks; 20-50 tasks from real failures is the recommended starting size")
    return out


class SplitManager:
    """Deterministic salted hash split. Rotating the salt reshuffles membership for a fresh held-out."""

    def __init__(self, *, heldout_fraction: float = 0.3, salt: str = "split-v1") -> None:
        if not 0 < heldout_fraction < 1:
            raise ValueError("heldout_fraction must be in (0, 1)")
        self.heldout_fraction = heldout_fraction
        self.salt = salt

    def split_of(self, task_id: str) -> str:
        h = int(hashlib.sha256(f"{self.salt}\x00{task_id}".encode()).hexdigest()[:12], 16) / float(16**12)
        return "heldout" if h < self.heldout_fraction else "train"

    def split(self, task_ids: Iterable[str]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {"train": [], "heldout": []}
        for t in sorted(set(task_ids)):
            out[self.split_of(t)].append(t)
        if not out["train"] or not out["heldout"]:
            raise ValueError("split produced an empty side; add tasks or change heldout_fraction")
        return out

    def rotated(self, new_salt: str) -> "SplitManager":
        if new_salt == self.salt:
            raise ValueError("rotation needs a new salt")
        return SplitManager(heldout_fraction=self.heldout_fraction, salt=new_salt)


@dataclass
class ExposureLedger:
    """Append-only JSONL of optimizer exposures and held-out uses (cross-process safe).

    ``purpose``: ``proposal`` (the optimizer saw transcripts/scores of these tasks), ``screen``
    (scored on them to filter candidates) or ``acceptance`` (held-out decision). Tasks with any
    ``proposal``/``screen`` exposure are contaminated for held-out use.
    """

    path: Path
    _cache: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, *, optimizer: str, purpose: str, task_ids: Sequence[str], note: str = "") -> None:
        if purpose not in ("proposal", "screen", "acceptance"):
            raise ValueError("purpose must be proposal|screen|acceptance")
        line = json.dumps({"optimizer": optimizer, "purpose": purpose, "task_ids": sorted(set(task_ids)), "note": note}, sort_keys=True)
        with file_lock(self.path.with_suffix(self.path.suffix + ".lock")):
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def exposed(self) -> set[str]:
        return {t for e in self.entries() if e["purpose"] in ("proposal", "screen") for t in e["task_ids"]}

    def acceptance_uses(self) -> int:
        return sum(1 for e in self.entries() if e["purpose"] == "acceptance")

    def assert_clean_heldout(self, heldout: Iterable[str]) -> None:
        leaked = sorted(set(heldout) & self.exposed())
        if leaked:
            raise ContaminationError(f"held-out tasks were exposed to an optimizer: {leaked[:10]}{'...' if len(leaked) > 10 else ''}; rotate the split")


def canary_leaked(canary: str | None, *texts: str) -> bool:
    """True when a task's canary string appears in agent output (training-data or answer leak)."""
    return bool(canary) and any(canary in (t or "") for t in texts)


def _shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    toks = re.findall(r"\w+|[^\w\s]", text)
    return {tuple(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1))}


def gold_patch_similarity(candidate: str, reference: str, *, n: int = 5) -> float:
    """Jaccard similarity of token n-gram shingles (1.0 = verbatim reproduction)."""
    a, b = _shingles(candidate, n), _shingles(reference, n)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def gold_reproduction_flag(candidate: str, reference: str, *, threshold: float = 0.9) -> bool:
    """Flag near-verbatim reproduction of a reference solution for contamination review."""
    return gold_patch_similarity(candidate, reference) >= threshold


__all__ = [
    "BROKEN_MIN_TRIALS",
    "ExposureLedger",
    "GRADUATE_AT",
    "SUITE_SATURATED_AT",
    "SaturationReport",
    "SplitManager",
    "canary_leaked",
    "gold_patch_similarity",
    "gold_reproduction_flag",
    "noise_warnings",
    "saturation_report",
]
