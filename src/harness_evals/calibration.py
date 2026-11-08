"""Judge calibration against human labels (reference §9.7 step 6).

Calibrate on a human-labelled sample before relying on an LLM judge, and recalibrate whenever the
judge model or prompt version changes. The reference gap is GDPval: automated grader-expert
agreement 66% vs 71% human-human on open-ended work. :class:`JudgeCalibration` records agreement,
Cohen's kappa and (if two humans labelled) the human-human ceiling, and
:meth:`JudgeCalibration.acceptable` applies the gate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Hashable, Sequence

#: Minimum labelled items before a calibration counts (our choice; GDPval-scale studies use far more).
MIN_CALIBRATION_N = 30
#: Maximum allowed shortfall of judge-human vs human-human agreement (GDPval gap is 0.05).
MAX_AGREEMENT_GAP = 0.05


def agreement(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("label sequences must be non-empty and equally long")
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def cohens_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    po = agreement(a, b)
    n = len(a)
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


@dataclass(frozen=True)
class JudgeCalibration:
    judge_model: str
    prompt_version: str
    n: int
    judge_human_agreement: float
    kappa: float
    human_human_agreement: float | None = None

    @classmethod
    def compute(cls, judge_model: str, prompt_version: str, judge: Sequence[Hashable], human: Sequence[Hashable],
                human_b: Sequence[Hashable] | None = None) -> "JudgeCalibration":
        """Labels are verdicts (e.g. "pass"/"fail"/"unknown"). ``human_b`` is a second annotator."""
        return cls(judge_model, prompt_version, len(judge), agreement(judge, human), cohens_kappa(judge, human),
                   agreement(human, human_b) if human_b is not None else None)

    @property
    def gap(self) -> float | None:
        return None if self.human_human_agreement is None else self.human_human_agreement - self.judge_human_agreement

    def acceptable(self, *, min_n: int = MIN_CALIBRATION_N, max_gap: float = MAX_AGREEMENT_GAP, min_agreement: float = 0.66) -> bool:
        if self.n < min_n:
            return False
        if self.gap is not None:
            return self.gap <= max_gap
        return self.judge_human_agreement >= min_agreement

    def valid_for(self, judge_model: str, prompt_version: str) -> bool:
        """A calibration is void once the judge model or prompt changes."""
        return judge_model == self.judge_model and prompt_version == self.prompt_version

    def to_dict(self) -> dict[str, Any]:
        return {"judge_model": self.judge_model, "prompt_version": self.prompt_version, "n": self.n, "judge_human_agreement": self.judge_human_agreement,
                "kappa": self.kappa, "human_human_agreement": self.human_human_agreement, "gap": self.gap}


__all__ = ["JudgeCalibration", "MAX_AGREEMENT_GAP", "MIN_CALIBRATION_N", "agreement", "cohens_kappa"]
