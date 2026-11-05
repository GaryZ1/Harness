"""Retry policy for model calls.

Research basis (reference report section 2.7 step 3; build plan error-class table):

* Codex defaults ``DEFAULT_REQUEST_MAX_RETRIES = 4`` (and 5 stream retries, a 300 s stream idle
  timeout) with exponential backoff; ``ContextWindowExceeded``, ``UsageLimitReached``,
  ``SessionBudgetExceeded`` and ``Fatal`` are returned without retry.
* Honour ``retry_after`` on rate limits ("pause and resume at reset"); use a fallback model only
  after retries are exhausted; never retry terminal errors (authentication, exhausted credit,
  budget, overflow after compaction), which only wastes money.

The loop owns the retry *loop* (so it can emit events and count retries in the ``chat`` span);
this module owns the arithmetic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class RetryPolicy:
    """``max_retries=4`` is Codex's request default; delays and the rate-limit cap are our choices."""

    max_retries: int = 4
    max_rate_limit_retries: int = 6
    initial_delay_s: float = 1.0
    multiplier: float = 2.0
    max_delay_s: float = 60.0
    jitter: float = 0.2              # +/- fraction of the computed delay
    max_retry_after_s: float = 900.0  # longest server-requested wait we honour in-process

    def __post_init__(self) -> None:
        if self.max_retries < 0 or self.max_rate_limit_retries < 0:
            raise ValueError("retry counts must be >= 0")
        if not 0 <= self.jitter < 1:
            raise ValueError("jitter must be in [0, 1)")

    def backoff(self, attempt: int, rng: random.Random | None = None) -> float:
        """Delay before retry number ``attempt`` (1-based)."""
        base = min(self.max_delay_s, self.initial_delay_s * self.multiplier ** max(0, attempt - 1))
        if self.jitter:
            r = (rng or random).uniform(-self.jitter, self.jitter)
            base *= 1 + r
        return max(0.0, base)

    def rate_limit_delay(self, attempt: int, retry_after: float | None, rng: random.Random | None = None) -> float:
        if retry_after is not None and retry_after >= 0:
            return min(float(retry_after), self.max_retry_after_s)
        return self.backoff(attempt, rng)


NO_RETRY = RetryPolicy(max_retries=0, max_rate_limit_retries=0)
