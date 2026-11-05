"""Retry taxonomy with jittered exponential backoff: retry transient errors only, fail fast on terminal ones.

Research basis (reference §10.3 "Retry taxonomy", checklist item 9, pitfall "Retrying terminal errors"):

* **Retryable:** 429 / rate limit, 5xx (incl. 529 overloaded), timeouts, dropped or idle streams,
  connection resets. **Terminal:** auth, validation, context overflow (after compaction failed),
  usage limit reached, session budget exceeded, fatal.
* Codex defaults: ``DEFAULT_REQUEST_MAX_RETRIES = 4``, ``DEFAULT_STREAM_MAX_RETRIES = 5``,
  ``DEFAULT_STREAM_IDLE_TIMEOUT_MS = 300_000``. Backoff ~0-2 s, 0-4 s, 0-8 s ("full jitter"),
  honouring ``retry_after`` (LiteLLM semantics).
* Usage limits and long rate-limit windows **pause** the session (:class:`SessionPaused`) to be
  auto-resumed when the limit resets (Claude Code ``/goal``), rather than failing it.
* Every retry adds latency and counts against rate limits; ``retry_count`` is reported so it can
  be recorded on the ``chat`` span.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Any, Callable, TypeVar

from harness_core.types import Clock, ModelClient, ModelRequest, ModelResponse, utcnow

from .errors import RetryExhaustedError, SessionPaused

T = TypeVar("T")

DEFAULT_REQUEST_MAX_RETRIES = 4  # Codex
DEFAULT_STREAM_MAX_RETRIES = 5  # Codex
DEFAULT_STREAM_IDLE_TIMEOUT_S = 300.0  # Codex DEFAULT_STREAM_IDLE_TIMEOUT_MS = 300_000

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
TERMINAL_STATUS = frozenset({400, 401, 402, 403, 404, 405, 413, 422})


class ErrorKind(str, Enum):
    RATE_LIMITED = "rate_limited"
    OVERLOADED = "overloaded"
    SERVER = "server_error"
    TIMEOUT = "timeout"
    STREAM_IDLE = "stream_idle"
    CONNECTION = "connection"
    AUTH = "auth"
    VALIDATION = "validation"
    CONTEXT_WINDOW = "context_window_exceeded"
    USAGE_LIMIT = "usage_limit_reached"
    BUDGET = "session_budget_exceeded"
    FATAL = "fatal"
    UNKNOWN = "unknown"


RETRYABLE_KINDS = frozenset(
    {ErrorKind.RATE_LIMITED, ErrorKind.OVERLOADED, ErrorKind.SERVER, ErrorKind.TIMEOUT, ErrorKind.STREAM_IDLE, ErrorKind.CONNECTION}
)
PAUSE_KINDS = frozenset({ErrorKind.USAGE_LIMIT})


@dataclass(frozen=True)
class ErrorClassification:
    kind: ErrorKind
    retryable: bool
    retry_after: float | None = None
    status: int | None = None

    @property
    def pause(self) -> bool:
        return self.kind in PAUSE_KINDS


class StreamIdleTimeout(TimeoutError):
    """Raise from a streaming reader when no event arrived for ``DEFAULT_STREAM_IDLE_TIMEOUT_S``."""


# Name fragments of provider SDK exception classes (Anthropic, OpenAI, httpx...), lower-cased.
# These terminal/pause names beat the HTTP status (a 429 "usage limit" pauses; a 400 "context window" is not validation).
_PRIORITY_NAME_RULES: tuple[tuple[str, ErrorKind], ...] = (
    ("usagelimit", ErrorKind.USAGE_LIMIT),
    ("budget", ErrorKind.BUDGET),
    ("contextwindow", ErrorKind.CONTEXT_WINDOW),
    ("contextlength", ErrorKind.CONTEXT_WINDOW),
)
_NAME_RULES: tuple[tuple[str, ErrorKind], ...] = (
    ("ratelimit", ErrorKind.RATE_LIMITED),
    ("overloaded", ErrorKind.OVERLOADED),
    ("internalserver", ErrorKind.SERVER),
    ("serviceunavailable", ErrorKind.SERVER),
    ("timeout", ErrorKind.TIMEOUT),
    ("connection", ErrorKind.CONNECTION),
    ("remoteprotocol", ErrorKind.CONNECTION),
    ("authentication", ErrorKind.AUTH),
    ("permissiondenied", ErrorKind.AUTH),
    ("badrequest", ErrorKind.VALIDATION),
    ("unprocessable", ErrorKind.VALIDATION),
    ("validation", ErrorKind.VALIDATION),
    ("notfound", ErrorKind.VALIDATION),
)


def _status_of(exc: BaseException) -> int | None:
    for attr in ("status_code", "status", "http_status"):
        v = getattr(exc, attr, None)
        if isinstance(v, int) and not isinstance(v, bool):
            return v
    resp = getattr(exc, "response", None)
    v = getattr(resp, "status_code", None)
    return v if isinstance(v, int) else None


def _retry_after_of(exc: BaseException) -> float | None:
    v = getattr(exc, "retry_after", None)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return max(0.0, float(v))
    headers = getattr(getattr(exc, "response", None), "headers", None) or getattr(exc, "headers", None)
    if headers is not None:
        try:
            raw = headers.get("retry-after") or headers.get("Retry-After")
        except Exception:  # noqa: BLE001 - foreign header objects
            raw = None
        if raw is not None:
            try:
                return max(0.0, float(raw))
            except (TypeError, ValueError):
                return None
    return None


def classify_error(exc: BaseException) -> ErrorClassification:
    """Classify by explicit ``kind`` attribute, then HTTP status, then exception type/name.

    Unknown errors are **not** retried: retrying an unclassified failure risks duplicate side
    effects and burned rate limit (research pitfall).
    """
    retry_after = _retry_after_of(exc)
    explicit = getattr(exc, "kind", None)
    if isinstance(explicit, ErrorKind):
        return ErrorClassification(explicit, explicit in RETRYABLE_KINDS, retry_after)
    if isinstance(explicit, str):
        try:
            k = ErrorKind(explicit)
            return ErrorClassification(k, k in RETRYABLE_KINDS, retry_after)
        except ValueError:
            pass
    name = type(exc).__name__.lower()
    for fragment, kind in _PRIORITY_NAME_RULES:
        if fragment in name:
            return ErrorClassification(kind, False, retry_after, _status_of(exc))
    status = _status_of(exc)
    if status is not None:
        if status == 429:
            return ErrorClassification(ErrorKind.RATE_LIMITED, True, retry_after, status)
        if status == 529:
            return ErrorClassification(ErrorKind.OVERLOADED, True, retry_after, status)
        if status in (408,):
            return ErrorClassification(ErrorKind.TIMEOUT, True, retry_after, status)
        if status in RETRYABLE_STATUS or 500 <= status < 600:
            return ErrorClassification(ErrorKind.SERVER, True, retry_after, status)
        if status in (401, 403):
            return ErrorClassification(ErrorKind.AUTH, False, retry_after, status)
        if status in TERMINAL_STATUS:
            return ErrorClassification(ErrorKind.VALIDATION, False, retry_after, status)
    if isinstance(exc, StreamIdleTimeout):
        return ErrorClassification(ErrorKind.STREAM_IDLE, True, retry_after)
    if isinstance(exc, TimeoutError):
        return ErrorClassification(ErrorKind.TIMEOUT, True, retry_after)
    if isinstance(exc, (ConnectionError,)):
        return ErrorClassification(ErrorKind.CONNECTION, True, retry_after)
    for fragment, kind in _NAME_RULES:
        if fragment in name:
            return ErrorClassification(kind, kind in RETRYABLE_KINDS, retry_after, status)
    return ErrorClassification(ErrorKind.UNKNOWN, False, retry_after, status)


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retries with full-jitter exponential backoff.

    Delay before retry ``n`` (1-based) is ``uniform(0, min(max_delay, base_delay * 2**n))`` -
    with the default ``base_delay=1.0`` that is 0-2 s, 0-4 s, 0-8 s, ... - or ``retry_after`` when
    the server sent one. A ``retry_after`` longer than ``max_retry_after`` turns into a pause.
    """

    max_retries: int = DEFAULT_REQUEST_MAX_RETRIES
    base_delay: float = 1.0
    max_delay: float = 32.0
    max_retry_after: float = 60.0
    classifier: Callable[[BaseException], ErrorClassification] = field(default=classify_error, compare=False)

    def __post_init__(self) -> None:
        if self.max_retries < 0 or self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("retry policy values must be >= 0")

    @classmethod
    def for_stream(cls, **kw: Any) -> "RetryPolicy":
        return cls(max_retries=kw.pop("max_retries", DEFAULT_STREAM_MAX_RETRIES), **kw)

    def backoff(self, retry_number: int, rng: random.Random | None = None) -> float:
        cap = min(self.max_delay, self.base_delay * (2**retry_number))
        return (rng or random).uniform(0.0, cap)


@dataclass
class RetryStats:
    attempts: int = 0
    retries: int = 0
    kinds: list[str] = field(default_factory=list)
    slept: float = 0.0


def retry_call(
    fn: Callable[[], T],
    policy: RetryPolicy | None = None,
    *,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    clock: Clock = utcnow,
    on_retry: Callable[[int, ErrorClassification, float], None] | None = None,
    stats: RetryStats | None = None,
) -> T:
    """Call ``fn`` under ``policy``.

    * terminal error -> re-raised immediately, unwrapped;
    * pause-class error (usage limit, or ``retry_after`` beyond ``max_retry_after``) -> :class:`SessionPaused`;
    * retryable error on the last attempt -> :class:`RetryExhaustedError` (``__cause__`` = last error).
    """
    policy = policy or RetryPolicy()
    stats = stats if stats is not None else RetryStats()
    for attempt in range(policy.max_retries + 1):
        stats.attempts = attempt + 1
        try:
            return fn()
        except Exception as exc:
            c = policy.classifier(exc)
            if c.pause or (c.retryable and c.retry_after is not None and c.retry_after > policy.max_retry_after):
                resume = clock() + timedelta(seconds=c.retry_after) if c.retry_after is not None else None
                raise SessionPaused(c.kind.value, resume) from exc
            if not c.retryable:
                raise
            if attempt >= policy.max_retries:
                raise RetryExhaustedError(exc, attempt + 1) from exc
            delay = c.retry_after if c.retry_after is not None else policy.backoff(attempt + 1, rng)
            stats.retries += 1
            stats.kinds.append(c.kind.value)
            stats.slept += delay
            if on_retry is not None:
                on_retry(attempt + 1, c, delay)
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


class RetryingModelClient:
    """``ModelClient`` wrapper applying :func:`retry_call`; ``last_stats`` is per thread.

    Retries hit the same client (deployment) first; model fallback is a gateway concern and should
    happen only after retries are exhausted and at turn boundaries (each model has its own cache).
    """

    def __init__(
        self,
        inner: ModelClient,
        policy: RetryPolicy | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        on_retry: Callable[[int, ErrorClassification, float], None] | None = None,
    ) -> None:
        self.inner = inner
        self.policy = policy or RetryPolicy()
        self._sleep = sleep
        self._rng = rng
        self._on_retry = on_retry
        self._local = threading.local()

    @property
    def last_stats(self) -> RetryStats | None:
        return getattr(self._local, "stats", None)

    def create(self, request: ModelRequest) -> ModelResponse:
        stats = RetryStats()
        self._local.stats = stats
        return retry_call(lambda: self.inner.create(request), self.policy, sleep=self._sleep, rng=self._rng, on_retry=self._on_retry, stats=stats)
