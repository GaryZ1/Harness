"""Idempotency keys for side-effecting tools, and the retry taxonomy with jittered backoff."""

from __future__ import annotations

import random
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from harness_core.types import ModelRequest, ModelResponse, RunContext, StopReason, ToolCall, ToolOutcome, ToolSpec
from harness_ops import (
    ErrorKind,
    IdempotencyConflictError,
    IdempotencyInProgressError,
    IdempotencyStore,
    IdempotentToolExecutor,
    RetryExhaustedError,
    RetryingModelClient,
    RetryPolicy,
    RetryStats,
    SessionPaused,
    StreamIdleTimeout,
    classify_error,
    idempotency_key,
    retry_call,
)

SRC = str(Path(__file__).resolve().parents[2] / "src")


def test_key_is_stable_per_step_and_distinct_across_steps():
    k = idempotency_key("s1", 3, "tu_1")
    assert k == idempotency_key("s1", 3, "tu_1") and len(k) == 64
    assert len({k, idempotency_key("s1", 4, "tu_1"), idempotency_key("s2", 3, "tu_1"), idempotency_key("s1", 3, "tu_2")}) == 4


def test_execute_once_replays_and_rejects_conflicting_reuse(tmp_path):
    store = IdempotencyStore(tmp_path / "idem.db")
    sent = []
    r1 = store.execute("k", lambda: sent.append(1) or {"pr": 17}, fingerprint="fp-a")
    r2 = store.execute("k", lambda: sent.append(1) or {"pr": 18}, fingerprint="fp-a")
    assert (r1.replayed, r2.replayed, r2.value, len(sent)) == (False, True, {"pr": 17}, 1)
    with pytest.raises(IdempotencyConflictError):
        store.execute("k", lambda: None, fingerprint="fp-b")
    with pytest.raises(RuntimeError):
        store.execute("k2", lambda: (_ for _ in ()).throw(RuntimeError("503")))
    assert store.execute("k2", lambda: "second try").value == "second try"  # exceptions release the claim


def test_concurrent_duplicates_execute_the_side_effect_once(tmp_path):
    store = IdempotencyStore(tmp_path / "idem.db")
    executions, results, in_progress = [], [], []
    barrier = threading.Barrier(10)

    def side_effect():
        executions.append(1)
        time.sleep(0.2)
        return "charged"

    def worker():
        barrier.wait()
        try:
            results.append(store.execute("pay-1", side_effect).value)
        except IdempotencyInProgressError:
            in_progress.append(1)

    ts = [threading.Thread(target=worker) for _ in range(10)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(executions) == 1 and len(results) + len(in_progress) == 10 and set(results) == {"charged"}
    assert store.execute("pay-1", side_effect).replayed and len(executions) == 1


def test_crashed_claim_is_taken_over_after_its_lease(tmp_path, clock):
    store = IdempotencyStore(tmp_path / "idem.db", clock=clock, lease_seconds=60)
    script = textwrap.dedent(
        f"""
        import os, sys
        sys.path.insert(0, {SRC!r})
        from datetime import datetime, timezone
        from harness_ops import IdempotencyStore
        fixed = lambda: datetime(2026, 11, 3, 10, 0, tzinfo=timezone.utc)  # same instant as the test's clock fixture
        IdempotencyStore({str(tmp_path / "idem.db")!r}, clock=fixed, lease_seconds=3600).execute("k", lambda: os._exit(9))
        """
    )
    assert subprocess.run([sys.executable, "-c", script], timeout=60).returncode == 9
    with pytest.raises(IdempotencyInProgressError):
        store.execute("k", lambda: "x")
    clock.advance(hours=2)
    assert store.execute("k", lambda: "recovered").value == "recovered"


class SideEffectTools:
    def __init__(self):
        self.calls = []

    def specs(self):
        return [ToolSpec("read_file", "read", read_only=True), ToolSpec("open_pr", "open a PR")]

    def execute(self, call, ctx):
        self.calls.append((call.name, ctx.metadata.get("idempotency_key")))
        if call.input.get("fail"):
            return ToolOutcome("remote 500", is_error=True)
        return ToolOutcome(f"done:{call.name}", metadata={"n": len(self.calls)})


def test_idempotent_tool_executor_dedupes_side_effects_only(tmp_path):
    inner = SideEffectTools()
    ex = IdempotentToolExecutor(inner, IdempotencyStore(tmp_path / "idem.db"))
    ctx = RunContext("sess-1", metadata={"step_idx": 2})
    pr = ToolCall("tu_1", "open_pr", {"title": "fix"})
    first, retry = ex.execute(pr, ctx), ex.execute(pr, ctx)
    assert first.content == retry.content == "done:open_pr" and retry.metadata["idempotent_replay"] is True
    assert retry.metadata["idempotency_key"] == idempotency_key("sess-1", 2, "tu_1")
    assert [c for c in inner.calls if c[0] == "open_pr"] == [("open_pr", idempotency_key("sess-1", 2, "tu_1"))]  # key passed downstream
    ex.execute(ToolCall("tu_2", "read_file", {"p": "a"}), ctx)
    ex.execute(ToolCall("tu_2", "read_file", {"p": "a"}), ctx)
    assert [c[0] for c in inner.calls].count("read_file") == 2 and inner.calls[-1][1] is None  # read-only: retry-safe bypass
    bad = ToolCall("tu_3", "open_pr", {"fail": True})
    assert ex.execute(bad, ctx).is_error and ex.execute(bad, ctx).is_error
    assert [c[0] for c in inner.calls].count("open_pr") == 3  # error outcomes are not cached by default
    unknown = ToolCall("tu_4", "mcp__x__deploy", {})
    ex.execute(unknown, ctx)
    assert ex.execute(unknown, ctx).metadata["idempotent_replay"]  # unknown tools treated as side-effecting


class HTTPError(Exception):
    def __init__(self, status, retry_after=None):
        super().__init__(f"http {status}")
        self.status_code = status
        self.retry_after = retry_after


class UsageLimitReached(Exception):
    status_code = 429


class ContextWindowExceededError(Exception):
    status_code = 400


@pytest.mark.parametrize(
    "exc, kind, retryable",
    [
        (HTTPError(429), ErrorKind.RATE_LIMITED, True),
        (HTTPError(529), ErrorKind.OVERLOADED, True),
        (HTTPError(503), ErrorKind.SERVER, True),
        (HTTPError(401), ErrorKind.AUTH, False),
        (HTTPError(400), ErrorKind.VALIDATION, False),
        (TimeoutError(), ErrorKind.TIMEOUT, True),
        (StreamIdleTimeout(), ErrorKind.STREAM_IDLE, True),
        (ConnectionResetError(), ErrorKind.CONNECTION, True),
        (UsageLimitReached(), ErrorKind.USAGE_LIMIT, False),
        (ContextWindowExceededError(), ErrorKind.CONTEXT_WINDOW, False),
        (type("APITimeoutError", (Exception,), {})(), ErrorKind.TIMEOUT, True),
        (type("AuthenticationError", (Exception,), {})(), ErrorKind.AUTH, False),
        (KeyError("x"), ErrorKind.UNKNOWN, False),
    ],
)
def test_error_taxonomy(exc, kind, retryable):
    c = classify_error(exc)
    assert (c.kind, c.retryable) == (kind, retryable)


def test_backoff_is_full_jitter_doubling_and_honours_retry_after():
    p = RetryPolicy()
    rng = random.Random(7)
    for n, cap in ((1, 2.0), (2, 4.0), (3, 8.0), (10, 32.0)):
        samples = [p.backoff(n, rng) for _ in range(200)]
        assert 0 <= min(samples) and max(samples) <= cap and max(samples) > cap * 0.8
    slept, seq = [], iter([HTTPError(429, retry_after=3), HTTPError(503), "ok"])

    def fn():
        v = next(seq)
        if isinstance(v, Exception):
            raise v
        return v

    stats = RetryStats()
    assert retry_call(fn, p, sleep=slept.append, rng=random.Random(1), stats=stats) == "ok"
    assert slept[0] == 3 and 0 <= slept[1] <= 4 and stats.retries == 2 and stats.kinds == ["rate_limited", "server_error"]


def test_terminal_errors_are_never_retried_and_exhaustion_wraps():
    calls = []

    def auth():
        calls.append(1)
        raise HTTPError(401)

    with pytest.raises(HTTPError):
        retry_call(auth, RetryPolicy(), sleep=lambda s: None)
    assert len(calls) == 1
    with pytest.raises(RetryExhaustedError) as ei:
        retry_call(lambda: (_ for _ in ()).throw(HTTPError(503)), RetryPolicy(max_retries=4), sleep=lambda s: None)
    assert ei.value.attempts == 5 and isinstance(ei.value.__cause__, HTTPError)
    assert RetryPolicy.for_stream().max_retries == 5 and RetryPolicy().max_retries == 4


def test_usage_limits_and_long_retry_after_pause_the_session(clock):
    with pytest.raises(SessionPaused) as ei:
        retry_call(lambda: (_ for _ in ()).throw(UsageLimitReached()), sleep=lambda s: None, clock=clock)
    assert ei.value.reason == "usage_limit_reached"
    with pytest.raises(SessionPaused) as ei:
        retry_call(lambda: (_ for _ in ()).throw(HTTPError(429, retry_after=3600)), sleep=lambda s: None, clock=clock)
    assert ei.value.resume_at.isoformat() == "2026-11-03T11:00:00+00:00"


def test_retrying_model_client_reports_retry_count():
    class Flaky:
        def __init__(self):
            self.n = 0

        def create(self, request):
            self.n += 1
            if self.n < 3:
                raise HTTPError(529)
            return ModelResponse([], StopReason.END_TURN)

    client = RetryingModelClient(Flaky(), RetryPolicy(), sleep=lambda s: None, rng=random.Random(0))
    client.create(ModelRequest(system="", messages=[]))
    assert client.last_stats.retries == 2 and client.last_stats.attempts == 3
