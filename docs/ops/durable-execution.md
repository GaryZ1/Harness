# Durable execution: step journal, leases, idempotency keys, retries

## Purpose

A crashed or redeployed worker resumes a session from its last committed step. It does not re-bill model calls or repeat
completed side effects. Only transient errors are retried, and only as often as the policy allows.

## Research basis

* **Durable workflows** (reference §10.3, checklist item 7, build plan "Operations"). Each session is a workflow keyed by `session_id`. Every model call is an activity whose result is persisted ("replay never re-bills"), and every side-effecting tool carries an idempotency key. The references are Temporal (OpenAI Agents SDK integration GA 2026-03-23), Restate, DBOS, and Managed Agents `wake(sessionId)` / "resume from the last event". Anthropic: "minor failures compound", so the system resumes "from where the agent was when the errors occurred".
* **Deterministic replay** (checklist item 8; Claude Code workflows). Completed agents return cached results up to the first agent whose prompt differs, and that agent and every later one re-run. Wall-clock and randomness are injected as recorded values.
* **Idempotency keys**: `hash(session_id, step_idx, tool_call_id)`, with read-only tools marked retry-safe. Do not trust MCP `idempotentHint`. Log the key on the `execute_tool` span.
* **Retry taxonomy** (checklist item 9). Retryable: 429, 5xx, timeouts, stream idle >300 s, connection resets. Terminal: auth, validation, context overflow, usage limit, budget. Backoff is ~0–2 s, 0–4 s, 0–8 s with jitter, honouring `retry_after`. Codex retries 4 times per request and 5 per stream. On usage limits, pause and auto-resume instead of failing.

## Step journal

```python
from harness_ops import StepJournal, RetryPolicy, durable_step

journal = StepJournal("state/journal.db")              # SQLite WAL, shared by all workers
with journal.open_run(session_id) as run:              # leases the run; heartbeat thread renews it
    resp = run.model_call("turn-3/model", client, request, retry=RetryPolicy())   # replayed on resume
    plan = run.step("turn-3/plan", lambda: make_plan(resp), input={"resp_id": resp.id})
    started = run.now("turn-3/started")                # recorded wall clock: deterministic on replay
    out = durable_step("turn-3/tool/tu_1", lambda: write_file(...))   # same thing via the active-run contextvar
    run.complete({"answer": out})
```

* `open_run(run_id, worker_id=None, lease_seconds=30, heartbeat=True, strict_determinism=False, metadata=None, tracer=None)` raises `LeaseHeldError` while another live worker holds the run.
* `run.step(key, fn, input=None, codec=JSON_CODEC, retry=None)` returns the recorded value or executes `fn` and commits it. **First execution and replay return the same value** (`codec.loads(codec.dumps(x))`). A value the codec cannot store raises `StepSerializationError` and is recorded as failed.
* `input` is a JSON-able description of what the step depends on (a prompt, tool args). A changed input sets `run.diverged_at`, and that step and every later step re-run. With `strict_determinism=True` it raises `NonDeterminismError` instead.
* `run.model_call(key, client, request)` uses the request fingerprint (system, messages, tools, max_tokens, extra) as the input and `MODEL_RESPONSE_CODEC`.
* `run.was_replayed(key)` tells telemetry not to re-count a replayed response (`ChatCall.record(replayed=True)`).
* `run.idempotency_key(step_key)` = `sha256(run_id:step_key)`, to pass to downstream APIs.
* `journal.run(run_id, workflow)` opens the run, calls `workflow(run)`, marks it complete and releases the lease. `journal.resumable_runs()` lists `running` runs with no live lease: crashed workers to wake. `journal.steps(run_id)` / `step_record` / `run_info` are for inspection.
* `run.fail(err)` marks a run terminally failed. Leaving the `with` block without `complete()` keeps it resumable, and the lease is released.

## Leases, heartbeats, fencing

`LeaseStore(db, clock=)` provides `acquire` / `try_acquire` / `renew` / `release` / `holder` / `expired(prefix)`. When a
*different* owner acquires a lease, its **fencing token** increments. Every journal write checks `owner + token` inside the
same `BEGIN IMMEDIATE` transaction. A worker that stalled past its lease (GC pause, partition) therefore gets
`LeaseLostError` on its next write and cannot commit stale results. `Heartbeat(renew_fn, interval_s)` renews in a daemon
thread (by default every `lease_seconds / 3`) and sets `lost` if the lease was taken.

## Idempotency

```python
from harness_ops import IdempotencyStore, IdempotentToolExecutor, idempotency_key

store = IdempotencyStore("state/idempotency.db")
executor = IdempotentToolExecutor(inner_executor, store)        # implements harness_core ToolExecutor
outcome = executor.execute(call, RunContext(session_id, metadata={"step_idx": 7}))
outcome.metadata["idempotency_key"], outcome.metadata["idempotent_replay"]
store.execute(key, fn, fingerprint=...)                         # generic: IdempotentResult(value, replayed)
```

* Read-only tools (`ToolSpec.read_only`) bypass the store. Unknown tools are treated as side-effecting.
* The key reaches the tool as `ctx.metadata["idempotency_key"]` so it can be forwarded downstream.
* The same key with a different request fingerprint raises `IdempotencyConflictError`. A concurrent duplicate while the first is still running raises `IdempotencyInProgressError` (retry later). A crashed claim is taken over after its lease (300 s by default).
* Error outcomes are not cached by default (`cache_errors=False`), and raised exceptions never are, so a retry can re-attempt a failed side effect.

## Retries

```python
from harness_ops import RetryPolicy, retry_call, RetryingModelClient, classify_error, SessionPaused
client = RetryingModelClient(raw_client, RetryPolicy())          # client.last_stats.retries -> chat span retry_count
try:
    retry_call(lambda: stream_once(), RetryPolicy.for_stream())
except SessionPaused as p:                                       # usage limit / long retry-after
    schedule_resume(session_id, p.resume_at)
```

`classify_error(exc)` checks, in order: an explicit `kind` attribute, terminal names (`UsageLimit`, `Budget`,
`ContextWindow`/`ContextLength`), the HTTP status (`status_code`/`status`/`response.status_code`), Python
`TimeoutError`/`ConnectionError`/`StreamIdleTimeout`, and finally SDK class-name fragments (`RateLimit`, `Overloaded`,
`Timeout`, `Authentication`, ...). **Unknown errors are not retried.** After the last attempt a retryable error raises
`RetryExhaustedError` (with `__cause__` and `attempts`). Terminal errors are re-raised unwrapped.

## Defaults

| Setting | Default | Anchor |
|---|---|---|
| Request / stream retries | 4 / 5 | vendor default (Codex `DEFAULT_REQUEST_MAX_RETRIES`, `DEFAULT_STREAM_MAX_RETRIES`) |
| Stream idle timeout | 300 s (`DEFAULT_STREAM_IDLE_TIMEOUT_S`) | vendor default (Codex `DEFAULT_STREAM_IDLE_TIMEOUT_MS`) |
| Backoff | full jitter `uniform(0, min(32, 1·2ⁿ))` → 0–2, 0–4, 0–8 s … | vendor semantics (LiteLLM windows); 32 s cap is our choice |
| `retry_after` | honoured; > 60 s becomes `SessionPaused` | vendor semantics (LiteLLM, `/goal`); 60 s threshold is our choice |
| Retryable statuses | 408, 409, 425, 429, 500, 502, 503, 504, 529, any 5xx | research taxonomy + Anthropic 529 overloaded |
| Run lease / heartbeat | 30 s / every 10 s | our choice |
| Idempotency in-progress lease | 300 s | our choice |
| Replay on changed input | re-run that step and all later ones | vendor default (Claude Code workflows) |
| Error outcomes cached | no | our choice |

## Invariants

* A step whose result was committed never runs again for that run. Duplicate keys within one execution raise `DuplicateStepError`.
* Only the current lease holder (matching owner and fencing token) can write steps or finish the run.
* Exactly one caller executes a given idempotency key at a time, across threads and processes.
* A terminal error is never retried. A non-idempotent tool is never retried without a key.

## Failure modes and runbook

* **Crash between `fn` finishing and the commit.** The step runs again on resume (at-least-once). That is why side-effecting steps need idempotency keys: wrap them in `IdempotencyStore` or pass `run.idempotency_key(...)` downstream.
* **`LeaseLostError` in a worker.** Another worker took over (this one stalled longer than the lease). Stop that worker. Its work is redone by the new owner.
* **Run stuck in `running`.** It appears in `resumable_runs()` once its lease lapses. Re-invoke the workflow with `open_run(run_id)` (the `wake` pattern).
* **Prompt edits between deploys.** Expect re-execution from the first changed model call (`run.diverged_at`). Pin sessions to versions (see [deployments.md](deployments.md)) so in-flight sessions do not diverge.
* **Retention.** Call `IdempotencyStore.purge(older_than_seconds)` on a schedule. Journals can be archived per completed run.
