# Runner seam and returns: `harness_multiagent.runner` + `harness_multiagent.result`

## Purpose

The real agent loop lives in `harness_core`, which another package builds, so this package talks to it through one protocol:

```python
class AgentRunner(Protocol):
    def run(self, task: SubagentTask, ctx: RunContext) -> SubagentResult: ...
```

`SimpleAgentRunner` is a complete, minimal implementation over any `ModelClient`. It is used standalone and in the tests. `result.py` defines what comes back: a typed status, a bounded report, and the untrusted tool rendering.

## Research basis

* "Make returns bounded and typed. Schema-validated JSON with up to 5 retries, then a typed failure (`error_max_structured_output_retries`), or an artifact path plus a summary of 2k tokens or less. Label all subagent output as untrusted, scan it before the parent reads it, and never let it carry approvals" (§8.7 item 6).
* Only the final message returns: "the parent receives a concise summary, not every file the subagent read". "API errors are never delivered as a subagent's result" (Agent SDK, §8.6).
* The pitfall "Result flooding": many detailed reports flood the parent context, so cap returns at ~2k tokens.
* "Within one agent, run read-only tools concurrently and mutating tools sequentially" (item 11).
* Status names reuse the Claude Agent SDK `ResultMessage.subtype` values.

## The runner contract (what the integrator implements)

`SubagentTask` is fully prepared by the manager. A runner must:

1. Resolve the model through `task.model_role` (for example via `ModelRouter`, which maps role to model id from caller config). Never hard-code an id. Send `ModelRequest(system=task.system_prompt, messages=task.initial_messages(), tools=task.tools.specs(), max_tokens=task.max_output_tokens, extra=task.request_extra(model_id))`. The id travels as `extra["model"]` and effort as `extra["effort"]`.
2. After **every** response, call `task.charge(response.usage, model)` and then stop with `task.stop_reason()` if it is not `None` (cancellation, timeout or budget).
3. Execute tool calls **only** via `task.tools.execute(call, ctx)`, never via the raw executor. Return all results of one response in one user message.
4. Enforce `task.max_turns` (`error_max_turns`) and `task.max_tool_calls` (excess calls get an `is_error` result asking the model to finish).
5. Parse the final text with `parse_report`. On `StructuredOutputError`, re-prompt up to `task.max_structured_retries` (5) times, then return `error_max_structured_output_retries`.
6. Return `SubagentResult(task.agent, task.run_id, status, report=..., usage=..., turns=..., tool_calls=..., model=...)`. The transcript stays inside the runner.

If a runner does not charge as it goes, the manager charges `result.usage` at the end so run totals stay correct. In that case mid-run budget enforcement is lost, so charge incrementally.

```python
router = ModelRouter({"lead": cfg.lead_model, "worker": cfg.worker_model}, client)   # or factory=model_id -> client
runner = SimpleAgentRunner(router, read_parallelism=8)
```

## Report schema and statuses

```json
{"summary": "...", "findings": [{"claim": "...", "evidence": "...", "source": "...", "confidence": 0.8}],
 "files": ["..."], "open_questions": ["..."], "confidence": 0.9}
```

`parse_report` accepts a bare object, a fenced ```` ```json ```` block, or one object surrounded by prose. Confidence is in [0, 1], or `low` / `medium` / `high` (0.25 / 0.5 / 0.85).

| `SubagentStatus` | Meaning | Retryable in workflows |
|---|---|---|
| `success` | Valid report | — |
| `error_max_turns` | Turn cap hit (partial) | no |
| `error_max_budget` | Per-agent or run budget hit | no |
| `error_timeout` | Deadline passed (stall) | **yes** |
| `cancelled` | Run cancelled | no |
| `error_during_execution` | Runner/model/API error, refusal, misconfiguration | **yes** |
| `error_max_structured_output_retries` | No valid report after 5 repairs | no |
| `refused` | A cap refused the spawn (`refusal_code`) | no |

## Condensation and untrusted rendering

* `condense(report, max_tokens)` estimates tokens as characters / 4 and trims deterministically. It trims evidence first (to 200 characters, claims to 400), then drops trailing open questions, then drops trailing files down to 5, then drops trailing findings, then shortens the summary, and finally drops the remaining files. It sets `truncated: true` when it removes anything. Earlier findings survive, so agents are told to rank.
* `SubagentResult.to_tool_outcome(max_tokens)` renders `{"type": "subagent_report", "trust": "untrusted", "note": ..., "agent", "run_id", "status", "report", "error", "warnings", "workspace"}`. It reserves 160 tokens for the envelope so the whole payload fits `max_tokens`. `metadata["untrusted"] = True`, and `is_error` is true unless the status is `success`.
* `scan_untrusted(text)` flags approval claims ("the user has approved..."), authority claims ("you are authorized"), instruction overrides ("ignore previous instructions") and safety-bypass requests. The warnings travel with the result. They never grant anything.

## Defaults

| Setting | Value | Source |
|---|---|---|
| Structured-output retries | 5 | Vendor (`MAX_STRUCTURED_OUTPUT_RETRIES`) |
| Return cap | 2,000 tokens | Vendor guidance (~2k) |
| Read-only tool parallelism | 8 | Ours |
| `max_tokens` stop reason | Re-prompt "reply with ONLY the compact report JSON" (bounded by `max_turns`) | Ours |

## Invariants

* The parent never sees intermediate messages or tool outputs: `SubagentResult` has no transcript field.
* Tool results for one response keep the model's call order, even when reads run concurrently.
* A mutating call never starts before the reads that precede it in the same response finish.

## Failure modes

The model raising, refusing (`stop_reason=refusal`) or naming an unconfigured role ends the run with `error_during_execution`, plus the error text in `error`. A cancellation observed between steps maps the token reason to `cancelled`, `error_timeout` or `error_max_budget`.
