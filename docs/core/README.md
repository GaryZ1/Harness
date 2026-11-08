# Core: agent architecture and loop (`harness_core`)

Production Python implementation of the architecture and loop design in
[`reports/SOTA agentic harness reference.md` §1 and §2](../../reports/SOTA%20agentic%20harness%20reference.md),
[`reports/SOTA agentic harness build plan.md`](../../reports/SOTA%20agentic%20harness%20build%20plan.md) ("Architecture" and "The loop"), and
[`research_notes/.../agent_architecture_and_loops.md`](../../research_notes/SOTA%20agentic%20harness%20build%20plan/agent_architecture_and_loops.md).
`harness_core.types` is the shared contract every `harness_*` package imports; everything else in this
package is the **session log**, the **harness** (the loop) and the interfaces to the **sandbox**.

| Component | What it does | Module | Doc |
|---|---|---|---|
| **Session log** | Append-only JSONL events with `seq`, fsync, torn-tail repair; replay to messages; compaction as a projection; fork/checkpoint; per-session lease | `session.py` | [session-log.md](session-log.md) |
| **Agent loop** | Canonical stop predicate, every `stop_reason`, one tool-result message per turn, steering queue, interrupts, crash resume | `loop.py`, `results.py` | [agent-loop.md](agent-loop.md) |
| **Tools and permissions** | `ToolExecutor` implementations (function, composite, lazy sandbox), read-only parallelism, permission policies, approvals and durable approval pauses | `tools.py`, `permissions.py` | [tools-and-permissions.md](tools-and-permissions.md) |
| **Hooks and stop evaluators** | Claude Code-compatible hooks (`session_start` … `session_end`, command hooks), evaluators that veto a stop (goal, completion promise, callable) | `hooks.py`, `evaluators.py` | [hooks-and-evaluators.md](hooks-and-evaluators.md) |
| **Budgets and recovery** | Turns, tokens, wall clock, USD; wrap-up turn; error and denial limits; doom-loop detection; retry/backoff/fallback; compaction hook and thrash guard | `budgets.py`, `retry.py`, `stuck.py`, `context.py` | [budgets-and-recovery.md](budgets-and-recovery.md) |
| **Model adapters and test doubles** | Optional Anthropic `ModelClient` (lazy import); `ScriptedModelClient` and friends for offline tests | `anthropic_adapter.py`, `testing.py` | [model-adapters.md](model-adapters.md) |
| **Long-horizon drivers** | Initializer + coding sessions with fresh context; Ralph loop with required max-iterations and verifier; prompt templates | `long_horizon.py`, `prompts.py` | [long-horizon.md](long-horizon.md) |

## Why this shape

Each point is a research finding the code enforces, not a prompt suggestion.

1. **Session, harness and sandbox are separate and each is replaceable.** Anthropic's Managed Agents split them so a container failure no longer loses the session; lazy sandbox provisioning alone cut p50 time-to-first-token by about 60%. Here the session is an `EventLog`, the harness is `AgentLoop`, and the sandbox is any `ToolExecutor` (`LazyExecutor` provisions on first use and converts sandbox death into a tool error).
2. **The log is the source of truth; context is a projection.** Compaction, memory writes and trimming are irreversible, so the log is never rewritten. Compaction is a new `compaction` event holding the replacement view plus a hash of the canonical prefix (OpenHands tombstones, Cline's hash-validated `compaction.json`).
3. **A thin loop with an explicit termination predicate.** Stop only when the last response has no tool calls, no user input is queued, and every stop hook and evaluator agrees (Claude Agent SDK loop, Codex `run_turn`). Results are typed (`success`, `error_max_turns`, `error_max_budget_usd`, `error_during_execution`, …) and carry `stop_reason`, usage, cost, turns and `session_id`.
4. **Append-only history in native shapes.** The full assistant `content` (thinking and compaction blocks unchanged) is appended; all tool results for a turn go back in one user message in the `tool_result` role. Simulating tool results as user text breaks reasoning replay on several models.
5. **Thick controls around the loop.** Budgets, time warnings and loop detection drove LangChain's +13.7-point Terminal-Bench 2.0 gain with the model held fixed; the 8-consecutive-blocks override, ignoring blocks without a reason, and a compaction thrash guard prevent runaway loops.
6. **Errors are classified by action.** Transient errors retry with backoff (Codex: 4 request retries), rate limits honour `retry_after`, terminal errors stop, overflow compacts, and tool failures are returned to the model as tool results so it can adapt.
7. **Every scaffold is a flag.** Labs keep deleting scaffolds (context resets, sprints, todo lists) as models improve, so the stuck detector, wrap-up turn, budget warnings, evaluators and long-horizon drivers are all optional configuration.

## Architecture

```text
             host (CLI / IDE / CI / server)               steer(text) ──┐   interrupt() ──┐
                         │ run(prompt | None, approvals)                │                 │
                         ▼                                              ▼                 ▼
 ┌──────────────────────────────────────────── AgentLoop (the harness) ───────────────────────────────────┐
 │  session_start ─▶ resume repair of a half-done tool batch ─▶ user_prompt_submit                                │
 │  ┌──────────────── loop ──────────────────────────────────────────────────────────────────────────────────┐   │
 │  │ compaction threshold? ─▶ Compactor ─▶ session.record_compaction (projection; log untouched)            │   │
 │  │ ContextAssembler.prepare(system, session.view(), frozen tools) ─▶ pre_model ─▶ ModelClient.create      │   │
 │  │      (RetryPolicy: transient backoff, retry_after, fallback model; terminal -> error_during_execution) │   │
 │  │ post_model ─▶ stop_reason: overflow→compact │ max_tokens→escalate/continue │ pause_turn→resend │ refusal │   │
 │  │ append FULL assistant content ─▶ tool_uses?                                                            │   │
 │  │   yes: budgets ─▶ stuck detector ─▶ per group: pre_tool ─▶ policy ─▶ approver ─▶ execute (read-only ∥) │   │
 │  │        ─▶ post_tool ─▶ ONE user message [tool_results…, steering, nudges, warnings] ─▶ error limits    │   │
 │  │   no:  steering queued? ─▶ continue │ stop hooks + evaluators: block(reason)? ─▶ continuation          │   │
 │  │        (8 consecutive blocks ⇒ override) │ else ─▶ RunResult(success)                                  │   │
 │  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
 └───────────────┬────────────────────────────────────────────────┬───────────────────────────────┬──────────────┘
                 │ emit(Event) — every message, call, result,      │ execute(call, ctx)            │ Tracer spans
                 ▼   decision, compaction, evaluation, stop        ▼                               ▼
   Session ── EventLog (JSONL, fsync, seq, lease)       ToolExecutor (FunctionToolExecutor,   invoke_agent ▸ chat
   SessionStore: create / open / wake / fork / list     CompositeExecutor, LazyExecutor ─▶     ▸ execute_tool
   <id>.jsonl  <id>.meta.json  <id>.lease               sandbox provisioned on first call)     ▸ compaction …
```

Outer loops sit above `AgentLoop`: `LongRunDriver` (one fresh session per context window,
initializer then coding prompts) and `RalphLoop` (same prompt re-fed inside one session).

## Quick start

```python
import os
from harness_core import AgentLoop, Budgets, FunctionToolExecutor, LoopConfig, SessionStore
from harness_core.anthropic_adapter import AnthropicModelClient

tools = FunctionToolExecutor()

@tools.tool("read_file", "Read a UTF-8 file", {"type": "object", "properties": {"path": {"type": "string"}},
            "required": ["path"]}, read_only=True)
def read_file(inp, ctx):
    return open(os.path.join(ctx.cwd, inp["path"]), encoding="utf-8").read()

store = SessionStore("~/.harness/sessions")
session = store.create()
loop = AgentLoop(
    AnthropicModelClient(os.environ["HARNESS_MODEL"], effort="medium"),   # model id from configuration
    tools,
    system="You are a coding agent.",
    session=session,
    budgets=Budgets(max_turns=50, max_wall_clock_s=1800),
    config=LoopConfig(context_window_tokens=200_000),
    approver=lambda call, spec, ctx, decision: input(f"allow {call.name}? ") == "y",
)
result = loop.run("Summarise README.md")
print(result.subtype.value, result.final_text, result.num_turns, result.usage)

# After a crash, any process can continue the same session from the log:
result = AgentLoop(AnthropicModelClient(os.environ["HARNESS_MODEL"]), tools, session=store.wake(session.id)).run()
```

Offline tests use `harness_core.testing.ScriptedModelClient` instead of a real model; see
[model-adapters.md](model-adapters.md).

## Integration notes

* `ContextAssembler` defaults to `PassthroughAssembler`; plug in the context package's cache-aware assembler.
* `Compactor` is a protocol; the context package supplies the summariser. Without one, overflow ends the run with `error_context_window`.
* `PermissionPolicy` defaults to `ReadOnlyPolicy`; the permissions package's layered pipeline is a drop-in policy.
* Task-state invariants (flip-only feature list, immutable tests) belong in a `pre_tool` hook, for example `harness_memory.task_state.guard_tool_call` wrapped in a `HookResult`.
