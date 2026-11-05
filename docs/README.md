# The harness: how the packages fit together

This repository builds an agentic harness as ten packages under `src/`. Each package is useful alone and has its own docs. The top-level package `harness` (`src/harness/`) wires them into one working harness: `build_harness(HarnessConfig) -> Harness`.

The design follows the build plan ([`reports/SOTA agentic harness build plan.md`](../reports/SOTA%20agentic%20harness%20build%20plan.md)) and the reference ([`reports/SOTA agentic harness reference.md`](../reports/SOTA%20agentic%20harness%20reference.md)). An agent is split into a **session** (an append-only event log), a **harness** (the loop plus everything that decides what the model sees and may do) and a **sandbox** (where commands actually run). The harness never runs inside the box it controls.

How each integration seam was resolved is in [integration.md](integration.md).

## Packages

| Package | What it provides | Docs |
|---|---|---|
| `harness` | `HarnessConfig` (YAML), `build_harness`, `Harness`, `LoopAgentRunner`, `LoopAgentUnderTest`, the core `Compactor` adapter | this page, [integration.md](integration.md) |
| `harness_core` | Shared contract (`types.py`), `AgentLoop`, durable `SessionStore`, hooks, budgets, retries, `AnthropicModelClient`, `testing.ScriptedModelClient` | [core/README.md](core/README.md) |
| `harness_context` | `LayeredContextAssembler`, prompt layers, cache breakpoints, tool-result clearing, compaction, re-injection, offloading | [context/README.md](context/README.md) |
| `harness_memory` | Instruction, working, semantic, episodic and procedural memory; task state; `MemorySystem`; the `memory_20250818` tool handler | [memory/README.md](memory/README.md) |
| `harness_tools` | `ToolRegistry` (validation, timeouts, output budgets), the kernel (`read`/`write`/`edit`/`bash`/`grep`/`glob`), tool search, MCP client | [tools/README.md](tools/README.md) |
| `harness_sandbox` | Sandbox policies, OS runners (bubblewrap, Seatbelt, hardened local fallback), egress proxy, snapshots | [sandbox/README.md](sandbox/README.md) |
| `harness_permissions` | `PolicyEngine` (L0-L8 pipeline), shell analysis, taint and injection probes, `ApprovalBroker`, policy files | [permissions/README.md](permissions/README.md) |
| `harness_multiagent` | `SubagentManager` (spawn tool, caps, budgets), the `AgentRunner` protocol, leases, workflows | [multiagent/README.md](multiagent/README.md) |
| `harness_evals` | `EvalRunner`, task format, trial isolation, graders, metrics and reports, the `AgentUnderTest` protocol | [evals/README.md](evals/README.md) |
| `harness_rl` | RL environments and rollouts over eval tasks, rewards, trajectory export, the self-improvement loop | [rl/README.md](rl/README.md) |
| `harness_ops` | `OpsTracer` (OTel GenAI span tree), metrics and dashboards, pricing, durable steps, idempotency, deploys | [ops/README.md](ops/README.md) |

## Architecture

```text
                          ┌───────────────────────── harness (build_harness) ─────────────────────────┐
  HarnessConfig (YAML) ──▶│                                                                            │
                          │   ┌───────────── AgentLoop (harness_core) ─────────────┐                   │
  user prompt / steer ───▶│   │ stop predicate · budgets · retries · hooks         │                   │
                          │   │ tool scheduling · approvals · resume               │                   │
                          │   └──┬──────────┬──────────────┬─────────────┬─────────┘                   │
                          │      │assembler │policy        │executor     │tracer + sinks               │
                          │      ▼          ▼              ▼             ▼                             │
                          │  context     permissions     tools          ops                            │
                          │  Layered     PolicyEngine    Offloading(    OpsTracer → spans.jsonl        │
                          │  Context     + Approval      ToolRegistry:  OpsEventAdapter → Fanout:      │
                          │  Assembler     Broker          kernel ──────┐  HarnessMetrics, events.jsonl│
                          │   ▲  │         ▲ record_       memory tool  │                              │
                          │   │  │compact  │ result        spawn tool ─┐│                              │
                          │   │  ▼         │               tool_search ││ MCP                         │
                          │  PromptStack ◀── memory        ...)        ││                              │
                          │  (instructions,  MemorySystem  ◀───────────┘│ multiagent                   │
                          │   memory index,  (memory tool,  SubagentManager → LoopAgentRunner          │
                          │   skills, task)  citations)        (a fresh AgentLoop per sub-agent)       │
                          └──────┬──────────────────────────────────────┼─────────────────────────────┘
                                 │ every message and event              │ bash → CommandRunner
                                 ▼                                      ▼
                    session log (harness_core.SessionStore)     sandbox (harness_sandbox runner:
                    append-only JSONL, compaction markers,       bubblewrap / Seatbelt / local fallback,
                    wake / resume / fork                         egress proxy, violations)

   evals / RL:  EvalRunner / RolloutRunner ──▶ LoopAgentUnderTest ──▶ build_harness per trial ──▶ Trajectory
```

* **Session log.** `harness_core.SessionStore` keeps one append-only, fsync-ed JSONL file per session. The model window is always a *projection* of it. Compaction markers, tool results, permission decisions, sub-agent events and sandbox events all land there, so `build_harness(cfg, session_id=...)` resumes a session after a crash.
* **Loop.** `harness_core.AgentLoop` owns control flow only: the stop predicate, every `stop_reason`, budgets, retries, steering, interrupts, approvals and resume. It delegates *what the model sees* to the assembler, *what may run* to the policy, *how tools run* to the executor, and *what is recorded* to the tracer and sinks.
* **Context.** `harness_context.LayeredContextAssembler` builds each request from the canonical log: a stable prompt stack (base prompt, sandbox description, memory context), the latest compaction marker plus the kept tail, cache breakpoints and a turn-scoped dynamic tail. It also owns auto-compaction.
* **Memory.** `harness_memory.MemorySystem` supplies the session-start context (instructions, memory index, skills, task state), the `memory_20250818` tool, the transcript and taint record, citation telemetry and the end-of-session commit.
* **Tools.** `harness_tools.ToolRegistry` holds the kernel tools confined to the workspace, the memory tool, the optional spawn tool, tool search and MCP tools. The registry is frozen at session start so the tool prefix stays cache-stable. `harness_context.OffloadingToolExecutor` wraps it so large outputs go to files.
* **Sandbox.** `bash` runs through the runner that `harness_sandbox.create_runner` picks. Whether that runner is a real boundary (`choice.auto_allow_in_boundary`) decides whether the permission engine may auto-allow shell commands inside the workspace.
* **Permissions.** `harness_permissions.PolicyEngine` is the loop's `PermissionPolicy`. ASK goes to an approver or the `ApprovalBroker`. After each tool runs, `record_result` probes and taints, and headless sessions that hit the denial limit are halted.
* **Multi-agent.** `harness_multiagent.SubagentManager` exposes `spawn_subagent`. Each sub-agent runs in its own `AgentLoop` (via `LoopAgentRunner`) with a scoped tool set, its own budget ledger and the same permission engine at depth + 1.
* **Evals / RL.** `harness_evals.EvalRunner` and `harness_rl.RolloutRunner` drive `LoopAgentUnderTest`, which builds a fresh harness inside each trial sandbox and converts the session log into a `Trajectory`.
* **Ops.** `harness_ops.OpsTracer` records the `invoke_agent → chat → execute_tool` span tree. `HarnessMetrics` sits behind a `FanoutEventSink` and turns loop events into the dashboard. Everything is written under `telemetry_dir`.

## Request and turn lifecycle

One `harness.run(prompt)`:

1. **Prompt.** The prompt is recorded in the memory transcript. The loop takes the session lease, emits `session_start` (with the tool names and the system-prompt hash), runs `user_prompt_submit` hooks and appends the user message to the durable log.
2. **Assemble.** `SessionBoundAssembler.prepare` passes the session's canonical log to `LayeredContextAssembler`. The assembler projects the latest marker plus the tail, applies the ingest cap and sticky clearing, and estimates tokens (calibrated from earlier responses). Above the compaction trigger it calls `harness_context.Compactor`, which appends a marker to the durable log through `append_to_log`. It then renders the prompt stack, the dynamic tail and up to four cache breakpoints. If the window still cannot fit, the request is flagged and the next step turns it into an overflow.
3. **Pre-model hooks.** If the permission engine says a headless session must terminate, the run halts here.
4. **Model call.** `OverflowGuardClient` raises `ContextWindowExceededError` for a flagged request. Otherwise it calls the model client (`AnthropicModelClient`, or an injected one). Retries, rate limits and the fallback model are handled by the loop. On a context-window error the loop calls `ContextCompactorAdapter`, which runs `LayeredContextAssembler.compact`. The loop records the derived view (`Session.record_compaction`) and retries.
5. **Post-model hooks.** `assembler.observe_response` feeds `usage` back for calibration and cache-hit tracking. The response is appended in full, with thinking and other provider blocks unchanged. Its event seq becomes `ctx.metadata["step_idx"]`.
6. **Tool calls.** For each `tool_use`:
   * the `pre_tool` hooks run: the must-terminate check, the file-activity tracker for re-injection, and the task-state `guard_tool_call`, which denies deleting tests or rewriting the feature list;
   * `EnginePolicy.check` (the `PolicyEngine`) decides. DENY comes back as exactly `denial_outcome(decision)`. ASK goes to the approver or the `ApprovalBroker`;
   * the call runs in `OffloadingToolExecutor → ToolRegistry`. `bash` runs in the sandbox runner. The context carries `tool_use_id`, `idempotency_key`, `step_idx`, `permission_mode` and `interactive`;
   * the `post_tool` hook calls `engine.record_result`, which handles taint, injection probes and annotations, and records sandbox violations. The output is logged to the memory transcript.
   Read-only calls run in parallel, mutating calls run in order, and all results go back in **one** user message.
7. **Stop.** When a response has no tool calls and no steering is queued, stop hooks and evaluators may veto the stop. On success the final reply goes through `MemorySession.finish_reply`, which records citations and strips the trailer, and `metrics.prom` is rewritten.
8. **End.** `harness.end_session()` emits `session_end` (the ops adapter adds outcome, cost and turns), calls `MemorySystem.end_session`, which persists taint and commits inline memory writes, closes MCP clients and the sandbox runner, and writes the final metrics.

## Configuration reference (`HarnessConfig`)

`HarnessConfig.from_yaml(path)` loads YAML. Relative paths resolve against the file's directory. `${VAR}` and `${VAR:-default}` expand from the environment. Unknown keys are errors. **`model` is required and never defaulted.**

| Key | Default | Meaning |
|---|---|---|
| `model` | *(required)* | Model id for the main agent. Use `${HARNESS_MODEL}`. |
| `effort` | unset | `low`/`medium`/`high`/`xhigh`/`max`, sent as `output_config.effort`. |
| `workspace` | `.` | Workspace root: kernel tools are confined to it and `bash` runs in it. |
| `base_dir` | `~/.harness` | Memory root (`org/`, `users/`, `projects/`, `sessions/`). |
| `state_dir` | `<base_dir>/harness` | Session logs (`sessions/`), assembler state (`context/`), `approvals.json`. |
| `user_id` | `default` | Memory user scope; also the approver id recorded by the broker. |
| `system_prompt` | built-in | Base system prompt. The prompt stack adds the sandbox, memory and delegation layers. |
| `context_window_tokens` | unset | The model's window. Unset means no auto-compaction (the provider's overflow still triggers compaction). |
| `compaction_threshold_tokens` | 150000 | Quality threshold. The effective trigger is `min(this, 90% of the window)`. |
| `max_tokens` | 16000 | Output tokens per request. |
| `tool_profile` | `generic` | `generic` (`read`, `bash`...) or `anthropic` (`Read`, `Bash`...) kernel names. |
| `durable_sessions` | `true` | JSONL `SessionStore` (resumable), or an in-memory session. |
| `permissions.mode` | policy default | `plan` / `dont-ask` / `default` / `accept-edits` / `auto`. |
| `permissions.interactive` | `true` | `false` = headless: ASK becomes DENY and repeated denials terminate the run. |
| `permissions.policy_files` | `[]` | Project-layer policy files (TOML/YAML/JSON). Also `managed_policy_files` and `user_policy_files`. |
| `permissions.wait_for_broker` | `false` | With no approver, block on the `ApprovalBroker` until someone answers or the timeout denies. |
| `permissions.approval_timeout_s` | policy (300) | Broker wait before denying. |
| `sandbox.policy_file` | defaults | `harness_sandbox` policy YAML/JSON. |
| `sandbox.mode` | policy | `read-only` / `workspace-write` / `external-sandbox` / `full-access` (the last needs admin). |
| `sandbox.require_isolation` | `false` | Fail instead of falling back to the local runner. |
| `budgets.*` | none | `max_turns`, `max_total_tokens`, `max_wall_clock_s`, `max_budget_usd` (the last needs `pricing_file`). |
| `features.memory` | `true` | `MemorySystem`, the memory tool, memory context and citations. |
| `features.memory_git` | `true` | Version memory stores with git. |
| `features.subagents` | `false` | The `spawn_subagent` tool. Definitions come from `<workspace>/.harness/agents` and `agents_dirs`. |
| `features.tool_search` | `false` | Client `tool_search`. It is switched on automatically when any tool is deferred, for example MCP tools. |
| `features.offload` | `true` | Offload large tool outputs to `<workspace>/.harness/tool-outputs`. |
| `features.mcp_servers` | `[]` | `{name, command, args, env, trusted}` stdio servers. Their tools are deferred and open-world. |
| `pricing_file` | unset | `harness_ops` price table, used for the cost estimate, USD budgets and cost metrics. |
| `telemetry_dir` | `<state_dir>/telemetry` | `spans.jsonl`, `events.jsonl`, `audit.jsonl`, `metrics.prom`. |
| `model_roles` | `{}` | Sub-agent role to model id. Roles that are not listed use `model`. |
| `agents_dirs` | `[]` | Extra sub-agent definition directories. |

Sample `harness.yaml`:

```yaml
model: ${HARNESS_MODEL}            # required
effort: high
workspace: .
base_dir: ~/.harness
user_id: alice
context_window_tokens: 200000
permissions:
  mode: auto
  interactive: true
  policy_files: [harness-policy.toml]
sandbox:
  policy_file: sandbox.yaml
budgets:
  max_turns: 60
  max_budget_usd: 5.0
features:
  memory: true
  subagents: true
  mcp_servers:
    - {name: tracker, command: tracker-mcp, args: [--stdio]}
pricing_file: prices.yaml
model_roles:
  worker: ${HARNESS_WORKER_MODEL}
```

## Quick start

```bash
pip install -e ".[dev]"                  # add ,anthropic for real model calls
python examples/harness_quickstart.py    # offline: scripted model, temp workspace
HARNESS_MODEL=<model id> python examples/harness_quickstart.py --real "Summarise README.md"
```

```python
from harness import HarnessConfig, build_harness

cfg = HarnessConfig.from_yaml("harness.yaml")
with build_harness(cfg, approver=lambda call, spec, ctx, decision: input(f"allow {call.name}? ") == "y") as h:
    result = h.run("Fix the failing test in tests/test_parse.py")
    print(result.subtype.value, result.final_text)
    print(h.metrics()["cache_hit_rate"])
```

To use it in tests and offline runs, inject a client: `build_harness(cfg, model_client=harness_core.testing.ScriptedModelClient([...]))`. In evals, pass `LoopAgentUnderTest(cfg, model_client=...)` to `harness_evals.EvalRunner` or `harness_rl.RolloutRunner`. The harness tests live in `tests/integration/` and run offline with `python3 -m pytest -p no:cacheprovider tests/integration`.
