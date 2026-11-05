# Integration seams and how they were resolved

The ten packages were built separately against `harness_core/types.py`. This page lists each place where they had to be joined and how `src/harness/` resolves it. Each seam has a test in `tests/integration/`.

Package code changes were kept minimal. Only one existing file changed: `harness_core/anthropic_adapter.py`, for seam 3. `harness_core/types.py` is untouched.

## 1. Context assembly against a durable log (`harness_context` ↔ `harness_core`)

**Problem.** `AgentLoop` calls `assembler.prepare(system, session.view(), tools, ctx)` with a *copy* of the history. `LayeredContextAssembler` appends compaction markers and re-injection bundles to the list it receives (through `append_to_log`, which defaults to `list.append`), so markers were lost after each call. The loop also has its own compaction path: the `Compactor` protocol plus `Session.record_compaction`. Its recorded view would put the marker at index 0, and the assembler's `derive_history` would then reject it (the covered message is no longer before the marker).

**Resolution: the assembler owns compaction and projection.**

* `SessionBoundAssembler` (`harness/adapters.py`) is the loop's assembler. It always gives `LayeredContextAssembler` the session's **canonical** log (`session.messages()`), which holds every marker, rather than the loop's view. Its `append_to_log` appends to that list *and* persists the marker with `session.append_message(..., kind="context")`, forwarding the event to the telemetry sinks. Markers survive a restart because `derive_history` re-verifies the prefix hash after the JSON round trip.
* Proactive (threshold) compaction belongs only to the assembler. The loop's own threshold is switched off (`LoopConfig.context_window_tokens=None`).
* Reactive (overflow) compaction keeps the loop's control flow. When the provider reports a context-window error, the loop calls `ContextCompactorAdapter.compact`. That calls `LayeredContextAssembler.compact`, so the same `harness_context.Compactor` and the same marker apply, and returns `assembler.derive_history(log)`, which is `[marker] + kept tail`. That view is smaller, so the loop's "freed nothing" and "did not shrink" checks pass. `Session.record_compaction` then stores a view consistent with what the assembler will project next.
* `LayeredContextAssembler.prepare` raises `ContextOverflowError` when the window cannot fit. The loop calls `prepare` *outside* its overflow handler, so this would have ended the run as `error_during_execution`. Instead, `SessionBoundAssembler` catches it and returns a request flagged with `OVERFLOW_KEY`. `OverflowGuardClient` turns the flag into `ContextWindowExceededError` without calling the provider. The result is the loop's normal overflow path, with its thrash guard and a typed `error_context_window` result.
* The loop never called `assembler.observe_response`. A `post_model` hook now calls it after every model call, which drives token calibration and cache-hit tracking.

Tests: `test_integration_context.py` covers threshold compaction (persisted, woken from disk), provider overflow through the adapter, local overflow (typed result, provider not called), `observe_response`, and the prompt stack.

## 2. Prompt stack from memory (`harness_memory` → `harness_context`)

`PromptStack` layers, in order:

* the `sandbox` layer, from `describe_sandbox(policy, workspace, choice)`;
* `delegation-policy`, only when the spawn tool is offered;
* `memory`, from `MemorySystem.session_start_context(session, include_task_state=False)`: instructions (`AGENTS.md`), the memory protocol, citation instructions, mounts, the memory summary or index, and the skills catalog;
* `task-state`, from `TaskState.orient()`, only when task state is initialised.

All layers are byte-stable for the session. After compaction, re-injection uses `TextSource("memory", MemorySystem.post_compaction_context)` and `RecentFilesSource`, fed by a `pre_tool` `FileActivityTracker`.

`session_start_context` returns one string, so instructions, the index and skills share one layer. Splitting them would mean calling `MemorySystem` private helpers.

## 3. `extra["effort"]` and `extra["model"]` (`harness_multiagent` → `harness_core.anthropic_adapter`)

`SubagentTask.request_extra()` sends `{"model": id, "effort": e}`. `extra["model"]` already overrode the client model through `params.update`. `extra["effort"]`, however, would have been sent as an invalid top-level parameter.

**Fix** (in `build_params`): `effort` is popped from the merged extras, validated against `EFFORT_LEVELS`, and merged into `output_config.effort`. An explicit `extra["output_config"]` is still merged on top. Tests: `test_integration_adapter_effort.py`, plus `test_effort_and_model_reach_the_anthropic_api_correctly`, which goes through `LoopAgentRunner` with a fake SDK.

## 4. Permissions as the loop's policy (`harness_permissions` ↔ `harness_core`)

* `EnginePolicy` wraps `PolicyEngine.check`. For DENY it rewrites the reason, so the loop's tool result (`"Permission denied: " + reason`) is exactly `denial_outcome(decision).content`. ALLOW and ASK pass through.
* ASK is routed by `make_approver`, in this order:
  1. `approval_ui` submits to the `ApprovalBroker`. A UI callback approves or denies as `Approver.human(user_id)`, and the broker resolves it. The queue is persisted in `approvals.json`, and the broker records the human decision with the engine.
  2. A core-style `approver` callback; the harness calls `engine.record_human_decision` for it.
  3. A blocking `broker.request` when `permissions.wait_for_broker` is set.
  With none of these, the loop treats ASK as DENY.
* `record_result` runs as a `post_tool` and `post_tool_failure` hook. It updates taint, runs the injection probe and annotates the output, and the updated outcome is what the model sees. It receives a copy of the context, so the loop itself flips `ctx.tainted` from `metadata.untrusted` and logs its `tainted` event. Sandbox violations in a result call `engine.record_sandbox_violation`.
* `must_terminate` is honoured by `pre_model` and `pre_tool` hooks, which return `continue_=False`. The run ends as `halted_by_hook` with the engine's reason before the next model call.
* `ctx.metadata` that the integration populates:
  * `permission_mode` and `interactive`, from config, on the loop's context;
  * `parent_session_id`, on every sub-agent loop's context, so `PolicyEngine.session` registers the child at depth + 1 under the parent's mode;
  * the reviewer transcript, through the engine's `transcript_provider`, which returns the live session's messages and also covers sub-agent sessions registered by the runner. This replaces copying `messages` into `ctx.metadata` on every call. The `ctx.metadata["messages"]` fallback is still honoured for unknown sessions.
* The engine's audit trail goes to `telemetry/audit.jsonl`, not the session log, because the engine serves several sessions, including sub-agents.

Tests: `test_integration_permissions.py`.

## 5. Sandbox boundary → permission engine (`harness_sandbox` → `harness_permissions`)

Shell commands are auto-allowed in the workspace only when `runner.choice.auto_allow_in_boundary` is true. When it is false (local fallback, `full-access`, or an `external-sandbox` with no detected container), the harness adds a **session-layer** policy document with `defaults.sandbox = "none"`. The session layer ranks above project and user files, so a project file cannot widen it. Code-executing shell commands then go to rules, the reviewer or a human. Test: `test_sandbox_boundary_feeds_the_engine_config`.

`bash` runs through the runner from `create_runner(policy, workspace, session_id=...)`. `harness_tools.as_command_runner` adapts its keyword-style `run(...)` to the `CommandRunner` protocol. Runner events (`sandbox.exec`) go into the session log.

## 6. Tools (`harness_tools` + `harness_memory` + `harness_multiagent` + `harness_context`)

* The **kernel** is registered against a `Workspace(workspace, checkpointer=MemoryCheckpointer())` with the sandbox runner.
* The **memory tool** is a registry `Tool` whose spec has `provider_type="memory_20250818"`, so the API sees `{"type": "memory_20250818", "name": "memory"}`. Its handler is `MemorySystem` session's `MemoryToolHandler`. It carries `permission_kind: memory`.
* The **spawn tool** is `SubagentManager.spawn_tool_spec()`, with `permission_kind: subagent` (so the engine treats it as a sub-agent launch) and no registry timeout (the manager enforces its own deadlines). Sub-agents get `ExcludingExecutor(registry, {"memory", "spawn_subagent"})`: they read memory through mounts and never write it. Nesting is controlled by the manager.
* `tool_search` is enabled when configured or when anything is deferred, for example MCP tools. It carries `permission_kind: internal`. The registry is frozen after setup.
* `OffloadingToolExecutor` wraps the registry. Offload files live in `<workspace>/.harness/tool-outputs`, so the model can read them back with the kernel `read` tool.
* `guard_tool_call` from `harness_memory.task_state` runs as a `pre_tool` hook. It blocks deleting tests and rewriting `feature_list.json`, and the denial comes back as a tool result.
* The `memory` tool's output is not echoed into the memory transcript; all other tool outputs are, through `record_tool_output`, which handles memory taint.

## 7. Sub-agents on the real loop (`LoopAgentRunner`)

`LoopAgentRunner` implements `AgentRunner.run(task, ctx)` with one fresh `AgentLoop` per task, using an in-memory session (or `session_factory`):

* Requests use `task.system_prompt`. The brief is the only user message; for `context: fork`, the parent history is appended first. Tools are `task.tools.specs()`. A `PassthroughAssembler` carries `extra=task.request_extra(model_id)`. The model id comes from `model_for_role(task.model_role)`, which is `HarnessConfig.model_roles` falling back to `model`.
* Tools run only through `task.tools.execute`, wrapped in `ToolCallCap(max_tool_calls)`.
* `_ChargingClient` calls `task.charge(usage, model)` after **every** response. A `post_model` hook halts as soon as `task.stop_reason()` is set, and a `pre_model` hook halts once the cancellation token fires.
* `max_turns` is `Budgets(max_turns, scope="session")`, so it also counts report-repair turns. The wrap-up turn is disabled, so the cap is exact.
* The final text goes through `parse_report`, with up to `task.max_structured_retries` repair prompts. Loop subtypes map to `SubagentStatus`.
* In the harness, sub-agent loops use the same `EnginePolicy`, approver, tracer and ops sink. The child session is registered for the reviewer transcript, and `engine.subagent_finished` releases the slot and propagates taint.

Tests: `test_integration_multiagent.py`.

## 8. Evals and RL (`LoopAgentUnderTest`)

`run(task, workspace, ctx)` builds a fresh harness per trial:

* `workspace` is the harness workspace. `ctx.metadata["memory_dir"]` is the only memory root; memory is disabled when `memory_mode == "frozen"`, because that snapshot must not change. Harness state goes under the trial's isolated `HOME`.
* `env` is the environment for `bash`, and `seed` seeds the loop's RNG.
* A watcher thread turns `cancel_event` into `harness.interrupt()`.
* Limits map to `max_turns`, `max_wall_clock_s` and a tool-call cap. In RL, `ctx.metadata["tools"]` (the rollout-scoped backend) replaces the registry, so every call goes through it, and `rollout_id` is copied into the tool context.
* Provider failures that persist after the loop's retries are raised as `InfraError`, so the eval runner retries the trial instead of scoring it.

`trajectory_from_session` converts the session log into the payloads `Trajectory.from_events` documents:

* message events become `{"message": ...}`;
* per-call `TOOL_RESULT` events are not replayed, because the results are already inside the user message and would otherwise be duplicated;
* discarded responses are not counted as turns, but their usage is;
* compaction events are counted once.

Tests: `test_integration_evals.py`.

## 9. Observability (`harness_ops`)

* The tracer is `OpsTracer([JsonlSpanExporter])`. The loop, the registry, the assembler and the manager all use it, which gives the span tree `invoke_agent`, `chat`, `execute_tool`, `context.prepare` and `invoke_agent <subagent>`.
* `OpsEventAdapter` translates loop payloads to the keys `HarnessMetrics` reads before forwarding a sanitised copy to `FanoutEventSink(HarnessMetrics, JsonlEventExporter)`:
  * `tool` on `TOOL_RESULT` and `PERMISSION`;
  * `STOP.reason`, set to the subtype;
  * `SESSION_END` `outcome`, `total_cost_usd` and `num_turns`;
  * `ERROR.error_class`.
  The loop's delegated compaction record is not counted twice. The durable event is never modified.
* `ctx.metadata["step_idx"]` is the seq of the assistant message being executed, maintained by `StepIndexSink`. `ctx.metadata["idempotency_key"]` is the loop's `sha256(session:step:tool_use_id)[:32]`, which is a prefix of `harness_ops.idempotency_key` with the same inputs. That means `IdempotentToolExecutor` and `StepJournal` can be layered on without new plumbing.
* With `pricing_file`, `PriceTable` gives both the loop's cost estimator (for USD budgets) and the metrics prices.

## Not wired, and why

* **MCP servers run as ordinary subprocesses, not inside the sandbox.** `harness_sandbox.prepare_mcp_server` can produce a sandboxed argv, but `harness_tools.MCPClient` launches the process itself and has no hook for that. Vet servers before enabling them; their tools are open-world and deferred.
* **The auto-mode reviewer (`TwoStageClassifier`) and the egress proxy are not built by default.** They need a reviewer model id and network policy decisions. Pass a configured `PolicyEngine`, or extend `build_harness`, to use them.
* **`StepJournal`, `IdempotentToolExecutor`, `DeploymentRegistry` and workspace snapshots are not in the default wiring.** The metadata they need (`step_idx`, `idempotency_key`, session ids) is populated, so they can be layered on.
* **The memory background pipeline (`MemorySystem.run_background`) is not run by the harness.** It is an idle-time job for a separate worker.
* **Known cross-package quirk.** `PolicyEngine.record_result` probes sub-agent reports, which are a taint source. The manager's own "untrusted data" preamble can trip the heuristic `instruction_override` probe, so sub-agent reports are usually returned with a security notice and the parent is marked tainted. This is safe, but noisy.
