# Open-Source Agent Harness Repositories and Lab Reference Implementations (state as of 2026-11-03)

Method note: Every repo below was shallow-cloned at its default branch on 2026-11-03, and its source tree, READMEs, docs and changelogs were read directly. Version numbers come from `git ls-remote --tags`. Star counts come from GitHub repo pages, fetched through a summarizing web-fetch tool on 2026-11-03. They are approximate, so spot-check them before quoting. The GitHub REST API, huggingface.co, arxiv.org and api-docs.deepseek.com were blocked from this session. As a result, DeepSeek findings rely partly on search-engine summaries of those pages, and each such finding is flagged.

## Q1. OpenAI repos: codex (codex-rs), openai-agents-python/-js, swarm, harmony/gpt-oss. Key files and design choices

### Takeaway
`openai/codex` is the most complete open-source (Apache-2.0) production coding-agent harness available. It is a roughly 150-crate Rust workspace with these parts:
- an OS-native sandbox: Seatbelt on macOS, bubblewrap/Landlock on Linux, and a Windows sandbox
- an `apply_patch` edit format defined by a Lark grammar and exposed as a "freeform" grammar-constrained tool
- local plus server-side ("remote v2") compaction
- hierarchical AGENTS.md loading
- Claude-Code-style hooks
- a JSON-RPC "app-server" thread/turn/item protocol that every OpenAI surface (TUI, IDE, desktop, SDK) is built on

The Agents SDK (Python/JS) is the general-purpose "Runner loop" library. Its 2026 additions are `SandboxAgent` (workspace manifest, capabilities, snapshots) and `responses.compact`-backed sessions. Swarm is officially superseded.

### Cited Findings

**openai/codex: overview and status**
- ~132.9k stars, ~21.0k forks (page fetch 2026-11-03). Latest stable Rust tag is `rust-v0.166.0` (alpha tags go higher). Last commit was `fbf5f0d` on 2026-11-03 — [GitHub openai/codex](https://github.com/openai/codex); tags from `git ls-remote`.
- Codex ships as a CLI, IDE extension, desktop app (`codex app`) and cloud "Codex Web". Install is via a standalone installer, npm `@openai/codex` or Homebrew. It can be used with a ChatGPT plan or an API key — [README](https://github.com/openai/codex/blob/main/README.md).
- The codex-rs workspace contains crates including:
  - `core`, `apply-patch`, `app-server`, `app-server-protocol`, `exec`, `exec-server`, `execpolicy`, `hooks`, `skills`, `memories`
  - `linux-sandbox`, `bwrap`, `windows-sandbox-rs`, `network-proxy`, `sandboxing`
  - `rmcp-client`, `codex-mcp`, `code-mode`, `rollout`, `thread-store`, `tui`, `otel`, `worktree`, `agent-roles`, `message-board`, `responses-api-proxy`
  
  [codex-rs tree](https://github.com/openai/codex/tree/main/codex-rs)
- `codex-core` "implements the business logic for Codex … used by the various Codex UIs written in Rust" — [core/README.md](https://github.com/openai/codex/blob/main/codex-rs/core/README.md).
- The model-specific base prompts live as markdown files in core: `gpt_5_codex_prompt.md`, `gpt-5.1-codex-max_prompt.md`, `gpt-5.2-codex_prompt.md`, `gpt_5_1_prompt.md`, `gpt_5_2_prompt.md` — [codex-rs/core](https://github.com/openai/codex/tree/main/codex-rs/core).

**Codex agent loop and tools (core/src)**
- Key modules in `core/src`:
  - `codex_thread.rs`, `thread_manager.rs`, `session/`, `tasks/`
  - `client.rs` (Responses API client), `responses_retry.rs`
  - `context_manager/`, `turn_diff_tracker.rs`, `rollout.rs`, `shell_snapshot.rs`, `exec_policy.rs`, `guardian_review.rs`
  - `mcp_tool_call.rs`, `skills.rs`, `agents_md.rs`, `compact*.rs`
  
  [core/src](https://github.com/openai/codex/tree/main/codex-rs/core/src)
- Tool plumbing lives under `core/src/tools/`: `registry.rs`, `router.rs`, `orchestrator.rs`, `parallel.rs`, `approvals.rs`, `sandboxing.rs`, `network_approval.rs` and `spec_plan.rs` — [core/src/tools](https://github.com/openai/codex/tree/main/codex-rs/core/src/tools).
- Tool names found in the handler specs:
  - shell/PTY: `exec_command`, `write_stdin` ("unified exec", PTY sessions keyed by `session_id`), plus a legacy `shell`
  - editing and planning: `apply_patch`, `update_plan`, `view_image`, `request_user_input`, `request_permissions`, `tool_search`
  - MCP resources: `list_mcp_resources`, `list_mcp_resource_templates`, `read_mcp_resource`
  - multi-agent: `spawn_agent`, `send_input`, `send_message`, `wait_agent`, `close_agent`, `resume_agent`, `interrupt_agent`, `list_agents`, `followup_task`
  - context and plugins: `get_context_remaining`, `new_context_window`, `request_plugin_install`
  
  [core/src/tools/handlers](https://github.com/openai/codex/tree/main/codex-rs/core/src/tools/handlers); [unified_exec.rs](https://github.com/openai/codex/blob/main/codex-rs/core/src/tools/handlers/unified_exec.rs).
- Sandbox/approval prompts are templated per mode:
  - `permissions/sandbox_mode/{read_only,workspace_write,danger_full_access}.md`
  - `approval_policy/{never,on_request,unless_trusted}.md`, plus `approved_prefixes.md`
  - a "guardian" auto-review classifier prompt (`guardian/policy.md`, `classifier_instructions.md`)
  
  [prompts/templates](https://github.com/openai/codex/tree/main/codex-rs/prompts/templates)

**Codex apply_patch**
- The format is defined by an official Lark grammar: `*** Begin Patch` / `*** Add File: ` / `*** Delete File: ` / `*** Update File: ` / `*** Move to: ` / `@@ <context>` / `+`/`-`/` ` lines / `*** End of File` / `*** End Patch`. A new optional `*** Environment ID:` header supports multi-environment threads. The parser is deliberately lenient; the code comments say that only gpt-4.1 "knowingly requires lenient parsing" — [apply-patch/src/parser.rs](https://github.com/openai/codex/blob/main/codex-rs/apply-patch/src/parser.rs).
- Context matching (`seek_sequence`) falls back progressively: exact match, then trim-end, then trim, then Unicode-normalised match — [seek_sequence.rs](https://github.com/openai/codex/blob/main/codex-rs/apply-patch/src/seek_sequence.rs).
- The tool is exposed as a FREEFORM tool with `format: {type: "grammar", syntax: "lark"}`, loaded from `assets/tools/apply_patch.lark`. Its description reads: "This is a FREEFORM tool, so do not wrap the patch in JSON" — [apply_patch_spec.rs](https://github.com/openai/codex/blob/main/codex-rs/core/src/tools/handlers/apply_patch_spec.rs).
- The crate also has a `streaming_parser.rs`, which lets patches be parsed as they stream in, and a standalone `apply_patch` executable — [apply-patch/src](https://github.com/openai/codex/tree/main/codex-rs/apply-patch/src).

**Codex compaction**
- The local compaction prompt reads: "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task." It asks for progress and decisions, constraints and preferences, next steps, and critical data — [prompts/templates/compact/prompt.md](https://github.com/openai/codex/blob/main/codex-rs/prompts/templates/compact/prompt.md).
- The summary is re-injected with a prefix: "Another language model started to solve this problem and produced a summary of its thinking process…" — [summary_prefix.md](https://github.com/openai/codex/blob/main/codex-rs/prompts/templates/compact/summary_prefix.md).
- Compacted history keeps prior real user messages up to `COMPACT_USER_MESSAGE_MAX_TOKENS = 20_000` plus the summary, and re-inserts the initial context before the last real user message or summary. Compaction can run inline when the token limit is hit (`run_inline_auto_compact_task`) or be triggered manually — [core/src/compact.rs](https://github.com/openai/codex/blob/main/codex-rs/core/src/compact.rs).
- A separate server-side path (`compact_remote_v2*.rs`) keeps certain tool outputs (`compact_retained_tool_outputs.rs`), budgets images, falls back to another model, and runs `PreCompact`/`PostCompact` hooks — [compact_remote_v2.rs](https://github.com/openai/codex/blob/main/codex-rs/core/src/compact_remote_v2.rs).

**Codex AGENTS.md handling**
- Discovery collects every `AGENTS.md` from the project root down to the cwd. `AGENTS.override.md` is the preferred local override. User and project docs are joined with `--- project-doc ---`. A byte budget (`project_doc_max_bytes`) is enforced across all files — [core/src/agents_md.rs](https://github.com/openai/codex/blob/main/codex-rs/core/src/agents_md.rs); [docs/agents_md.md](https://github.com/openai/codex/blob/main/docs/agents_md.md).

**Codex sandboxing and exec policy**
- On macOS, Codex uses `/usr/bin/sandbox-exec` (Seatbelt). Under workspace-write it allows writes under the configured roots but keeps `.git`, the resolved gitdir and `.codex` read-only.
- On Linux, the binary re-execs itself as `codex-linux-sandbox` (arg0 trick). It prefers system `bwrap`, falls back to a bundled bwrap, and keeps a legacy Landlock path. WSL1 is unsupported.
- On Windows, there is an elevated sandbox with split filesystem policies.

[core/README.md](https://github.com/openai/codex/blob/main/codex-rs/core/README.md)
- CLI subcommands include `exec` (alias `e`), `review`, `mcp` (manage external servers: list/get/add/remove/login/logout), `plugin`, `app-server`, `remote-control`, `sandbox`, `execpolicy` (check rule files), `apply`, `resume`, `fork`, `queue`, `cloud` and `exec-server` — [cli/src/main.rs](https://github.com/openai/codex/blob/main/codex-rs/cli/src/main.rs).

**Codex hooks**
- The hooks crate implements these events: `PreToolUse`, `PostToolUse`, `SessionStart`, `UserPromptSubmit`, `Stop`, `PreCompact`, `PostCompact`, `SubagentStop`, `PermissionRequest`. These are the same names Claude Code uses — [codex-rs/hooks/src](https://github.com/openai/codex/tree/main/codex-rs/hooks/src).

**Codex MCP**
- The MCP client uses the official Rust SDK `rmcp` (pinned `=3.5.1`) — [codex-rs/Cargo.toml](https://github.com/openai/codex/blob/main/codex-rs/Cargo.toml).
- `codex-mcp` "manages Codex's MCP connections, tool calls, and their captured authority". Servers can opt into an experimental `codex/sandbox-state` capability, which lets them receive sandbox policy in `_meta` — [codex-mcp/README.md](https://github.com/openai/codex/blob/main/codex-rs/codex-mcp/README.md).

**Codex headless (exec) mode and app-server protocol**
- `codex exec --json` emits JSONL `ThreadEvent`s: `thread.started`, `turn.started`, `turn.completed`, `turn.failed`, `item.started`, `item.updated`, `item.completed`, `error` — [exec/src/exec_events.rs](https://github.com/openai/codex/blob/main/codex-rs/exec/src/exec_events.rs).
- The app-server protocol is JSON-RPC over a thread/turn/item model. Methods include:
  - threads: `thread/start|resume|fork|read|list|search|compact/start|revert|archive`, `thread/goal/set|get|clear`, `thread/queue/*`, `thread/memory/read`
  - turns: `turn/start`, `turn/steer`, `turn/interrupt`
  - streamed items: `item/agentMessage/delta`, `item/commandExecution/requestApproval`, `item/fileChange/requestApproval`, `item/plan/delta`, `item/reasoning/summaryTextDelta`
  - other: `review/start`, `skills/list`, `mcpServer/elicitation/request`
  
  [app-server-protocol](https://github.com/openai/codex/tree/main/codex-rs/app-server-protocol); [app-server/README.md](https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md).
- The TypeScript `@openai/codex-sdk` "wraps the `codex` CLI … spawns the CLI and exchanges JSONL events over stdin/stdout". Usage is `codex.startThread()` and then `thread.run(...)`. A Python SDK exists in `sdk/python` — [sdk/typescript/README.md](https://github.com/openai/codex/blob/main/sdk/typescript/README.md); [sdk/](https://github.com/openai/codex/tree/main/sdk).
- In the current main branch, `cli/src/main.rs` has no top-level `mcp-server` subcommand (it existed in 2025-era versions). App-server and the SDK appear to be the supported embedding paths; this was checked only against the subcommand enum — [cli/src/main.rs](https://github.com/openai/codex/blob/main/codex-rs/cli/src/main.rs).

**Codex memories**
- Memories run as a two-phase background pipeline, only for non-ephemeral root (non-subagent) sessions:
  - Phase 1 extracts a structured memory from each eligible idle rollout (`raw_memory`, `rollout_summary`, `rollout_slug`), with secret redaction and DB leases and retry backoff.
  - Phase 2 holds a global lock and runs a consolidation agent that writes filesystem memory artifacts. Memories unused for `max_unused_days` are pruned.
  
  [codex-rs/memories/README.md](https://github.com/openai/codex/blob/main/codex-rs/memories/README.md)

**openai/openai-agents-python and openai-agents-js**
- Python: ~30.5k stars; tag `v0.24.0`; last commit 2026-10-30. JS: tag `v0.21.0`; last commit 2026-11-02 — [openai-agents-python](https://github.com/openai/openai-agents-python); [openai-agents-js](https://github.com/openai/openai-agents-js).
- The SDK describes itself as "provider-agnostic, supporting the OpenAI Responses and Chat Completions APIs, as well as 100+ other LLMs". Core concepts are Agents, Sandbox agents, Realtime agents, Voice, Agents-as-tools/Handoffs, Tools (function, MCP, hosted), Guardrails, Human in the loop, Sessions and Tracing — [README](https://github.com/openai/openai-agents-python/blob/main/README.md).
- The Runner loop works like this: call the LLM; if the output is final, return it; if it is a handoff, switch the current agent and re-run; if it is tool calls, run them, append the results and re-run. Input can be a string, Responses-format items, or a `RunState` when resuming a paused run — [docs/running_agents.md](https://github.com/openai/openai-agents-python/blob/main/docs/running_agents.md).
- The loop internals are split into `run_internal/`: `run_loop.py`, `turn_preparation.py`, `turn_resolution.py`, `tool_planning.py`, `tool_execution.py`, `approvals.py`, `guardrails.py`, `session_persistence.py`, `prompt_cache_key.py`, `model_retry.py` and `streaming.py` — [src/agents/run_internal](https://github.com/openai/openai-agents-python/tree/main/src/agents/run_internal).
- Handoffs are exposed to the model as tools named `transfer_to_{agent.name}` — [handoffs/__init__.py](https://github.com/openai/openai-agents-python/blob/main/src/agents/handoffs/__init__.py).
- Guardrails are `InputGuardrail` and `OutputGuardrail`, which return `GuardrailFunctionOutput(tripwire_triggered=…)`; a tripwire halts execution — [guardrail.py](https://github.com/openai/openai-agents-python/blob/main/src/agents/guardrail.py). Tool-level guardrails are in `tool_guardrails.py` — [src/agents](https://github.com/openai/openai-agents-python/tree/main/src/agents).
- Sessions: `SQLiteSession`, `OpenAIConversationsSession` and a Redis option. `OpenAIResponsesCompactionSession` is a decorator that "triggers responses.compact when stored history grows" (Responses-API models only) — [memory/openai_responses_compaction_session.py](https://github.com/openai/openai-agents-python/blob/main/src/agents/memory/openai_responses_compaction_session.py).
- Tracing lives in `tracing/` (spans, traces, processors, provider), which exports to the OpenAI Traces dashboard by default — [src/agents/tracing](https://github.com/openai/openai-agents-python/tree/main/src/agents/tracing).
- **New in 2026: `SandboxAgent`.** It takes a `Manifest` of workspace entries (`GitRepo`, `LocalDir`), `capabilities` (`Capabilities.default()` plus `Skills(lazy_from=…)`), and `SandboxRunConfig(client=UnixLocalSandboxClient()|DockerSandboxClient|hosted)`.
  - The SDK provides "file staging, filesystem tools, shell access, sandbox lifecycle, snapshots."
  - It warns that UnixLocal "adds no OS-level confinement" on Linux.
  
  [docs/sandbox_agents.md](https://github.com/openai/openai-agents-python/blob/main/docs/sandbox_agents.md); [README](https://github.com/openai/openai-agents-python/blob/main/README.md)
- Sandbox capabilities modules: `compaction.py`, `filesystem.py`, `memory.py`, `shell.py`, `skills.py`. Hosted sandbox extensions: `e2b`, `modal`, `daytona`, `cloudflare`, `vercel`, `blaxel`. Also `apply_patch.py`, `snapshot.py` and `manifest.py` — [src/agents/sandbox](https://github.com/openai/openai-agents-python/tree/main/src/agents/sandbox); [extensions/sandbox](https://github.com/openai/openai-agents-python/tree/main/src/agents/extensions/sandbox).
- The JS SDK has the same concept set and a monorepo split into `agents`, `agents-core`, `agents-openai`, `agents-extensions` and `agents-realtime`. It runs on Node 22+, Deno and Bun — [openai-agents-js README](https://github.com/openai/openai-agents-js/blob/main/README.md).

**openai/swarm (deprecated)**
- The README says: "Swarm is now replaced by the OpenAI Agents SDK … We recommend migrating to the Agents SDK for all production use cases." It is described as "experimental, educational" and runs almost entirely client-side and stateless — [swarm README](https://github.com/openai/swarm/blob/main/README.md).

**openai/harmony and openai/gpt-oss**
- Harmony is the response format gpt-oss was trained on. It has these parts:
  - roles `system`/`developer`/`user`/`assistant`
  - channels `analysis`/`commentary`/`final`; function calls go to `commentary`
  - tools declared as TypeScript-like `namespace functions { type f = (_: {...}) => any; }`
  - special tokens `<|start|>…<|message|>…<|end|>`
  - a Rust core with Python bindings for loss-free render/parse
  
  Latest tag is `v0.0.8` — [harmony README](https://github.com/openai/harmony/blob/main/README.md).
- gpt-oss ships reference tools the models were trained on: `gpt_oss/tools/simple_browser`, `python_docker` (a stateless Python tool), and `apply_patch.py`/`apply_patch.md`. It also includes a terminal chat (`python -m gpt_oss.chat … -a` enables apply_patch) and a Responses-API-compatible example server (`gpt_oss/responses_api`), and states that the models "should only be used with this format" — [gpt-oss README](https://github.com/openai/gpt-oss/blob/main/README.md); [gpt_oss/tools](https://github.com/openai/gpt-oss/tree/main/gpt_oss/tools). Latest tag `v0.0.9`, last commit 2026-10-29.

### Inferences
What a builder should copy from the OpenAI repos:
- **Edit format and tool shape.** Use the `apply_patch` envelope with a Lark grammar, exposed as a grammar-constrained "freeform" tool rather than JSON. Pair it with the tolerant `seek_sequence` matcher and a streaming parser. OpenCode, the Agents SDK and gpt-oss have all adopted the same envelope (see Q4), so it is becoming a de facto cross-vendor format.
- **Client protocol.** The app-server's thread/turn/item JSON-RPC (with `turn/steer`, `turn/interrupt`, item-level approval requests and delta streams) is the most complete open spec for a UI-agnostic harness/client boundary. A new harness could adopt it rather than invent its own.
- **Compaction.** Use a short "handoff summary" prompt plus selective retention of recent user messages (a token budget), and re-insert initial context after compaction. That is a small, battle-tested recipe. Codex also has a provider-side compaction path for the Responses API.
- **Shell.** Make it a PTY-backed `exec_command` plus `write_stdin` pair (interactive processes), not one-shot subprocesses. Note that this diverges from mini-swe-agent's stateless `subprocess.run` choice (Q4).
- **Policy is separate from the sandbox.** Execpolicy prefix rules, approval-policy templates and an LLM "guardian" auto-reviewer each sit on top of the OS sandbox.
- **For non-coding agent apps,** the Agents SDK's Runner, handoff, guardrail tripwire and session abstractions are the reference. Its SandboxAgent is OpenAI's answer to the Claude Agent SDK's "give the agent a computer" model.

### Gaps
- I could not read developers.openai.com/codex pages (the in-repo docs/*.md are stubs pointing there). Default auto-compaction thresholds and execpolicy rule syntax were therefore not captured.
- Star counts for agents-js, harmony and gpt-oss were not fetched.
- I could not verify the exact date of the `rust-v0.166.0` release, because the GitHub Releases API was blocked.

## Q2. Anthropic repos: claude-agent-sdk-python/-typescript, claude-code, skills, claude-cookbooks, sandbox-runtime, claude-code-action, reference harnesses

### Takeaway
Anthropic's harness itself (Claude Code) is not open source: the repo is licensed under "Commercial Terms of Service", "All rights reserved". What Anthropic does open-source is the scaffolding around it:
- the MIT Python SDK, which spawns the bundled Claude Code CLI and speaks a stream-JSON plus control-request protocol to it
- the Apache-2.0 `sandbox-runtime` (bubblewrap/Seatbelt plus a domain-allowlist proxy)
- the Agent Skills repo and spec (`SKILL.md`, now at agentskills.io)
- plugin/hook/"mod" sources and example settings
- cookbooks on context editing, compaction and memory
- quickstarts, including the two-agent "initializer + coding agent" long-running harness and Managed Agents samples

### Cited Findings

**claude-agent-sdk-python (MIT) and -typescript (Commercial Terms)**
- Python: ~8.4k stars; tag `v0.2.177`; last commit 2026-11-03. TypeScript: tag `v0.3.317`. The Python LICENSE is MIT. The TypeScript repo's LICENSE.md reads "© Anthropic PBC. All rights reserved. Use is subject to Anthropic's Commercial Terms of Service" — [python LICENSE](https://github.com/anthropics/claude-agent-sdk-python/blob/main/LICENSE); [TS LICENSE.md](https://github.com/anthropics/claude-agent-sdk-typescript/blob/main/LICENSE.md); [python repo](https://github.com/anthropics/claude-agent-sdk-python).
- "The Claude Code CLI is automatically bundled with the package." The SDK uses the bundled CLI by default, or a custom `cli_path`. There are two entry points: `query()` (one-shot) and `ClaudeSDKClient` (bidirectional, which enables custom tools and hooks) — [README](https://github.com/anthropics/claude-agent-sdk-python/blob/main/README.md).
- **Architecture.** The SDK is a single transport, `_internal/transport/subprocess_cli.py`, plus `_internal/query.py`. The query layer sends `control_request` JSON lines and handles inbound control requests with subtypes `initialize`, `can_use_tool` (permission callback), `hook_callback` and `mcp_message` (routing to in-process SDK MCP servers) — [_internal/query.py](https://github.com/anthropics/claude-agent-sdk-python/blob/main/src/claude_agent_sdk/_internal/query.py).
- **Custom tools are in-process MCP servers,** built with `@tool` and `create_sdk_mcp_server()`. Benefits listed are "No subprocess management … No IPC overhead … Single Python process". `allowed_tools` pre-approves tools; it "does not control tool availability" — [README](https://github.com/anthropics/claude-agent-sdk-python/blob/main/README.md).
- **Hook events** (`HookEvent`): `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `UserPromptSubmit`, `Stop`, `SubagentStop`, `PreCompact`, `Notification`, `SubagentStart`, `PermissionRequest`. A hook returns `hookSpecificOutput.permissionDecision: "deny"` with a reason to block a call — [types.py](https://github.com/anthropics/claude-agent-sdk-python/blob/main/src/claude_agent_sdk/types.py); [README hooks example](https://github.com/anthropics/claude-agent-sdk-python/blob/main/README.md).
- **Permission modes:** `default`, `acceptEdits`, `plan`, `bypassPermissions`, `dontAsk`, `auto` — [types.py](https://github.com/anthropics/claude-agent-sdk-python/blob/main/src/claude_agent_sdk/types.py).
- **Subagents** are `AgentDefinition(description, prompt, tools, disallowedTools, model, skills, memory: user|project|local, mcpServers, initialPrompt, maxTurns, background, effort, permissionMode)` — [types.py](https://github.com/anthropics/claude-agent-sdk-python/blob/main/src/claude_agent_sdk/types.py).
- **`ClaudeAgentOptions` fields** include `tools`, `allowed_tools`, `disallowed_tools`, `system_prompt`, `mcp_servers`, `permission_mode`, `can_use_tool`, `hooks`, `agents`, `setting_sources`, `skills`, `sandbox`, `plugins`, `max_turns`, `max_budget_usd`, `task_budget`, `fallback_model`, `thinking`, `effort`, `output_format`, `enable_file_checkpointing`, `session_store`, `fork_session`, `resume` and `include_partial_messages` — [types.py](https://github.com/anthropics/claude-agent-sdk-python/blob/main/src/claude_agent_sdk/types.py).
- The examples directory covers `agents.py`, `filesystem_agents.py`, `hooks.py`, `tool_permission_callback.py`, `mcp_calculator.py`, `max_budget_usd.py`, `plugin_example.py`, `session_stores/`, `setting_sources.py` and `streaming_mode*.py` — [examples/](https://github.com/anthropics/claude-agent-sdk-python/tree/main/examples).

**anthropics/claude-code (proprietary product; repo holds issues, changelog, examples and plugin/mod sources)**
- ~154.3k stars; tag `v2.1.317`. LICENSE.md reads "© Anthropic PBC. All rights reserved … Commercial Terms of Service" — [LICENSE.md](https://github.com/anthropics/claude-code/blob/main/LICENSE.md); [repo](https://github.com/anthropics/claude-code).
- Official plugins have moved to `anthropics/claude-plugins-official`. Plugins bundle "custom slash commands, specialized agents, hooks, and MCP servers" and are installed via `/plugin` marketplaces — [plugins/README.md](https://github.com/anthropics/claude-code/blob/main/plugins/README.md).
- **New "mods":** a plugin whose behaviour is a hooks module with `register(on, options)` hooking engine events as `($, e, next)` middleware. Four built-in mods ship as source:
  - `sec-default` (shields managed policy from user-installed plugins)
  - `diff`
  - `telemetry`
  - `agents-md`, which loads AGENTS.md where no CLAUDE.md exists (default `claude-md-or-agents-md`), beside it (`claude-md-and-agents-md`), `managed-only`, or `claude-md` only
  
  [mods/README.md](https://github.com/anthropics/claude-code/blob/main/mods/README.md)
- Example settings exist for strict, lax, bash-sandbox and HIPAA configurations (`settings-strict.json`, `settings-bash-sandbox.json`, `managed-mcp-hipaa.json`), along with a hook example `bash_command_validator_example.py` — [examples/](https://github.com/anthropics/claude-code/tree/main/examples).
- The v2.1.317 changelog mentions Agent SDK hook callbacks `onFailure: "block"`, auto mode, agent teams, background agents, `/loop`, `/schedule`, compaction fixes ("Fixed very long conversations losing their earlier summary when compaction had to shorten its own request") and OTel attributes — [CHANGELOG.md](https://github.com/anthropics/claude-code/blob/main/CHANGELOG.md).

**anthropics/skills**
- ~186.8k stars. Skills are "folders of instructions, scripts, and resources that Claude loads dynamically", each with a `SKILL.md` that has YAML frontmatter (`name`, `description`). The spec moved to agentskills.io ("The spec is now located at https://agentskills.io/specification"). Many skills are Apache-2.0; the document skills (docx/pdf/pptx/xlsx) are source-available. The repo installs as a plugin marketplace (`/plugin marketplace add anthropics/skills`) — [README](https://github.com/anthropics/skills/blob/main/README.md); [spec/agent-skills-spec.md](https://github.com/anthropics/skills/blob/main/spec/agent-skills-spec.md); [template/SKILL.md](https://github.com/anthropics/skills/blob/main/template/SKILL.md).

**anthropic-experimental/sandbox-runtime (`srt`, Apache-2.0, "Beta Research Preview")**
- `srt` provides "filesystem and network restrictions on arbitrary processes at the OS level, without requiring a container" via `sandbox-exec` (macOS) or `bubblewrap` (Linux), plus "proxy-based network filtering". It works as a CLI or a library, and can wrap MCP servers (`"command": "srt"`) — [README](https://github.com/anthropic-experimental/sandbox-runtime/blob/main/README.md).
- Policy semantics:
  - reads use deny-then-allow (`allowRead` beats `denyRead`, but more-specific denies stay denied)
  - writes are allow-only with `denyWrite` precedence
  - network is "By default, all network access is denied". HTTP(S) goes through an HTTP proxy and other TCP through SOCKS5, both enforcing domain allowlists.
  - Windows uses WFP filters; violation attribution is keyed by `commandId`; an ask-callback can deny with a reason that is surfaced to the model in a `<sandbox_violations>` block
  - `srt proxy` mode works with an external decider
  
  [README](https://github.com/anthropic-experimental/sandbox-runtime/blob/main/README.md). Tag `v0.0.81`, last commit 2026-10-25.

**anthropics/claude-code-action**
- A GitHub Action (tag `v1.0.267`) that runs Claude Code on your own runner. It responds to `@claude` mentions and issue assignments, or runs explicit automation via unified `prompt` and `claude_args` inputs. It auto-detects its mode, tracks progress with checkboxes, and supports structured JSON outputs. Source is under `src/{modes,mcp,github,create-prompt,entrypoints}` — [README](https://github.com/anthropics/claude-code-action/blob/main/README.md); [src/](https://github.com/anthropics/claude-code-action/tree/main/src).

**anthropics/claude-quickstarts (reference harnesses)**
- `autonomous-coding` is "a minimal harness demonstrating long-running autonomous coding with the Claude Agent SDK" using a two-agent pattern:
  - An Initializer agent writes `feature_list.json` (200 test cases), `init.sh`, and a git repo.
  - A Coding agent works in fresh-context sessions that resume from `feature_list.json`, `claude-progress.txt` and git commits, and auto-continue.
  - Security is defense in depth: an OS sandbox, a filesystem restricted to the project dir, and a bash allowlist enforced by a security hook (`security.py`).
  
  [autonomous-coding/README.md](https://github.com/anthropics/claude-quickstarts/blob/main/autonomous-coding/README.md)
- `agents/` is "a minimal educational implementation of LLM agents using the Claude API … NOT an SDK" (tools-in-a-loop plus MCP) — [agents/README.md](https://github.com/anthropics/claude-quickstarts/blob/main/agents/README.md).
- `managed-agents/` holds apps built on "Claude Managed Agents: agents Anthropic runs for you, with server-side sessions, sandboxed tools, and an event stream your app consumes". Examples include assistant-ui, Vercel Chat SDK, AG-UI/CopilotKit, Slack, Linear, Sentry autofix and self-hosted sandboxes; bash is configurable as `always_ask` with resumable approval gates — [managed-agents/README.md](https://github.com/anthropics/claude-quickstarts/blob/main/managed-agents/README.md).

**anthropics/claude-cookbooks**
- `tool_use/context_engineering/context_engineering_tools.ipynb` compares compaction, tool-result clearing and memory as context levers ("context rot") — [notebook](https://github.com/anthropics/claude-cookbooks/blob/main/tool_use/context_engineering/context_engineering_tools.ipynb).
- `tool_use/automatic-context-compaction.ipynb` uses the Python SDK tool runner's `runner.compact_before_next_turn()` to "summarize the history server-side" — [notebook](https://github.com/anthropics/claude-cookbooks/blob/main/tool_use/automatic-context-compaction.ipynb).
- `misc/session_memory_compaction.ipynb` covers proactive background session-memory building for "instant compaction" using threading plus prompt caching (claimed "~80%" cost reduction for the background updates) — [notebook](https://github.com/anthropics/claude-cookbooks/blob/main/misc/session_memory_compaction.ipynb).
- Also present: `tool_use/memory_cookbook.ipynb` and `memory_tool.py` (a client-side memory tool implementation), `programmatic_tool_calling_ptc.ipynb`, `tool_search_with_embeddings.ipynb`, `patterns/agents/` (workflows, orchestrator-workers, evaluator-optimizer, async multi-agent) and `claude_agent_sdk/` notebooks 00–08 (research, chief-of-staff, observability, SRE, migrating from the OpenAI Agents SDK, session browser, hosting, dynamic workflows) — [tool_use/](https://github.com/anthropics/claude-cookbooks/tree/main/tool_use); [claude_agent_sdk/](https://github.com/anthropics/claude-cookbooks/tree/main/claude_agent_sdk); [patterns/agents](https://github.com/anthropics/claude-cookbooks/tree/main/patterns/agents).

### Inferences
What a builder should copy from the Anthropic repos:
- **Embedding protocol.** Anthropic's model is "the harness is a subprocess; the SDK is a protocol client". Copy the control protocol shape: newline-delimited JSON messages plus `control_request`/`control_response`, with `can_use_tool`, `hook_callback` and `mcp_message` callbacks into the host process. It cleanly separates the agent loop from host-supplied policy and tools.
- **Hooks.** The vocabulary `PreToolUse`/`PostToolUse`/`UserPromptSubmit`/`Stop`/`SubagentStop`/`PreCompact`/`PermissionRequest` with `permissionDecision` outputs is now shared with Codex, which makes it a de facto standard to implement (see Q5).
- **Sandbox.** `sandbox-runtime` is the most reusable open-source OS sandbox for agent bash and MCP servers that does not need a container. Its "deny network by default, allow by domain through a local proxy" design and its violation-to-model feedback loop are worth copying directly; it is Apache-2.0 and usable as a library.
- **Long-running work.** The two-agent long-running pattern from the quickstarts uses a feature list as ground truth, a progress file, git checkpoints and fresh context per session. It is the canonical open reference for multi-session work, and it is model-agnostic in principle.
- **Licensing caveat.** Because Claude Code and the TS SDK are under Commercial Terms, a builder cannot fork Anthropic's harness. They can only use it via the SDK or replicate its open interfaces (hooks, skills, settings, plugins, AGENTS.md).

### Gaps
- The Claude Code core (system prompt, built-in tool implementations such as Edit/TodoWrite, compaction thresholds) is closed. No primary source for its internal tool schemas was inspected here; the docs at docs.claude.com / platform.claude.com were not fetched.
- I did not find a separately published Anthropic "long-running agent harness" repo beyond `claude-quickstarts/autonomous-coding`. There may be engineering-blog-only material.
- Star counts for sandbox-runtime, claude-code-action, cookbooks and quickstarts were not fetched.

## Q3. DeepSeek repos and docs: V3.x/V3.2/V4, chat templates with tool calls and thinking, agent templates, DSA kernels, Anthropic-API compatibility

### Takeaway
DeepSeek publishes model and inference code (MIT), kernels (FlashMLA, DeepGEMM, DeepEP, TileLang examples) and Python "encoding" reference scripts for chat formats on Hugging Face. It does not publish an agent harness. Its agent strategy is to train for, and serve compatibility with, third-party harnesses. Its API exposes an Anthropic-format endpoint (`https://api.deepseek.com/anthropic`) so that Claude Code (and OpenCode and others) can drive DeepSeek models. V4 (Pro/Flash, 1M context) reportedly launched as an open-weight preview on 2026-04-24, but primary sources (HF, the API docs, arXiv) were blocked from this session.

### Cited Findings
- **`deepseek-ai/DeepSeek-V3.2-Exp`** (MIT; last commit 2025-11-18): V3.2-Exp "builds upon V3.1-Terminus by introducing DeepSeek Sparse Attention (DSA)". Indexer logit kernels are in DeepGEMM (PR 200), sparse attention kernels in FlashMLA (PR 98), readable TileLang kernels in `tile-ai/tilelang/examples/deepseek_v32`, and SGLang/vLLM have day-0 support — [README](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/README.md).
- V3.2-Exp versus V3.1-Terminus on agentic benchmarks:

  | Benchmark | V3.1-Terminus | V3.2-Exp |
  | :--- | :---: | :---: |
  | BrowseComp | 38.5 | 40.1 |
  | SWE Verified | 68.4 | 67.8 |
  | SWE-bench Multilingual | 57.8 | 57.9 |
  | Terminal-bench | 36.7 | 37.7 |
  | Aider-Polyglot | 76.1 | 74.5 |

  [README](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/README.md)
- On 2025-11-17, DeepSeek disclosed an inference-demo bug: the RoPE layout in the DSA indexer must be non-interleaved, while MLA RoPE is interleaved — [README "Update"](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/README.md).
- The inference demo is `inference/{model.py,kernel.py,generate.py,convert.py,config_671B_v3.2.json}`, launched via torchrun — [inference/README.md](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/README.md).
- `github.com/deepseek-ai/DeepSeek-V3.2` resolves to the same HEAD commit as V3.2-Exp (`87e509a`). Probing for `deepseek-ai/DeepSeek-V4`, `DeepSeek-V4-Pro`, `DeepSeek-V3.1` and `DeepSeek-Agent` on GitHub returned no public repo. Repos that do exist include FlashMLA, DeepGEMM, DeepEP, 3FS, DualPipe, smallpond, Engram, DeepSeek-OCR, DeepSeek-Math-V2, DeepSeek-Prover-V2 and awesome-deepseek-integration — [git ls-remote probes against github.com/deepseek-ai](https://github.com/deepseek-ai).
- **V3.2 chat template** (per mirrors of the HF model card, since the official HF page was blocked):
  - It introduced "a revised format for tool calling and … 'thinking with tools'".
  - An `encoding/` folder provides Python scripts and tests, with `encode_messages` and `parse_message_from_completion_text`, configured as `encode_config = dict(thinking_mode="thinking", drop_thinking=True, add_default_bos_token=True)`.
  - A `developer` role exists "only for search-agent scenarios" and is not accepted by the official API.
  - The parser "is designed to handle well-formatted strings only".
  - V3.2-Speciale does not support tool calling.
  
  [HF mirror of DeepSeek-V3.2 card](https://huggingface.co/Deniss8686/DeepSeek-V3.2); [official commit page](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/commit/ecea77f188a446a72e5edbc4b41de9c3764be7f0) (search-summarized; unverified)
- **V4 encoding** (search-summarized from the HF `encoding/README.md` of DeepSeek-V4 repos):
  - Thinking mode emits reasoning before tool calls or answers; chat mode closes the thinking block immediately.
  - `drop_thinking` defaults to True but is automatically disabled when tools are defined, so that tool-calling conversations keep the full reasoning.
  - Tool calls use a DSML `tool_calls` block with named invoke/parameter elements; `string="false"` marks JSON-typed params.
  - Tool results are wrapped in `tool_result` tags inside user messages.
  - Three reasoning-effort levels are set as text prefixes.
  - The reference file is `encoding_dsv4.py`.
  
  [DeepSeek-V4 Encoding README (HF)](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark/raw/main/encoding/README.md); [DeepSeek-V4-Flash encoding](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/encoding/README.md) (search-summarized; unverified)
- **V4 release** (secondary sources only): an open-weight preview on 2026-04-24 with V4-Pro (reported ~1.6T total / 49B active) and V4-Flash (~284B / 13B active), both with 1M context and MIT weights. Sources conflict on parameter counts and pricing, and on SWE-bench Verified (80.6 "vendor-reported" versus ~91.2) — [letsdatascience](https://letsdatascience.com/news/deepseek-v4-preview-adds-open-weight-1m-context-models-4a8f21c6); [framia](https://framia.converge.ai/page/en-US/news/deepseek-v4-announcement-april-2026); [mindstudio review](https://www.mindstudio.ai/blog/deepseek-v4-open-source-frontier-model-review); [framia coding guide](https://framia.converge.ai/page/en-US/news/deepseek-v4-coding).
- Secondary coverage says V4 shipped "with documented support for Claude Code, OpenClaw, OpenCode, and CodeBuddy", integrated by swapping the base URL, and that DeepSeek runs V4-Pro as its internal coding agent — [handyai substack](https://handyai.substack.com/p/model-drop-deepseek-v4); [atlascloud](https://www.atlascloud.ai/blog/deepseek-v4-preview-launch).
- **Anthropic-API compatibility** (from search-engine summaries of the official docs page, which was blocked):
  - The base URL is `https://api.deepseek.com/anthropic`, used with Claude Code via `ANTHROPIC_BASE_URL` plus `ANTHROPIC_API_KEY` (community guides also use `ANTHROPIC_AUTH_TOKEN`).
  - `anthropic-beta` and `anthropic-version` headers are ignored.
  - The Chinese docs reportedly map `claude-opus*` → `deepseek-v4-pro`, and `claude-sonnet*`/`claude-haiku*` and unknown names → `deepseek-v4-flash`.
  - A third-party summary says MCP tool-use blocks are unsupported.
  
  [api-docs.deepseek.com anthropic_api](https://api-docs.deepseek.com/guides/anthropic_api); [zh-cn version](https://api-docs.deepseek.com/zh-cn/guides/anthropic_api); [developersdigest compatibility limits](https://www.developersdigest.tech/reference/api-compatibility/deepseek); [aiengineerguide TIL](https://aiengineerguide.com/til/deepseek-llm-in-claude-code/)
- Aider's changelog added "DeepSeek Reasoner model and updated DeepSeek model metadata with costs and prompt caching". DeepSeek is a first-class provider in third-party harnesses rather than shipping its own — [aider HISTORY.md](https://github.com/Aider-AI/aider/blob/main/HISTORY.md).

### Inferences
What a builder should take from DeepSeek:
- **A cheap model-agnostic test.** The pattern "Anthropic Messages-compatible endpoint plus model-name aliasing" means that a harness written against the Anthropic Messages API (or using `ANTHROPIC_BASE_URL`) can run DeepSeek V3.2/V4 with no code changes. Supporting a configurable base URL plus model-alias mapping is therefore the cheapest way to make a harness model-agnostic.
- **For self-hosted DeepSeek weights,** do not hand-roll chat templates. Use DeepSeek's `encoding_dsv32.py`/`encoding_dsv4.py` reference encoders (and the vLLM/SGLang parsers built on them).
- **Reasoning retention matters.** The rule "keep reasoning across tool calls, drop it at a new user turn" (which `drop_thinking` approximates) is a model-specific context-management requirement the harness must honour.
- **DSA is not a harness concern.** It mainly lowers the cost of long agent trajectories at the serving layer. Its relevance to harness design is cheaper long contexts, which makes less aggressive compaction viable for self-hosted DeepSeek.

### Gaps
- huggingface.co, api-docs.deepseek.com and arxiv.org were all blocked, so I could not verify the following from primary sources:
  - the exact DSML token syntax
  - the V3.1 "search agent" / "code agent" trajectory templates
  - the V3.2 tech report's agentic-task-synthesis numbers
  - the V4 tech report
  - the current API compatibility table
  
  All DeepSeek V3.2/V4 template and API details above come from search summaries.
- I found no evidence of a DeepSeek-authored agent framework or harness release in 2025–2026. This is absence of evidence from name probes and searches, not proof.

## Q4. Other influential open-source harnesses: architecture, loop, tools, context management, innovations, license, maturity

### Takeaway
By late 2026 the field has consolidated:
- **Leaders by usage and activity:** OpenCode (TS, MIT), Gemini CLI (TS, Apache), Pi (TS, MIT), OpenHands (Python, MIT; now an "Agent Canvas" over ACP agents plus a separate Software Agent SDK), Cline (TS, Apache; now with an SDK), Goose (Rust, Apache; moved to the Linux Foundation AAIF), Qwen Code (a Gemini CLI fork), Crush (Go, FSL), deepagents (Python, MIT) and mini-swe-agent (the research baseline).
- **Archived or shut down:** Roo Code (shut down 2026-05-15) and Kimi CLI (Python), which was replaced by Kimi Code CLI (TS, MIT).
- **Slowing:** Aider (no tagged release since v0.86.x, ~Aug 2025).
- **Effectively in maintenance:** SWE-agent; its authors point users to mini-swe-agent.
- **Not open source:** Amp.

### Cited Findings

**OpenHands (All-Hands-AI → OpenHands org)**
- The main repo is now "Agent Canvas: the self-hosted developer control center for coding agents and automations". It runs "OpenHands, Claude Code, Codex, Gemini, Pi, OpenCode, or any ACP-compatible agent across local, remote, and cloud backends". ~92.5k stars; tag `v1.34.0`; MIT — [OpenHands README](https://github.com/OpenHands/OpenHands/blob/main/README.md); [GitHub page](https://github.com/OpenHands/OpenHands).
- **Software Agent SDK** (`OpenHands/software-agent-sdk`, MIT, tag `v1.59.0`):
  - It provides Python, TypeScript and REST APIs. The packages are `openhands-sdk`, `openhands-tools`, `openhands-workspace` and `openhands-agent-server`.
  - It is "the engine behind the OpenHands CLI and OpenHands Cloud".
  - It runs locally or in Docker/Kubernetes workspaces via the Agent Server.
  - The README badge claims SWEBench 77.6, and the tech report is arXiv 2511.03690.
  
  [software-agent-sdk README](https://github.com/OpenHands/software-agent-sdk/blob/main/README.md)
- SDK modules include `agent`, `context` (condenser, view, memory), `conversation` (event_store, stuck_detector, goal), `event`, `critic`, `security` (LLM analyzer, shell AST parser, confirmation policy, defense_in_depth), `hooks`, `skills`, `subagent`, `mcp`, `plugin`, `marketplace` and `workspace` — [openhands/sdk](https://github.com/OpenHands/software-agent-sdk/tree/main/openhands-sdk/openhands/sdk).
- Tools include `terminal`, `file_editor`, `apply_patch`, `glob`, `grep`, `browser_use`, `task_tracker`, `delegate`, `task`, `child_conversation`, `planning_file_editor`, `ask_oracle` and `tom_consult` — [openhands-tools](https://github.com/OpenHands/software-agent-sdk/tree/main/openhands-tools/openhands/tools).
- **Event-stream and condenser design:**
  - The conversation is "an append-only event log". Condensation is recorded as `Condensation` events ("similar to tombstones in … Cassandra and Kafka"). A `View` applies them to produce the LLM-visible context.
  - The default `LLMSummarizingCondenser` replaces "the first half of all events with a single summary event", with `max_size=240` events, `keep_first=2` and an optional `max_tokens`.
  - Condensation is either "soft" (resource limit; it can be skipped if it would break message-structure invariants) or "hard" (explicitly requested, for example after a context-window error).
  - The README states the rationale explicitly: regular condensation keeps prompt-cache rebuild costs low.
  
  [condenser/README.md](https://github.com/OpenHands/software-agent-sdk/blob/main/openhands-sdk/openhands/sdk/context/condenser/README.md); [llm_summarizing_condenser.py](https://github.com/OpenHands/software-agent-sdk/blob/main/openhands-sdk/openhands/sdk/context/condenser/llm_summarizing_condenser.py)
- A `StuckDetector` scans recent events for repetitive patterns (for example 4 repeats × 2 events per cycle) — [stuck_detector.py](https://github.com/OpenHands/software-agent-sdk/blob/main/openhands-sdk/openhands/sdk/conversation/stuck_detector.py).

**SWE-agent and mini-swe-agent (Princeton/Stanford, MIT)**
- SWE-agent (tag `v1.1.0`, last commit 2026-07-16) says "Most of our current development effort is on mini-swe-agent … Our general recommendation is to use mini-SWE-agent instead of SWE-agent going forward". Its agent-computer interface (ACI) ships as tool bundles in `tools/`: `windowed`, `windowed_edit_linting`, `windowed_edit_replace`, `windowed_edit_rewrite`, `edit_anthropic`, `search`, `filemap`, `submit`, `review_on_submit_m`, `diff_state`, `web_browser` — [SWE-agent README](https://github.com/SWE-agent/SWE-agent/blob/main/README.md); [tools/](https://github.com/SWE-agent/SWE-agent/tree/main/tools).
- mini-swe-agent: ~9.0k stars; tag `v2.4.6`. Its README makes these design claims:
  - "Just some 100 lines of python for the agent class"; `agents/default.py` is 190 lines including config
  - "Scores >74% on the SWE-bench verified benchmark"; the July 2025 version scored 65%
  - "Does not have any tools other than bash — it doesn't even need to use the tool-calling interface"
  - "completely linear history … no difference between the trajectory and the messages"
  - actions run via "`subprocess.run` — every action is completely independent (as opposed to keeping a stateful shell session)", so sandboxing is just swapping in `docker exec`; environments are `local.py`, `docker.py` and `singularity.py`
  - it powers the SWE-bench "bash only" leaderboard
  
  [mini-swe-agent README](https://github.com/SWE-agent/mini-swe-agent/blob/main/README.md); [agents/default.py](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/agents/default.py); [environments/local.py](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/environments/local.py); [SWE-agent README news](https://github.com/SWE-agent/SWE-agent/blob/main/README.md)

**Aider (Apache-2.0)**
- ~49.9k stars. Latest tag `v0.86.2`; the v0.86.0 analytics timestamps are from Aug 2025. The main branch has unreleased changes, with the last commit on 2026-05-22 (adding Claude 4.5–4.7, GPT-5.x and Gemini 3 aliases) — [HISTORY.md](https://github.com/Aider-AI/aider/blob/main/HISTORY.md); [GitHub page](https://github.com/Aider-AI/aider).
- **Repo map:** `aider/repomap.py` uses tree-sitter tag queries (`aider/queries/tree-sitter-language-pack`) and ranks symbols with a token budget (`map_tokens=1024` by default, scaled when no files are in chat) — [repomap.py](https://github.com/Aider-AI/aider/blob/main/aider/repomap.py).
- **Edit formats:**
  - `whole`, `diff` (SEARCH/REPLACE blocks), `diff-fenced`, `udiff`
  - `editor-diff`/`editor-whole`, used by the architect/editor two-model split
  - a `patch_coder.py`, an apply_patch-style coder
  
  [edit-formats doc](https://github.com/Aider-AI/aider/blob/main/aider/website/docs/more/edit-formats.md); [aider/coders](https://github.com/Aider-AI/aider/tree/main/aider/coders)

**OpenCode (sst/opencode → anomalyco/opencode; MIT)**
- ~224.1k stars (as displayed on the page fetch). The README badges point to `anomalyco/opencode`. `@opencode/core` is at version 2.2.5 (tags `v2.2.x`). The TypeScript/Bun monorepo uses Effect plus drizzle/SQLite; packages include `core`, `server`, `tui`, `desktop`, `sdk`, `plugin`, `codemode` and `protocol` — [README](https://github.com/sst/opencode/blob/dev/README.md); [packages/core/package.json](https://github.com/sst/opencode/blob/dev/packages/core/package.json); [GitHub anomalyco/opencode](https://github.com/anomalyco/opencode).
- It has two built-in primary agents switched with Tab: `build` (full access) and `plan` (read-only, denies edits, asks before bash). There is also a `general` subagent invoked as `@general` — [README](https://github.com/sst/opencode/blob/dev/README.md).
- Its `patch` tool uses the Codex envelope verbatim ("*** Begin Patch … *** Add File / Update File / Delete File / Move to … *** End Patch"). There is also a `todowrite` tool. Core modules include `session/compaction.ts`, `session/subagent-job.ts`, `instruction-discovery.ts`, `permission.ts`, `snapshot.ts`, `worktree.ts`, `persistent-pty.ts` and `lsp` handling — [packages/core/src/tool/patch.txt](https://github.com/sst/opencode/blob/dev/packages/core/src/tool/patch.txt); [packages/core/src](https://github.com/sst/opencode/tree/dev/packages/core/src).

**Goose (block/goose → aaif-goose/goose; Apache-2.0; Rust)**
- ~56.2k stars. Stable tag `v1.57.0`, with v2.0 release candidates in progress. Goose is "a general-purpose AI agent" (desktop, CLI and API) with "15+ providers", ACP providers that let it use "Claude, ChatGPT, or Gemini subscriptions", and "70+ extensions via MCP". It "is part of the Agentic AI Foundation (AAIF) at the Linux Foundation". Crates include `goose-agent`, `goose-context-management`, `goose-mcp`, `goose-providers`, `goose-sdk` and `goose-acp-macros` — [README](https://github.com/block/goose/blob/main/README.md); [crates/](https://github.com/block/goose/tree/main/crates); [GitHub aaif-goose/goose](https://github.com/aaif-goose/goose).

**Cline (Apache-2.0; TypeScript)**
- ~71.2k stars; tag `v4.1.26`. Cline now ships as a CLI (headless for CI), desktop app, VS Code and JetBrains extensions, and an SDK — [README](https://github.com/cline/cline/blob/main/README.md).
- **SDK layering:** `@cline/shared` → `@cline/llms` → `@cline/agents` (the "stateless runtime loop": iteration, tool orchestration, events, hooks) → `@cline/core` (sessions, persistence, compaction policy, RPC) → host apps.
- **Compaction design:** the canonical session history is kept at full fidelity, while compaction state lives in a separate `${sessionId}.compaction.json` that is reused on resume only after validating a hash of the canonical prefix. Automatic compaction requires a known context window ("unknown custom-model limits do not trigger compaction").
- [sdk/ARCHITECTURE.md](https://github.com/cline/cline/blob/main/sdk/ARCHITECTURE.md)

**Roo Code (Apache-2.0) — shut down**
- The last commit, "Remove roocode.com web app", is dated 2026-05-15; the last tag is `v3.54.0` — [repo](https://github.com/RooCodeInc/Roo-Code).
- The sunset was announced around 2026-04-20/21, and the extension, Cloud and Router ended on 2026-05-15. The team pivoted to "Roomote". Third-party sources point users to Cline or the community fork "Zoo Code" — [distillintelligence](https://www.distillintelligence.com/news/roomote); [kilo.ai](https://kilo.ai/compare/roo-code-shutdown-roomote); [thenewstack](https://thenewstack.io/roo-code-cloud-ides-ai-coding/).

**Gemini CLI (Google; Apache-2.0; TypeScript)**
- ~108.7k stars. Stable `v0.66.0`; `v0.67.0-preview.0`; nightly builds. Packages: `cli`, `core`, `sdk`, `a2a-server`, `vscode-ide-companion`, `devtools` — [repo](https://github.com/google-gemini/gemini-cli); [packages/](https://github.com/google-gemini/gemini-cli/tree/main/packages).
- **Tools:** `read-file`, `read-many-files`, `write-file`, `edit` (`old_string`/`new_string`), `glob`, `grep`/`ripGrep`, `ls`, `shell` plus `shellBackgroundTools`, `web-fetch`, `web-search`, `memoryTool` (`GEMINI.md` is the default context filename), `write-todos`, `trackerTools`, `enter-plan-mode`/`exit-plan-mode`, `activate-skill`, `ask-user`, `complete-task` and MCP resource tools — [core/src/tools](https://github.com/google-gemini/gemini-cli/tree/main/packages/core/src/tools); [memoryTool.ts](https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/tools/memoryTool.ts).
- **Hook events** (a different vocabulary from Claude/Codex): `BeforeTool`, `AfterTool`, `BeforeAgent`, `AfterAgent`, `BeforeModel`, `AfterModel`, `BeforeToolSelection`, `PreCompress`, `SessionStart`, `SessionEnd`, `Notification` — [hooks/types.ts](https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/hooks/types.ts).
- **Policy and agents:** a TOML policy engine (`policies/{read-only,write,plan,yolo,sandbox-default,non-interactive,agents,conseca}.toml`); built-in subagents (`codebase-investigator`, `generalist-agent`, `cli-help-agent`); and remote agents over A2A (`a2a-client-manager.ts`) — [policy/policies](https://github.com/google-gemini/gemini-cli/tree/main/packages/core/src/policy/policies); [core/src/agents](https://github.com/google-gemini/gemini-cli/tree/main/packages/core/src/agents).
- **Context management:** `DEFAULT_COMPRESSION_TOKEN_THRESHOLD = 0.5` and `COMPRESSION_PRESERVE_THRESHOLD = 0.3` (keep the latest 30%), plus a `toolOutputMaskingService`, `toolDistillationService` and a context pipeline with processors — [chatCompressionService.ts](https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/context/chatCompressionService.ts); [context/](https://github.com/google-gemini/gemini-cli/tree/main/packages/core/src/context).

**Qwen Code (Alibaba; Apache-2.0)**
- ~28.9k stars; `v0.26.6`. "Originally based on Google Gemini CLI v0.8.2 … Starting from Qwen Code v0.1, we stopped syncing with upstream". It is multi-protocol (OpenAI, Anthropic, Gemini and Qwen APIs) and markets "Auto-Memory, Auto-Skills, SubAgents, Agent Teams, and MCP" — [README](https://github.com/QwenLM/qwen-code/blob/main/README.md). Its instruction-file names include QWEN.md, AGENTS.md, CLAUDE.md and GEMINI.md — [packages/core/src](https://github.com/QwenLM/qwen-code/tree/main/packages/core/src).

**Kimi CLI → Kimi Code CLI (Moonshot)**
- `MoonshotAI/kimi-cli` (Python) reads: "This project has been archived and is no longer maintained … replaced by Kimi Code CLI" — [kimi-cli README](https://github.com/MoonshotAI/kimi-cli/blob/main/README.md).
- `MoonshotAI/kimi-code` is TypeScript, MIT, with its last commit on 2026-11-03. Packages include `agent-core-v2`, `acp-server`, `kap-server`, `node-sdk`, `session-import`, `transcript`, `tree-sitter-bash` and `pi-tui`. It has built-in `coder`/`explore`/`plan` subagents, lifecycle hooks, a plugin marketplace with trust levels, and `kimi acp` for Zed/JetBrains — [kimi-code README](https://github.com/MoonshotAI/kimi-code/blob/main/README.md); [packages/](https://github.com/MoonshotAI/kimi-code/tree/main/packages).

**Crush (Charm; FSL-1.1 "Functional Source License"; Go)**
- ~28.9k stars; `v0.99.0`. It is built on `charm.land/fantasy` (agent/LLM library) and `catwalk` (model catalog). Features: "LSP-Enhanced", "switch LLMs mid-session while preserving context", and MCP over http/stdio/sse — [README](https://github.com/charmbracelet/crush/blob/main/README.md); [go.mod](https://github.com/charmbracelet/crush/blob/main/go.mod); [LICENSE](https://github.com/charmbracelet/crush/blob/main/LICENSE.md).
- Tools include `bash`, `edit`, `multiedit`, `glob`, `grep`, `ls`, `fetch`, `download`, `diagnostics`, `job_output`/`job_kill`, `question`, and LSP tools (`lsp_definition`, `lsp_rename`, `lsp_replace_symbol`, `lsp_symbols`, `lsp_call_hierarchy`). Tool descriptions are `.md`/`.md.tpl` templates next to the Go code — [internal/agent/tools](https://github.com/charmbracelet/crush/tree/main/internal/agent/tools).

**Pi (badlogic/pi-mono → earendil-works/pi; MIT; TypeScript)**
- ~119.6k stars. Description: "AI agent toolkit: unified LLM API, agent loop, TUI, coding agent CLI". Pi is "a minimal, extensible agent harness … ships with powerful defaults but skips features like sub-agents and plan mode". It is extended via extensions, skills, prompt templates and packages, and runs interactively, in print/JSON mode, over RPC, or via a TS SDK — [coding-agent README](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/README.md); [GitHub earendil-works/pi](https://github.com/earendil-works/pi).
- The default coding toolset is exactly four tools: `read`, `bash`, `edit`, `write`. The read-only set is `read`, `grep`, `find`, `ls` — [tools/index.ts](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/src/core/tools/index.ts).
- OpenHands Agent Canvas lists Pi as a supported ACP agent, and Kimi Code uses `pi-tui` — [OpenHands README](https://github.com/OpenHands/OpenHands/blob/main/README.md); [kimi-code packages](https://github.com/MoonshotAI/kimi-code/tree/main/packages).

**LangChain deepagents (MIT; Python, plus deepagents.js)**
- ~30.9k stars; `deepagents==0.7.32`. "The batteries-included agent harness", built on LangGraph (streaming, persistence, checkpointing). It includes sub-agents with isolated context, a pluggable filesystem backend, summarization and offloading of tool outputs to disk, shell, persistent memory, HITL, skills and MCP. A terminal coding agent, "Deep Agents Code", is also available — [README](https://github.com/langchain-ai/deepagents/blob/main/README.md).
- Implementation is middleware-based: `summarization.py`, `_blob_offload.py`, `_message_eviction.py`, `_overflow_clip.py`, `_prompt_caching.py`, `subagents.py`, `async_subagents.py`, `skills.py`, `memory.py`, `permissions.py`, `patch_tool_calls.py`. Backends are `filesystem`, `local_shell`, `sandbox`, `state`, `store` and `composite`. The libs include `acp` — [middleware/](https://github.com/langchain-ai/deepagents/tree/main/libs/deepagents/deepagents/middleware); [backends/](https://github.com/langchain-ai/deepagents/tree/main/libs/deepagents/deepagents/backends); [libs/](https://github.com/langchain-ai/deepagents/tree/main/libs).

**smolagents (Hugging Face; Apache-2.0)**
- Tag `v1.26.0`; last commit 2026-10-06. "The logic for agents fits in ~1,000 lines". `CodeAgent` "writes its actions in code" (Python snippets), alongside a standard `ToolCallingAgent`. Sandboxes are E2B, Blaxel and Modal (plus Docker/WASM options). It warns that "`LocalPythonExecutor` is not a security sandbox" — [README](https://github.com/huggingface/smolagents/blob/main/README.md).

**Letta (MemGPT) → Letta Code (Apache-2.0)**
- `letta-ai/letta` now says "The current source code lives in letta-ai/letta-code"; the server's last tag is `0.16.8` — [letta README](https://github.com/letta-ai/letta/blob/main/README.md).
- Letta Code is "a stateful agent harness": memory blocks, system-prompt learning, MemFS ("All context (including memory blocks) is tracked via git"), skills (`~/.letta`, `.agents/skills`), subagents (general-purpose, forked, recall), hooks, permissions, crons/heartbeats, and channels (Slack, Telegram, Discord) — [letta-code README](https://github.com/letta-ai/letta-code/blob/main/README.md).

**Amp (Sourcegraph → Amp Inc.)**
- Proprietary and closed-source. It was spun out of Sourcegraph in Dec 2025; a rebuilt CLI ("Neo") began rollout in May 2026 — [tessl](https://tessl.io/blog/sourcegraph-spins-out-ai-coding-agent-amp-as-a-standalone-company/); [latent.space wiki](https://latent.space/wiki/amp-code/); [amplifying.ai](https://amplifying.ai/coding-agents/amp).

**Summary table** (stars from page fetches on 2026-11-03, approximate)

| Harness | Lang | License | Stars | Latest tag | Status |
|---|---|---|---|---|---|
| OpenAI Codex | Rust | Apache-2.0 | 132.9k | rust-v0.166.0 | Very active |
| OpenCode | TS/Bun | MIT | 224.1k | v2.2.5 | Very active; repo now anomalyco/opencode |
| Claude Code | TS (closed) | Commercial ToS | 154.3k | v2.1.317 | Very active; not open source |
| Pi | TS | MIT | 119.6k | — | Very active; earendil-works/pi |
| Gemini CLI | TS | Apache-2.0 | 108.7k | v0.66.0 | Very active |
| OpenHands (Agent Canvas) / SDK | Py/TS | MIT | 92.5k | v1.34.0 / SDK v1.59.0 | Very active; pivot to multi-agent control center |
| Cline | TS | Apache-2.0 | 71.2k | v4.1.26 | Active; SDK split |
| Goose | Rust | Apache-2.0 | 56.2k | v1.57.0 (v2 RC) | Active; Linux Foundation AAIF |
| Aider | Python | Apache-2.0 | 49.9k | v0.86.2 | Slowing; no release since ~Aug 2025 |
| deepagents | Python | MIT | 30.9k | 0.7.32 | Active |
| OpenAI Agents SDK (py) | Python | MIT | 30.5k | v0.24.0 | Active |
| Qwen Code | TS | Apache-2.0 | 28.9k | v0.26.6 | Active; Gemini CLI fork |
| Crush | Go | FSL-1.1 | 28.9k | v0.99.0 | Active; source-available, not OSI |
| mini-swe-agent | Python | MIT | 9.0k | v2.4.6 | Active research baseline |
| Claude Agent SDK (py) | Python | MIT | 8.4k | v0.2.177 | Very active |
| Kimi Code CLI | TS | MIT | n/a | — | Active (replaces archived kimi-cli) |
| Roo Code | TS | Apache-2.0 | n/a | v3.54.0 | Shut down 2026-05-15 |
| SWE-agent | Python | MIT | n/a | v1.1.0 | Maintenance; superseded by mini |

### Inferences
What a builder should copy from each:
- **mini-swe-agent:** the minimal baseline and eval harness (bash-only, linear history, stateless subprocess actions, a swappable environment). Build your harness so it can be ablated down to this; it is the control condition.
- **OpenHands SDK:**
  - an append-only event log plus tombstone-style condensation events plus a "View" projection
  - halving condensation with `keep_first` and prompt-cache-aware cadence
  - a stuck detector and a security analyzer
  
  This is the best open reference for durable, replayable agent state.
- **Cline SDK:** the clean "stateless loop package vs stateful core" layering, and canonical transcript plus separately persisted, hash-validated compaction state. Together these are the best pattern for resumable compaction without losing the original transcript.
- **Gemini CLI:** a TOML policy engine with named profiles (plan, read-only, yolo, non-interactive), model-level hooks (`BeforeModel`/`AfterModel`/`BeforeToolSelection`) that Claude/Codex lack, tool-output masking/distillation, and A2A remote subagents.
- **Aider:** the tree-sitter repo map and the empirical edit-format matrix, which remain the reference for edit-format and model matching and for weak-model support.
- **Crush:** LSP-as-tools (rename, definition, diagnostics) is a distinctive idea worth copying for refactor-heavy work.
- **Pi:** strong evidence (high adoption) that a 4-tool core plus extensibility is competitive. Plan mode and subagents can be add-ons rather than core.
- **deepagents:** a middleware stack as the extension mechanism (summarization, offload-to-filesystem, eviction, overflow clipping) is the best Python reference for "context management as composable middleware".
- **Letta Code:** git-tracked memory (MemFS) and self-editing memory blocks are the reference for persistent, learning agents. This goes beyond per-session compaction.
- **Maturity signals:** Roo Code's shutdown and Kimi CLI's archival show that IDE-extension forks and single-vendor CLIs churn. Building on protocols (ACP, MCP, AGENTS.md, Agent Skills) is the hedge.

### Gaps
- The OpenHands arXiv tech report (2511.03690) and SWE-agent/ACI papers were not fetched (arxiv blocked). The SWE-bench numbers above (OpenHands 77.6, mini >74%) are the repos' own claims, not independently verified.
- I did not inspect Goose's context-management crate internals, Qwen Code's divergences in depth, or Kimi Code's agent-core-v2 loop.
- I could not confirm the exact OpenCode org move date or Pi repo-move details beyond README links.
- Star counts for Kimi Code, Letta Code, smolagents and SWE-agent were not fetched.

## Q5. Cross-cutting: which design decisions have converged, and where they diverge

### Takeaway
There is strong convergence on:
- a small core toolset (shell plus read/edit/write plus grep/glob)
- a todo/plan tool
- subagents with isolated context
- AGENTS.md as the instruction file, now Linux-Foundation-stewarded
- MCP as the tool-extension bus
- Agent Skills (SKILL.md) as the capability-packaging format
- LLM-summary compaction
- hook events with Claude-Code-compatible names (Codex adopted the same vocabulary)
- OS-level sandboxes (Seatbelt/bubblewrap) plus approval policies
- headless JSON-streaming modes and an SDK

Divergence persists on:
- **edit format:** apply_patch envelope vs str_replace vs SEARCH/REPLACE vs bash-only
- **shell model:** stateful PTY vs stateless subprocess
- **client protocol:** Codex app-server JSON-RPC vs Claude stream-JSON control protocol vs ACP vs A2A
- **implementation language:** Rust vs TS vs Python vs Go
- **where state and compaction live:** client-side vs provider-side `responses.compact` / Anthropic server-side compaction / Managed Agents

### Cited Findings

**AGENTS.md convergence**
- "AGENTS.md is now stewarded by the Agentic AI Foundation under the Linux Foundation" — [agents.md AboutSection](https://github.com/agentsmd/agents.md/blob/main/components/AboutSection.tsx). Goose is also in AAIF — [goose README](https://github.com/block/goose/blob/main/README.md).
- Codex loads hierarchical AGENTS.md files plus `AGENTS.override.md` — [agents_md.rs](https://github.com/openai/codex/blob/main/codex-rs/core/src/agents_md.rs). Claude Code loads AGENTS.md through its built-in `agents-md` mod — [mods/README.md](https://github.com/anthropics/claude-code/blob/main/mods/README.md).
- Instruction-file names found in source, by harness:

  | Harness | Instruction files |
  |---|---|
  | OpenCode | AGENTS.md |
  | Kimi Code | AGENTS.md |
  | Pi | AGENTS.md, CLAUDE.md |
  | Cline | AGENTS.md, .clinerules |
  | Goose | AGENTS.md, CLAUDE.md, .goosehints |
  | Crush | AGENTS.md, CLAUDE.md, CRUSH.md, GEMINI.md |
  | Qwen Code | AGENTS.md, QWEN.md, CLAUDE.md, GEMINI.md |
  | Gemini CLI | GEMINI.md (default, configurable) |

  [opencode core](https://github.com/sst/opencode/tree/dev/packages/core/src); [kimi-code packages](https://github.com/MoonshotAI/kimi-code/tree/main/packages); [pi coding-agent](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent/src); [cline sdk](https://github.com/cline/cline/tree/main/sdk/packages); [goose crates](https://github.com/block/goose/tree/main/crates); [crush internal](https://github.com/charmbracelet/crush/tree/main/internal); [qwen-code core](https://github.com/QwenLM/qwen-code/tree/main/packages/core/src); [gemini memoryTool.ts](https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/tools/memoryTool.ts)

**Hooks convergence**
- Codex hooks use `PreToolUse`/`PostToolUse`/`SessionStart`/`UserPromptSubmit`/`Stop`/`PreCompact`/`PostCompact`/`SubagentStop`/`PermissionRequest` — [codex hooks](https://github.com/openai/codex/tree/main/codex-rs/hooks/src). These match the Claude Agent SDK's `HookEvent` names — [types.py](https://github.com/anthropics/claude-agent-sdk-python/blob/main/src/claude_agent_sdk/types.py).
- Gemini CLI diverges with `BeforeTool`/`AfterTool`/`BeforeModel`/`AfterModel`/`BeforeToolSelection`/`PreCompress` — [hooks/types.ts](https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/hooks/types.ts).
- Hooks are also listed as features by Kimi Code, Letta Code, OpenHands SDK (`hooks/`) and Cline (extension/hook system) — [kimi-code README](https://github.com/MoonshotAI/kimi-code/blob/main/README.md); [letta-code README](https://github.com/letta-ai/letta-code/blob/main/README.md); [openhands sdk](https://github.com/OpenHands/software-agent-sdk/tree/main/openhands-sdk/openhands/sdk); [cline ARCHITECTURE.md](https://github.com/cline/cline/blob/main/sdk/ARCHITECTURE.md).

**Skills convergence**
- SKILL.md folders, with the spec at agentskills.io — [anthropics/skills](https://github.com/anthropics/skills/blob/main/README.md).
- Adopted by:
  - Codex (`skills` crate, `skills/list` RPC) — [codex-rs](https://github.com/openai/codex/tree/main/codex-rs)
  - OpenAI Agents SDK `Skills` capability (lazy-loaded) — [sandbox_agents.md](https://github.com/openai/openai-agents-python/blob/main/docs/sandbox_agents.md)
  - Gemini CLI `activate-skill` — [tools](https://github.com/google-gemini/gemini-cli/tree/main/packages/core/src/tools)
  - deepagents `skills.py` — [middleware](https://github.com/langchain-ai/deepagents/tree/main/libs/deepagents/deepagents/middleware)
  - Letta Code `.agents/skills` — [letta-code README](https://github.com/letta-ai/letta-code/blob/main/README.md)
  - Pi skills — [pi README](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/README.md)
  - OpenHands SDK `skills/` — [sdk](https://github.com/OpenHands/software-agent-sdk/tree/main/openhands-sdk/openhands/sdk)
  - Crush `internal/skills` — [crush](https://github.com/charmbracelet/crush/tree/main/internal)

**Todo/plan tools**
- Codex `update_plan`, OpenCode `todowrite`, Gemini CLI `write-todos` plus plan mode, OpenHands `task_tracker`, and OpenCode's read-only `plan` agent — [codex handlers](https://github.com/openai/codex/tree/main/codex-rs/core/src/tools/handlers); [opencode tool](https://github.com/sst/opencode/tree/dev/packages/core/src/tool); [gemini tools](https://github.com/google-gemini/gemini-cli/tree/main/packages/core/src/tools); [openhands-tools](https://github.com/OpenHands/software-agent-sdk/tree/main/openhands-tools/openhands/tools).
- Pi deliberately omits plan mode — [pi README](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/README.md).

**Subagents**
- Codex `spawn_agent`/`wait_agent`/…; Claude `AgentDefinition`; Gemini CLI built-in subagents plus A2A; OpenCode `@general`; Kimi Code `coder`/`explore`/`plan`; deepagents `subagents.py`/`async_subagents.py`; OpenHands `delegate`/`task`; Letta Code forked/recall subagents. Pi omits them by design. (All sources cited in Q1, Q2 and Q4.)

**Edit formats (main divergence)**
- **apply_patch envelope:** Codex (Lark grammar, freeform), OpenCode `patch`, OpenAI Agents SDK sandbox `apply_patch.py`, gpt-oss `apply_patch`, OpenHands `apply_patch` tool, Aider `patch_coder.py` — [codex parser.rs](https://github.com/openai/codex/blob/main/codex-rs/apply-patch/src/parser.rs); [opencode patch.txt](https://github.com/sst/opencode/blob/dev/packages/core/src/tool/patch.txt); [agents sandbox](https://github.com/openai/openai-agents-python/tree/main/src/agents/sandbox); [gpt-oss tools](https://github.com/openai/gpt-oss/tree/main/gpt_oss/tools); [openhands-tools](https://github.com/OpenHands/software-agent-sdk/tree/main/openhands-tools/openhands/tools); [aider coders](https://github.com/Aider-AI/aider/tree/main/aider/coders).
- **str_replace / old_string→new_string:** Gemini CLI `edit`, Crush `edit`/`multiedit`, Pi `edit`, SWE-agent `edit_anthropic` bundle, OpenHands `file_editor` — [gemini edit.ts](https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/tools/edit.ts); [crush tools](https://github.com/charmbracelet/crush/tree/main/internal/agent/tools); [pi tools](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent/src/core/tools); [SWE-agent tools](https://github.com/SWE-agent/SWE-agent/tree/main/tools).
- **SEARCH/REPLACE blocks, udiff and whole-file (no tool calling):** Aider — [edit-formats.md](https://github.com/Aider-AI/aider/blob/main/aider/website/docs/more/edit-formats.md).
- **bash-only (sed/heredoc):** mini-swe-agent — [README](https://github.com/SWE-agent/mini-swe-agent/blob/main/README.md).
- **Code-as-action:** smolagents `CodeAgent` — [README](https://github.com/huggingface/smolagents/blob/main/README.md). Codex and OpenCode both have `code-mode`/`codemode` packages — [codex-rs](https://github.com/openai/codex/tree/main/codex-rs); [opencode packages](https://github.com/sst/opencode/tree/dev/packages).

**Shell model divergence**
- Codex uses PTY sessions (`exec_command` plus `write_stdin`). OpenCode has `persistent-pty`. Crush has `job_output`/`job_kill`. Gemini has background shell tools. mini-swe-agent uses stateless `subprocess.run` per action ("a big deal" for sandboxing and scaling) — [codex unified_exec.rs](https://github.com/openai/codex/blob/main/codex-rs/core/src/tools/handlers/unified_exec.rs); [opencode core](https://github.com/sst/opencode/tree/dev/packages/core/src); [crush tools](https://github.com/charmbracelet/crush/tree/main/internal/agent/tools); [mini README](https://github.com/SWE-agent/mini-swe-agent/blob/main/README.md).

**Compaction (converged on LLM summary; diverges on retention and placement)**

| Harness | Recipe | Source |
|---|---|---|
| Codex | Handoff-summary prompt plus ≤20k tokens of prior user messages; separate remote-v2 path | [compact.rs](https://github.com/openai/codex/blob/main/codex-rs/core/src/compact.rs) |
| Gemini CLI | Compress at 50% of window, keep latest 30%, plus tool-output masking | [chatCompressionService.ts](https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/context/chatCompressionService.ts) |
| OpenHands | Summarize first half of events, `keep_first=2`, max 240 events, tombstone events | [condenser README](https://github.com/OpenHands/software-agent-sdk/blob/main/openhands-sdk/openhands/sdk/context/condenser/README.md) |
| Cline | Projection stored separately from the canonical transcript | [ARCHITECTURE.md](https://github.com/cline/cline/blob/main/sdk/ARCHITECTURE.md) |
| deepagents | Summarization plus filesystem offload middleware | [middleware](https://github.com/langchain-ai/deepagents/tree/main/libs/deepagents/deepagents/middleware) |
| OpenAI Agents SDK | Provider-side `responses.compact` | [compaction session](https://github.com/openai/openai-agents-python/blob/main/src/agents/memory/openai_responses_compaction_session.py) |
| Anthropic SDK tool runner | Server-side summarization via `compact_before_next_turn()` | [cookbook](https://github.com/anthropics/claude-cookbooks/blob/main/tool_use/automatic-context-compaction.ipynb) |

**Sandboxing**
- **OS-native:** Codex (Seatbelt/bwrap/Landlock/Windows) and Anthropic `srt` (Seatbelt/bwrap plus proxy) — [codex core README](https://github.com/openai/codex/blob/main/codex-rs/core/README.md); [srt README](https://github.com/anthropic-experimental/sandbox-runtime/blob/main/README.md).
- **Container or hosted:** OpenAI SandboxAgent clients (Docker, E2B, Modal, Daytona, Cloudflare, Vercel, Blaxel), OpenHands Agent Server (Docker/K8s), smolagents (E2B/Blaxel/Modal), mini-swe-agent (docker/singularity) — [agents sandbox ext](https://github.com/openai/openai-agents-python/tree/main/src/agents/extensions/sandbox); [OpenHands SDK README](https://github.com/OpenHands/software-agent-sdk/blob/main/README.md); [smolagents README](https://github.com/huggingface/smolagents/blob/main/README.md); [mini environments](https://github.com/SWE-agent/mini-swe-agent/tree/main/src/minisweagent/environments).
- **Policy layers on top:** Codex execpolicy plus guardian, Gemini TOML policies, Claude permission modes, OpenHands security analyzers.

**Client/embedding protocols (divergent)**
- Codex app-server JSON-RPC (thread/turn/item) — [app-server-protocol](https://github.com/openai/codex/tree/main/codex-rs/app-server-protocol).
- Claude stream-JSON plus `control_request` — [query.py](https://github.com/anthropics/claude-agent-sdk-python/blob/main/src/claude_agent_sdk/_internal/query.py).
- ACP (Agent Client Protocol), spoken by Kimi Code, Goose, deepagents and Pi, and used by OpenHands Agent Canvas to run any of them — [kimi-code README](https://github.com/MoonshotAI/kimi-code/blob/main/README.md); [goose README](https://github.com/block/goose/blob/main/README.md); [deepagents libs/acp](https://github.com/langchain-ai/deepagents/tree/main/libs); [OpenHands README](https://github.com/OpenHands/OpenHands/blob/main/README.md).
- A2A in Gemini CLI — [a2a-server package](https://github.com/google-gemini/gemini-cli/tree/main/packages/a2a-server).
- MCP is universal for tools. Codex uses the official `rmcp` SDK — [Cargo.toml](https://github.com/openai/codex/blob/main/codex-rs/Cargo.toml).

**Language and hosting split**
- **Rust:** Codex, Goose.
- **TypeScript:** OpenCode, Gemini CLI, Qwen Code, Cline, Kimi Code, Pi, Letta Code, Claude Code (closed).
- **Python:** OpenHands SDK, SWE-agent/mini, Aider, smolagents, deepagents, OpenAI Agents SDK, Claude Agent SDK (wrapper).
- **Go:** Crush.
- **Hosted counterparts:** Codex Web/Cloud tasks, OpenHands Cloud, Claude Managed Agents.
- Sources: per-repo citations above.

### Inferences
- **Recommended reference stack** for a new harness:
  - **Loop and protocol:** Codex `codex-rs/core` and `app-server-protocol` (production-grade, Apache-2.0). For a lighter TS reference, use Pi or the Cline SDK `@cline/agents` (stateless loop). For Python, use the OpenHands Software Agent SDK (event-sourced) or mini-swe-agent (baseline).
  - **Edit tool:** implement both the `apply_patch` envelope (Codex grammar; best for OpenAI models and adopted by OpenCode) and `str_replace` (old_string/new_string; best for Claude-style models), and select per model, as Aider's per-model edit-format table does.
  - **Sandbox:** `anthropic-experimental/sandbox-runtime` (library, Apache-2.0) or Codex's sandbox crates, plus a container/remote option à la OpenAI SandboxAgent clients.
  - **Hooks:** use the Claude/Codex event names and the `permissionDecision` output schema for ecosystem compatibility. Consider adding Gemini-style `BeforeModel`/`AfterModel`.
  - **Instructions and packaging:** AGENTS.md (hierarchical, with override and byte budget à la Codex), plus SKILL.md skills, plus MCP, plus plugin-marketplace manifests.
  - **Compaction:** handoff-summary compaction with recent-user-message retention (Codex), tool-output masking (Gemini), and an append-only canonical log with separately persisted compaction projections (OpenHands/Cline).
  - **Long-horizon:** the initializer/coding-agent pattern with feature list, progress file and git checkpoints (Anthropic quickstart), plus a background memory consolidation pipeline (Codex memories / Letta MemFS).
  - **Embedding:** expose both a JSONL headless mode (Codex `exec --json` event schema) and an ACP server, so that the harness can be driven by IDEs (Zed, JetBrains) and control centers (OpenHands Agent Canvas).
- **Primitives are standardizing faster than harnesses.** Two of the notable 2025-era harnesses died or were replaced within a year (Roo Code, Kimi CLI), while AGENTS.md, MCP, Skills and ACP spread across almost all of them. Investing in protocol compatibility is lower-risk than tight coupling to any one harness.

### Gaps
- Claude Code's internal tool schemas (Edit/MultiEdit/TodoWrite/Task) could not be cited from source because the harness is closed. Statements about its edit format rely on SDK-visible behaviour only and were therefore omitted.
- ACP spec details (agentclientprotocol.com) and the Agent Skills spec text (agentskills.io) were not fetched.
- No head-to-head benchmark comparing edit formats across 2026 models was found in these repos. Aider's leaderboard data is the closest, but was not extracted.
