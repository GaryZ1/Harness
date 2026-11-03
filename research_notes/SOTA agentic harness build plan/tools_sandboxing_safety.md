# Tool Layer, Protocols, Execution Environments, Permissions and Prompt-Injection Defense for Agent Harnesses (state as of 2026-11-03)

Scope note: these notes cover tool interfaces, tool protocols, code execution, sandboxes, permissions/HITL and injection defenses. Multi-agent orchestration and evals are covered elsewhere. Primary sources were preferred. Several primary hosts could not be fetched from this research session because of network policy (modelcontextprotocol.io, developers.openai.com, simonwillison.net, arxiv.org). To work around this, the MCP spec and the Codex CLI source were read directly from shallow git clones of their public GitHub repos (openai/codex @ fbf5f0d, 2026-11-03; modelcontextprotocol/modelcontextprotocol @ a40e90c, 2026-11-02; anthropic-experimental/sandbox-runtime @ 98a4016, 2026-10-25). For OpenAI docs, simonwillison.net and arXiv, the notes rely on search-engine summaries and are marked accordingly.

---

## 1. Tool design principles (Anthropic "Writing effective tools", "Building effective agents" ACI, OpenAI function calling, strict schemas, parallel calls, tool counts)

### Takeaway
The settled guidance (Anthropic Sep 2025, building on the Dec 2024 ACI appendix) is to design tools as an agent-facing UX, not as API wrappers. That means few consolidated, task-shaped and namespaced tools; high-signal, token-bounded responses (pagination, truncation, `response_format`); actionable errors; poka-yoke arguments (for example, absolute paths); and strict schemas. Descriptions should then be iterated against an eval harness, because small description edits produce large measured gains.

### Cited Findings
**Anthropic, "Writing effective tools for AI agents — with agents" (Sep 11, 2025)**
- Prototype first. Wrap tools in a local MCP server or Desktop extension (DXT) and connect it with `claude mcp add <name> <command> [args...]`. Give Claude Code the docs of the underlying SDKs (e.g., `llms.txt`) — [Anthropic](https://www.anthropic.com/engineering/writing-tools-for-agents)
- Eval methodology: generate dozens of realistic prompt/response pairs grounded in real services, avoiding toy sandboxes. Strong tasks may need dozens of tool calls. Pair each with a verifier (exact match or LLM judge) and don't over-penalize formatting. Run a simple `while`-loop agent per task and have the model emit reasoning/feedback blocks before tool calls (or use interleaved thinking). Collect top-level accuracy, per-call and per-task runtime, total tool calls, total tokens and tool errors — [Anthropic](https://www.anthropic.com/engineering/writing-tools-for-agents)
- Diagnostics: many redundant calls suggest adjusting pagination or token-limit params. Many invalid-parameter errors suggest the descriptions need clarifying or examples. Read raw transcripts. Example: Claude kept appending "2025" to web-search `query`, and a description fix resolved it. Use held-out test sets to avoid overfitting when Claude Code refactors the tools — [Anthropic](https://www.anthropic.com/engineering/writing-tools-for-agents)
- Choose tools carefully and don't wrap raw endpoints. Prefer `search_contacts`/`message_contact` over `list_contacts`. Consolidate: `schedule_event` replaces `list_users` + `list_events` + `create_event`, `search_logs` replaces `read_logs`, and `get_customer_context` replaces three getters. Avoid overlapping tools — [Anthropic](https://www.anthropic.com/engineering/writing-tools-for-agents)
- Namespacing: group by service and resource (`asana_search`, `jira_search`, `asana_projects_search`). Prefix vs suffix naming "has non-trivial effects" on evals, so choose empirically — [Anthropic](https://www.anthropic.com/engineering/writing-tools-for-agents)
- Return meaningful context: `name`, `image_url`, `file_type` rather than `uuid`, `256px_image_url`, `mime_type`. Resolving UUIDs to natural-language names or 0-indexed IDs reduced hallucination and improved retrieval precision — [Anthropic](https://www.anthropic.com/engineering/writing-tools-for-agents)
- Expose a `response_format` enum (`DETAILED = "detailed"`, `CONCISE = "concise"`). In the Slack example a detailed response was 206 tokens and a concise one 72 (about 1/3). Detailed responses carry IDs needed for follow-up calls. Response structure (XML/JSON/Markdown) also affects performance — [Anthropic](https://www.anthropic.com/engineering/writing-tools-for-agents)
- Token efficiency: use pagination, range selection, filtering and truncation with sensible defaults. Claude Code caps tool responses at 25,000 tokens by default. Truncation messages should steer toward many small, targeted searches. Errors should be specific and actionable, with a correct-input example, rather than opaque codes or tracebacks — [Anthropic](https://www.anthropic.com/engineering/writing-tools-for-agents)
- Write descriptions "as if onboarding a new hire" and make implicit context explicit. Use unambiguous params (`user_id` not `user`), enforce inputs and outputs with strict data models, and use MCP tool annotations to flag open-world or destructive tools. Claude Sonnet 3.5 reached SOTA on SWE-bench Verified after precise tool-description refinements; no exact delta is given — [Anthropic](https://www.anthropic.com/engineering/writing-tools-for-agents)

**Anthropic, "Building effective agents", Appendix 2 ACI (Dec 19, 2024; background)**
- Give the model enough tokens to "think" before it writes itself into a corner. Keep formats close to naturally occurring internet text. Avoid formatting overhead such as accurate line counts over thousands of lines, diff hunk headers that need line counts before the code is written, or code wrapped in JSON that needs escaping of newlines and quotes — [Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- "Poka-yoke your tools. Change the arguments so that it is harder to make mistakes." For the SWE-bench agent they "spent more time optimizing our tools than the overall prompt". The model erred with relative paths after `cd`, so the tool was changed to always require absolute paths, which fixed it — [Anthropic](https://www.anthropic.com/engineering/building-effective-agents)

**OpenAI function calling (via search summaries; developers.openai.com not directly fetchable here)**
- OpenAI recommends always enabling `strict` mode, which makes calls adhere to the schema rather than best-effort. It requires `additionalProperties: false` on every object and every property listed in `required` (optional fields are expressed as nullable unions) — [OpenAI function calling guide (search summary)](https://developers.openai.com/api/docs/guides/function-calling)
- `parallel_tool_calls: false` guarantees zero or one tool call per turn. Older docs said strict mode was incompatible with parallel calls. Current reports say this applies only to fine-tuned models, where strict is disabled when multiple functions are called in one turn — [OpenAI guide (search summary)](https://developers.openai.com/api/docs/guides/function-calling); [OpenAI community](https://community.openai.com/t/parallel-tool-use-documentation-for-api-models/1304519). Current status is unverified from the primary page.
- `allowed_tools` restricts the callable subset per turn without changing the `tools` list, which keeps the prompt-cache prefix stable (third-party description). `tool_search` defers rarely used tools and is supported only on gpt-5.4+ models per the guide — [OpenAI guide (search summary)](https://developers.openai.com/api/docs/guides/function-calling)

**DeepSeek strict function calling (Beta)**
- Enable it with `base_url="https://api.deepseek.com/beta"` and `strict: true` on every function. It works in both thinking and non-thinking modes. The server validates the schema and errors on unsupported types. All properties must be `required` and `additionalProperties: false`. Strings support `pattern` and the formats email/hostname/ipv4/ipv6/uuid but not `minLength`/`maxLength`. Numbers support `minimum`/`maximum`/`multipleOf`. Arrays do not support `minItems`/`maxItems`. `$ref`/`$def` are supported. Strict mode was introduced with V3.1 (Aug 2025) — [DeepSeek Tool Calls guide](https://api-docs.deepseek.com/guides/tool_calls/); [DeepSeek V3.1 release](https://api-docs.deepseek.com/news/news250821/)

**MCP spec-level tool design guidance**
- MCP 2025-11-25 added tool-name guidance (SEP-986). It also clarified that input-validation errors must be returned as Tool Execution Errors (`isError` results), not protocol errors, "to enable model self-correction" (SEP-1303). JSON Schema 2020-12 became the default dialect (SEP-1613) — [MCP 2025-11-25 changelog](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2025-11-25/changelog.mdx)
- MCP 2026-07-28 says servers SHOULD return `tools/list` in a deterministic order "to enable client-side caching and improve LLM prompt cache hit rates". `inputSchema`/`outputSchema` may use any JSON Schema 2020-12 keywords, with `$ref` resolution and composition bounds (SEP-2106) — [MCP 2026-07-28 changelog](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2026-07-28/changelog.mdx)

### Inferences
- Harness defaults to adopt: absolute-path-only file tools; a `response_format` or verbosity parameter on any tool that can return large payloads; a per-result token cap (Claude Code uses 25k) with truncation messages that coach narrower queries; validation errors returned as tool results so the model can self-correct; deterministic tool ordering for prompt-cache stability; and strict schemas on every provider that supports them. DeepSeek's and OpenAI's strict subsets differ, so a schema compiler or linter per provider is needed.
- Tool count is not a fixed number any more. The practical threshold for switching to deferred loading is about 10+ tools or >10K tokens of definitions (Anthropic guidance in §4), not the folk "<20 tools" rule.

### Gaps
- No primary-source OpenAI numeric tool-count limit or "fewer than 20 tools" guidance could be verified; developers.openai.com was unreachable.
- Anthropic's held-out Slack/Asana tool-optimization graphs give no exact percentages in the text.

---

## 2. Core tool sets in leading harnesses, and why "bash + filesystem" became the universal tool

### Takeaway
Every leading coding harness converged on a small kernel: a shell (one-shot or PTY/session), file read/search, a structured edit primitive (string-replace in Claude Code, `apply_patch` V4A-style diffs in Codex), a plan/todo tool, web search/fetch, subagent spawn, and MCP passthrough. Anthropic's stated rationale is "give your agents a computer". Bash plus files lets agents compose arbitrary programs, do agentic search with grep/tail, and use the filesystem as context engineering.

### Cited Findings
**Claude Code / Claude Agent SDK**
- Built-in tool names (`Bash`, `Read`, `Edit`, `Write`, `Glob`, `Grep`, `WebFetch`, `WebSearch`, `NotebookEdit`, Task/Agent subagents, TodoWrite) are the exact strings used in permission rules, subagent tool lists and hook matchers. The tools reference has a "Permission required" column for Manual mode — [Claude Code tools reference](https://code.claude.com/docs/en/tools-reference). The per-tool table was not retrieved in full.
- Read-only tools (file reads, LS, Grep, Glob) need no approval. Bash and file edit/write need approval. File-modification approval lasts until session end, while Bash approvals can be persisted. Launching a subagent doesn't prompt, but its tool calls are checked against the rules — [Claude Code IAM docs](https://code.claude.com/docs/en/iam); [Claude Code permissions (search summary)](https://code.claude.com/docs/en/permissions)
- "Give your agents a computer, allowing them to work like humans do." The loop is "gather context -> take action -> verify work -> repeat". Agentic search (grep/tail over files) is recommended before semantic search: "only adding semantic search if you need faster results". "The folder and file structure of an agent becomes a form of context engineering." Subagents "use their own isolated context windows, and only send relevant information back". Compaction summarizes near limits. Verification comes via rules/linting (TypeScript gives more feedback than JS), visual feedback (Playwright MCP) and LLM-as-judge ("generally not a very robust method") — [Claude Agent SDK post, Sep 29, 2025](https://claude.com/blog/building-agents-with-the-claude-agent-sdk)

**Anthropic API client/server tools (versioned type strings)**
- `text_editor_20250728` fixed earlier issues and added `max_characters` for truncating large file views (Jul 28, 2025). Earlier (May 1, 2025), the bash and text-editor tools were decoupled from the computer-use system prompt — [Claude API release notes](https://docs.claude.com/en/release-notes/api)
- `memory_20250818` is a client-side tool for storing and retrieving information across conversations via a memory file directory — [LangChain Anthropic integration docs](https://docs.langchain.com/oss/javascript/integrations/tools/anthropic). It was reported as requiring the `context-management-2025-06-27` beta header (third-party, unverified).
- Code execution server tool `code_execution_20250825` (bash + file ops in an Anthropic-hosted sandbox) is the substrate for programmatic tool calling — [Anthropic advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use). The web fetch tool (beta) retrieves full web pages and PDFs — [Claude API release notes](https://docs.claude.com/en/release-notes/api)

**Codex CLI (openai/codex, read from source @ fbf5f0d)**
- Tool names registered for the model in current source include `exec_command` + `write_stdin` (PTY-session shell), legacy `shell`/`shell_command`, `apply_patch`, `update_plan`, `view_image`, `web_search`, `tool_search`, `request_user_input`, `spawn_agent`/`wait_agent`, `list_mcp_resources`/`read_mcp_resource`, and a code-mode `exec` — [openai/codex codex-rs/tools, core/src/tools](https://github.com/openai/codex/tree/main/codex-rs/tools)
- `apply_patch` grammar (official Lark grammar in source): `start: begin_patch environment_id? hunk+ end_patch`. The markers are `*** Begin Patch`, `*** Add File: <path>`, `*** Delete File: <path>`, `*** Update File: <path>` with optional `*** Move to: <path>`, `@@` / `@@ <context>` anchors, lines prefixed `+`/`-`/space, an optional `*** End of File`, and `*** End Patch`. Line numbers are deliberately absent (context-anchored), and the parser tolerates surrounding whitespace. A streaming parser exists, and the tool is exposed as a "Freeform" (grammar-constrained custom) tool spec — [codex-rs/apply-patch/src/parser.rs](https://github.com/openai/codex/blob/main/codex-rs/apply-patch/src/parser.rs)
- Codex system prompt: "You have access to an `update_plan` tool which tracks steps and progress and renders them to the user". It also says to send brief preambles (8–12 words) before grouped tool calls, and to follow AGENTS.md scoping rules (deeper files win; direct prompt instructions beat AGENTS.md) — [codex-rs/models-manager/prompt.md](https://github.com/openai/codex/blob/main/codex-rs/models-manager/prompt.md)

**OpenAI hosted tools (search summaries)**
- The Responses API `shell` tool supports local execution (client-run) and hosted execution in OpenAI-managed containers. Hosted shell has no interactive TTY and no sudo. Containers can be created and reused across Responses calls. `apply_patch` is a built-in tool alongside `shell` — [OpenAI shell tool guide (search summary)](https://developers.openai.com/api/docs/guides/tools-shell); [Microsoft Foundry shell docs](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/shells); [ruby_llm-responses_api](https://github.com/9elements/ruby_llm-responses_api)

### Inferences
- Why bash + filesystem won: it offers (a) an unbounded action space via composition; (b) the context-economy primitives grep/head/tail/jq; (c) state persistence across turns in files; (d) direct reuse of model pretraining on shell transcripts; and (e) a single choke-point for sandboxing, since one OS sandbox covers every subprocess. That is why the security story in §5 is "sandbox the shell" rather than "secure each tool".
- Edit primitive choice: string-replace (Claude `Edit`/`str_replace`) and line-number-free context diffs (Codex `apply_patch`) both follow the ACI advice to avoid line counting. A new harness should support at least one of them and should validate edits atomically.

### Gaps
- The full Claude Code per-tool "Permission required" table, and the current computer-use tool version string (e.g., `computer_2025xxxx`), were not retrieved.
- It is unverified whether the OpenAI hosted `apply_patch` executes server-side in the container or only client-side.

---

## 3. Protocols: MCP (2025-06-18 → 2025-11-25 → 2026-07-28), governance (AAIF), A2A, Agent Skills, AGENTS.md

### Takeaway
MCP is the de facto tool protocol, now under the Linux Foundation's Agentic AI Foundation (AAIF, Dec 9, 2025). Its current revision, **2026-07-28**, made the core stateless. It has no `initialize` handshake and no `Mcp-Session-Id`; it adds `server/discover`, the Multi Round-Trip Requests (`input_required`) pattern replacing server→client requests, header-based routing (`Mcp-Method`/`Mcp-Name`), cacheable deterministic lists, tasks as an extension, and Roots/Sampling/Logging deprecated. A2A reached v1.0 (2026) for agent↔agent interop. Agent Skills (`SKILL.md`, Dec 18, 2025) and AGENTS.md are the de facto standards for packaging instructions.

### Cited Findings
**MCP 2025-06-18 (background)**
- Removed JSON-RPC batching. Added structured tool output (`structuredContent` + `outputSchema`) and resource links in tool results. MCP servers are classified as OAuth Resource Servers (protected resource metadata), and clients must implement RFC 8707 Resource Indicators. Added elicitation. HTTP requires the `MCP-Protocol-Version` header — [MCP 2025-06-18 changelog](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2025-06-18/changelog.mdx)

**MCP 2025-11-25**
- Added OpenID Connect Discovery for AS discovery (#797); icons on tools/resources/prompts (SEP-973); incremental scope consent via `WWW-Authenticate` (SEP-835); tool-name guidance (SEP-986); standards-based `EnumSchema` with single/multi-select (SEP-1330); URL-mode elicitation (SEP-1036); tool calling inside sampling via `tools`/`toolChoice` (SEP-1577); OAuth Client ID Metadata Documents (CIMD) as the recommended registration (SEP-991); experimental tasks for durable requests with polling and deferred results (SEP-1686); input validation errors as Tool Execution Errors (SEP-1303); polling SSE (SEP-1699); RFC 9728 alignment (SEP-985); JSON Schema 2020-12 default (SEP-1613); governance, WGs/IGs and SDK tiering (SEP-932/994/1302/1730) — [MCP 2025-11-25 changelog](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2025-11-25/changelog.mdx)

**MCP 2026-07-28 (current; RC May 21, 2026; GA Jul 28, 2026)**
- Removes protocol sessions and `Mcp-Session-Id` (SEP-2567). Cross-call state uses "explicit, server-minted handles passed as ordinary tool arguments". Removes the `initialize`/`notifications/initialized` handshake. Every request carries `_meta` `io.modelcontextprotocol/protocolVersion` and `io.modelcontextprotocol/clientCapabilities` (and SHOULD carry `clientInfo`), and results carry `serverInfo`. Mismatch returns `UnsupportedProtocolVersionError` (SEP-2575) — [MCP 2026-07-28 changelog](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2026-07-28/changelog.mdx)
- New `server/discover` RPC (MUST implement). `subscriptions/listen` replaces the GET stream and `resources/subscribe`. `ping`, `logging/setLevel` and `notifications/roots/list_changed` are removed, with log level set per request via `io.modelcontextprotocol/logLevel`. SSE resumability (`Last-Event-ID`) is removed, so a broken stream means re-issuing the request with a new ID — [MCP 2026-07-28 changelog](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2026-07-28/changelog.mdx)
- Tasks moved to the official extension `io.modelcontextprotocol/tasks`. Polling uses `tasks/get`, new `tasks/update` carries client→server input, `tasks/list` is removed, and servers may return task handles unsolicited (SEP-2663) — [MCP 2026-07-28 changelog](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2026-07-28/changelog.mdx)
- Multi Round-Trip Requests (SEP-2322): the server returns `InputRequiredResult` (`resultType: "input_required"`) with `inputRequests`, and the client retries the original request with `inputResponses`. This replaces server-initiated `roots/list`, `sampling/createMessage` and `elicitation/create`. All results carry a required `resultType` (`"complete"`/`"input_required"`). The URL-elicitation `elicitationId` and `notifications/elicitation/complete` are removed, and servers use `requestState` instead — [MCP 2026-07-28 changelog](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2026-07-28/changelog.mdx)
- Minor changes: `extensions` field in capabilities; OpenTelemetry `traceparent`/`tracestate`/`baggage` in `_meta` (SEP-414); required `Mcp-Method`/`Mcp-Name` headers and `x-mcp-header` for custom headers from tool params (SEP-2243); `CacheableResult` with required `ttlMs` and `cacheScope` (`public`/`private`) on list/read results (SEP-2549); resource-not-found changed to `-32602`; RFC 9207 `iss` validation (SEP-2468); `application_type` in DCR (SEP-837); credentials bound to issuer (SEP-2352); error-code range `-32020..-32099` reserved — [MCP 2026-07-28 changelog](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2026-07-28/changelog.mdx)
- Deprecated: Roots, Sampling and Logging (SEP-2577), with suggested migrations to tool params/URIs, direct LLM APIs and stderr/OTel; the HTTP+SSE transport; `includeContext` `thisServer`/`allServers`; and Dynamic Client Registration in favor of CIMD (PR #2858). A feature lifecycle policy sets a ≥12-month deprecation window (SEP-2596) — [MCP 2026-07-28 changelog](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2026-07-28/changelog.mdx)
- GA announcement: "close to half-a-billion downloads a month" across Tier 1 SDKs, and the TS and Python SDKs each crossed 1B total downloads. The TypeScript, Python, Go and C# SDKs were updated. Extensions include Tasks, MCP Apps and Enterprise Managed Authorization (EMA). Rationale for handles: "the model can see the handle and thread it between tools" — [MCP blog, "The 2026-07-28 Specification"](https://blog.modelcontextprotocol.io/posts/2026-07-28/) (read from [repo copy](https://github.com/modelcontextprotocol/modelcontextprotocol/tree/main/blog/content/posts/2026-07-28-spec-ga)). Anthropic announced adoption in Claude — [Claude blog](https://claude.com/blog/bringing-mcp-2026-07-28-to-claude) (title only seen).
- MCP Registry preview launched at `registry.modelcontextprotocol.io` (Sep 8, 2025) as an open catalog and API with an OpenAPI spec that sub-registries (public/private) build on — [MCP blog](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/blog/content/posts/2025-09-08-mcp-registry-preview.md). The MCP Apps extension was announced Nov 21, 2025 and Jan 26, 2026, and MCPB (bundles) Nov 20, 2025 (blog file list in the repo).

**Governance**
- Dec 9, 2025: Anthropic donated MCP to the Agentic AI Foundation (AAIF), a directed fund under the Linux Foundation co-founded by Anthropic, Block and OpenAI, with support from Google, Microsoft, AWS, Cloudflare and Bloomberg. The founding projects are MCP, goose (Block) and AGENTS.md (OpenAI). At that time MCP had "97 million monthly SDK downloads, 10,000 active servers" and client support in ChatGPT, Claude, Cursor, Gemini, Copilot and VS Code — [MCP blog](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/blog/content/posts/2025-12-09-mcp-joins-agentic-ai-foundation.md)

**A2A**
- The latest released version is 1.0.0. v1.0 added multi-protocol bindings and version negotiation, multi-tenancy, and signed Agent Cards. A2A was launched under the Linux Foundation on Jun 23, 2025 and joined AAIF (reported Aug 2026). Some docs still show `A2A-Version: 0.3` headers — [A2A spec](https://a2a-protocol.org/latest/specification/); [AAIF blog](https://aaif.io/blog/a2a-joins-aaif) (via search summary; exact v1.0 date conflicts: March 2026 GA vs. "RC" per [dev.to](https://dev.to/eclaw/a2a-protocol-tech-update-20260316-4fii))

**Agent Skills / AGENTS.md**
- Agent Skills was released as an open standard on Dec 18, 2025. A skill is a folder with a required `SKILL.md` (YAML frontmatter `name` + `description`, then markdown instructions loaded when relevant) plus optional `scripts/`, `references/` and `assets/`. Adopters reportedly include OpenAI Codex, GitHub Copilot, VS Code, Cursor and Gemini CLI, with about 40–44 clients listed on agentskills.io by mid-2026 (secondary sources). One analysis found 22% of 673 sampled skills failed structural validation — [Atlan](https://atlan.com/know/ai-agent/ai-agent-skills/what-are-agent-skills/); [aictrl.dev](https://aictrl.dev/blog/skill-md-adoption-trajectory) (secondary; counts unverified)
- Codex's `codex-rs/skills` crate and `docs/skills.md` exist in the repo, and Anthropic's code-execution-with-MCP post proposes `SKILL.md` to persist reusable code as skills — [openai/codex docs](https://github.com/openai/codex/tree/main/docs); [Anthropic](https://www.anthropic.com/engineering/code-execution-with-mcp)
- AGENTS.md semantics as implemented by Codex: scope is the directory tree rooted at the file, more deeply nested files take precedence, direct system/developer/user instructions beat AGENTS.md, and root-to-CWD AGENTS.md files are auto-injected — [codex prompt.md](https://github.com/openai/codex/blob/main/codex-rs/models-manager/prompt.md)

### Inferences
- A harness built in late 2026 should implement an MCP client against 2026-07-28. Concretely: stateless requests with `_meta` version/capabilities; `server/discover` with fallback to `initialize` for older servers; handling `resultType: input_required` by surfacing elicitation to the user and retrying; caching `tools/list` by `ttlMs` with deterministic order to keep prompt caches warm; and CIMD-first OAuth with RFC 9207 `iss` checks and RFC 8707 resource indicators. Don't build new features on Sampling or Roots.
- Long-running tools should use the `io.modelcontextprotocol/tasks` extension (poll with `tasks/get`) rather than holding streams open.

### Gaps
- modelcontextprotocol.io was unreachable, so content came from the GitHub repo (which is the spec source). Claude's specific 2026-07-28 adoption details were not read.
- The A2A v1.0 exact GA date and the official agentskills.io spec text were not fetched directly.

---

## 4. Advanced tool use: tool search / deferred loading, programmatic tool calling, tool-use examples, code execution with MCP, OpenAI and DeepSeek equivalents

### Takeaway
Three Anthropic features (Nov 24, 2025, beta header `advanced-tool-use-2025-11-20`) address the three tool bottlenecks. **Tool Search** (`defer_loading`) cuts definition tokens by about 85% and raised Opus 4.5 MCP-eval accuracy from 79.5% to 88.1%. **Programmatic Tool Calling** (`allowed_callers` from `code_execution_20250825`) cut tokens 37% and raised GAIA from 46.5% to 51.2%. **`input_examples`** raised complex-parameter accuracy from 72% to 90%. The "code execution with MCP" pattern (tools as a TypeScript filesystem API) reported 150K→2K tokens (98.7%). OpenAI has `tool_search` (gpt-5.4+) and Codex has a code-mode, and DeepSeek V3.2 trained thinking-with-tools on 1,827 synthesized environments.

### Cited Findings
**Tool Search Tool**
- Mark tools `"defer_loading": true` (keep 3–5 hot tools `false`). The type string is `tool_search_tool_regex_20251119` (regex and BM25 variants; custom/embedding search is supported). For MCP, use `mcp_toolset` with `default_config: {"defer_loading": true}` plus per-tool `configs`. Deferred tools are excluded from the initial prompt, so prompt caching is preserved — [Anthropic, Nov 24, 2025](https://www.anthropic.com/engineering/advanced-tool-use)
- Numbers: 50+ MCP tools take ~72K tokens (~77K before work) traditionally, versus ~500 upfront + ~3K for 3–5 discovered tools (~8.7K total), an "85% reduction". Five servers with 58 tools take ~55K tokens, Jira alone ~17K, and definitions have been seen reaching 134K. The post also cites 191,300 vs 122,800 tokens preserved, a figure inconsistent with the others that the page doesn't reconcile — [Anthropic](https://www.anthropic.com/engineering/advanced-tool-use)
- MCP eval accuracy: Opus 4 went from 49% to 74% and Opus 4.5 from 79.5% to 88.1%. Use it when definitions exceed 10K tokens, there are 10+ tools, multiple servers, or selection errors. It is not worth it for <10 tools or compact definitions — [Anthropic](https://www.anthropic.com/engineering/advanced-tool-use)

**Programmatic Tool Calling (PTC)**
- Add `{"type": "code_execution_20250825", "name": "code_execution"}` and set `"allowed_callers": ["code_execution_20250825"]` on opted-in tools. Calls from code carry a `caller` field. The model's code invocation appears as `server_tool_use` and the output as `code_execution_tool_result` — [Anthropic](https://www.anthropic.com/engineering/advanced-tool-use)
- Results: complex research tasks went from 43,588 to 27,297 average tokens (−37%). Internal knowledge retrieval went from 25.6% to 28.5%. GAIA went from 46.5% to 51.2%. 20+ calls in one code block save 19+ inference passes. In the budget example, 2,000+ line items (~200KB) were reduced to a ~1KB result in context. Claude for Excel uses PTC. Best for large data, 3+ dependent calls, filtering, and parallel fan-out. Document return formats, and prefer idempotent, parallel-safe tools — [Anthropic](https://www.anthropic.com/engineering/advanced-tool-use)

**Tool Use Examples**
- Add an `input_examples` array to the tool definition. Complex-parameter accuracy went from 72% to 90%. Use 1–5 realistic examples (minimal/partial/full) that show format conventions (`YYYY-MM-DD`, `USR-XXXXX`), nesting, and correlated optional params. Not needed for simple tools or formats JSON Schema already enforces — [Anthropic](https://www.anthropic.com/engineering/advanced-tool-use)

**Code execution with MCP (Nov 4, 2025)**
- Present each MCP server as a folder `servers/<server>/` with one typed TS file per tool, each wrapping `callMCPTool('google_drive__get_document', …)`. The agent explores the filesystem to load only the tools it needs (progressive disclosure, optionally via a `search_tools` tool with detail levels). It writes scripts that chain calls, so intermediate data stays in the sandbox (e.g., it sees 5 rows instead of 10,000, and a 2-hour transcript doesn't add ~50K tokens) — [Anthropic](https://www.anthropic.com/engineering/code-execution-with-mcp)
- Reported "150,000 tokens to 2,000 tokens", a 98.7% saving. Extra benefits: loops and conditionals run in code, PII can be tokenized by the MCP client (`[EMAIL_1]`) and untokenized between tools, state persists in workspace files, and reusable code can be saved as `SKILL.md` skills. Caveat: it requires a secure sandbox with resource limits and monitoring, which is an operational cost — [Anthropic](https://www.anthropic.com/engineering/code-execution-with-mcp)

**OpenAI / Codex equivalents**
- The OpenAI `tool_search` defers rarely used functions and is gpt-5.4+ only — [OpenAI guide (search summary)](https://developers.openai.com/api/docs/guides/function-calling). Codex source registers a `tool_search` tool and code-mode crates (`code-mode`, `code-mode-runtime`, `code-mode-host`). It augments tool descriptions "with code-mode-specific exec samples" (`augment_tool_spec_for_code_mode`) — [openai/codex codex-rs/tools/src/code_mode.rs](https://github.com/openai/codex/blob/main/codex-rs/tools/src/code_mode.rs)

**DeepSeek V3.1 / V3.2**
- V3.2 (Dec 1, 2025) is "our first model to integrate thinking directly into tool-use", and supports tool use in both thinking and non-thinking modes — [DeepSeek V3.2 release](https://api-docs.deepseek.com/news/news251201)
- Agent post-training in the V3.2 report: a synthesis agent built 1,827 task-oriented environments and 85,000+ complex prompts, designed to be "hard to solve but easy to verify". It iteratively raised task difficulty, updated solution and verification functions, and augmented toolsets when insufficient. It also built tens of thousands of reproducible issue-resolution environments from GitHub issue/PR pairs — [DeepSeek-V3.2 tech report, arXiv 2512.02556](https://arxiv.org/pdf/2512.02556) (via search summary)
- API: thinking is enabled via `extra_body={"thinking": {"type": "enabled"}}` and reasoning is returned in `reasoning_content`. In multi-turn tool use within a turn, reasoning, content and tool calls should be kept together. The official API returns `None` (vLLM: `[]`) when there are no tool calls — [vLLM DeepSeek-V3.2 recipe](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2.html); [DeepSeek Thinking Mode guide](https://api-docs.deepseek.com/guides/thinking_mode/). The DeepSeek API also exposes Responses-API and Anthropic-API compatible endpoints — [DeepSeek docs](https://api-docs.deepseek.com/guides/anthropic_api/)
- Conflict: Vercel AI Gateway says the "V3.2 Thinking" variant doesn't support tool use, while the official notes say thinking + tools are integrated — [Vercel](https://vercel.com/ai-gateway/models/deepseek-v3.2-thinking) vs [DeepSeek](https://api-docs.deepseek.com/news/news251201)

### Inferences
- Harness design: a tool registry with `hot` / `deferred` tiers and a search tool over deferred definitions, preferably server-side to keep the cache prefix stable. A sandboxed code tool can call registered tools as functions (PTC/code-mode), with results returned to the model only via `print`/return. `input_examples` should be attached to complex tools. These are composable: Anthropic recommends starting with the biggest bottleneck (context bloat → search; large intermediates → PTC; param errors → examples).
- Code-mode turns the sandbox into a security-critical component, because tool calls now originate from model-written code. Per-tool `allowed_callers`-style gating is needed so destructive tools stay direct-call (and approval-gated).

### Gaps
- No BM25 tool-search type string was given in the Anthropic post. The current GA status of these betas in Nov 2026 was not verified.
- No public numbers were found for OpenAI `tool_search` or Codex code-mode, and the code-mode runtime engine (JS/V8?) was not confirmed from source.
- The official DeepSeek tool-calls page was not fetched directly (only search snippets).

---

## 5. Execution environments and sandboxing

### Takeaway
The state of the art is OS-level sandboxing of the whole process tree plus proxy-mediated egress allowlists, with filesystem and network isolation both mandatory. Local tools use Seatbelt on macOS and bubblewrap + seccomp on Linux (Codex has moved its Linux default from Landlock to bubblewrap); Windows uses restricted tokens/WFP. Cloud agents use per-session VMs/containers (Firecracker microVMs for untrusted code, gVisor for lighter isolation) with credentials held outside the sandbox and injected by a proxy. Anthropic measured an 84% reduction in permission prompts from sandboxing.

### Cited Findings
**Claude Code sandboxing (Oct 20, 2025)**
- Filesystem: read/write in CWD and modification outside it blocked. Network: all traffic goes via a Unix domain socket to a proxy outside the sandbox that enforces a domain allowlist and asks about new domains (custom rules are possible). Enforcement uses Linux bubblewrap and macOS Seatbelt and covers spawned subprocesses. "Both layers are required": without network isolation a compromised agent can exfiltrate SSH keys, and without FS isolation it can escape to gain network — [Anthropic](https://www.anthropic.com/engineering/claude-code-sandboxing)
- "In our internal usage, we've found that sandboxing safely reduces permission prompts by 84%." Threat model: prompt injection, so "a successful prompt injection is fully isolated" and the agent "can't steal your SSH keys, or phone home" — [Anthropic](https://www.anthropic.com/engineering/claude-code-sandboxing)
- Claude Code on the web: each session runs in an isolated cloud sandbox. Git credentials and signing keys are never inside it. A custom git proxy authenticates the sandbox's scoped credential, verifies the operation (e.g., push only to the configured branch), then attaches the real token — [Anthropic](https://www.anthropic.com/engineering/claude-code-sandboxing)

**anthropic-experimental/sandbox-runtime (`srt`, npm `@anthropic-ai/sandbox-runtime`)**
- Mechanisms: macOS `sandbox-exec` with dynamically generated Seatbelt profiles. Linux bubblewrap with the network namespace removed entirely, so all traffic goes via host HTTP and SOCKS5 proxies on bind-mounted Unix sockets (needs `bwrap` + `socat`). Windows (alpha) runs as a dedicated `srt-sandbox` user with a WFP egress filter keyed on its SID, a write-restricted token and ACEs. On macOS the Seatbelt profile allows only one localhost proxy port — [sandbox-runtime README](https://github.com/anthropic-experimental/sandbox-runtime)
- Policy semantics: reads are allowed by default (deny-then-allow; `allowRead` beats `denyRead`, but a more-specific `denyRead` like `**/.env` stays denied). Writes are denied by default (allow-only; `denyWrite` beats `allowWrite`). Network is denied by default (`allowedDomains`, `deniedDomains`, `allowUnixSockets`, `allowLocalBinding`). Other knobs: `ignoreViolations`, `enableWeakerNestedSandbox`, `allowNestedUserNamespaces`, `enableWeakerNetworkIsolation`, `allowAppleEvents`. There's a CLI, a library, `--control-fd` for live config updates, and `srt proxy` with an external decider. It's explicitly intended to sandbox local MCP servers too — [sandbox-runtime README](https://github.com/anthropic-experimental/sandbox-runtime)

**Codex CLI sandbox (source @ fbf5f0d)**
- `SandboxPolicy` variants: `danger-full-access`; `read-only {network_access=false}`; `external-sandbox {network_access}` (full disk, for when already inside a container/VM); and `workspace-write {writable_roots, network_access=false, exclude_tmpdir_env_var, exclude_slash_tmp}`. Writable roots carry `read_only_subpaths` so that `.git` (notably `.git/hooks`) and `.codex` stay read-only even under a writable root, to prevent privilege escalation — [codex protocol.rs](https://github.com/openai/codex/blob/main/codex-rs/protocol/src/protocol.rs)
- Linux helper `codex-linux-sandbox`: "uses bubblewrap for filesystem and namespace isolation, and seccomp for syscall filtering" (`PR_SET_NO_NEW_PRIVS`). Landlock is retained only as a legacy implementation. Bubblewrap is the default FS sandbox; it prefers system `bwrap` on PATH outside CWD, falls back to a bundled `bwrap`, and warns if user namespaces are unavailable. WSL2 is supported and WSL1 rejected — [codex linux-sandbox README](https://github.com/openai/codex/blob/main/codex-rs/linux-sandbox/README.md)
- macOS uses Seatbelt profiles (`seatbelt_network_policy.sbpl`, `seatbelt_read_only_platform_defaults.sbpl`, etc.). Windows has dedicated `windows-sandbox-rs` and `windows-sandbox-service` crates — [codex-rs/sandboxing](https://github.com/openai/codex/tree/main/codex-rs/sandboxing)
- `codex-network-proxy` runs an HTTP proxy (default `127.0.0.1:3128`) and SOCKS5 (`127.0.0.1:8081`) and enforces allow/deny policy. A `mode = "limited"` provides read-only network, with automatic HTTPS MITM and a managed CA exported to child processes. Local and private networking is rejected by default, and hostnames resolving to private IPs stay blocked even if allowlisted. Config lives under `[permissions.<profile>.network]` with `default_permissions = "workspace"`. Non-loopback binds require `dangerously_allow_non_loopback_proxy` — [codex network-proxy README](https://github.com/openai/codex/blob/main/codex-rs/network-proxy/README.md)
- `--dangerously-bypass-approvals-and-sandbox` (alias `--yolo`) is documented as "EXTREMELY DANGEROUS. Intended solely for running in environments that are externally sandboxed" — [codex shared_options.rs](https://github.com/openai/codex/blob/main/codex-rs/utils/cli/src/shared_options.rs)

**Cloud sandboxes (secondary/vendor sources; numbers vary widely)**
- E2B runs each sandbox in a Firecracker microVM (dedicated kernel) with pause/fork/snapshot. Modal uses gVisor (user-space kernel; lighter, GPU support, 24h cap). Daytona is described as Docker containers in some sources and Firecracker in others. Reported cold starts: E2B ~150ms, Daytona "sub-90ms" (vendor) vs 200–500ms (third-party), Modal ~100ms–3s. The common guidance is Firecracker for untrusted LLM code, Modal for GPU, Daytona for persistent/self-hosted workspaces — [Northflank](https://northflank.com/blog/daytona-vs-e2b-ai-code-execution-sandboxes); [Spheron](https://www.spheron.network/blog/ai-agent-code-execution-sandbox-e2b-daytona-firecracker/); [Beam](https://www.beam.cloud/blog/best-e2b-alternatives) (vendor-adjacent; treat as rough)
- OpenAI hosted shell runs in OpenAI-managed containers (no TTY, no sudo; reusable container IDs) — [OpenAI shell guide (search summary)](https://developers.openai.com/api/docs/guides/tools-shell)
- Codex has a `worktree` crate in source, so git worktrees are used for isolation of parallel tasks — [openai/codex codex-rs](https://github.com/openai/codex/tree/main/codex-rs) (inferred from crate name only).

### Inferences
- Reference architecture for a new harness:
  - **Local mode:** wrap every shell/tool subprocess (and local MCP servers) in an srt-style OS sandbox: write-allow-only in the workspace and tmp, protected subpaths (`.git/hooks`, harness config dirs, `.env`), network namespace removed, and egress through a host proxy with a domain allowlist and private-IP blocking.
  - **Cloud mode:** use a per-session Firecracker/gVisor VM with no long-lived secrets inside, and a credential-injecting egress proxy (git proxy pattern) that also validates operations such as branch-scoped pushes.
  - **Both modes:** provide an `external-sandbox` setting so the harness doesn't double-sandbox inside containers.
- The 84% prompt reduction is the main UX argument. Sandbox boundaries let the permission system auto-allow in-boundary actions and only ask at boundary crossings.

### Gaps
- No independent, benchmarked comparison of sandbox cold-start or escape resistance was found; vendor numbers conflict.
- No primary Codex cloud doc on per-environment internet-access settings was retrievable (developers.openai.com blocked). Docker-only isolation trade-offs weren't separately sourced.

---

## 6. Permissions and human-in-the-loop

### Takeaway
Harnesses converged on layered decision pipelines: static allow/deny rules (with deny precedence), then mode (plan/read-only, accept-edits, auto, bypass), then hooks/policy engines, then a model-based auto-reviewer, and only then the human. Anthropic's auto mode data shows why: users approved 93% of prompts anyway, and a two-stage Sonnet 4.6 transcript classifier achieves 0.4% FPR, with an honestly reported 17% FNR on real overeager actions. Codex has an equivalent `approvals_reviewer = "auto_review"` subagent, Starlark `prefix_rule` exec policies, and `untrusted`/`on-request`/`granular`/`never` approval policies. Checkpoint/rewind is the recovery layer.

### Cited Findings
**Claude Code auto mode (engineering post Mar 25, 2026)**
- Motivation: users "accept 93% of them anyway" for manual prompts, a sign of approval fatigue — [Anthropic](https://www.anthropic.com/engineering/claude-code-auto-mode)
- Two layers. The input layer is a server-side prompt-injection probe that scans tool outputs and adds a warning to treat suspect content cautiously. The output layer is a transcript classifier on Sonnet 4.6 that evaluates each action before execution — [Anthropic](https://www.anthropic.com/engineering/claude-code-auto-mode)
- The classifier has two stages. Stage 1 is a single-token yes/no filter told to "err on the side of blocking". Stage 2 is chain-of-thought re-evaluation, run only if stage 1 flags. It sees only user messages and tool calls; assistant text and tool results are stripped ("reasoning-blind by design"). The cost is weaker provenance — [Anthropic](https://www.anthropic.com/engineering/claude-code-auto-mode)
- Tiers:
  - Tier 1 is a fixed allowlist of non-state-modifying tools plus user rules. Entering auto mode drops blanket shell, wildcard interpreter (python/node/ruby) and package-manager-run rules.
  - Tier 2 lets in-project file writes and edits skip the classifier.
  - Tier 3 sends everything else to the classifier: shell, web fetch, external tools, subagent spawns and out-of-project FS.

  — [Anthropic](https://www.anthropic.com/engineering/claude-code-auto-mode)
- Results:

  | Dataset | Stage 1 only | Full pipeline |
  |---|---|---|
  | Real internal traffic (n=10,000), FPR | 8.5% | 0.4% |
  | Real overeager actions (n=52), FNR | 6.6% | 17% ("the honest number") |
  | Synthetic exfiltration (n=1,000), FNR | 1.8% | 5.7% |

  Most misses were cases where the classifier saw the danger but misjudged whether user consent covered the blast radius — [Anthropic](https://www.anthropic.com/engineering/claude-code-auto-mode)
- On block, the denial returns as a tool result instructing the agent to find a safer path and not route around it. 3 consecutive or 20 total denials escalate to the human, and headless `claude -p` terminates. Subagents are checked at delegation (can deny) and at return (warn only). Configuration has three slots: Environment (trust boundary; default only the current git repo), Block rules (20+ defaults in four groups: destroy/exfiltrate, degrade security posture, cross trust boundaries, bypass review/affect others) and Allow exceptions. Inspect with `claude auto-mode defaults` — [Anthropic](https://www.anthropic.com/engineering/claude-code-auto-mode)
- Auto mode launched as a research preview (Sonnet 4.6 / Opus 4.6). Anthropic says it "reduces risk compared to --dangerously-skip-permissions but doesn't eliminate it" and recommends isolated environments — [Claude blog](https://claude.com/blog/auto-mode). The docs reportedly make it the default for new Pro/Max/Team sessions from Aug 14, 2026 — [Claude Code docs (search summary)](https://code.claude.com/docs/en/auto-mode-config) (not verified against release notes)

**Claude Code modes, rules, hooks and checkpoints (docs via search summaries)**
- Modes:
  - `acceptEdits` auto-runs reads, edits and common FS commands (mkdir/touch/mv/cp).
  - `plan` runs only reads and read-only shell, with no source edits.
  - `bypassPermissions` skips prompts except actions no mode auto-approves, and should be used only in a container/VM/sandbox.
  - `auto` is the classifier mode described above.

  — [Claude Code permission modes](https://code.claude.com/docs/en/permission-modes)
- Hooks include `PreToolUse`, `PermissionRequest` (fires when a permission decision is needed) and `PermissionDenied` (auto-mode denials), and their input includes `permission_mode` — [Claude Code hooks reference](https://code.claude.com/docs/en/hooks). The exact `permissionDecision` schema was not verified.
- Checkpointing is on by default. Checkpoints are saved with the conversation and auto-cleaned after 30 days. `/rewind` (or Esc Esc) restores code+conversation, conversation only, or code only. The SDK's file checkpointing tracks only Write/Edit/NotebookEdit changes, not Bash side effects — [Claude Code SDK file checkpointing](https://code.claude.com/docs/en/agent-sdk/file-checkpointing)

**Codex approvals and policy (source)**
- `AskForApproval` variants:
  - `untrusted` (`UnlessTrusted`) means commands need approval unless an exec-policy rule allows them.
  - `on-request` (default) lets the model decide when to ask. `on-failure` is now just a serde alias for `on-request`.
  - `granular` (`GranularApprovalConfig`) has per-category booleans, and false means auto-reject rather than prompt.
  - `never` returns failures to the model and never escalates.

  — [codex protocol.rs](https://github.com/openai/codex/blob/main/codex-rs/protocol/src/protocol.rs)
- `ApprovalsReviewer`: `user` (default) or `auto_review` (alias `guardian_subagent`). The guardian module "hosts approval decisions and the isolated synchronous reviewer", and there's a separate network-access guardian prompt. Admin `requirements` layers can constrain `allowed_approvals_reviewers` — [codex core/src/guardian](https://github.com/openai/codex/tree/main/codex-rs/core/src/guardian); [codex config](https://github.com/openai/codex/tree/main/codex-rs/config/src)
- `execpolicy` uses Starlark: `prefix_rule(pattern=[...], decision="allow"|"prompt"|"forbidden", justification=..., match=[...], not_match=[...])`. The `match`/`not_match` examples are validated at load as unit tests. `host_executable(name="git", paths=[...])` pins which absolute binaries may match basename rules. Check with `codex execpolicy check` — [codex execpolicy README](https://github.com/openai/codex/blob/main/codex-rs/execpolicy/README.md)

**OpenAI Agents SDK HITL (secondary)**
- Approval pauses the run, and `RunResult.interruptions` holds `ToolApprovalItem`s. Hosted-MCP sticky decisions are keyed by `server_label` + tool name. Guardrails (input/output/tool) raise tripwire exceptions. A vendor claims guardrails don't cover hosted tools, MCP, or computer/shell — [openai-agents-python PR #4280](https://github.com/openai/openai-agents-python/pull/4280/files); [Arcjet](https://arcjet.com/compare/openai-agents-guardrails-vs-arcjet) (vendor claim; unverified)

**MCP annotations as policy input**
- `ToolAnnotations {title, readOnlyHint (default false), destructiveHint (default true), idempotentHint (default false), openWorldHint (default true)}`. All are hints and are untrusted unless the server is trusted. Pending SEPs: #1913 trust/sensitivity (GitHub + OpenAI), #1984 governance, #1561 `unsafeOutputHint`, #1560 `secretHint`, #1487 `trustedHint`. GitHub's MCP read-only mode is used by ~17% of users. No client yet filters by annotations or shows them in approval prompts — [MCP blog, Mar 16, 2026](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/blog/content/posts/2026-03-16-tool-annotations.md)

### Inferences
- Recommended decision pipeline for each tool call:
  1. Deny rules (hard).
  2. Sandbox-boundary check: in-boundary actions are auto-allowed.
  3. Allow rules (narrow; never blanket interpreters).
  4. Hooks / exec-policy (Starlark-style prefix rules with tested examples).
  5. MCP annotation- and taint-aware policy.
  6. Model auto-reviewer that sees user intent plus the action, not tool outputs.
  7. Human, with escalation after N denials.
- Pair this with checkpoints that also snapshot Bash side effects, e.g., git-stash or worktree snapshots, since SDK checkpoints miss Bash edits.
- Make the auto-reviewer's denial a tool result with "don't route around" guidance, and count denials to prevent grinding.

### Gaps
- The exact PreToolUse JSON output fields (`permissionDecision: allow|deny|ask`, `updatedInput`) were not verified from the primary doc in this session.
- No measured precision/recall for Codex's `auto_review` guardian was found.

---

## 7. Security: prompt injection, lethal trifecta, CaMeL/dual-LLM/design patterns, measured attack rates, MCP-specific threats, secrets and egress

### Takeaway
Prompt injection is unsolved at the model level. Vendors report large but non-zero reductions: Opus 4.5 browser-use ASR was 1.4% with safeguards in Nov 2025, and Aug 2026 evals show 0% for Opus 5/Sonnet 5 with probes + classifier, against an adaptive attacker. Robust designs therefore rely on architecture:
- break the lethal trifecta (private data + untrusted content + exfiltration);
- isolate untrusted data from control flow, with CaMeL as the reference at 77% AgentDojo tasks solved with provable security vs 84% undefended;
- enforce egress allowlists and keep secrets outside the sandbox;
- pin and hash MCP tool descriptions against poisoning and rug pulls.

### Cited Findings
**Lethal trifecta (Simon Willison, Jun 16, 2025)**
- The trifecta is access to private data, exposure to untrusted content, and the ability to communicate externally. Any agent combining all three can be tricked into leaking data. LLMs "are unable to reliably distinguish the importance of instructions based on where they came from". Guardrails catching "95% of attacks" are "a failing grade" in security. MCP's mix-and-match makes users assemble the trifecta themselves, and "the only way to stay safe there is to avoid that lethal trifecta combination entirely" — [Simon Willison](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/) (via search summary; site blocked here)

**Design patterns paper (Beurer-Kellner et al., arXiv 2506.08837, Jun 2025)**
- The paper proposes six patterns that trade flexibility for security:
  - **Action-Selector:** an LLM picks from fixed actions with no feedback.
  - **Plan-Then-Execute:** the tool-call plan is fixed before untrusted data is read. Injections can alter args but not add calls.
  - **LLM Map-Reduce:** untrusted documents are processed in isolation, then aggregated.
  - **Dual LLM:** a privileged LLM never sees untrusted data, while a quarantined LLM processes it and returns symbolic references.
  - **Code-Then-Execute:** the CaMeL approach.
  - **Context-Minimization:** drop unneeded content, including the user prompt, after use.

  — [arXiv 2506.08837](https://arxiv.org/pdf/2506.08837); [Simon Willison summary](https://simonwillison.net/2025/Jun/13/prompt-injection-design-patterns/) (via search summaries)

**CaMeL (Debenedetti et al., Google DeepMind, arXiv 2503.18813, Mar 2025)**
- A privileged LLM writes a program from the trusted query. Untrusted data is processed by a quarantined LLM. Values carry capabilities (provenance plus allowed readers), and security policies are enforced at tool-call time, preventing exfiltration over unauthorized data flows without changing the model. It solves 77% of AgentDojo tasks with provable security vs 84% undefended. Code: github.com/google-research/camel-prompt-injection — [arXiv 2503.18813](https://arxiv.org/abs/2503.18813); [MIT 6.858 reading](https://css.csail.mit.edu/6.858/2026/readings/camel.pdf)

**Anthropic browser-use measurements**
- Nov 24, 2025, adaptive Best-of-N attacker, 100 attempts per environment, ~2,200–3,200 attempts per bar. ASR (model only / old safeguards / improved safeguards):

  | Model | Model only | Old safeguards | Improved safeguards |
  |---|---|---|---|
  | Sonnet 4 | 13.8% | 10.7% | 2.0% |
  | Sonnet 4.5 | 13.8% | 10.8% | 1.9% |
  | Haiku 4.5 | 12.0% | 11.1% | 2.4% |
  | Opus 4.5 | 8.0% | 7.6% | 1.4% |

  Defenses: RL on injections in simulated web content, classifiers scanning all untrusted content entering context, and continuous human red-teaming. "A 1% attack success rate—while a significant improvement—still represents meaningful risk." — [Anthropic research](https://www.anthropic.com/research/prompt-injection-defenses)
- Claude in Chrome GA (Aug 26, 2026), stronger red-team eval:
  - Opus 4.5 before safeguards: 17.6%. Opus 4.5 with probes: 16.7%.
  - Opus 5 before safeguards: 3.8%.
  - Sonnet 5, Opus 5 and Mythos 5 with probes + safety classifier: 0%. Fable 5: 0.3%.

  An earlier Cowork-harness eval saturated at 0% and was retired. All breaks were low-severity. Safeguards: training on a growing attack library, probes on tool results (warn and maybe check with the user), an action classifier that compares each action to the original request and blocks mismatches, auto-approval using the auto-mode mechanism, and admin domain allowlists — [Claude in Chrome GA](https://claude.com/resources/articles/claude-in-chrome-generally-available)

**OpenAI**
- "Hardening Atlas against prompt injection" (Dec 22, 2025): an LLM attacker trained end-to-end with RL to find long-horizon injections (dozens to hundreds of steps). Newly found attack classes feed adversarial training of the agent model (a "rapid response loop"), shipped to all Atlas users. OpenAI calls prompt injection an open challenge for years to come. Example: an email trying to get the agent to forward tax documents — [OpenAI](https://openai.com/index/hardening-atlas-against-prompt-injection) (via search summary; no ASR numbers captured)

**MCP-specific threats**
- Tool poisoning (Invariant Labs, Apr 2025): hidden instructions in tool descriptions that the model reads but users rarely do. In the WhatsApp demo, a benign-looking `get_fact_of_the_day()` from a malicious server redirected `send_message` on a separate, trusted WhatsApp MCP server. It needed no compromise of that server, only co-residence in one agent. Rug pull: the description changes after user approval. One analysis reports 5.5% of MCP servers show tool poisoning and 33% allow unrestricted network access. Mitigations: hash approved tool descriptions and block or alert on change, show descriptions to users, and alert on diffs — [Invariant Labs](https://invariantlabs.ai/blog/whatsapp-mcp-exploited); [Docker](https://www.docker.com/blog/mcp-horror-stories-whatsapp-data-exfiltration-issue/); [CSA whitepaper, May 2026](https://labs.cloudsecurityalliance.org/research-rb/csa-whitepaper-mcp-security-tool-poisoning-20260506-csa-styl/)
- A malicious Google Calendar event, an MCP calendar server and a local code-exec tool achieved RCE/exfiltration (LayerX against Claude Desktop Extensions). The MCP maintainers note that "any agent with unrestrained shell access sits one injected instruction away from exfiltration". Annotations "don't make the model resist prompt injection", but a `seesUntrustedData`-style hint could let the host taint the session and tighten approvals — [MCP blog, Mar 16, 2026](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/blog/content/posts/2026-03-16-tool-annotations.md)
- Protocol-level hardening: RFC 8707 resource indicators (2025-06-18) stop token passthrough to malicious servers. Streamable HTTP must return 403 on an invalid `Origin` (2025-11-25). RFC 9207 `iss` validation and issuer-bound credentials, with DCR deprecated for CIMD (2026-07-28) — [MCP changelogs](https://github.com/modelcontextprotocol/modelcontextprotocol/tree/main/docs/specification)

**Secrets and egress**
- Keep credentials outside the sandbox and inject them at a proxy that validates the operation (Claude Code web git proxy). Egress is denied by default with domain allowlists (srt, Codex network proxy). Private/LAN destinations are blocked even when hostnames resolve there (Codex). Sensitive paths such as `~/.ssh` are read-denied (srt example config) — [Anthropic](https://www.anthropic.com/engineering/claude-code-sandboxing); [sandbox-runtime](https://github.com/anthropic-experimental/sandbox-runtime); [codex network-proxy](https://github.com/openai/codex/blob/main/codex-rs/network-proxy/README.md)
- The code-execution pattern enables PII tokenization: the MCP client replaces emails and phones with placeholders before data reaches the model and restores them when passing to tools — [Anthropic](https://www.anthropic.com/engineering/code-execution-with-mcp)
- Auto mode's classifier deliberately excludes tool outputs from its input, so injected text cannot argue with the reviewer — [Anthropic](https://www.anthropic.com/engineering/claude-code-auto-mode)

### Inferences
- Defense-in-depth stack for a new harness, in priority order:
  1. Sandbox with egress allowlist and no in-sandbox secrets. This removes the "external communication" leg by default.
  2. Session taint tracking. Once any tool with untrusted/open-world output runs, require approval or the classifier for any action with exfiltration potential (network writes, messaging, git push to non-session branches).
  3. Injection probes on tool results that annotate rather than drop.
  4. An action classifier that sees user intent plus actions only.
  5. MCP hygiene: pin server versions, hash tool descriptions and diff on change, sandbox local MCP servers with srt, namespace tools per server, and treat annotations from unvetted servers as untrusted.
  6. For high-assurance workflows (email/finance), offer CaMeL-style Plan-Then-Execute or Dual-LLM modes, accepting a ~7-point utility cost (77% vs 84% on AgentDojo).
- Vendor ASR numbers are internal and use differing attackers. Nov 2025 (Opus 4.5 at 1.4%) and Aug 2026 (Opus 4.5 at 16.7–17.6% under a "stronger" attacker) are not comparable. This shows that attacker strength dominates reported rates.

### Gaps
- Could not fetch arXiv, simonwillison.net or openai.com primary texts. Paper details (CaMeL policy language, AgentDojo attack-success numbers for each pattern) come from abstracts and summaries only.
- No published OpenAI attack-success-rate numbers for Atlas or Codex were captured. No independent (non-vendor) 2026 benchmark of coding-agent injection rates was verified. arXiv 2606.05233 (793-episode browser benchmark) and 2601.17548 (injection in skills/tools/protocol ecosystems) surfaced but were not read.
- Prevalence stats for MCP poisoning (5.5%) and network access (33%) trace to a secondary citation of an unnamed arXiv analysis and are unverified.
