# Context Engineering for LLM Agents and Agentic Harnesses (state of the art as of 2026-11-03)

_Source-access note: this session's network allowed anthropic.com, platform.claude.com, code.claude.com, claude.com, gist.github.com and raw.githubusercontent.com (fetched directly, including raw Markdown of the Claude docs and OpenAI Codex source). manus.im, research.trychroma.com, arxiv.org, developers.openai.com, api-docs.deepseek.com, dbreunig.com and langchain.com were blocked, so findings from those come from search-engine extracts of the primary pages or from secondary write-ups. Each such case is labeled. Model names such as "Claude Opus 5.5", "Fable 5.1" and "Sonnet 5.5" appear as the live Claude docs list them in Nov 2026._

## Q1. Core principles: Anthropic's context-engineering guidance, "right altitude", minimal high-signal tokens, just-in-time retrieval vs pre-loading, progressive disclosure, agentic search vs RAG; context rot and the evidence for it

### Takeaway
Anthropic's guidance (Sep 2025, now built into the Claude docs) is to find "the smallest set of high-signal tokens that maximizes the likelihood of the desired outcome." In practice that means: write system prompts at the "right altitude"; keep tool sets minimal and non-overlapping; use a few canonical examples; keep lightweight identifiers (paths, queries, URLs) in context and load data just in time with tools such as grep, glob, head and tail; and use a hybrid that pre-loads only small, stable context such as CLAUDE.md. The case rests on "context rot": the 2025 Chroma study found recall and reasoning degrade as input grows, even on trivial tasks and well inside the advertised window. Distractors, low needle–question similarity and coherent haystacks make it worse.

### Cited Findings
**Anthropic, "Effective context engineering for AI agents" (Sep 29, 2025)**
- **Definition.** Context engineering is "strategies for curating and maintaining the optimal set of tokens" at inference, including everything beyond the prompt. The guiding principle is the smallest set of high-signal tokens that maximizes the likelihood of the desired outcome. — [Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- **Context rot and the attention budget.**
  - Recall accuracy falls as the window fills. Every token draws down a finite "attention budget".
  - n tokens create n² pairwise attention relationships, and models see fewer long sequences in training.
  - The decline is a gradient, not a cliff. Bigger windows are not expected to remove it. — [Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- **System prompt "right altitude".**
  - Avoid both brittle hard-coded if/else logic and vague high-level guidance.
  - Organize sections with XML tags or Markdown headers.
  - Aim for the minimal information that fully specifies behavior. Minimal does not have to mean short.
  - Start with a minimal prompt on the best model, then add instructions in response to observed failures. — [Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- **Tools.** Keep them token-efficient in what they return. Keep the set minimal with little functional overlap; bloated sets and ambiguous choices are a common failure mode. — [Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- **Examples.** Use a small, diverse set of canonical examples rather than stuffing in long lists of edge-case rules. — [Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- **Just-in-time retrieval.**
  - Agents keep lightweight identifiers (file paths, stored queries, web links) and load data at runtime with tools.
  - Claude Code analyzes large data this way, writing queries and using `head`/`tail` without loading full objects.
  - Metadata is signal: folder hierarchy, naming and timestamps all carry meaning (e.g. `test_utils.py` under `tests/` vs under `src/core_logic/`).
  - Progressive disclosure: file sizes suggest complexity, names suggest purpose, and timestamps proxy relevance. — [Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- **Trade-offs and the hybrid.**
  - Runtime exploration is slower than pre-computed retrieval and needs good tools and heuristics, or the agent wastes context.
  - Claude Code puts CLAUDE.md into context up front and uses glob/grep for everything else.
  - A hybrid may suit less dynamic domains such as legal or finance.
  - General rule: "do the simplest thing that works." — [Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- **Context as a first-class concept in the API docs.**
  - The Claude docs define context rot as accuracy and recall degrading as token count grows, so curating what is in context matters as much as available space.
  - Everything counts toward the window: system prompt, messages, tool results, images and documents, tool definitions, and the output including extended thinking.
  - Cached prefixes still occupy the window; caching changes price, not size. — [Claude docs: Context windows](https://platform.claude.com/docs/en/build-with-claude/context-windows)

**Chroma, "Context Rot: How Increasing Input Tokens Impacts LLM Performance" (Jul 14, 2025; Kelly Hong, Anton Troynikov, Jeff Huber)** — primary not reachable; facts below come via search extracts and secondary sources
- The study tested 18 LLMs, including GPT-4.1, Claude 4, Gemini 2.5 and Qwen3. Performance "grows increasingly unreliable as input length grows," even when the task stays trivial. — [Chroma report (via search extract)](https://research.trychroma.com/context-rot)
- **LongMemEval experiment.** 306 prompts: "focused" inputs of about 300 tokens vs full histories of about 113k tokens containing the same answer. Focused inputs scored better across models. — [ZenML summary](https://www.zenml.io/llmops-database/context-rot-evaluating-llm-performance-degradation-with-increasing-input-tokens). Hamel Husain's talk notes give about 100 tokens for the focused condition, a conflict with the ~300 figure. — [hamel.dev](https://hamel.dev/notes/llm/rag/p6-context_rot.html)
- **Secondary-source findings** — [cedrickchee gist notes on the report](https://gist.github.com/cedrickchee/b9646b3007b4a15f528351abd331984f):
  - Lower semantic similarity between question and needle makes performance fall faster with length.
  - Plausible distractors cut accuracy, and different distractors hurt to different degrees.
  - Models did better on shuffled (structureless) haystacks than on logically coherent ones.
  - On repeated-word copy tasks, long outputs degrade: models hallucinate, refuse, or insert random tokens.
  - Claude Opus 4 tends to abstain while GPT-4.1 tends to hallucinate.
  - Thinking mode helps, but a gap between focused and full inputs remains.
- About 194,480 LLM calls in total (8 input lengths × 11 needle positions per condition). — [ZenML summary](https://www.zenml.io/llmops-database/context-rot-evaluating-llm-performance-degradation-with-increasing-input-tokens)

**Background (2023, not re-fetched in this session)**
- "Lost in the Middle" (Liu et al.): long-context QA accuracy is highest when relevant information sits at the start or end of the input and drops for the middle. — [arXiv 2307.03172](https://arxiv.org/abs/2307.03172)

### Inferences
- **Tokens in the window are a cost to minimize.** A harness should report "tokens in window" per category (tools, system, project instructions, history, tool results), as Claude Code's `/context` does. Each category should have a budget.
- **Agentic search scales better than RAG for code.** Grep, glob and read-with-range plus identifiers (paths, line ranges, URLs) is the default for code and file-heavy work. Embeddings and RAG fit best as an optional select-tool for large, unstructured corpora, not as wholesale pre-loading.
- **Relevance alone is not enough.** Chroma's distractor and similarity findings mean "semantically near but wrong" content is especially harmful. Retrieval and recall steps should prefer precision, and stale or near-duplicate tool results should be pruned.
- **Quality degrades long before the hard limit.** Compaction and clearing thresholds should be set for quality, not just to avoid overflow (see Q2).

### Gaps
- Could not fetch the Chroma report directly. Exact per-model accuracy curves and the precise shuffled-vs-coherent deltas are unverified here.
- I found no controlled 2026 study that quantifies context-rot curves for the current 1M-token frontier models (Opus 4.6+ / Sonnet 5.x / GPT-5.x). Vendors assert degradation qualitatively, but I located no public numbers.

## Q2. Compaction and summarization: Claude Code auto-compact, Anthropic context editing and tool-result clearing, server-side compaction, OpenAI Responses compaction and Codex CLI, DeepSeek reasoning content across turns — thresholds, triggers, and what is preserved

### Takeaway
As of Nov 2026, compaction is a first-class API primitive at both Anthropic and OpenAI.
- **Anthropic** offers two flavors.
  - *Threshold compaction* (`compact_20260112`): the server summarizes once input reaches a trigger (default 150k, minimum 50k).
  - *On-demand compaction* (`compact-2026-09-04`): the client asks for a signed summary block whenever it chooses, can keep recent turns verbatim, and can run it in the background.
  - Lighter context editing clears old tool results (default trigger 100k, keep the last 3 tool uses) and old thinking blocks.
- **OpenAI** offers a stateless `/responses/compact` endpoint and an inline `context_management.compact_threshold`. Both return an opaque encrypted compaction item.
- **Codex CLI** summarizes with a short "handoff summary" prompt. It rebuilds history as initial context + recent user messages (≤20k tokens) + the summary, and auto-compacts at min(configured, 90% of the window).
- **Claude Code** auto-compacts near the window limit (~967k on native 1M models). Afterwards it re-injects CLAUDE.md, memory, plan, up to 5 recent files and skill bodies.
- **DeepSeek thinking mode** requires `reasoning_content` to be passed back inside tool-call loops; omitting it returns a 400 error.

### Cited Findings
**Anthropic API — threshold (automatic) server-side compaction**
- **How to enable.** Add `{"type":"compact_20260112"}` to `context_management.edits` with beta header `compact-2026-01-12`. — [Claude docs: Threshold compaction](https://platform.claude.com/docs/en/build-with-claude/compaction-threshold)
- **Parameters.**
  - `trigger` defaults to `{"type":"input_tokens","value":150000}`. `input_tokens` is the only trigger type, and the value must be ≥50,000.
  - `pause_after_compaction` defaults to false. When true, the API stops with `stop_reason: "compaction"` so you can re-insert content.
  - `instructions` fully replaces the default summarization prompt. — [Threshold compaction](https://platform.claude.com/docs/en/build-with-claude/compaction-threshold)
- **Mechanics.**
  - The API detects the trigger, writes a summary into a `compaction` block at the start of the assistant response, and continues.
  - On later requests, everything before the block is dropped. Streamed compaction blocks arrive as a single delta.
  - With server tools, the trigger is checked at each sampling iteration, so one request can compact several times.
  - `count_tokens` applies existing compaction blocks but never triggers new ones. — [Threshold compaction](https://platform.claude.com/docs/en/build-with-claude/compaction-threshold)
- **Default summarization prompt (one variant).** "You have written a partial transcript for the initial task above. Please write a summary of the transcript. The purpose of this summary is to provide continuity so you can continue to make progress towards solving the task in a future context, where the raw history above may not be accessible and will be replaced with this summary. Write down anything that would be helpful, including the state, next steps, learnings etc. You must wrap your summary in a <summary></summary> block." — [Threshold compaction](https://platform.claude.com/docs/en/build-with-claude/compaction-threshold)
- **Limitations.**
  - The summary is written by the request's own model; you cannot pick a cheaper summarizer.
  - With tools defined, the model sometimes calls a tool instead of summarizing, which yields `content: null`. The fix is custom instructions that say "Do not call any tools while writing this summary; respond with text only."
  - On Claude 5.1+ models, thinking blocks before the compaction block are not carried forward, so the summary must hold everything needed. — [Threshold compaction](https://platform.claude.com/docs/en/build-with-claude/compaction-threshold)
- **Billing.** Compaction adds a sampling iteration. Top-level `usage.input_tokens`/`output_tokens` exclude it, so true cost must be summed over `usage.iterations`. — [Threshold compaction](https://platform.claude.com/docs/en/build-with-claude/compaction-threshold)
- **Caching.**
  - Put `cache_control` on the compaction block.
  - Put a separate breakpoint at the end of the system prompt so the system-prompt cache survives each compaction and only the summary is re-written. — [Threshold compaction](https://platform.claude.com/docs/en/build-with-claude/compaction-threshold)
- **Budget guard.** `pause_after_compaction` can be combined with a compaction counter to enforce a total token budget across a long task. — [Threshold compaction](https://platform.claude.com/docs/en/build-with-claude/compaction-threshold)
- **Managed Agents.** Compaction is always on and emits an `agent.thread_context_compacted` event. — [Threshold compaction](https://platform.claude.com/docs/en/build-with-claude/compaction-threshold)

**Anthropic API — on-demand compaction (beta `compact-2026-09-04`; Anthropic recommends it "wherever it is available")**
- **Request.**
  - Send the conversation with `"compaction": {"type":"summarize"}`. The response is a single signed `compaction` block with `stop_reason: "compaction"` and no reply.
  - Send the same `system` and `tools` as the conversation. The summarizer reads the tool definitions but never runs a tool.
  - `max_tokens` caps summary plus thinking, so allow several thousand.
  - Cannot be combined with `context_management` on the same request.
  - `instructions` takes up to 16,384 characters and replaces the default prompt. — [Compaction on demand](https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand)
- **Swap rules.**
  - The block goes first in `messages` and replaces the summarized messages. Send exactly one block, unmodified (with its signature), on every later request.
  - Common errors: `compaction_block_misplaced`, `compaction_signature_invalid`, `compaction_nothing_to_summarize`.
  - Two silent failure modes: leaving summarized messages after the block re-sends them; omitting the block means no summary.
  - "Compact again" summarizes the old summary plus everything after it. — [Compaction on demand](https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand)
- **When to compact.** Estimate the next request as `input_tokens + cache_read_input_tokens + cache_creation_input_tokens + output_tokens` from the last `usage` and compare it to a limit you choose below the window. The Python and TypeScript SDK tool runners expose `compact_before_next_turn()`. — [Compaction on demand](https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand)
- **Failure handling.** A summary is produced only when the call ends normally with text. Otherwise you get HTTP 200 with empty content and `stop_reason` of `max_tokens`, `model_context_window_exceeded`, `tool_use`, `refusal` or `end_turn`. You must compact before overflow, because the conversation plus summarization prompt must still fit. — [Compaction on demand](https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand)
- **What is lost.** Images, documents, `container_upload` blocks and fetched URLs inside summarized messages are gone. Mid-conversation `role:"system"` instructions inside the summarized range stop applying, and per-message effort is not carried over. — [Compaction on demand](https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand)
- **Keep-tail compaction.**
  - There is no parameter for it: the client picks a cut point, sends only the older turns to compaction, then sends block + kept turns verbatim (thinking included).
  - The cut must not split a tool call from its result.
  - The reference loop keeps the last 2 turns (cut 4 messages from the end). — [Keep recent turns](https://platform.claude.com/docs/en/build-with-claude/compaction-keep-recent-turns)
- **Background ("async") compaction.**
  - Fire the compaction request with the history as it stands and record the message count; keep working on the full history.
  - When the block arrives, replace exactly the first N messages with it and keep turns appended since. Do not edit history or start a second compaction while one is pending.
  - Start early enough that the window can absorb the turns that arrive in the meantime. — [Compaction in the background](https://platform.claude.com/docs/en/build-with-claude/compaction-background)

**Anthropic API — context editing (beta `context-management-2025-06-27`)**
- **`clear_tool_uses_20250919`.**
  - When the prompt exceeds `trigger` (default 100,000 input tokens; can also be expressed in `tool_uses`), the oldest tool results are replaced with placeholder text.
  - `keep` defaults to the last 3 tool use/result pairs.
  - `clear_at_least` sets a minimum number of tokens to clear, otherwise the edit is skipped; this exists to justify breaking the cache.
  - `exclude_tools` lists tools whose results are never cleared. `clear_tool_inputs` (default false) also clears the call parameters. — [Claude docs: Context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing)
- **`clear_thinking_20251015`.** `keep` = `{type:"thinking_turns", value:N}` or `"all"`. The default keeps all prior thinking on Opus 4.5+, Sonnet 4.6+ and Haiku 5.5+, and only the last turn on older models. It must be listed first when combined with tool clearing. — [Context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing)
- **Server-side behavior and caching.**
  - Edits are applied server-side; the client keeps the full history.
  - Tool-result clearing invalidates the cached prefix from the cleared point, so you pay a cache write each time; hence `clear_at_least`.
  - Keeping thinking preserves the cache.
  - The response carries `context_management.applied_edits` with `cleared_input_tokens`. `count_tokens` returns `input_tokens` and `original_input_tokens`. — [Context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing)
- **With the memory tool.** Near the clearing threshold, Claude receives an automatic warning so it can save important results to memory files before they are cleared. — [Context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing)
- **Measured results (Sep 29, 2025, internal agentic-search eval).**
  - Memory tool + context editing improved performance 39% over baseline; context editing alone improved it 29%.
  - In a 100-turn web-search eval, context editing let runs complete that otherwise exhausted context and cut token consumption 84%. — [Claude blog: Managing context](https://claude.com/blog/context-management)
- **Deprecated client-side SDK compaction (`compaction_control`).**
  - Default `context_token_threshold` was 100,000. History of ~100k shrank to "~2–3k tokens" after compaction.
  - Its default summary prompt was structured into 5 sections: Task Overview; Current State (files modified, artifacts); Important Discoveries (constraints, decisions, errors resolved, failed approaches); Next Steps (blockers, priorities); Context to Preserve (user preferences, commitments).
  - Removed from the Python SDK v1.0 in favor of server-side compaction. — [Context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing)

**Claude Code (CLI) compaction behavior**
- **Default auto-compact threshold.**
  - Compacts at the model's context limit, with these exceptions:
    - Native-1M models (Fable, Sonnet 5+, Haiku 5.5, Opus 4.7+) compact at ~967K by default.
    - Sonnet 4.6 and Opus 4.6 without extended context compact at the 200K boundary.
    - Cloud sessions compact as the conversation approaches the limit.
  - Override with `/autocompact 500k`, `autoCompactWindow` in settings, `--autocompact`, or `CLAUDE_CODE_AUTO_COMPACT_WINDOW` (highest precedence). Each subagent can have its own window. — [Claude Code: Model config](https://code.claude.com/docs/en/model-config)
- **What survives compaction.**
  - System prompt and output style still apply.
  - Re-injected from disk: project-root CLAUDE.md and unscoped rules, auto memory, the plan-mode plan, and a fresh git status snapshot.
  - Skill descriptions are rebuilt. Invoked skill bodies are re-injected, capped at 5,000 tokens per skill and 25,000 total, oldest dropped first, and truncated from the end.
  - Path-scoped rules and nested CLAUDE.md files reload on demand.
  - Up to 5 most recently modified read or edited files are re-read; files over 5,000 tokens come back as path-only "Referenced file".
  - Background commands and subagents keep running, with a reminder so they are not duplicated.
  - Hook-added context is summarized. SessionStart hooks matching `compact` run again.
  - Since v2.1.198 the summarization request inherits the session's thinking configuration. — [Claude Code: Context window](https://code.claude.com/docs/en/context-window)
- **User controls.** `/compact <focus instructions>`; `/rewind` → "Summarize from here / up to here"; `/clear` between tasks; delegating large reads to subagents; custom compaction instructions placed in CLAUDE.md. — [Claude Code: Context window](https://code.claude.com/docs/en/context-window); [Claude Code: Costs](https://code.claude.com/docs/en/costs)
- **Compaction cost and caching.**
  - Compaction invalidates the conversation layer, but the system-prompt and project-context layers stay cached.
  - The summarization request reuses the warm prefix, so a mid-session `/compact` costs "a fraction of what the context size suggests". After the TTL expires it reprocesses uncached.
  - Recommendation: compact at natural task breaks rather than mid-task. Prefer `/rewind` (truncate to an already cached prefix) when abandoning a path.
  - With the 1-hour TTL, Claude Code can idle-compact (`idleCompaction`). — [Claude Code: Prompt caching](https://code.claude.com/docs/en/prompt-caching)
- **Earlier description (Sep 2025).** Claude Code compaction preserves "architectural decisions, unresolved bugs, and implementation details" and discards redundant tool outputs. Tune compaction prompts by first maximizing recall, then improving precision. — [Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

**OpenAI Responses API compaction** — developers.openai.com was blocked; facts are from search extracts of the official guide
- **Standalone endpoint.** `/responses/compact` is stateless and ZDR-friendly. It takes a full context window and returns a new compacted window that includes an encrypted compaction item. Pass the returned window unchanged to the next `/responses` call: "Do not prune /responses/compact output." — [OpenAI Compaction guide (via search)](https://developers.openai.com/api/docs/guides/compaction)
- **Server-side option.**
  - `context_management` with `compact_threshold` on `POST /responses`: when the rendered token count crosses the threshold, the server compacts and streams the compaction item. The docs' examples use 200,000 with `store=false`.
  - When chaining with `previous_response_id`, do not prune manually. When chaining with an input array, you may drop items before the latest compaction item. — [OpenAI Compaction guide (via search)](https://developers.openai.com/api/docs/guides/compaction); [OpenAI Context management (via search)](https://developers.openai.com/api/docs/guides/context-management)
- **Gateway support.** LiteLLM and the Vercel AI Gateway expose `/v1/responses/compact`. — [LiteLLM](https://docs.litellm.ai/docs/response_api_compact.md); [Vercel](https://vercel.com/docs/ai-gateway/sdks-and-apis/responses/compaction.md)

**OpenAI Codex CLI (read from the source on GitHub `main`, Nov 2026)**
- **Compaction prompt** (`codex-rs/prompts/templates/compact/prompt.md`): "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task. Include: Current progress and key decisions made; Important context, constraints, or user preferences; What remains to be done (clear next steps); Any critical data, examples, or references needed to continue. Be concise, structured, and focused on helping the next LLM seamlessly continue the work." — [openai/codex prompt.md](https://github.com/openai/codex/blob/main/codex-rs/prompts/templates/compact/prompt.md)
- **Summary prefix** injected before the summary: "Another language model started to solve this problem and produced a summary of its thinking process. You also have access to the state of the tools that were used by that language model. Use this to build on the work that has already been done and avoid duplicating work…" — [openai/codex summary_prefix.md](https://github.com/openai/codex/blob/main/codex-rs/prompts/templates/compact/summary_prefix.md)
- **History rebuild.**
  - New history = initial context, then the user's own prior messages, then the summary text. The context is inserted before the last real user message or summary.
  - User messages are kept newest-first up to `COMPACT_USER_MESSAGE_MAX_TOKENS = 20_000`; over-budget messages are truncated, and images and audio are dropped.
  - `compact_prompt` in config overrides the prompt. — [openai/codex compact.rs](https://github.com/openai/codex/blob/main/codex-rs/core/src/compact.rs)
- **Remote compaction.** The code supports both remote (server-side) and local streamed compaction. — [compact.rs](https://github.com/openai/codex/blob/main/codex-rs/core/src/compact.rs)
- **Thresholds.**
  - `auto_compact_token_limit()` = min(configured `model_auto_compact_token_limit`, 90% of the context window). Without config it defaults to 90% of the window.
  - `effective_context_window_percent` defaults to 95%, reserving headroom for system prompt, tools and output.
  - Further config: `model_auto_compact_token_limit_scope` (full context vs tokens after the carried compaction prefix) and `model_post_turn_compact_threshold_percent` (turn-end compaction; 0 disables it).
  - The `token_budget` feature has `reminder_threshold_tokens`, `auto_compact_fallback_prompt` (≤2,000 bytes) and `auto_compact_fallback_buffer_tokens`. — [openai/codex openai_models.rs](https://github.com/openai/codex/blob/main/codex-rs/protocol/src/openai_models.rs); [openai/codex config/mod.rs](https://github.com/openai/codex/blob/main/codex-rs/core/src/config/mod.rs)
- **Known bug.** In codex-cli 0.117.0, changing `model_context_window` could make auto-compaction fail permanently after the first overflow because the check read a stale token count. — [openai/codex issue #16068](https://github.com/openai/codex/issues/16068)

**DeepSeek thinking mode** — api-docs.deepseek.com was blocked; from search extracts of the official guide
- In thinking mode with tool calls, the assistant message's `reasoning_content` must be passed back with `content` and `tool_calls`, or the API returns a 400. Without `tools`, it does not need to be passed back and is ignored if sent. The official sample appends the full assistant message. — [DeepSeek Thinking Mode guide (via search)](https://api-docs.deepseek.com/guides/thinking_mode)
- A secondary source says reasoning carries forward within one question and old reasoning should be cleared when a new user question begins (not DeepSeek's own wording). — [CometAPI guide](https://www.cometapi.com/how-to-use-deepseek-v3-2-api/)

**Anthropic thinking blocks across turns**
- On Opus 4.5+, Sonnet 4.6+, Haiku 5.5 and Fable/Mythos models, the API keeps previous thinking blocks by default, and they count as input. Older models strip them automatically.
- Thinking blocks paired with tool results must be returned unmodified, with their signature. — [Claude docs: Context windows](https://platform.claude.com/docs/en/build-with-claude/context-windows)

**Research on learned or optimized compression (2025–2026)**
- **ACON** (Microsoft/KAIST, arXiv 2510.00615; ICML 2026).
  - Optimizes natural-language compression guidelines by contrasting paired trajectories where full context succeeds and compressed context fails.
  - Cuts peak tokens 26–54% on AppWorld, OfficeBench and multi-objective QA while largely preserving accuracy.
  - Distilled compressors keep >95% of accuracy; smaller LMs as agents improve by up to 46%. — [arXiv 2510.00615 (via search)](https://arxiv.org/abs/2510.00615); [microsoft/acon](https://github.com/microsoft/acon); [ICML 2026](https://icml.cc/virtual/2026/poster/66270)
- **AgentFold** (arXiv 2510.24699; ICLR 2026).
  - Proactive "folding" of history: it criticizes ReAct context saturation and "fixed full-history summarization" that risks irreversible detail loss.
  - The 30B-A3B model scores 36.2% on BrowseComp vs DeepSeek-V3.1-671B's 30.0%, and 62.1% on WideSearch.
  - Context grows from about 3.5k to 7k tokens over 100 turns. — [arXiv 2510.24699 (via search)](https://arxiv.org/abs/2510.24699)
- **Context-Folding / FoldGRPO** (arXiv 2510.11967; ICML 2026).
  - `branch`/`return` actions open sub-trajectories and fold them on return, trained with RL and token-level process rewards.
  - Matches baselines with up to 10× smaller active context: 32K active budget, ≤10 branches.
  - Seed-OSS-36B scores 62.0% on BrowseComp-Plus and 58.0% on SWE-Bench Verified. — [arXiv 2510.11967 (via search)](https://arxiv.org/abs/2510.11967); [PMLR](https://proceedings.mlr.press/v306/sun26x.html)
- **Recursive Language Models** (Zhang, Kraska, Khattab; arXiv 2512.24601).
  - The prompt is held as a variable in a Python REPL; the model inspects and chunks it with code and recursively calls sub-models.
  - Reported to handle 10M+ tokens. On BrowseComp-Plus (1,000 docs, 6–11M tokens), RLM(GPT-5) scores 91.3 vs 70.5 for a compaction agent and 51.0 for CodeAct+BM25. Numbers are from secondary summaries. — [arXiv 2512.24601 (via search)](https://arxiv.org/abs/2512.24601)

### Inferences
- **Recommended compaction ladder** for a harness, cheapest and most reversible first:
  1. Cap and paginate tool output at ingestion.
  2. Offload large outputs to files and keep only a path or summary.
  3. Clear stale tool results with `keep` ≈ 3–5 and a `clear_at_least` large enough to amortize the cache write.
  4. Summarize older turns with a structured handoff prompt, keeping the last 1–2 turns verbatim.
  5. Re-inject durable state after compaction: instruction files, the plan or todo, a few recently touched files, and active skills. This mirrors Claude Code's list.
  6. For multi-session work, start a fresh context bootstrapped from progress files (see Q6).
- **Summary schema.** Converge on the structure shared by the Anthropic SDK prompt and the Codex prompt: task and constraints, current state and files touched, decisions and rationale, errors and failed approaches, next steps, and user preferences. Always say "do not call tools."
- **Trigger points.** Anthropic's 150k default and Manus's 128k–200k "pre-rot threshold" (Q3) suggest compacting well before 1M windows fill, for quality. Claude Code's ~967K default on 1M models optimizes for continuity and cost, not necessarily for quality, so a harness may set a lower `autoCompactWindow` for precision-sensitive work. This is inference; no public quality-vs-threshold curve was found.
- **Async compaction** removes the user-visible pause and should be the default where supported. It requires append-only history semantics.
- **Thinking blocks.** Never edit earlier turns client-side, because that breaks signatures and the prefix check on 5.1+ models. Use server-side edits instead.
- **Opaque summaries.** OpenAI's encrypted compaction items cannot be inspected or edited. A cross-provider harness must store its own readable state (todo, notes) rather than relying on the opaque summary.

### Gaps
- OpenAI's exact default compaction prompt, the contents of the encrypted item, and whether `compact_threshold` has a minimum value or default were not verifiable; the docs domain was blocked.
- Codex's per-model default `auto_compact_token_limit` values (from the model catalog) and whether OpenAI-hosted sessions default to remote compaction were not confirmed. Third-party posts conflict (64k vs ~200k) — [danielvaughan Apr 2026](https://codex.danielvaughan.com/2026/04/16/codex-cli-context-compaction-tuning-long-sessions/), [danielvaughan Aug 2026](https://codex.danielvaughan.com/2026/08/13/context-compaction-theory-formal-analysis-bounded-context-windows-codex-cli-selection-generation-communication-complexity/).
- The DeepSeek docs could not be fetched. Current model IDs (one extract references `deepseek-flash` with `extra_body={"thinking":{"type":"enabled"}}`, suggesting a post-V3.2 generation) and any server-side context-management features are unverified.
- I found no head-to-head public benchmark comparing Anthropic server compaction, OpenAI compaction and Codex's handoff prompt.

## Q3. KV-cache and prompt caching as a design constraint: Manus lessons, Anthropic prompt caching (breakpoints, TTLs, pricing), OpenAI automatic caching, DeepSeek disk caching — cost and latency numbers

### Takeaway
For agents, input tokens dominate cost; Manus reports about 100:1 input:output. KV-cache or prompt-cache hit rate is therefore the key production metric. Design rules that follow:
- Keep a stable, deterministic prefix (tools → system → project context), with no timestamps and no reordering.
- Keep history append-only.
- Never mutate tool definitions mid-session. Mask or defer tools instead, and on Claude use the newer mid-conversation system messages, `tool_addition`/`tool_removal`, and turn-scoped reminders, which add dynamic content without breaking the prefix.

Provider numbers in brief:
- **Anthropic.** Cache reads cost 0.1× base input (0.05× on Opus/Sonnet 5.5, 0.025× on Fable/Mythos 5.1). Writes cost 1.25× for the 5-minute TTL or 2× for 1 hour. Up to 4 breakpoints and a 20-block lookback.
- **OpenAI.** Caching is automatic for prompts ≥1,024 tokens, with up to 90% input-cost and 80% latency reduction, plus `prompt_cache_key` and 24h retention.
- **DeepSeek.** Automatic disk-based caching; hit tokens cost a small fraction of misses.

### Cited Findings
**Manus, "Context Engineering for AI Agents: Lessons from Building Manus"** (Yichao "Peak" Ji, Jul 18, 2025) — primary blocked; facts via search extracts and a gist summary
- **KV-cache hit rate** is "the single most important metric for a production-stage AI agent" because it drives both latency and cost. — [Manus blog (via search)](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- **Economics.** The average input:output ratio is about 100:1, with a typical task around 50 tool calls. With Claude Sonnet, cached input cost $0.30/MTok vs $3/MTok uncached, a 10× difference. — [Manus blog (via search)](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- **Rules: stable prefix, append-only, deterministic serialization.**
  - Keep the prompt prefix stable: a single-token change invalidates the cache from that point on.
  - Make context append-only.
  - Ensure deterministic serialization.
  - Tool definitions sit near the front, so changing them invalidates the cache for all later actions and observations. — [Manus blog (via search)](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- **Mask, don't remove.** Rather than dynamically adding or removing tools (or RAG-selecting tools), Manus keeps definitions fixed and uses logit masking during decoding. Action names share consistent prefixes (`browser_*`, `shell_*`) so whole groups can be constrained. — [gist summary](https://gist.github.com/cedrickchee/b9646b3007b4a15f528351abd331984f); [Manus blog (via search)](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- **File system as context.** The file system is "the ultimate context": unlimited, persistent, directly operable. Compression should be restorable, e.g. keep a URL or file path so content can be re-fetched instead of keeping the full content. — [gist summary](https://gist.github.com/cedrickchee/b9646b3007b4a15f528351abd331984f)
- **Recitation via `todo.md`.** The agent rewrites a todo file step by step, pushing the global plan into recent attention to fight lost-in-the-middle and goal drift over long loops. — [gist summary](https://gist.github.com/cedrickchee/b9646b3007b4a15f528351abd331984f)
- **Keep the wrong stuff in.** Failed actions and error traces stay in context so the model updates its beliefs; "error recovery is one of the clearest indicators of true agentic behavior." — [gist summary](https://gist.github.com/cedrickchee/b9646b3007b4a15f528351abd331984f)
- **Don't get few-shotted.** Uniform context makes agents brittle and imitative. Introduce structured variation in serialization templates, phrasing and order. — [gist summary](https://gist.github.com/cedrickchee/b9646b3007b4a15f528351abd331984f)
- **Manus × LangChain webinar (Oct 2025), via secondary summaries.**
  - Manus tracks a "pre-rot threshold", typically ~128K–200K tokens, and triggers context reduction before it.
  - Compaction is reversible: it strips content that also exists in the environment or file system and keeps a path, so the agent can re-read it.
  - Summarization is the irreversible fallback, used only when compaction no longer frees enough space. — [ZenML summary](https://www.zenml.io/llmops-database/context-engineering-for-production-ai-agents-at-scale); [Phil Schmid, Context Engineering Part 2](https://www.philschmid.de/context-engineering-part-2); [Luma event page](https://luma.com/819i5ime)

**Anthropic prompt caching (Claude docs, Nov 2026)**
- **Two modes.**
  - *Automatic caching*: a single top-level `cache_control` moves the breakpoint to the last cacheable block each turn.
  - *Explicit breakpoints*: up to 4 per request; the automatic one uses one slot.
  - Cache prefix order is `tools` → `system` → `messages`. — [Claude docs: Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- **Pricing multipliers.**
  - 5-minute writes cost 1.25× base input; 1-hour writes cost 2×. Reads and refreshes cost 0.1× base.
  - Exceptions: Opus 5.5 and Sonnet 5.5 reads cost 0.05×; Fable 5.1 and Mythos 5.1 reads cost 0.025×. Multipliers stack with batch, long-context and data-residency modifiers.
  - Examples ($/MTok, base / 5m write / 1h write / read / output):
    - Opus 5.5: 4 / 5 / 8 / 0.20 / 20
    - Sonnet 4.6: 3 / 3.75 / 6 / 0.30 / 15
    - Sonnet 5.5: 2 / 2.50 / 4 / 0.10 / 10
    - Haiku 5.5 (≤100k prompt): 0.10 / 0.125 / 0.20 / 0.01 / 0.50 — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- **TTL semantics.**
  - Default 5 minutes, refreshed free on each hit.
  - Lifetime is measured from the start of the request, so a 4-minute streamed response leaves about 1 minute to send the follow-up.
  - A cache entry becomes available only once the first response starts, so parallel fan-out should wait for the first response. — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- **Lookback.**
  - Writes occur only at breakpoints.
  - Reads walk back at most 20 blocks (runs of consecutive tool_use or tool_result blocks count as one position) looking for prior writes.
  - A growing conversation that adds 20 or more blocks between writes misses unless a second breakpoint is added.
  - Classic mistake: a breakpoint on a block containing a timestamp means a write every request and never a read. — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- **Minimum cacheable length.**
  - 512 tokens: Opus 5.5, Opus 5, Sonnet 5.5, Fable 5/5.1, Haiku 5.5
  - 1,024: Sonnet 4.6, Sonnet 5, Opus 4.8
  - 2,048: Opus 4.7
  - 4,096: Opus 4.6, Opus 4.5, Haiku 4.5
  - Shorter prompts silently don't cache. — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- **Invalidation matrix.**
  - Tool definition changes invalidate everything.
  - Web-search/citations toggles and the speed setting invalidate system + messages.
  - `tool_choice`, images, thinking parameters and the effort setting invalidate messages; thinking and effort can also invalidate tools and system on some models.
  - On older models, non-tool-result user content strips cached thinking. — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- **Cache-preserving dynamic content (Nov 2026 features).**
  - *Mid-conversation system messages*: append `{"role":"system"}` to `messages` instead of editing top-level `system`; no beta header on supported 5.x models.
  - *Mid-conversation tool changes*: beta `inline-tools-2026-09-15` with `tool_addition`/`tool_removal` blocks, including defining a new tool by value. The `tools` array and earlier messages stay byte-identical, so the cache hits.
  - *Turn-scoped system messages*: beta `mid-conversation-system-clear-at-2026-08-21`, `clear_at: "next_user_message"`. The message renders only until the next user or tool_result message, then costs 0 tokens. Designed for per-turn reminders in tool loops that never pile up; resend cleared messages verbatim. — [Claude docs: Mid-conversation system messages](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages)
- **Deferred tools and caching.** Deferred tools (`defer_loading: true`) are excluded from the system-prompt prefix. When discovered, a `tool_reference` is appended inline, so the prefix and cache are untouched. A deferred tool cannot carry `cache_control`. — [Tool use with prompt caching](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-use-with-prompt-caching); [Tool search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)
- **Rollout order.** Cache tool definitions from day one (the 25% write markup pays back on the second hit). Add tool search past about 20 tools, then context editing for long runs, then programmatic tool calling for chains. — [Claude docs: Manage tool context](https://platform.claude.com/docs/en/agents-and-tools/tool-use/manage-tool-context)

**Claude Code as a reference implementation of cache-aware design**
- **Layering.** Request order is system prompt + tool definitions → project context (CLAUDE.md, memory, unscoped rules) → conversation. Plan mode and skill loading append messages rather than editing the prefix. File edits produce an appended `<system-reminder>` instead of rewriting earlier reads. Editing CLAUDE.md applies only after `/clear`, `/compact` or a restart, precisely so the cache is preserved. — [Claude Code: Prompt caching](https://code.claude.com/docs/en/prompt-caching)
- **Cache-busting actions.** Switching model (each model has its own cache), changing effort (except on Opus/Sonnet/Haiku 5.5 and Fable 5.1), connecting or removing an MCP server, denying an entire tool, compaction, and removing batches of old images. — [Claude Code: Prompt caching](https://code.claude.com/docs/en/prompt-caching)
- **TTL policy.**
  - Main conversation: 1 hour on a subscription within plan usage, 5 minutes on API key or usage credits.
  - Subagents, compaction, forks and titles: 5 minutes.
  - Subagents start their own cache, while forks inherit the parent's prefix and read its cache.
  - Workflow fan-outs hold all but the first agent for up to 5 seconds so they read the first one's cached prefix. — [Claude Code: Prompt caching](https://code.claude.com/docs/en/prompt-caching)
- **Monitoring.** `/usage` shows a "Prompt cache (main)" line (share of input served from cache, misses, warm/cold) and recommendations such as "Trim ~/code/api/CLAUDE.md · 49k tokens per request". — [Claude Code: Costs](https://code.claude.com/docs/en/costs)
- **Usage accounting.** `input_tokens` counts only tokens after the last breakpoint. Total input = `cache_read_input_tokens + cache_creation_input_tokens + input_tokens`. — [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)

**OpenAI prompt caching** — official docs via search extracts
- **Basics.** Automatic for prompts ≥1,024 tokens; up to 80% latency and 90% input-cost reduction. The `cached_tokens` usage field is always present. — [OpenAI Prompt caching guide (via search)](https://developers.openai.com/api/docs/guides/prompt-caching)
- **`prompt_cache_key`.** Routes requests that share a prefix to the same cache. Keep each prefix+key combination under ~15 requests/minute to avoid overflow. — [OpenAI Prompt caching (via search)](https://developers.openai.com/api/docs/guides/prompt-caching)
- **`prompt_cache_retention`.**
  - Values are `in_memory` and `24h`, at the same price. Extended retention stores KV tensors on GPU-local storage and is not ZDR-eligible.
  - Defaults reportedly vary by org and platform; Azure says `in_memory` for gpt-5.4 and older. — [OpenAI (via search)](https://developers.openai.com/api/docs/guides/prompt-caching); [Azure Foundry](https://learn.microsoft.com/en-ca/Azure/foundry/openai/how-to/prompt-caching)
- **GPT-5.6+ changes (per search extract of current docs).** `prompt_cache_key` is required for the "more reliable matching". Retention becomes a minimum lifetime of ≥30 minutes. Explicit breakpoints and `prompt_cache_options` are introduced and rejected by older models. — [OpenAI Prompt caching (via search)](https://developers.openai.com/api/docs/guides/prompt-caching)

**DeepSeek context caching**
- Disk-based context caching launched Aug 2, 2024, at $0.014/MTok for cache hits at that time. — [DeepSeek news](https://deepseek.com/en/news/context-caching/)
- It is automatic. Usage reports separate hit and miss token counts, and a hit requires the prefix to already be persisted to disk. — [DeepSeek KV cache guide (via search)](https://api-docs.deepseek.com/guides/kv_cache)
- **Later pricing (secondary).** V4 Flash: $0.0028/MTok hit vs $0.14/MTok miss. V4 Pro: $0.003625 hit vs $0.435 miss. — [benchlm.ai](https://www.benchlm.ai/deepseek/api-pricing)

### Inferences
- **Fixed serialization order:** `[tools (stable; most deferred)] → [system (stable)] → [project instructions/skills index (stable per session)] → [append-only conversation]`. Recommended breakpoints:
  - end of tools
  - end of system or project context
  - the moving automatic breakpoint
  - an extra mid-history breakpoint whenever a turn can add 20 or more blocks (e.g. large parallel tool fan-outs)
- **Dynamic data must never sit in the prefix.** This covers time, environment state, file-change notices, budgets and todo recitation. Append it as messages; on Claude 5.x, use turn-scoped system messages so reminders render for one turn and then vanish without breaking the cache. This resolves the Manus tension between recitation and cache stability.
- **Tool gating.** Prefer `defer_loading` plus tool search, or `tool_addition`, over editing `tools`. For self-hosted models, the logit/prefill masking pattern with namespaced tool prefixes is the equivalent.
- **Model routing breaks the cache.** Avoid mid-session model or effort switches. Route side tasks such as compaction to subagents or forks that share the prefix. Note that Anthropic's server compaction cannot use a cheaper model.
- **Per-turn telemetry.** Log cache-hit ratio, cache-write tokens and "expected rebuilds" (compaction, clearing) separately from unexpected misses. Treat unexpected misses as regressions; one Anthropic-focused article reportedly calls cache misses "production incidents" — [implicator.ai](https://www.implicator.ai/anthropic-says-cache-misses-are-production-incidents-reveals-caching-shaped-claude-code/) (secondary, not verified).

### Gaps
- The full Manus post could not be fetched. Specifics such as its response-prefill modes for constraining tool choice and its vLLM/session-ID routing advice are recalled from training and not verified here, so they are omitted from the findings.
- The DeepSeek cache unit or granularity (e.g. a fixed token interval) and the current official price list were not verified.
- I found no independent measurement of real-world agent cache-hit rates across providers.

## Q4. Tool definitions as context: tool-count limits, tool search and deferred loading, programmatic tool calling, the MCP code-execution pattern, Agent Skills progressive disclosure

### Takeaway
Tool definitions are often the largest fixed context cost: about 55k tokens for 5 MCP servers, and up to 134k observed. Claude's tool-selection accuracy degrades past roughly 30–50 tools. The SOTA stack:
1. Defer most tools behind a regex or BM25 tool search, with 3–5 hot tools always loaded. This gives an 85%+ reduction and better MCP-eval accuracy.
2. Let the model call tools from code (programmatic tool calling: 37% fewer tokens), or expose MCP servers as a code API on a filesystem (150k → 2k tokens).
3. Package procedures as Skills with three-level progressive disclosure: ~100 tokens of metadata per skill, then a <5k-token SKILL.md, then resources loaded on demand.
4. Make tool outputs small: a 25k-token cap, pagination, and concise/detailed modes.

### Cited Findings
**Tool search and `defer_loading` (stable)**
- **Problem.** A typical multi-server setup (GitHub, Slack, Sentry, Grafana, Splunk) uses ~55k tokens in definitions. Tool search typically cuts this by over 85%, loading 3–5 tools per request. Claude's tool-selection accuracy "degrades once you exceed 30–50 available tools." — [Claude docs: Tool search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)
- **Variants.**
  - `tool_search_tool_regex_20251119`: Python `re.search` patterns, ≤200 characters.
  - `tool_search_tool_bm25_20251119`: natural-language queries, ≤500 characters.
  - Both search names, descriptions, argument names and argument descriptions. They return up to 5 `tool_reference` blocks by default (`limit` 1–10,000) and support up to 10,000 deferred tools per request.
  - Keep the 3–5 most-used tools non-deferred, and never defer the search tool itself.
  - Custom search (e.g. embeddings) is possible by returning `tool_reference` blocks from your own tool. — [Tool search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)
- **Anthropic engineering, "Advanced tool use" (Nov 24, 2025).**
  - 58 tools across 5 servers ≈ 55K tokens (Jira alone ≈ 17K); definitions of 134K seen before optimization.
  - Traditional loading ≈ 77K before work, vs ≈ 8.7K with tool search (≈500 upfront + ~3K discovered).
  - MCP-eval accuracy: Opus 4 rose from 49% to 74%, Opus 4.5 from 79.5% to 88.1%.
  - Use it when definitions exceed 10K tokens, you have 10+ tools, or multiple MCP servers. — [Anthropic: Advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use)
- **Claude Code.** MCP tool definitions are deferred by default; only tool names and server instructions enter context until a tool is used. — [Claude Code: Costs](https://code.claude.com/docs/en/costs)

**Programmatic tool calling (PTC)**
- **Mechanism.**
  - Mark tools with `allowed_callers: ["code_execution_20260120"]`. Claude writes Python that calls them as async functions (parallel via `asyncio.gather`) inside the code-execution container.
  - Each call pauses execution and returns a `tool_use` with a `caller` field. Intermediate results never enter the context window; only the final output does.
  - Requires code execution `code_execution_20260120`+. Not on Haiku 4.5. Not ZDR-eligible (data kept up to 30 days). `allowed_callers` is guidance, not a security boundary. — [Claude docs: Programmatic tool calling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling)
- **Measured results.**
  - Average tokens fell from 43,588 to 27,297 (−37%) on complex research tasks.
  - Internal knowledge retrieval rose from 25.6% to 28.5%; GIA from 46.5% to 51.2%.
  - In an expense example, 200KB+ of intermediate data became ~1KB in context.
  - Use for 3+ dependent calls, large data, filtering, or parallel fan-out. — [Anthropic: Advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use)
- **Tool use examples (`input_examples`).** Accuracy on complex parameters rose from 72% to 90%; use 1–5 realistic examples per tool. — [Anthropic: Advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use)

**MCP via code execution (Anthropic, Nov 4, 2025)**
- Expose MCP servers as a code API on a filesystem (e.g. `./servers/google-drive/getDocument.ts`). The agent lists directories and reads only the files it needs, or calls a `search_tools` tool with detail levels: name only, name + description, or full schema. — [Anthropic: Code execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp)
- **Reported savings.**
  - 150,000 → 2,000 tokens (98.7%).
  - A 10,000-row spreadsheet filtered to 5 rows before reaching the model.
  - A 2-hour meeting transcript (~50,000 tokens) need not pass through context twice.
  - PII can be tokenized (e.g. `[EMAIL_1]`) so the model never sees raw values.
  - Intermediate state persists to files, and working code can be saved as skills.
- **Caveat.** Requires a secure sandbox with resource limits and monitoring. — [Anthropic: Code execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp)

**Tool output hygiene (Anthropic, "Writing effective tools for agents", Sep 11, 2025)**
- **Limit output size.** Paginate, select ranges, filter and truncate with sensible defaults. "Claude Code restricts tool responses to 25,000 tokens by default." When truncating, steer the agent toward narrower, targeted searches. — [Anthropic: Writing tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
- **Verbosity modes.** A `response_format` enum (`concise`/`detailed`): one example used 206 tokens detailed vs 72 concise, and a Slack example was about 1/3 the tokens when concise. — [Anthropic: Writing tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
- **Design rules.**
  - Return semantic fields (names, file types), not UUIDs or MIME types; resolving IDs to names reduced hallucinations.
  - Consolidate multi-step operations into one tool (e.g. `schedule_event`, `search_logs` with context lines).
  - Namespace by service or resource; whether a prefix or suffix works better depends on the model.
  - Make errors actionable. — [Anthropic: Writing tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
- **Changing tools mid-session.** `tool_addition`/`tool_removal` (beta `inline-tools-2026-09-15`) add, change or withdraw tools mid-conversation without touching `tools`. Keep at least one non-deferred tool in `tools`, or the first by-value definition costs a full cache miss. — [Mid-conversation system messages](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages)

**Agent Skills progressive disclosure**
- **Three levels.**
  - Level 1: metadata (`name` + `description` from YAML frontmatter), always loaded, about 100 tokens per skill.
  - Level 2: the SKILL.md body, loaded when triggered, under 5k tokens.
  - Level 3+: bundled resources, zero tokens until read. Scripts run via bash, and only their output enters context. — [Claude docs: Agent Skills overview](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)
- **Authoring rules.**
  - `name` ≤64 characters. The `description` must say what the skill does and when to use it.
  - Keep the SKILL.md body under 500 lines.
  - Keep references one level deep from SKILL.md; add a table of contents to reference files over 100 lines.
  - Organize by domain so irrelevant schemas aren't loaded.
  - Ask "does this paragraph justify its token cost?" — [Claude docs: Skill authoring best practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices)
- **History and security.**
  - Skills launched Oct 16, 2025 and became an open standard (agentskills.io) on Dec 18, 2025.
  - Build skills from evaluation-observed gaps; watch real usage, especially name and description, which drive triggering.
  - Security: malicious skills can exfiltrate, so audit third-party skills. — [Anthropic: Equipping agents with Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- **After compaction (Claude Code).** Skill bodies are re-injected after compaction, capped at 5k tokens each and 25k total, truncating from the end. Put critical instructions at the top of SKILL.md. — [Claude Code: Context window](https://code.claude.com/docs/en/context-window)

**Tool count and confusion evidence (secondary)**
- Breunig's "context confusion" category cites tool overload. A secondary summary reports a quantized Llama 3.1 8B failing with 46 tools and succeeding with 19. — [Breunig, How to Fix Your Context](https://www.dbreunig.com/2025/06/26/how-to-fix-your-context.html) (via secondary summaries; not verified)

### Inferences
- **Default tool layout.** Always-on core tools (≤5–10: read, search/grep, edit, bash, todo, and tool_search) plus everything else deferred and indexed for regex or BM25 search. Use custom embedding search only for catalogs of thousands of tools.
- **Use code orchestration instead of chatty tool loops** for data-heavy, multi-call work. Mark bulk data tools as code-only callers so raw results never reach the transcript.
- **Every tool returns bounded, paginated, semantic output with a concise mode.** Large outputs go to files with a path plus a preview (this matches Manus's restorable compression).
- **Skills are where procedures live.** CLAUDE.md or AGENTS.md holds always-true facts (Q5), skills hold procedures, and tools hold actions. Each layer has its own loading policy and budget.

### Gaps
- No provider-neutral number for a "max tools before degradation" for GPT-5.x or DeepSeek models. The 30–50 figure is Anthropic's for Claude.
- No published comparison of BM25 vs regex vs embedding tool search accuracy was found.
- OpenAI's equivalent features (tool search, deferred tools, MCP connectors in Responses) were not researched in detail because the domain was blocked.

## Q5. Project instruction files: CLAUDE.md, AGENTS.md (cross-vendor standard), hierarchical/nested files, length and content best practices

### Takeaway
Instruction files are pre-loaded context paid on every request, so keep them short: under 200 lines per CLAUDE.md, per Anthropic. Fill them only with always-true facts: build and test commands, conventions, layout, and "always do X" rules. Push procedures to skills and area-specific guidance to path-scoped rules or nested files that load on demand. Avoid contradictions; Claude may pick either side. AGENTS.md is the cross-vendor "README for agents", stewarded since Dec 2025 by the Linux Foundation's Agentic AI Foundation. Claude Code reads AGENTS.md when no CLAUDE.md is present, or via `@AGENTS.md` import.

### Cited Findings
- **Content and size (Claude Code docs).**
  - Keep CLAUDE.md to facts needed every session; move multi-step procedures to skills and area-specific rules to path-scoped rules.
  - Target under 200 lines per file. "Longer files consume more context and reduce adherence."
  - Imports (`@path`) organize but don't reduce cost; they recurse up to 4 hops.
  - Contradictory instructions cause arbitrary choices, so audit periodically; `/doctor prompt-audit` does this.
  - Instructions are context, not enforcement; use PreToolUse hooks to block actions. — [Claude Code: Memory / CLAUDE.md](https://code.claude.com/docs/en/memory)
- **Hierarchy and loading.**
  - Load order runs broadest to most specific (user → project → local); a project instruction appears after a user instruction.
  - CLAUDE.md files above the working directory load at launch. Subdirectory CLAUDE.md files load on demand when Claude reads, writes or edits a file in that subtree.
  - `.claude/rules/*.md` with `paths:` glob frontmatter load only when matching files are touched.
  - `claudeMdExcludes` skips other teams' files in monorepos.
  - Block-level HTML comments are stripped before injection (free maintainer notes).
  - Auto memory loads its first 200 lines or 25KB. — [Claude Code: Memory](https://code.claude.com/docs/en/memory)
- **Compaction interaction.** Project-root CLAUDE.md is re-read and re-injected after `/compact`. Nested CLAUDE.md files and path-scoped rules are summarized away and reload only when a matching file is touched again; instructions given only in conversation are lost unless written to CLAUDE.md. — [Claude Code: Memory](https://code.claude.com/docs/en/memory); [Claude Code: Context window](https://code.claude.com/docs/en/context-window)
- **AGENTS.md handling in Claude Code.**
  - Only AGENTS.md (no CLAUDE.md) → reads AGENTS.md.
  - Both present → reads CLAUDE.md only, unless CLAUDE.md imports AGENTS.md.
  - Configurable via the "Project instructions" setting. — [Claude Code: Memory](https://code.claude.com/docs/en/memory)
- **AGENTS.md definition.** "A simple, open format for guiding coding agents… a README for agents: a dedicated, predictable place to provide context and instructions." The canonical example sections are dev environment tips, testing instructions and PR instructions. — [agentsmd/agents.md README](https://github.com/agentsmd/agents.md)
- **Stewardship.** On Dec 9, 2025 the Linux Foundation formed the Agentic AI Foundation (AAIF), co-founded by OpenAI, Anthropic and Block, with founding projects MCP, goose and AGENTS.md (donated by OpenAI). One report cites use across ~40,000 open-source projects. — [OpenAI: Agentic AI Foundation](https://openai.com/index/agentic-ai-foundation/); [AAIF announcement](https://aaif.io/news/linux-foundation-announces-formation-of-aaif); [InfoQ](https://infoq.com/news/2025/12/agentic-ai-foundation/)
- **Hybrid in practice.** Claude Code's hybrid strategy pre-loads CLAUDE.md and fetches everything else just in time. — [Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- **Cost visibility.** `/usage` can recommend "Trim ~/code/api/CLAUDE.md · 49k tokens per request", which shows oversized instruction files are a real cost driver. — [Claude Code: Costs](https://code.claude.com/docs/en/costs)

### Inferences
- **A harness should:**
  1. Discover AGENTS.md and CLAUDE.md from repo root down to the working directory and load them at session start, in the stable "project context" layer.
  2. Lazily load nested files and path-scoped rules as appended messages when the agent touches a matching path. Appending, rather than rewriting the prefix, keeps the cache.
  3. Re-inject root files after compaction.
  4. Lint files for length (>200 lines) and contradictions.
  5. Strip HTML comments.
- **Treat imported files as untrusted** when they resolve outside the repo; Claude Code asks for approval for external imports.

### Gaps
- I could not verify the AGENTS.md spec's nested-file precedence rule (e.g. "closest file wins") from a primary source; the agents.md site was blocked and the README doesn't state it.
- No controlled study was found on the instruction-file length vs adherence curve beyond Anthropic's "<200 lines" guidance.

## Q6. Long-context model features relevant to harnesses: 1M windows, context-awareness and token-budget signals, DeepSeek Sparse Attention, structured note-taking, sub-agent isolation

### Takeaway
1M-token windows are now the default on current Claude models (Opus 4.6+, Sonnet 4.6+, Haiku 5.5, Fable/Mythos), with no beta header and standard pricing except Haiku 5.5 above 100k. Bigger windows don't remove context rot, so harnesses still need:
- **Budget signals.** On some Claude models the API injects `<budget:token_budget>` and per-tool-call `<system_warning>` usage tags; on newer models, task budgets (beta, minimum 20k) inject a model-visible countdown.
- **External structured state.** Notes, todo, progress files, and the memory tool with its "ASSUME INTERRUPTION" protocol.
- **Sub-agent isolation.** Sub-agents return 1–2k-token summaries or file references.

DeepSeek's DSA (V3.2) makes long contexts cheaper by attending to only the top-2048 keys per query, but it doesn't change the curation problem.

### Cited Findings
- **1M windows.**
  - 1M-token windows on Fable 5/5.1, Mythos 5/5.1/Preview, Opus 5.5/5/4.8/4.7/4.6, Sonnet 5.5/5/4.6 and Haiku 5.5. Max output is 128k tokens per request.
  - 1M is the default with no beta header, billed at standard pricing except Haiku 5.5 above 100k.
  - Up to 600 images or PDF pages per request (100 on 200k models).
  - If input + `max_tokens` exceeds the window, generation stops with `stop_reason: "model_context_window_exceeded"` (Claude 4.5+). — [Claude docs: Context windows](https://platform.claude.com/docs/en/build-with-claude/context-windows); [Prompt caching pricing table](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- **Context awareness.**
  - Sonnet 5, Sonnet 4.6, Sonnet 4.5 and Haiku 4.5 receive `<budget:token_budget>200000</budget:token_budget>` (or 1M) in the system prompt, and after each tool call `<system_warning>Token usage: 35000/200000; 165000 remaining</system_warning>`. This is automatic, and image tokens are included.
  - Opus 4.7+, Sonnet 5.5, Fable and Haiku 5.5 do not receive these tags; use task budgets instead. — [Claude docs: Context windows](https://platform.claude.com/docs/en/build-with-claude/context-windows)
- **Task budgets.**
  - Beta `task-budgets-2026-03-13`: `output_config.task_budget = {type:"tokens", total, remaining?}`, covering thinking, tool calls, tool results and output across one agentic turn (until a user message without tool results).
  - The model sees a server-injected countdown, and the API returns no remaining field. Minimum `total` is 20,000.
  - Budgets that are too small can cause refusal-like early stopping.
  - Server compaction doesn't reset the budget. It is advisory, unlike `max_tokens`. — [Claude docs: Task budgets](https://platform.claude.com/docs/en/build-with-claude/task-budgets)
- **Structured note-taking.** The Sep 2025 post recommends agent-written notes outside the window (to-do lists, NOTES.md). Claude playing Pokémon tracked objectives such as "1,234 steps" of training and maintained maps and strategy notes across context resets. — [Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- **Memory tool** (`memory_20250818`).
  - Client-side file operations under `/memories`: view, create, str_replace, insert, delete, rename. `view` truncates text over 16,000 characters.
  - The API auto-injects "IMPORTANT: ALWAYS VIEW YOUR MEMORY DIRECTORY BEFORE DOING ANYTHING ELSE… ASSUME INTERRUPTION: Your context window might be reset at any moment, so you risk losing any progress that is not recorded in your memory directory."
  - The handler must block path traversal. — [Claude docs: Memory tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool)
- **Multi-window harness (Anthropic, Nov 26, 2025).**
  - Compaction alone was insufficient: Opus 4.5 looping across windows still fell short of a production web app.
  - Fix: an initializer agent writes `init.sh`, `claude-progress.txt`, a JSON feature list (200+ features, all initially failing; agents flip only `passes`) and an initial git commit. JSON was chosen because models are less likely to overwrite it than Markdown.
  - Each later coding session runs `pwd`, reads progress and the git log, runs a basic end-to-end test, then does one feature.
  - This counters "one-shotting" and premature victory. — [Anthropic: Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- **Sub-agent isolation.**
  - Sub-agents explore with tens of thousands of tokens but return condensed summaries, "often 1,000–2,000 tokens." — [Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
  - Anthropic's research system (Jun 13, 2025):
    - Multi-agent (Opus 4 lead + Sonnet 4 sub-agents) beat single-agent Opus 4 by 90.2%.
    - Token usage explains 80% of BrowseComp variance (95% with tool-call count and model).
    - Agents use ~4× chat tokens; multi-agent ~15×.
    - The lead saves its plan to memory because contexts over 200,000 tokens get truncated.
    - Sub-agents write outputs to external storage and return lightweight references to avoid a "game of telephone." — [Anthropic: Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)
  - In Claude Code, a subagent's large file reads stay out of the parent context; only the summary and a small metadata trailer return. Subagents have their own cache; forks share the parent's. — [Claude Code: Context window](https://code.claude.com/docs/en/context-window); [Claude Code: Prompt caching](https://code.claude.com/docs/en/prompt-caching)
- **DeepSeek Sparse Attention (V3.2).**
  - DSA is the only architectural change in DeepSeek-V3.2, introduced via continued training. — [DeepSeek-V3.2 report (arXiv 2512.02556, via search)](https://arxiv.org/pdf/2512.02556)
  - A "lightning indexer" selects the top-2048 key-value entries per query for main attention (`index_topk` default 2048). Training used a 2.1B-token dense warm-up of the indexer, then 943.7B tokens of sparse training. — [AMD technical article](https://www.amd.com/en/developer/resources/technical-articles/2025/amd-instinct-mi335x-gpu-supports-deepseek-v3-2-exp.html)
  - Core attention is O(L·k). — [Raschka architecture gallery](https://www.sebastianraschka.com/llm-architecture-gallery/deepseek-sparse-attention/)
  - The indexer itself remains O(N²) but with a small coefficient. Its top-k overlap with exact attention averages about 60% on a 128K HELMET variant, while end-to-end quality stays strong. — [arXiv 2512.03494 (via search)](https://arxiv.org/pdf/2512.03494)

### Inferences
- **A harness should give the model explicit budget awareness** on every provider: inject a short appended "tokens used / remaining / next compaction at N" note, ideally turn-scoped on Claude so it doesn't accumulate. It should also use native signals (context awareness, task budgets) where available.
- **Long-horizon state belongs in files, not in chat:** progress log, structured JSON task list, todo for recitation, NOTES, and git history. This makes compaction and fresh-context restarts cheap and robust.
- **Sub-agents are mainly a context-isolation tool,** worth it when exploration is large relative to the result. They cost about 15× tokens, so use them for breadth-first or parallel research and large reads, not for small edits. Return summaries or file references with an explicit length cap (e.g. ≤2k tokens).
- **DSA-style sparse attention** reduces long-prompt cost on DeepSeek, so pre-loading is cheaper there. Retrieval precision still matters because top-k selection is lossy (~60% overlap).

### Gaps
- OpenAI GPT-5.x context-awareness or budget-injection features and current context sizes were not verified (docs blocked).
- No public data was found on how much the `<system_warning>` usage tags or task budgets improve task completion.
- The DeepSeek V4 architecture and context length (secondary sources mention V4 Flash/Pro pricing) were not researched.

## Q7. Failure modes (poisoning, distraction, confusion, clash; lost-in-the-middle) and mitigation taxonomies (write/select/compress/isolate)

### Takeaway
Drew Breunig's June 2025 taxonomy names four long-context failure modes:
- **Poisoning:** a hallucination or error gets into context and is repeatedly referenced.
- **Distraction:** context so long the model over-focuses on it and neglects training knowledge.
- **Confusion:** superfluous content, especially excess tools, degrades output.
- **Clash:** conflicting information or tools accrue.

His fixes are RAG, tool loadout, context quarantine, pruning, summarization and offloading. LangChain's July 2025 umbrella, write / select / compress / isolate, maps onto the vendor features: memory and notes, tool search and JIT retrieval, compaction and clearing, and sub-agents. Research since late 2025 (ACE, ACON, AgentFold, Context-Folding, RLM) tries to learn or optimize these operations. ACE adds two failure modes of naive iterative summarization: "brevity bias" and "context collapse".

### Cited Findings
- **Breunig's taxonomy, "How Long Contexts Fail" (Jun 22, 2025)** — via a summarizing gist; primary blocked.
  - Context Poisoning: a hallucination or other error makes it into the context and is repeatedly referenced.
  - Context Distraction: the context grows so long that the model over-focuses on it, neglecting what it learned in training.
  - Context Confusion: superfluous content is used by the model to generate a low-quality response; MCP tool overload is the example.
  - Context Clash: new information and tools conflict with other information in context.
  - The examples come from the Gemini Pokémon agent. Large windows are mainly useful for summarization and fact retrieval, so beware of a model's "distraction ceiling." — [Breunig (via gist)](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html); [cedrickchee gist](https://gist.github.com/cedrickchee/b9646b3007b4a15f528351abd331984f)
- **Breunig's fixes, "How to Fix Your Context".**
  - RAG: selectively add relevant information.
  - Tool Loadout: select only relevant tool definitions.
  - Context Quarantine: isolate contexts in dedicated threads.
  - Context Pruning: remove irrelevant information.
  - Context Summarization: condense accrued context.
  - Context Offloading: store information outside the context, e.g. a "think"/scratchpad tool. He cites Anthropic's finding that the think tool with a domain prompt gave up to 54% improvement on a specialized-agent benchmark. — [Breunig (via gist)](https://www.dbreunig.com/2025/06/26/how-to-fix-your-context.html); [cedrickchee gist](https://gist.github.com/cedrickchee/b9646b3007b4a15f528351abd331984f)
- **LangChain, "Context Engineering for Agents" (Jul 2, 2025).**
  - Write: save context outside the window, e.g. scratchpads and memories.
  - Select: pull relevant context in, e.g. retrieval, few-shot examples, tools.
  - Compress: retain only the tokens needed, e.g. summarization and trimming.
  - Isolate: split context across sub-agents, state fields or sandboxes.
  - Context engineering is framed as "filling the context window with just the right information for the next step" (Karpathy's LLM-as-OS analogy). — [LangChain blog (via search)](https://www.langchain.com/blog/context-engineering-for-agents); [langchain-ai/context_engineering README](https://github.com/langchain-ai/context_engineering)
- **Error retention vs poisoning.** Manus's counterpoint is to keep failed actions and stack traces in context so the model adapts instead of repeating mistakes. — [gist summary of Manus](https://gist.github.com/cedrickchee/b9646b3007b4a15f528351abd331984f)
- **Few-shot rut.** Repetitive action-observation patterns induce drift. The mitigation is structured variation. — [gist summary of Manus](https://gist.github.com/cedrickchee/b9646b3007b4a15f528351abd331984f)
- **Lost in the middle and recitation.** Manus's `todo.md` recitation keeps goals in the recent-attention zone over ~50-step loops. — [gist summary of Manus](https://gist.github.com/cedrickchee/b9646b3007b4a15f528351abd331984f); background: [Liu et al. 2023](https://arxiv.org/abs/2307.03172)
- **Clash from instruction files.** Contradictory CLAUDE.md or rules lead Claude to pick one arbitrarily. — [Claude Code: Memory](https://code.claude.com/docs/en/memory)
- **ACE — Agentic Context Engineering** (arXiv 2510.04618, repo Nov 2025).
  - Treats context as an evolving "playbook" curated by Generator / Reflector / Curator roles.
  - Uses incremental delta updates with helpful/harmful counters and deterministic merge, de-duplication and pruning, to avoid "brevity bias" and "context collapse", where iterative rewriting erodes detail.
  - +10.6% on AppWorld (matching the top production GPT-4.1 agent with a smaller open model) and +8.6% on finance tasks.
  - 86.9% lower adaptation latency; −82.3% latency and −75.1% rollouts vs GEPA offline; −91.5% latency and −83.6% token cost vs Dynamic Cheatsheet online. — [ace-agent/ace README](https://github.com/ace-agent/ace); [arXiv 2510.04618](https://arxiv.org/abs/2510.04618)
- **2026 research signal.** "Uniform Decay, Heterogeneous Causes: Long-Horizon Failure in Long-Context Agents" appears as a NeurIPS 2026 listing — [NeurIPS 2026](https://neurips.cc/virtual/2026/173812) (title only; content not reviewed).

### Inferences
- **Harness guards mapped to failure modes:**
  - *Poisoning*: verification steps (tests, schema checks) before facts enter durable notes. Allow notes to be corrected and versioned. Prefer citing tool outputs over model restatements.
  - *Distraction*: quality-oriented compaction thresholds, tool-result clearing, and sub-agent quarantine for large reads.
  - *Confusion*: tool search with few hot tools, path-scoped instructions, skills instead of monolithic prompts.
  - *Clash*: instruction-file linting, a single source of truth for task state (JSON feature list or todo), and explicit supersession notes when plans change.
- **Keep errors, but compactly.** Retain errors (Manus) while clearing raw logs: keep the error message and the corrective decision in notes or the summary, and drop verbose traces after resolution.
- **Use delta-style updates for any persistent playbook or notes file** (ACE): append and edit bullets with counters, not full rewrites, to avoid context collapse across repeated compactions.

### Gaps
- I could not verify Breunig's quantitative citations directly: the Gemini 2.5 Pokémon ">100k tokens" observation, the Databricks long-context RAG degradation points, and the Berkeley Function-Calling Leaderboard tool-count effects. The primary blog was blocked and search results didn't include those numbers.
- I found no standardized benchmark that measures each failure mode separately for agent harnesses, or the effect of each mitigation.
