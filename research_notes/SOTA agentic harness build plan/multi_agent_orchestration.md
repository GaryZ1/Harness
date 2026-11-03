# Multi-Agent Systems and Orchestration for Agentic Harnesses (state as of 2026-11-03)

Research note scope: when and how to use multiple agents (subagents, orchestrator-worker, handoffs, parallel agents, agent teams, script-driven workflows), the evidence for and against, and concrete orchestration mechanics. Single-agent loop design, tools, memory and evals are covered elsewhere.

Access caveat: the session network blocked cognition.ai, cognition.com, arxiv.org, openai.github.io, research.google, blog.langchain.com and kimi.com. Findings from those sources come from search-result snippets or secondary write-ups and are flagged. The primary sources I fetched in full were anthropic.com engineering posts, code.claude.com docs, and the OpenAI Agents SDK docs (via raw.githubusercontent.com).

---

## Q1. Anthropic: multi-agent research system, Claude Code / Agent SDK subagents, agent teams, parallel Claudes, dynamic workflows

### Takeaway
Anthropic's practice has moved from LLM-driven orchestration toward code-driven orchestration. It started with an LLM orchestrator-worker research system (June 2025: +90.2% over single agent, about 15x chat tokens). It then added a generator/evaluator harness for long tasks (Mar 2026) and an orchestrator-free 16-agent swarm coordinated through git and lock files (Feb 2026). Claude Code now offers four tiers: subagents (default depth 3, 20 concurrent), experimental agent teams (shared task list and mailboxes), and script-driven "dynamic workflows" (May 2026, up to 1,000 agents per run, deterministic and resumable). Across all of it, Anthropic repeatedly warns that multi-agent setups pay off only for parallelizable, read-heavy or independently verifiable work.

### Cited Findings

**"How we built our multi-agent research system" (June 13, 2025)**: [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)
- Architecture: orchestrator-worker. A lead agent (Claude Opus 4) plans, saves the plan to memory, spawns subagents (Claude Sonnet 4) that search in parallel with interleaved thinking, and decides whether more research is needed. A separate **CitationAgent** then places citations against the source documents. — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)
- Opus 4 lead with Sonnet 4 subagents beat single-agent Opus 4 by **90.2%** on Anthropic's internal research eval. — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)
- On BrowseComp, three factors explained **95%** of performance variance. **Token usage alone explained 80%**; the other two were number of tool calls and model choice. Upgrading to Sonnet 4 gained more than doubling the token budget on Sonnet 3.7. Interpretation offered: multi-agent setups work largely because they spend more tokens across separate context windows. — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)
- Cost: agents use about **4x** the tokens of chat, and multi-agent systems about **15x**. The task value has to justify this. — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)
- Effort-scaling rules written into the prompt:
  - simple fact-finding: 1 agent, 3–10 tool calls
  - direct comparisons: 2–4 subagents, 10–15 calls each
  - complex research: more than 10 subagents with divided responsibilities

  — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)
- Parallelism: the lead spawns **3–5 subagents at once**, and each subagent uses **3+ tools in parallel**. This cut research time **by up to 90%** on complex queries. — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)
- Prompting principles:
  1. Simulate agents step by step with the exact prompts and tools.
  2. Give the lead detailed delegation briefs: **objective, output format, tools/sources guidance, clear task boundaries**. Without them, subagents duplicate work or leave gaps.
  3. Encode effort-scaling rules.
  4. Make tool design and selection explicit, with distinct purposes and clear descriptions, and prefer specialized tools.
  5. Let Claude improve its own prompts and tools. A tool-testing agent rewrote a flawed MCP tool description and **cut task completion time by 40%** for later agents.
  6. Start wide, then narrow.
  7. Use extended thinking for planning and interleaved thinking after tool results.
  8. Parallelize.

  — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)
- Evaluation: start with **about 20 real-usage queries**, because early changes have large effect sizes. The LLM-as-judge rubric covers factual accuracy, citation accuracy, completeness, source quality and tool efficiency, with a single call returning a 0.0–1.0 score plus pass/fail. Stateful agents are judged on **end state**, not per step. Human testers still caught failures automation missed, such as SEO content farms ranked above authoritative sources. — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)
- Production reliability:
  - Agents are stateful, so errors compound. The system **resumes from where the error occurred** instead of restarting, using checkpoints and retry logic.
  - Full production tracing is in place, but conversation contents are not monitored.
  - **Rainbow deployments** shift traffic gradually so running agents are not broken mid-flight.
  - Synchronous execution is a bottleneck: the lead waits for each batch, and subagents cannot steer or coordinate with each other. Async execution was planned.

  — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)
- Long-horizon and context handling:
  - The plan is saved to memory because context beyond **200k tokens** is truncated.
  - Completed phases are summarized into external memory, and fresh subagents with clean contexts are spawned as limits approach.
  - Subagents can write outputs to an **external artifact system/filesystem** and pass **lightweight references** back, which avoids a "game of telephone" and saves tokens.

  — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)
- Poor fit: tasks where all agents need the same context, tasks with many inter-agent dependencies, and tasks with few parallelizable subtasks, "e.g., most coding tasks." — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)

**"Effective harnesses for long-running agents" (Nov 26, 2025)**: [Anthropic](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- Two-agent sequential design: an **initializer agent** sets up the environment on the first run, and a **coding agent** makes incremental progress in each later fresh-context session. — [Anthropic](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- State passes between agents and sessions as files:
  - `init.sh`
  - `claude-progress.txt`
  - git commits with descriptive messages
  - a JSON feature list (over 200 features, all initially failing) in which agents may only flip `passes`. JSON was chosen because the model is less likely to edit it inappropriately than Markdown.

  — [Anthropic](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- Failure modes addressed: one-shotting (fixed with one feature per session), declaring victory early, and marking features done without end-to-end tests (fixed with Puppeteer MCP browser testing "as a human user would"). — [Anthropic](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- The post says it is "still unclear whether a single, general-purpose coding agent performs best." Specialized testing, QA or cleanup agents are listed as future work. — [Anthropic](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)

**"Building a C compiler with a team of parallel Claudes" (Feb 5, 2026, Nicholas Carlini)**: [Anthropic](https://www.anthropic.com/engineering/building-c-compiler)
- Scale: **16 parallel Claude Opus 4.6 agents**, nearly **2,000 Claude Code sessions** over two weeks, **just under $20,000** in API cost, about **2B input and 140M output tokens**. The result was a roughly **100k-line** clean-room Rust C compiler that builds Linux 6.9 on x86, ARM and RISC-V, plus QEMU, FFmpeg, SQLite, PostgreSQL and Redis, with about 99% pass on most compiler test suites including GCC torture tests. — [Anthropic](https://www.anthropic.com/engineering/building-c-compiler)
- Coordination had **no orchestrator agent and no inter-agent messaging**:
  - A bash `while true` loop re-launches `claude` with a prompt file.
  - Each agent runs in its own Docker container, cloned from a bare git repo that acts as "upstream."
  - Agents claim tasks by writing a lock file to `current_tasks/`. A git push conflict forces the loser to pick another task.
  - Merge conflicts are frequent, and Claude resolves them itself.
  - Agents pick the "next most obvious" task.

  — [Anthropic](https://www.anthropic.com/engineering/building-c-compiler)
- Specialization by role: one agent deduplicated code, one improved compiler performance, one improved generated-code efficiency, one critiqued design as a Rust developer, and one maintained docs. — [Anthropic](https://www.anthropic.com/engineering/building-c-compiler)
- Key failure: on one monolithic task (the Linux kernel), all 16 agents hit the same bug and overwrote each other's fixes, so 16 agents did not help. The fix was to use **GCC as a known-good oracle**: compile most kernel files with GCC and a random subset with Claude's compiler, then delta-debug failing file pairs. That made the monolith parallelizable again. — [Anthropic](https://www.anthropic.com/engineering/building-c-compiler)
- Harness lessons:
  - The verifier/test suite must be **nearly perfect**, or agents solve the wrong problem.
  - Keep test output to a few lines, send details to log files, and tag errors with `ERROR` so they are grep-able. This prevents context pollution.
  - A `--fast` mode runs a deterministic per-agent 1%/10% random sample to compensate for "time blindness."
  - CI was added late, after regressions kept happening.
  - Agents maintain READMEs and progress files because each one starts fresh.
  - One agent ran `pkill -9 bash` and killed its own loop.

  — [Anthropic](https://www.anthropic.com/engineering/building-c-compiler)
- Model progression: Opus 4 barely produced a working compiler; Opus 4.5 passed test suites but could not build large projects; Opus 4.6 succeeded. — [Anthropic](https://www.anthropic.com/engineering/building-c-compiler)

**"Harness design for long-running application development" (Mar 24, 2026, Prithvi Rajasekaran)**: [Anthropic](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- GAN-inspired **generator/evaluator** split. "Separating the agent doing the work from the agent judging it proves to be a strong lever." A separate evaluator is easier to make skeptical than a generator is to make self-critical. — [Anthropic](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- Three-agent full-stack harness:
  - **Planner**: expands a 1–4 sentence prompt into a product spec, kept high level so spec errors do not cascade.
  - **Generator**: builds in sprints with git.
  - **Evaluator**: drives the running app with Playwright MCP, checking UI, API and database state against per-criterion hard thresholds. Failing any criterion fails the sprint.

  — [Anthropic](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- **Sprint contracts**: before each sprint, the generator proposes what it will build and how success will be verified. The evaluator reviews and they iterate to agreement. Sprint 3 alone had 27 criteria. Agents communicate **through files**. — [Anthropic](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- Numbers:
  - Retro game maker on Opus 4.5: solo run took **20 min / $9** and produced a broken game. The full harness took **6 hr / $200** (over 20x the cost) and produced 16 features across 10 sprints, playable.
  - DAW on Opus 4.6 with a simplified harness: about **3 hr 50 min / $124.70**. Planner 4.7 min / $0.46; build round 1 2 hr 7 min / $71.08; QA round 1 8.8 min / $3.24. QA still caught stubbed features.
  - Frontend design loop: 5–15 iterations, runs up to 4 hours.

  — [Anthropic](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- Context resets (fresh agent plus structured handoff) were essential for Sonnet 4.5, which showed "context anxiety." With Opus 4.5 that behavior largely disappeared, so resets were dropped in favor of automatic compaction. — [Anthropic](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- Simplify as models improve. With Opus 4.6 the sprint construct was removed and the evaluator moved to a single end-of-run pass. The planner was kept, because without it the generator under-scoped the app. Components were removed one at a time to find which were load-bearing. "Every component in a harness encodes an assumption about what the model can't do on its own." Claude out of the box is a poor QA agent; the evaluator prompt needed several tuning rounds against logs, plus few-shot calibration examples. — [Anthropic](https://www.anthropic.com/engineering/harness-design-long-running-apps)

**Claude Code subagents (current docs)**: [Claude Code docs: subagents](https://code.claude.com/docs/en/sub-agents)
- The **Agent** tool was formerly the **Task** tool, and `Task(...)` still aliases it. Subagents have their own context window, system prompt, tool allowlist and permissions, and **only the final report returns** to the parent. Claude delegates based on each agent's `description`. — [Claude Code docs](https://code.claude.com/docs/en/sub-agents)
- Built-ins:
  - **Explore**: read-only, with thoroughness quick/medium/very thorough; skips CLAUDE.md and git status.
  - **Plan**: read-only, used in plan mode.
  - **general-purpose**: all tools.
  - Also `claude`, `statusline-setup`, `claude-code-guide`.

  — [Claude Code docs](https://code.claude.com/docs/en/sub-agents)
- Definitions are Markdown with YAML frontmatter. Scope priority: managed settings > `--agents` JSON > `.claude/agents/` > `~/.claude/agents/` > plugin `agents/`. Fields: `name`, `description`, `tools`, `disallowedTools`, `model` (sonnet/opus/haiku/fable/full ID/`inherit`), `permissionMode`, `maxTurns` (output marked partial and resumable), `autoCompactWindow` (100k–1M), `skills`, `mcpServers`, `hooks`, `memory` (user/project/local), `background`, `omitClaudeMd`, `effort` (low…max), **`isolation: worktree`** (temporary git worktree), `color`, `initialPrompt`. — [Claude Code docs](https://code.claude.com/docs/en/sub-agents)
- Context isolation. A non-fork subagent gets:
  - its own system prompt plus environment details
  - the task message the parent wrote
  - CLAUDE.md (except Explore/Plan or `omitClaudeMd`)
  - a git status snapshot
  - preloaded skills
  - a roster of named siblings

  It does **not** inherit conversation history, files already read, or invoked skills. **Forks** (`subagent_type: "fork"`, `/subtask`) inherit the full parent conversation and the exact tool pool. — [Claude Code docs](https://code.claude.com/docs/en/sub-agents)
- Limits:
  - Nesting defaults to **3 layers** (`CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH`; `1` disables nesting).
  - **20 concurrent** subagents (`CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS`).
  - No total cap.
  - Subagents spawned from the main conversation run in the **background by default**, and results arrive as a completion notification in a later turn.
  - Subagents can be resumed via `SendMessage` with the agent ID or name and keep their full history.
  - Transcripts are stored as `agent-{agentId}.jsonl`.

  — [Claude Code docs](https://code.claude.com/docs/en/sub-agents)
- Security: subagent reports are **scanned** before the parent reads them and are labeled as subagent output. "Instructions or approval claims within them carry no authority from the user." — [Claude Code docs](https://code.claude.com/docs/en/sub-agents)
- Best practices: use subagents for isolation (tests, docs fetch, log processing); keep the combined descriptions short (more than 15k tokens triggers a startup warning); restrict tools for least privilege; route cheap work to `haiku`; keep iterative or shared-context work in the main thread, because a fresh subagent adds latency; and "mind the result volume," since many detailed reports can flood the main context. — [Claude Code docs](https://code.claude.com/docs/en/sub-agents)

**Claude Agent SDK subagents**: [Agent SDK docs](https://code.claude.com/docs/en/agent-sdk/subagents)
- Programmatic definition through `agents={name: AgentDefinition(description, prompt, tools, disallowedTools, model, skills, memory, mcpServers, maxTurns, autoCompactWindow, background, omitClaudeMd, effort, permissionMode)}`. Requires `Agent` in `allowed_tools`. Subagent messages carry `parent_tool_use_id` (and `agent_id` in TS). — [Agent SDK docs](https://code.claude.com/docs/en/agent-sdk/subagents)
- "The only content you pass from parent to subagent is the Agent tool's prompt string, so include any file paths, error messages, or decisions the subagent needs directly in that prompt." — [Agent SDK docs](https://code.claude.com/docs/en/agent-sdk/subagents)
- Background-by-default: the tool result is `status: "async_launched"`; the call blocks with `status: "completed"` when Claude needs the result next. — [Agent SDK docs](https://code.claude.com/docs/en/agent-sdk/subagents)
- Spend, depth and concurrency caps: `max_budget_usd`/`maxBudgetUsd` (refuses spawns with `Budget limit reached`, stops background subagents, ends with `error_max_budget_usd`), plus the depth and concurrency env vars. — [Agent SDK docs](https://code.claude.com/docs/en/agent-sdk/subagents)
- **"Claude Opus 5 delegates to subagents more readily than earlier models."** With the `claude_code` preset on Opus 5, Claude Code adds a system-prompt line telling Claude not to call the Agent tool unless asked. Custom prompts must add their own delegation instruction, and hard limits should be set regardless. — [Agent SDK docs](https://code.claude.com/docs/en/agent-sdk/subagents)

**Claude Code agent teams (experimental)**: [Claude Code docs: agent teams](https://code.claude.com/docs/en/agent-teams)
- Enabled with `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`. Components:
  - a **team lead**
  - **teammates**, each a full independent Claude Code session
  - a **shared task list** (pending/in-progress/completed, with dependencies and auto-unblocking; **claiming uses file locking**)
  - a **mailbox**, implemented as JSON files at `~/.claude/teams/{team}/inboxes/{agent}.json`

  Teammates message each other directly via `SendMessage`. Interactive sessions only; not in `-p` or the Agent SDK. — [Claude Code docs](https://code.claude.com/docs/en/agent-teams)
- Subagents compared with teams: subagents report results to the caller, with lower token cost and coordination by the main agent. Teammates are fully independent, self-coordinate, and cost more because each is a separate instance. — [Claude Code docs](https://code.claude.com/docs/en/agent-teams)
- Guidance:
  - Start with **3–5 teammates**; "three focused teammates often outperform five scattered ones."
  - Aim for **5–6 tasks per teammate**.
  - Size tasks as self-contained deliverables.
  - **Avoid file conflicts** by giving each teammate its own file set.
  - Start with research and review before parallel implementation.
  - The lead sometimes starts implementing instead of waiting, or stops early, so steer it.

  — [Claude Code docs](https://code.claude.com/docs/en/agent-teams)
- Quality gates via hooks: `TeammateIdle`, `TaskCreated` and `TaskCompleted`. Exit code 2 blocks the action and sends feedback. Teammates in plan mode send plan-approval requests to the lead. — [Claude Code docs](https://code.claude.com/docs/en/agent-teams)
- Recommended patterns: a parallel code review where each reviewer gets a distinct lens (security, performance, tests), and "competing hypotheses" debugging where teammates try to disprove each other's theories, to counter anchoring. — [Claude Code docs](https://code.claude.com/docs/en/agent-teams)
- Limitations: no nested teams, one team per session, the lead is fixed, task status can lag and block dependents, in-process teammates are not restored on `/resume`, and inter-agent messages cannot grant permissions. — [Claude Code docs](https://code.claude.com/docs/en/agent-teams)

**Claude Code dynamic workflows (introduced around May 28, 2026, v2.1.154+ per secondary sources)**: [Claude Code docs: workflows](https://code.claude.com/docs/en/workflows); date and version from [agentic-ai docs mirror](https://agentic-ai.readthedocs.io/en/latest/WorkflowBuilders/dynamic-workflows/) and [alexop.dev](https://alexop.dev/posts/claude-code-workflows-deterministic-orchestration)
- Claude writes a **JavaScript orchestration script** that a background runtime executes. The script holds the loop, the branching and the intermediate results in variables, so "Claude's context holds only the final answer." — [Claude Code docs](https://code.claude.com/docs/en/workflows)
- Script API: `agent(prompt, {schema, label, stallMs, model…})`, `parallel()`, `pipeline(items, fn)`, `phase()`, `log()`, and an `args` global. `schema` forces JSON output, validated with up to 5 retries (`MAX_STRUCTURED_OUTPUT_RETRIES`). `agent()` resolves to `null` on stop or unrecoverable error. — [Claude Code docs](https://code.claude.com/docs/en/workflows)
- Comparison table: subagents are "a few delegated tasks per turn"; agent teams are "a handful of long-running peers"; workflows are "dozens to hundreds of agents per run" and "resumable in the same session." — [Claude Code docs](https://code.claude.com/docs/en/workflows)
- Limits:
  - **16 concurrent agents** by default (`CLAUDE_CODE_WORKFLOW_MAX_CONCURRENT_AGENTS`, 1–256)
  - **4,096 items** per `parallel()`/`pipeline()`
  - **1,000 agents per run**
  - no mid-run user input
  - no filesystem access from the script itself
  - no `import()`

  — [Claude Code docs](https://code.claude.com/docs/en/workflows)
- Determinism and resume: `Date.now()`, `Math.random()` and `new Date()` throw inside scripts, so a relaunch repeats the same `agent()` calls. Completed agents return cached results until the first agent whose prompt differs; that agent and every later one re-run. — [Claude Code docs](https://code.claude.com/docs/en/workflows)
- Fault handling:
  - A stalled agent auto-restarts up to 5 times.
  - Inside `parallel`/`pipeline`, a failure leaves `null` and the run continues.
  - A directly awaited failure ends the run.
  - Usage-limit hits pause and auto-resume.

  — [Claude Code docs](https://code.claude.com/docs/en/workflows)
- Cost controls:
  - A "Large workflow" warning fires above **25 agents or 1.5M projected tokens**.
  - Size guideline values: `small` <5, `medium` <10 (default), `large` <50 agents.
  - Prompt-cache sharing in fan-out: agents with an identical model/effort/tools/schema/cwd prefix share cache. The runtime holds siblings until the first response begins (`CLAUDE_CODE_WORKFLOW_PREFIX_STAGGER_MS`, default 5000).
  - "Run the workflow on a small slice first."

  — [Claude Code docs](https://code.claude.com/docs/en/workflows)
- Built-in quality patterns: adversarial cross-review of findings before reporting, drafting a plan from several angles, "keep fixing until check passes or two rounds make no progress," and "find issues until the list stops growing." The bundled `/deep-research` workflow fans out searches, cross-checks, **votes on each claim**, and filters claims that do not survive. Claims that cannot be checked are labeled unverified, not refuted. — [Claude Code docs](https://code.claude.com/docs/en/workflows)
- `ultracode` effort mode lets Claude plan a workflow for every substantive task. A single request can become "one to understand the code, one to make the change, and one to verify it," at higher token cost. — [Claude Code docs](https://code.claude.com/docs/en/workflows)

### Inferences
- Anthropic's own trajectory is the strongest signal for a harness roadmap: (a) default to a single strong agent; (b) add read-only, context-isolated subagents for search and verification; (c) separate generator and evaluator for quality; (d) use code-driven fan-out (workflows) for large homogeneous jobs; and (e) use peer swarms only when there is a near-perfect external verifier (test suite or oracle) and a file-level work partition.
- The "token usage explains 80% of variance" result implies much of the multi-agent gain is **more total reasoning tokens through more context windows**. A harness should treat multi-agent as a compute-scaling knob with a budget, not as a free quality improvement.
- Interfaces that pass references instead of content recur everywhere: artifact or file references, `claude-progress.txt`, JSON feature lists, `current_tasks/` lock files, mailbox JSON files, and script variables. Durable files plus git are the de facto shared memory or blackboard.
- Newer models (Opus 5) over-delegate by default. A harness needs explicit delegation policy in the prompt **and** hard caps on depth, concurrency and budget.

### Gaps
- The research-system post gives no per-task breakdown of the 90.2% figure and does not publish the eval set. Asynchronous lead/subagent execution was "planned," and I found no follow-up primary post confirming it shipped in the research product.
- The exact release date of agent teams (experimental) was not found in the docs page.
- I found no published Anthropic quantitative comparison of agent teams against subagents against workflows.

---

## Q2. OpenAI: Agents SDK (handoffs, agents-as-tools, guardrails, tracing), Codex parallel tasks, Swarm, AgentKit/Agent Builder

### Takeaway
The OpenAI Agents SDK formalizes two composition primitives. With handoffs, the specialist takes over the conversation. With agents-as-tools (`Agent.as_tool()`), a manager stays in charge. The SDK explicitly recommends code-driven orchestration (structured-output routing, chaining, evaluator `while` loops, `asyncio.gather`) when predictability matters. Codex runs parallel cloud tasks and subagents. AgentKit's visual Agent Builder is reportedly being deprecated in 2026 in favor of the code-first SDK (secondary sources, conflicting dates).

### Cited Findings
- The OpenAI Agents SDK docs describe two orchestration modes. **Via LLM**: an agent with instructions, tools and handoffs plans autonomously. Tactics: good prompts, monitoring, self-critique loops, specialized agents over general ones, and evals. **Via code**: "more deterministic and predictable, in terms of speed, cost and performance." — [OpenAI Agents SDK docs (multi_agent.md)](https://raw.githubusercontent.com/openai/openai-agents-python/main/docs/multi_agent.md)
- Code-orchestration patterns named in the docs:
  - structured outputs to classify the task and pick the next agent
  - chaining (research → outline → draft → critique → revise)
  - an evaluator loop: a task agent runs in a `while` loop with an evaluator agent until criteria pass
  - parallel independent agents via `asyncio.gather`

  Examples live in `examples/agent_patterns`. — [OpenAI Agents SDK docs](https://raw.githubusercontent.com/openai/openai-agents-python/main/docs/multi_agent.md)
- **Agents as tools** (`Agent.as_tool()`): the manager keeps control. Best when one agent should own the final answer, combine specialist outputs, or enforce shared guardrails. **Handoffs**: a triage agent routes and the specialist becomes the active agent for the rest of the turn. Best when the specialist should respond directly or routing is part of the workflow. The two combine, for example a triage agent hands off to a specialist that calls other agents as tools. — [OpenAI Agents SDK docs](https://raw.githubusercontent.com/openai/openai-agents-python/main/docs/multi_agent.md)
- Rule of thumb from third-party guides: "if the specialist owns the answer, handoff; if the manager owns the answer, tool." With handoffs, debugging misrouting requires understanding the routing logic, and choosing what history to filter on handoff "is a real design problem." — [team400.ai guide (Apr 2026)](https://team400.ai/blog/2026-04-openai-agent-orchestration-handoffs-guide); framing consistent with the [official JS multi-agent guide](https://openai.github.io/openai-agents-js/guides/multi-agent) (snippet only)
- Codex app for macOS launched **Feb 2, 2026** as a "command center" for managing multiple coding agents in parallel. Codex also spawns parallel subagents and consolidates their results. — [IntuitionLabs (secondary)](https://intuitionlabs.ai/articles/openai-codex-app-ai-coding-agents); [heise](https://heise.de/-11165126)
- Agent Builder, AgentKit's visual canvas launched at DevDay in Oct 2025, reportedly got deprecation notices on **June 3, 2026**, with shutdown cited as **Nov 30, 2026**. Sources disagree on dates. — [osher.com.au (secondary)](https://osher.com.au/blog/openai-agentkit/); [timewell.jp (secondary)](https://timewell.jp/en/columns/agent-builder)

### Inferences
- The OpenAI and Anthropic docs converge: (1) a manager or agents-as-tools pattern for bounded subtasks where one agent owns the answer, (2) handoffs only for routing or triage-style UX, and (3) code-level orchestration (gather, loops, structured routing) when cost and latency predictability matter.
- If the Agent Builder deprecation is accurate, it is a market signal that visual low-code multi-agent canvases lost to code-first SDKs for production.

### Gaps
- I could not fetch openai.github.io (blocked), so guardrails (input/output/tool guardrails, tripwires), tracing span types, `handoff_input_filter`/`nest_handoff_history` details and Sessions were not verified this session.
- I did not verify Swarm's history (experimental educational framework that preceded the Agents SDK) or the Agents SDK launch date against a primary source. Treat these as background only.
- I found no primary OpenAI numbers on Codex parallel-task speedups. A secondary claim of "5–7x for read-heavy tasks" could not be verified.
- I found no primary OpenAI multi-agent research paper from 2025–2026 comparable to Anthropic's posts.

---

## Q3. Counter-evidence and trade-offs: Cognition, MAST, controlled scaling studies, single-agent-with-skills, debate

### Takeaway
The best evidence says multi-agent helps on **parallelizable, decomposable, read-heavy** tasks. It hurts on **sequential, tightly coupled** tasks: Google, DeepMind and MIT found +81% on Finance-Agent but −39% to −70% on sequential planning. Independent parallel agents amplify errors 17.2x unless a centralized coordinator verifies. Most failures are specification and verification problems, not model capability. Cognition's April 2026 update converges on one rule: many agents may **contribute intelligence**, but **writes stay single-threaded**. A fresh-context reviewer is the multi-agent pattern that clearly works.

### Cited Findings

**Cognition (Walden Yan)**
- "Don't Build Multi-Agents" (June 2025): context engineering, not agent count, drives reliability.
  - Principle 1: share full context and full agent traces, not just messages.
  - Principle 2: actions carry implicit decisions, and conflicting decisions produce bad results.
  - Flappy Bird example: two subagents (background and bird) build mismatched pieces because they lack each other's context.
  - Recommendation: single-threaded linear agents, plus an LLM to compress history for long tasks.

  — Original at cognition.ai (blocked); summarized via [jxnl.co](https://jxnl.co/writing/2025/09/11/why-cognition-does-not-use-multi-agent-systems/), [agentic-ai docs summary](https://agentic-ai.readthedocs.io/en/latest/ContextEngineering/devin/), and search snippets
- "Multi-Agents: What's Actually Working" (April 2026) claims, from search snippets of the original:
  - "Most multi-agent setups in the world are limited to 'readonly' subagents... these mostly resemble tool calls rather than true multi-agent collaboration."
  - Devin and Devin Review iterate against each other. Devin Review catches **about 2 bugs per PR on Devin-written PRs, about 58% severe**.
  - "The coding agent and review agent do not share any context beforehand," and "having a clean context makes the agent smarter because of the math of attention."

  — [Cognition blog (via search snippet)](https://cognition.com/blog/multi-agents-working); [X article](https://x.com/walden_yan/article/2047054401341370639)
- Secondary summaries say Yan holds that the 2025 warnings still apply to **parallel-writer swarms**, and that the working pattern is multiple agents contributing intelligence with **single-threaded writes**. I could not confirm this phrasing against the original. Cognition also describes a lead/executor split (frontier lead for planning and review, cheaper executor such as SWE-2). — [AGNT summary](https://agnt.gg/articles/engineering/single-agent-vs-multi-agent); [Cognition product page via snippet](https://ai.engineer/orgs/cognition)
- LangChain's response: even read-heavy multi-agent systems need sophisticated context engineering, and write-heavy multi-agent is harder. (Snippet only; page blocked.) — [LangChain blog](https://blog.langchain.com/how-and-when-to-build-multi-agent-systems)

**MAST, "Why Do Multi-Agent LLM Systems Fail?" (Cemri et al., UC Berkeley; NeurIPS 2025, arXiv 2503.13657)**
- **14 failure modes** in **3 categories**: (i) specification / system design issues, (ii) inter-agent misalignment, (iii) task verification. v3 is based on **1,642 MAS execution traces**; v1 covered 5 frameworks and 150+ tasks; v2 covered 7 frameworks and 200+ tasks. — [arXiv 2503.13657 (search snippet)](https://arxiv.org/html/2503.13657v3); [NeurIPS 2025](https://neurips.cc/virtual/2025/122442)
- Category shares reported by a secondary source: specification/system design **44.2%**, task verification **23.5%** (so inter-agent misalignment is about 32%). Failure rates across studied systems ran **41%–86.7%**, but are not comparable across benchmarks. These figures are unverified against the paper. — [Antoine Buteau blog (secondary)](https://www.antoinebuteau.com/multi-agent-systems-fail-like-organizations/)

**Google Research, DeepMind and MIT, "Towards a Science of Scaling Agent Systems" (arXiv 2512.08296, Dec 2025)**
- Controlled comparison of 5 architectures: single-agent, plus independent, centralized, decentralized and hybrid multi-agent. Three LLM families; 180 configurations in v1, **260 configurations across 6 benchmarks** in a later version. — [arXiv 2512.08296 (search snippet)](https://arxiv.org/pdf/2512.08296); [HF papers](https://huggingface.co/papers/2512.08296)
- Results:
  - Multi-agent gave **+81% on Finance-Agent** (parallelizable).
  - **All multi-agent variants degraded sequential reasoning (PlanCraft) by 39–70%.**
  - **Independent** agents amplified errors **17.2x**; **centralized** coordination contained errors best.
  - **Capability saturation**: diminishing or negative returns once the single-agent baseline exceeds about **45%**.
  - A predictive model chose the optimal architecture for **87%** of unseen configurations.

  — [arXiv 2512.08296 / Google Research blog (via search snippets)](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/)

**Single agent with skills vs multi-agent (2026)**
- "When Single-Agent with Skills Replace Multi-Agent Systems and When They Fail" (arXiv 2601.04748, Jan 2026) compiles a MAS into one agent that selects from a skill library. It matched MAS accuracy (+0.7% on average) with **54% fewer tokens and 50% lower latency**. Selection accuracy **degrades sharply beyond about 50–100 skills**, and semantically similar skills make it worse. — [arXiv 2601.04748 (search snippet)](https://arxiv.org/pdf/2601.04748)
- Tran and Kiela (arXiv 2604.02460, Apr 2026): at a fixed thinking-token budget, single agents match or beat MAS on multi-hop reasoning. MAS pulls ahead only when the single agent's context use is already degraded. — [alphaXiv 2604.02460 (snippet)](https://www.alphaxiv.org/abs/2604.02460v1); [beancount.io summary](https://beancount.io/bean-labs/research-logs/2026/05/31/single-agent-outperforms-multi-agent-equal-token-budget)
- Contrary result: a clinical-workload study (npj Health Systems 2026) found single-agent accuracy degraded as batch size grew, while an orchestrated MAS held up and used up to **65x fewer tokens**. — [BIDMC research listing](https://research.bidmc.org/radiology-research/publications/orchestrated-multi-agents-sustain-accuracy-under-clinical-scale-workloads-compared)

**Debate and critique**
- Equal-response multi-agent debate does no better than majority voting; gains come from consistency, not self-correction ("LLMs Cannot Self-Correct Reasoning Yet"). "Talk Isn't Always Cheap" (2025) finds debate can make answers worse than a single agent's. "The Cost of Consensus" (2026 preprint) reports debate uses **2.1–3.4x more tokens** than isolated self-correction for equal or lower accuracy. — [arXiv 2310.01798](https://arxiv.org/pdf/2310.01798); [arXiv 2509.05396](https://arxiv.org/html/2509.05396v2); [arXiv 2605.00914 (snippet)](https://arxiv.org/pdf/2605.00914)
- Contrary result: ACL 2026 SRW, "Multi-Agent Reasoning Improves Compute Efficiency," reports debate beats self-consistency by about 1.3 points on MMLU-Pro at equal compute, and self-consistency saturates earlier on hard tasks. — [ACL Anthology preview](https://preview.aclanthology.org/ingest-acl/2026.acl-srw.1/)

**Practitioner cost data (Anthropic)**
- Multi-agent uses about 15x chat tokens. — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)
- The full planner/generator/evaluator harness cost over 20x the solo run ($200 vs $9) but was the only version that worked. — [Anthropic](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- 16 agents gave zero benefit on a single shared bug until an oracle was used to partition the work. — [Anthropic](https://www.anthropic.com/engineering/building-c-compiler)

### Inferences
- Decision rule supported by the convergent evidence. Use multi-agent when **all** of the following hold:
  1. The task decomposes into subtasks with low mutual dependency, such as breadth-first search, per-file audits, or independent failing tests.
  2. Each subtask's output can be checked or merged without re-deriving the other subtasks' context.
  3. The single-agent baseline is weak, or context-limited and degraded.
  4. The task value justifies 4–20x the token cost.

  Otherwise prefer one agent with good context engineering, skills and compaction.
- Error containment needs **centralized verification**. Independent parallel agents without a checker amplify errors (17.2x). Any fan-out should end in a verifier, judge or merge step, not in naive concatenation.
- The most robust write pattern is "**many readers or critics, one writer**." Parallel writers are acceptable only with strict partitioning (file ownership, worktrees, task locks) **and** an external oracle (tests, compiler, CI).
- Fresh-context reviewers (Cognition's Devin Review, Anthropic's evaluator) are a cheap, high-yield use of a second agent, because they avoid the generator's self-assessment bias and context bloat.
- Debate between homogeneous agents is weak value per token. Prefer independent sampling plus a verifier or vote, or heterogeneous critics with different lenses.

### Gaps
- I could not read the full MAST paper (arXiv blocked), so per-mode percentages (for example step repetition, reasoning-action mismatch, premature termination) and the intervention results were not captured. The commonly cited ChatDev +14% from intervention was not verified.
- I could not read Cognition's April 2026 post in full. The "parallel writers" and "smart friend" framings come from secondary sources.
- The Google scaling paper's exact error-amplification figure for centralized systems (secondary sources say about 4.4x) was not verified.

---

## Q4. Pattern catalog: orchestrator-workers, hierarchy, debate, evaluator-optimizer, generator-verifier, map-reduce, best-of-N, background/async, worktrees, A2A, blackboard

### Takeaway
Production-proven patterns, in rough order of evidence:
1. Orchestrator-worker fan-out for breadth research.
2. Generator/evaluator with a fresh-context, skeptical evaluator.
3. Map-reduce over items with schema-constrained outputs and an adversarial verify/dedup stage.
4. Swarms partitioned by file or task, coordinated by git, lock files and a near-perfect test oracle.
5. Best-of-N or parallel rollouts with a selector.

Inter-agent protocols (A2A v1.0) matter for cross-vendor interop, not for in-harness orchestration, where files and git dominate.

### Cited Findings
- **Background taxonomy** from Anthropic's "Building effective agents" (Dec 2024): workflows (prompt chaining, routing, parallelization in its sectioning and voting forms, orchestrator-workers, evaluator-optimizer) versus autonomous agents. The advice is to start with the simplest pattern. — [Anthropic (background)](https://www.anthropic.com/engineering/building-effective-agents)
- **Orchestrator-worker**: lead plans, 3–5 parallel subagents with explicit briefs, synthesis, then a citation pass. — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)
- **Generator-evaluator / evaluator-optimizer**: sprint contracts negotiated in files, an evaluator with hard per-criterion thresholds, Playwright-driven end-state checks, and few-shot calibrated grading. — [Anthropic](https://www.anthropic.com/engineering/harness-design-long-running-apps). The OpenAI SDK equivalent is an evaluator `while` loop. — [OpenAI Agents SDK docs](https://raw.githubusercontent.com/openai/openai-agents-python/main/docs/multi_agent.md)
- **Fresh-context reviewer**: Devin and Devin Review iterate with no shared prior context, catching about 2 bugs per PR. — [Cognition (snippet)](https://cognition.com/blog/multi-agents-working)
- **Map-reduce / pipeline**: `pipeline(items, item => agent(...))`, then one agent ranks and deduplicates. Examples include per-file reviewers merged into one ranked summary, and a 500-file migration "each file in its own isolated copy." — [Claude Code workflows docs](https://code.claude.com/docs/en/workflows)
- **Adversarial verification and voting**: `/deep-research` cross-checks sources and "votes on each claim," dropping claims that do not survive. Workflows can "have independent agents adversarially review each other's findings before they're reported." — [Claude Code workflows docs](https://code.claude.com/docs/en/workflows)
- **Iterate-until-convergence loops**: "keep fixing until the type check passes or two rounds in a row make no progress"; "stop once two rounds in a row find nothing new." — [Claude Code workflows docs](https://code.claude.com/docs/en/workflows)
- **Competing-hypotheses debate**: 5 teammates each try to disprove the others' root-cause theories to counter anchoring. — [Claude Code agent teams docs](https://code.claude.com/docs/en/agent-teams)
- **Multi-lens parallel review**: separate security, performance and test-coverage reviewers. — [Claude Code agent teams docs](https://code.claude.com/docs/en/agent-teams); [Agent SDK docs](https://code.claude.com/docs/en/agent-sdk/subagents)
- **Peer swarm with task locks**: `current_tasks/` lock files, a bare git upstream, per-agent Docker containers, role-specialized agents (dedup, performance, docs, design critic), and an oracle-based partition for monolithic tasks. — [Anthropic C compiler](https://www.anthropic.com/engineering/building-c-compiler)
- **Git worktrees for parallel coding agents**: `isolation: worktree` runs a subagent in a temporary git worktree. — [Claude Code subagents docs](https://code.claude.com/docs/en/sub-agents). Agent teams warn that "two teammates editing the same file leads to overwrites." — [Claude Code agent teams docs](https://code.claude.com/docs/en/agent-teams)
- **Shared state / blackboard**: shared task list with dependencies and file-locked claiming, plus JSON mailboxes. — [Claude Code agent teams docs](https://code.claude.com/docs/en/agent-teams). Also progress files and JSON feature lists. — [Anthropic long-running harness](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents). Google ADK `ParallelAgent` branches write to shared `session.state` and need distinct `output_key`s. — [community ADK guide](https://zack.onrender.com/docs/GEN%20AI%20DEV/Agents/Google%20ADK)
- **Best-of-N / parallel test-time compute**: Anthropic has reported SWE-bench Verified with "parallel test-time compute," for example Sonnet 4.5 at 77.2% single versus **82.0% with parallel compute** (secondary table; sampling count undisclosed, internal scoring model). — [Verdent/third-party (secondary)](https://www.verdent.ai/guides/claude-sonnet-5-swe-bench-verified-results); discussion at [GreaterWrong](https://www.greaterwrong.com/posts/hmZHPE4ZJvEc3khgQ/checking-in-on-ai-2027/comment/LzyFabszAsB6b7sJm). CodeMonkeys (Jan 2025) shows that equal coverage does not guarantee equal final score, because **selection quality** is the bottleneck. — [arXiv 2501.14723 (snippet)](https://arxiv.org/pdf/2501.14723)
- **A2A protocol**:
  - Google donated it to the Linux Foundation in June 2025. **v1.0** shipped around Mar–Apr 2026 (sources disagree), with v1.0.1 around May 2026.
  - Each server publishes an Agent Card at `/.well-known/agent-card.json` listing name, skills, interfaces, capabilities and security schemes.
  - v1.0 adds signed Agent Cards, multi-tenancy and the AP2 payments extension.
  - Bindings: JSON-RPC 2.0/HTTPS, gRPC, HTTP+JSON/REST.

  — [letsdatascience (secondary)](https://letsdatascience.com/news/a2a-solidifies-cross-vendor-agent-interoperability-53400ea4); [tyk.io](https://tyk.io/blog/a2a-protocol/); [olostep](https://www.olostep.com/blog/a2a-protocol-agent2agent)
- Microsoft Agent Framework A2A client and hosting packages moved to the A2A v1 SDK but are **still in preview**. — [Visual Studio Magazine](https://visualstudiomagazine.com/articles/2026/04/06/microsoft-ships-production-ready-agent-framework-1-0-for-net-and-python.aspx); [Microsoft devblog](https://devblogs.microsoft.com/agent-framework/?p=5212)

### Inferences
- For a coding harness, parallel coding agents should get (1) one worktree or container per writer, (2) an explicit file-ownership or task-lock scheme, (3) a shared test oracle and CI gate before merge, and (4) a single integrator (the lead or a merge agent) that owns the main branch. This fits Cognition's "single-threaded writes" and the C-compiler experience.
- Best-of-N is only as good as the selector. Pair parallel rollouts with an executable verifier (tests) first, then an LLM judge as tie-breaker.
- A2A is for crossing trust and organization boundaries between independently deployed agents. Inside one harness, function calls, files and git are simpler and better understood.

### Gaps
- I found no primary 2025–2026 source with Anthropic's exact parallel-test-time-compute method (number of samples, selection model) for Claude 4.x SWE-bench results.
- I found no rigorous public benchmark of git-worktree parallel coding (N agents vs 1) with quality and merge-conflict rates.
- I could not reach the A2A official spec (a2a-protocol.org) or the Linux Foundation announcement, so the release dates are unconfirmed.

---

## Q5. Mechanics: delegation prompts, condensed returns, context isolation and handoff, budgets, failure handling, dedup, code-driven vs LLM-driven orchestration

### Takeaway
Delegation should be an explicit, self-contained contract: objective, output format or schema, tool and source guidance, boundaries, and effort budget. Results come back condensed, or as references to artifacts, through a labeled and untrusted channel. Budgets are enforced by the harness (depth, concurrency, dollars, turns), not only by prompt. Failure handling relies on checkpoints and resume-not-restart. Deterministic code orchestration is now the recommended way to run large fan-outs.

### Cited Findings
- **Delegation brief contents**: "an objective, an output format, guidance on the tools and sources to use, and clear task boundaries." Vague briefs such as "research the semiconductor shortage" led subagents to duplicate each other's work. — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)
- **Self-contained prompts**: "The only content you pass from parent to subagent is the Agent tool's prompt string, so include any file paths, error messages, or decisions the subagent needs directly in that prompt." — [Agent SDK docs](https://code.claude.com/docs/en/agent-sdk/subagents). Teammates "don't inherit the lead's conversation history," so spawn prompts should include task-specific details, for example "The app uses JWT tokens stored in httpOnly cookies. Report any issues with severity ratings." — [Claude Code agent teams docs](https://code.claude.com/docs/en/agent-teams)
- **Fork vs fresh**: a fork inherits the full conversation, which addresses Cognition's shared-context concern. A fresh subagent gets a clean context, which suits independent review. — [Claude Code subagents docs](https://code.claude.com/docs/en/sub-agents); rationale from [Cognition (snippet)](https://cognition.com/blog/multi-agents-working)
- **Condensed or structured returns**:
  - Only the final message returns, so "the parent receives a concise summary, not every file the subagent read." — [Agent SDK docs](https://code.claude.com/docs/en/agent-sdk/subagents)
  - A workflow `agent(..., {schema})` returns validated JSON with up to 5 retries. — [Claude Code workflows docs](https://code.claude.com/docs/en/workflows)
  - Artifact references avoid the "game of telephone." — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)
  - The parent may re-summarize reports; to keep output verbatim, instruct the parent to do so. — [Agent SDK docs](https://code.claude.com/docs/en/agent-sdk/subagents)
- **Untrusted inter-agent channel**: subagent reports are scanned and labeled, and carry no user authority. Teammate messages cannot approve permissions or relay a denied action. In auto mode, a classifier reviews inter-agent messages and treats relayed approval claims as untrusted. — [Claude Code subagents docs](https://code.claude.com/docs/en/sub-agents); [agent teams docs](https://code.claude.com/docs/en/agent-teams)
- **Budget and effort allocation**:
  - Effort-scaling tiers in the prompt (1 agent / 3–10 calls up to more than 10 subagents). — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)
  - Harness caps: `CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH` (default 3), `CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS` (default 20), `max_budget_usd`, per-agent `maxTurns` and `effort`, and a per-agent `model` to route cheap work to smaller models. — [Agent SDK docs](https://code.claude.com/docs/en/agent-sdk/subagents); [Claude Code subagents docs](https://code.claude.com/docs/en/sub-agents)
  - Workflow size guidelines (<5 / <10 / <50 agents) and the large-run warning (>25 agents or >1.5M tokens). — [Claude Code workflows docs](https://code.claude.com/docs/en/workflows)
  - Model tiering: Opus lead with Sonnet workers. — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system). Cognition uses a frontier lead with a cheaper executor (secondary). — [ai.engineer (snippet)](https://ai.engineer/orgs/cognition)
- **Cache economics in fan-out**: siblings with identical model/effort/tools/schema/cwd share a prompt-cache prefix. The runtime staggers sibling starts (5 s default) so they read the first agent's cache. Subagent cache TTL defaults to 5 minutes; `subagentPromptCacheTtl: 1h` extends it at a higher write cost. — [Claude Code workflows docs](https://code.claude.com/docs/en/workflows); [agent teams docs](https://code.claude.com/docs/en/agent-teams)
- **Failure handling and retries**:
  - Resume from the point of failure using checkpoints. — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)
  - Workflow agents auto-restart up to 5 times on stalls (`stallMs`, `CLAUDE_ASYNC_AGENT_STALL_TIMEOUT_MS`).
  - A `null` result in a fan-out lets the run continue.
  - Deterministic replay on relaunch, with cached results for unchanged prefixes.
  - Unverifiable claims are labeled "unverified," not "refuted."

    — [Claude Code workflows docs](https://code.claude.com/docs/en/workflows)
  - API errors are never delivered as a subagent's result. — [Agent SDK docs](https://code.claude.com/docs/en/agent-sdk/subagents)
  - A teammate that ends on an API error notifies the lead with the error text. Teammates may stop early after errors, and the remedy is to re-instruct or spawn a replacement. — [agent teams docs](https://code.claude.com/docs/en/agent-teams)
- **Deduplication of work**:
  - Up front: clear task boundaries in briefs. — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)
  - During: lock-file task claiming (git-arbitrated) — [C compiler](https://www.anthropic.com/engineering/building-c-compiler) — and file-locked task claiming with dependencies — [agent teams docs](https://code.claude.com/docs/en/agent-teams).
  - After: a merge or rank agent deduplicates per-item findings. — [workflows docs](https://code.claude.com/docs/en/workflows)
  - A dedicated "dedup" role agent in the swarm. — [C compiler](https://www.anthropic.com/engineering/building-c-compiler)
- **Quality gates**: hooks `TaskCompleted` and `TeammateIdle` (exit 2 blocks completion or idling and sends feedback) enforce tests or lint before a task is accepted. — [agent teams docs](https://code.claude.com/docs/en/agent-teams)
- **Code-driven vs LLM-driven orchestration**:
  - "Orchestrating via code makes tasks more deterministic and predictable, in terms of speed, cost and performance." — [OpenAI Agents SDK docs](https://raw.githubusercontent.com/openai/openai-agents-python/main/docs/multi_agent.md)
  - Claude Code workflows move "the plan into code… Claude's context holds only the final answer"; they are repeatable, savable (`.claude/workflows/`), parameterized via `args`, and resumable. — [Claude Code workflows docs](https://code.claude.com/docs/en/workflows)
  - Practitioner advice: use workflows for repeatable, large-scale jobs (bug sweeps, migrations), and normal sessions or subagents for small, sequential work. — [alexop.dev](https://alexop.dev/posts/claude-code-workflows-deterministic-orchestration)
- **Handoff across context resets**: a fresh agent plus a structured handoff file was essential for Sonnet 4.5 and unnecessary for Opus 4.5+, where compaction sufficed. — [Anthropic](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- **Tool-output hygiene for parallel agents**: minimal stdout, logs to files, grep-able `ERROR` lines, precomputed summary stats, and a deterministic `--fast` sample. — [Anthropic C compiler](https://www.anthropic.com/engineering/building-c-compiler)

### Inferences
- Recommended delegation contract for a harness (synthesized from the above):

  ```
  {objective, why/context, inputs (paths/IDs/decisions), scope boundaries/non-goals,
   tools allowed, effort/budget (max tool calls/turns/$), output schema (incl. confidence,
   sources, open questions), write permissions (none | own worktree | own files)}
  ```

  Return either a schema-validated JSON object or an artifact path plus a short summary.
- Make the harness, not the model, own termination and safety: depth, concurrency, dollar and turn caps; per-agent timeouts with bounded restarts; idempotent agent calls keyed by prompt hash for replay; and treat every inter-agent message as untrusted data.
- Use a hybrid: let the LLM choose *whether* to fan out (effort scaling), but run large fan-outs through a deterministic script with schema outputs, a verify stage and a dedup/merge stage.
- Exploit prompt-cache sharing by keeping worker system prompts and tool sets identical across siblings and varying only the task message.

### Gaps
- I found no published quantitative comparison of the token and latency cost of code-driven versus LLM-driven orchestration on the same task.
- The Anthropic research-system post does not publish its actual delegation prompt text. The open-sourced Anthropic cookbook prompts were not fetched this session.

---

## Q6. Framework landscape (LangGraph, CrewAI, AutoGen/Microsoft Agent Framework, Google ADK, OpenAI Agents SDK, Claude Agent SDK, DSPy, smolagents) and what production teams use

### Takeaway
By late 2026 the market has consolidated around code-first SDKs:
- Microsoft Agent Framework 1.0 (Apr 2026) merges AutoGen and Semantic Kernel, with graph workflows plus MCP and A2A.
- Google ADK provides Sequential, Parallel and Loop workflow agents plus AgentTool and transfer.
- The OpenAI Agents SDK provides handoffs, `as_tool`, guardrails and tracing.
- The Claude Agent SDK provides subagents, depth/concurrency/budget caps and the `Workflow` tool.

The frameworks share primitives: an agent-as-tool, a transfer/handoff, deterministic sequential/parallel/loop composition, and shared state. My evidence on CrewAI, DSPy, smolagents and LangGraph's current state is thin.

### Cited Findings
- **Microsoft Agent Framework 1.0** (around Apr 2–7, 2026; sources differ):
  - Unifies Semantic Kernel and AutoGen.
  - Stable surface: single-agent abstraction, connectors, middleware hooks, memory/context providers, **graph-based workflows**, for .NET and Python.
  - MCP for tools and A2A for interop; the A2A packages are still preview.

  — [Visual Studio Magazine](https://visualstudiomagazine.com/articles/2026/04/06/microsoft-ships-production-ready-agent-framework-1-0-for-net-and-python.aspx); [Microsoft Agent Framework devblog](https://devblogs.microsoft.com/agent-framework/?p=5212); [techstrong.ai](https://techstrong.ai/features/microsoft-ships-agent-framework-1-0-a-production-ready-foundation-for-multi-agent-ai/)
- **Google ADK**: `SequentialAgent` (deterministic pipeline), `ParallelAgent` (concurrent sub-agents writing to shared `session.state`, needing distinct `output_key`s), `LoopAgent` (until `escalate=True` or `max_iterations`), `AgentTool` (specialist called as a function, result returned inline) versus sub-agent transfer (`transfer_to_agent`). Built-in MCP and A2A support. Community sources only. — [community ADK guide](https://zack.onrender.com/docs/GEN%20AI%20DEV/Agents/Google%20ADK); [Google Cloud blog](https://cloud.google.com/blog/products/ai-machine-learning/build-multi-agentic-systems-using-google-adk); [nexu.io comparison](https://nexu.io/blog/microsoft-agent-framework-maf-semantic-kernel-autogen)
- **OpenAI Agents SDK**: `Agent.as_tool()`, handoffs, and code orchestration (structured outputs, chaining, evaluator loops, `asyncio.gather`). — [OpenAI Agents SDK docs](https://raw.githubusercontent.com/openai/openai-agents-python/main/docs/multi_agent.md)
- **Claude Agent SDK**: `agents`/`AgentDefinition`, forks, the `Workflow` tool (TS SDK v0.3.149+), and caps on depth, concurrency and spend. — [Agent SDK docs](https://code.claude.com/docs/en/agent-sdk/subagents)
- Analyst caution: production constraints (latency, cost, observability) "reveal themselves only in live deployments." — [techstrong.ai](https://techstrong.ai/features/microsoft-ships-agent-framework-1-0-a-production-ready-foundation-for-multi-agent-ai/)

### Inferences
- The cross-framework primitives a harness needs are: (1) agent-as-tool, returning a value; (2) handoff/transfer, changing the active agent; (3) deterministic sequential, parallel and loop combinators; (4) shared state with namespaced keys; (5) per-agent tool and model config; and (6) tracing that links child spans to the parent tool call (Claude's `parent_tool_use_id`, OpenAI tracing).
- Coding-agent harnesses (Claude Code, Codex, Devin) build orchestration natively instead of on LangGraph or CrewAI. The general frameworks matter more for enterprise workflow apps than for frontier coding or research harnesses. This is inferred from the docs reviewed, not from a usage survey.

### Gaps
- LangGraph (supervisor/swarm libraries, durable execution, LangChain "Deep Agents" subagents), CrewAI (crews/flows), DSPy (multi-module optimization) and smolagents (managed agents) were not researched this session because of the tool-call budget and the blocked LangChain blog. I have no verified 2026 status for them.
- I found no reliable 2026 survey of which frameworks production teams actually use.

---

## Q7. Chinese labs and "heavy"/parallel test-time compute modes (Kimi Agent Swarm, Grok Heavy, GPT-5 Pro, DeepSeek/Qwen)

### Takeaway
Moonshot's Kimi K2.5 **Agent Swarm** (early 2026) is the leading example of *trained* (RL) orchestration. A trainable orchestrator dynamically spawns up to 100 frozen sub-agents, with an RL reward that first encourages parallelism and then shifts to task success, plus a "critical steps" latency metric. Moonshot reports up to 4.5x faster execution. "Heavy" modes (Grok 4 Heavy) run multiple independent reasoning agents and then compare and aggregate. That is the parallel-sampling-plus-aggregation pattern, directly relevant to best-of-N harness modes.

### Cited Findings
- **Kimi K2.5 Agent Swarm** (research preview/beta on Kimi.com):
  - Self-directs up to **100 sub-agents** across up to **1,500 coordinated steps or tool calls** (units differ across sources).
  - A **trainable orchestrator** decomposes the task, and **dynamically instantiated frozen sub-agents** execute. "No predefined subagents or workflow."
  - **PARL** (Parallel-Agent RL) addresses "**serial collapse**," where the orchestrator defaults to single-agent execution. An auxiliary reward encourages parallelism early, then shifts to task success, preventing "fake parallelism."
  - Latency is measured by "**critical steps**," so spawning helps only if it shortens the critical path.
  - Reported gains: up to **4.5x** less execution time versus a single agent, and **80%** lower end-to-end runtime in internal evals.

  — [Kimi K2.5 blog (via search snippet)](https://kimi.com/en/blog/kimi-k2-5); [techloy](https://www.techloy.com/moonshot-ais-kimi-k2-5-deploys-100-sub-agents-simultaneously-cuts-coding-time-by-4-5x/); [morphllm (third-party)](https://www.morphllm.com/kimi-k2-5-agent-swarm)
- **Kimi K2.6** reportedly raises this to **300 sub-agents and 4,000 steps**. — [DataCamp tutorial (secondary)](https://www.datacamp.com/ko/tutorial/kimi-k2-agent-swarm-guide)
- **Grok 4 Heavy** (July 2025, background) launches parallel reasoning agents that "work independently, then compare notes," about 10x test-time compute. HLE: **44.4%** (Heavy) vs **25.4%** (Grok 4); the two figures may reflect different tool settings. Pricing is reported inconsistently ($300/mo vs $60/mo). — [OfficeChai](https://officechai.com/miscellaneous/elon-musk-explains-how-grok4-heavy-xais-multi-agent-reasoning-model-works/); [DataCamp](https://www.datacamp.com/blog/grok-4)
- **GPT-5 Pro / GPT-5.x Pro**: I found no primary source confirming a parallel multi-agent architecture. One weak source claims "scaled parallel compute… multiple reasoning shards… aggregating results." — [Ken Huang Substack (weak)](https://kenhuangus.substack.com/p/frontier-ai-models-technical-deep)

### Inferences
- Kimi's "serial collapse" finding mirrors Anthropic's note that models under-parallelize without explicit effort-scaling prompts, and the reverse problem with Opus 5, which over-delegates. Either way, the harness must **steer the degree of parallelism**: in the prompt, with caps, and ideally with a critical-path-aware cost model.
- The "critical steps" metric (latency equals the longest dependent chain, not total steps) is a useful harness telemetry metric for judging whether a fan-out helped.
- A "heavy mode" in a harness can be built as N independent rollouts, then a verifier or aggregator agent that reads all N and produces a final answer. The selector or aggregator is the quality bottleneck (see CodeMonkeys in Q4).

### Gaps
- I could not access Moonshot's technical report (kimi.com blocked), so I have no exact BrowseComp/WideSearch swarm-vs-single numbers or PARL reward formulas.
- I found **no reliable sources** on DeepSeek or Qwen multi-agent or parallel-thinking agent modes (for example "heavy" modes, ParaThinker-style parallel reasoning). Searches returned only Kimi material.
- I found no primary OpenAI description of GPT-5 Pro's parallel compute mechanism, and no head-to-head between heavy modes.
