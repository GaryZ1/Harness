# Context engineering (`harness_context`)

This package turns the append-only durable message log into the window the model sees on each call. It is the production implementation of
[`reports/SOTA agentic harness reference.md` §3](../../reports/SOTA%20agentic%20harness%20reference.md) (build checklist §3.7, pitfalls §3.8),
[`research_notes/.../context_engineering.md`](../../research_notes/SOTA%20agentic%20harness%20build%20plan/context_engineering.md) and the
"Context" section of the [build plan](../../reports/SOTA%20agentic%20harness%20build%20plan.md).

The entry point is `LayeredContextAssembler`, an implementation of `harness_core.types.ContextAssembler`:
`prepare(system, messages, tools, ctx) -> ModelRequest`.

| Component | What it does | Module | Doc |
|---|---|---|---|
| **Assembler** | Builds the window, walks the compaction ladder L0–L4, places breakpoints, reports per-category usage | `assembler.py` | [assembler.md](assembler.md) |
| **Prompt layers** | Stable system and project context: instructions, skills index, memory index, repo map, reference index | `layers.py`, `references.py` | [prompt-layout.md](prompt-layout.md) |
| **Dynamic channel** | Turn-scoped tail (budget note, warnings, git status, todo recitation) and persistent reminders | `dynamic.py` | [prompt-layout.md](prompt-layout.md#dynamic-channel) |
| **Caching** | TTL policy, a budget of ≤4 breakpoints, a checker for silent invalidators, prefix-diff and hit-rate monitoring | `cache.py` | [caching.md](caching.md) |
| **Tokens and budget** | Pluggable `TokenCounter` (4 chars/token heuristic, or `count_tokens`), thresholds, calibration from `usage` | `tokens.py`, `budget.py` | [budget-and-tokens.md](budget-and-tokens.md) |
| **Tool results** | 25k-token ingest cap, file offload with path and preview, sticky restorable clearing | `offload.py`, `clearing.py` | [tool-results.md](tool-results.md) |
| **Compaction** | Structured handoff summary through a `ModelClient`, kept tail, Codex user-message retention, re-injection bundle | `compaction.py`, `reinjection.py` | [compaction.md](compaction.md) |
| **Server-side** | Emits `context_management` edits (`clear_tool_uses`, `clear_thinking`, `compact_20260112`) and betas into `ModelRequest.extra` | `server.py` | [server-side.md](server-side.md) |
| **Sub-agents and guards** | Delegation briefs, depth and concurrency caps, condensed untrusted results, stuck and tool-count guards | `subagent.py`, `guards.py` | [subagents-and-guards.md](subagents-and-guards.md) |

## Why this shape

The package enforces each of these research conclusions in code. None of them depends on a prompt.

1. **Every token is a cost, a cache key and a distraction.** Context rot is a gradient. It is not a cliff, and Chroma found focused ~300-token inputs beat ~113k-token histories. Thresholds are therefore set for quality: compaction triggers at 150k, or `min(configured, 90% of the window)`, not when the window overflows.
2. **The cache-hit rate is the first-order cost metric.** Manus measures ~100:1 input:output, and a cached read costs 0.1× or less. The request layout is fixed (`tools → system → project → append-only messages`), serialization is canonical, the window has ≤4 breakpoints, and time-varying content goes only into a tail placed after the moving breakpoint. A checker flags timestamps, UUIDs and run ids in the prefix, and tracks expected rebuilds separately from unexpected misses.
3. **History is append-only.** Editing earlier turns breaks the prefix cache and, on the newest models, thinking signatures. Each window is therefore a *derived projection* of the log. Compaction is recorded by *appending* a marker message, and the marker holds a hash of the log prefix it covers. Clearing is a deterministic, sticky transform of the window. Neither one edits the log.
4. **Cheapest, most reversible rung first.** The ladder runs L0 (ingest cap), L1 (offload to file), L2 (clear stale tool results, only when the pass frees at least `clear_at_least` tokens), L3 (summarize, keeping the last 2 turns verbatim), then L4 (re-inject durable state). Placeholders keep paths, URLs and queries, so cleared content can be fetched again ("restorable compression").
5. **Compaction must not lose what summaries drop.** The summary schema always includes user preferences and learnings, decisions, open tasks and files touched. Files touched are merged from tool-call inputs, so the list does not depend on the model. Earlier user messages (≤20k tokens) and mid-conversation operator instructions carry forward. Instructions, the memory index, the plan, recent files and active skills are re-loaded from disk.
6. **Prefer server mechanisms where they exist.** The same policy can be emitted as Anthropic `context_management` edits instead. Server compaction blocks are kept verbatim, get a breakpoint, and trigger re-injection.
7. **Isolate and bound.** Sub-agents return condensed results of ≤2k tokens, labeled untrusted, with offloaded detail. Depth is capped at 3 and concurrency at 20.

## Architecture

```text
 durable log (append-only, owned by the loop)                         ModelRequest (derived, per call)
 ─────────────────────────────────────────────                        ─────────────────────────────────
 m0 m1 … m9 [tail m10…m13] M1(marker: covers m9) m14 … m20 …   ──▶     tools  (canonical, sorted)           [BP optional]
                                                                      system: base + SYSTEM + PROJECT layers [BP]
             LayeredContextAssembler.prepare()                        messages:
  ┌────────────────────────────────────────────────────────────┐        M1 summary + re-injection           [BP]
  │ 1 server compaction seen? → append re-injection bundle      │        m10 … m20 verbatim (thinking kept)
  │ 2 project: [latest marker] + messages after covers_through  │          (sticky <cleared-tool-result> …)  [BP mid if ≥20 blocks]
  │   (verify prefix sha256)                                    │        last durable message                [BP tail]
  │ 3 L0 cap → sticky clearing → estimate (calibrated)          │        dynamic tail: warnings, budget note,
  │ 4 ≥ 100k  → L2 clearing pass (if frees ≥ clear_at_least)    │          git status, todo recitation  (turn-scoped)
  │ 5 ≥ trigger → L3 Compactor → append marker → re-project     │      extra: request_extra (+ context_management, betas)
  │ 6 overflow? raise; warn once; dynamic tail; breakpoints ≤ 4 │
  │ 7 stability scan + prefix diff → ContextReport, events      │   observe_response(usage) → calibration + cache monitor
  └────────────────────────────────────────────────────────────┘
        ▲ L0/L1 at ingest: OffloadingToolExecutor(ToolOutputOffloader) → path + preview in the tool_result
```

## Quick start

```python
from harness_context import (
    AssemblerConfig, BudgetPolicy, Compactor, DynamicContext, FileActivityTracker, LayeredContextAssembler,
    OffloadingToolExecutor, PromptStack, RecentFilesSource, ReinjectionBuilder, SkillEntry, TextSource,
    ToolOutputOffloader,
)

stack = (PromptStack()
         .add_instructions(instructions_text, source="AGENTS.md")       # e.g. harness_memory.InstructionLoader(...).load().render()
         .add_skills_index([SkillEntry("pdf", "Fill and parse PDF forms")])
         .add_memory_index(memory_index_text))

tracker = FileActivityTracker()
reinjection = ReinjectionBuilder([
    TextSource("instructions", instructions_text),
    TextSource("plan", working_memory.recitation),                       # any zero-arg callable
    RecentFilesSource(tracker, repo_root),
])
dynamic = DynamicContext()
dynamic.register("git", lambda ctx: git_status_short())                  # turn-scoped, never cached
dynamic.register_recitation(working_memory.recitation)                   # always last

assembler = LayeredContextAssembler(
    stack=stack,
    config=AssemblerConfig(budget=BudgetPolicy(window_tokens=config.window_tokens),
                           request_extra={"thinking": {"type": "adaptive"}, "output_config": {"effort": config.effort}}),
    compactor=Compactor(model_client, reinjection=reinjection, events=event_log),
    dynamic=dynamic, events=event_log, state_dir=".harness/context",
)
tools = OffloadingToolExecutor(executor, ToolOutputOffloader(".harness/tool-outputs"))

# agent loop
request = assembler.prepare(BASE_SYSTEM_PROMPT, log, tools.specs(), ctx)   # may append a compaction marker to `log`
response = model_client.create(request)
assembler.observe_response(ctx, response)                                  # calibration + cache-hit tracking
log.append(Message.assistant(response.content))                            # full content, thinking blocks unchanged
for use in response.tool_uses:
    tracker.observe_call(use)
    ...
```

Tests: `python3 -m pytest -p no:cacheprovider tests/context` runs offline, with fake model clients, fake clocks and `tmp_path`.
