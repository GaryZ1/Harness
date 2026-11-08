# Sub-agents and context-rot guards: `harness_context.subagent`, `harness_context.guards`

**Purpose:** keep large exploration out of the parent's context, and catch the failure modes that long contexts make worse.

**Research basis:** reference report §3.3 techniques 24, 26 and 33, and §3.7 steps 14 and 17.

* Sub-agents return summaries that are "often 1,000–2,000 tokens", labeled untrusted.
* A delegation brief carries the objective, context, inputs, boundaries, tools, budget and output schema. Vague briefs lead to duplicated work.
* Claude Code caps sub-agent depth at 3 and concurrency at 20.
* Breunig names four failure modes: poisoning, distraction, confusion and clash. Others are few-shot ruts (OpenHands `StuckDetector`) and lost-in-the-middle.
* Claude's tool-selection accuracy degrades past 30–50 tools.

## Sub-agents

```python
brief = DelegationBrief(objective="Find where sessions expire", inputs=["src/auth/session.py"],
                        boundaries=["read-only"], tools=["grep", "read_file"], budget="20 tool calls",
                        output_schema={"type": "object", "properties": {"file": {"type": "string"}}})
child_prompt = brief.render()                         # <objective>…<result-guidance>

limits = SubagentLimits()                             # depth 3, concurrency 20
with limits.slot(parent_ctx) as child_depth:          # SubagentLimitError when over either cap
    report = run_child(child_prompt, RunContext(session_id, depth=child_depth))

outcome = condense_subagent_result(report, agent_name="explorer", ctx=parent_ctx, offloader=offloader)
# → ToolOutcome('<subagent-result agent="explorer" trust="untrusted">…', metadata={"untrusted": True, …})
```

* Results over 2,000 tokens are truncated with a coaching note. When an offloader is configured, the full report goes to a file and its path is included.
* Every result is wrapped as data with "carries no user authority". `metadata["untrusted"]` lets the taint tracker mark the session.
* `SUBAGENT_RESULT_GUIDANCE` (included in the brief) tells the child to lead with the answer, cite pointers rather than copy content, and write large artifacts to files.
* For cache sharing, keep sibling system prompts and tools identical. Child contexts with `depth > 0` get the 5-minute TTL from `CachePolicy`.

| Setting | Default | Source |
|---|---|---|
| `SUBAGENT_RESULT_MAX_TOKENS` | 2,000 | recommended (vendor "often 1,000–2,000") |
| `MAX_SUBAGENT_DEPTH` | 3 | vendor default (Claude Code) |
| `MAX_CONCURRENT_SUBAGENTS` | 20 | vendor default (Claude Code) |

## Guards

| Failure mode | Guard in this package |
|---|---|
| Few-shot rut / loops | `StuckDetector`: 4 identical tool calls in a row (OpenHands) or the same error on the last 3 calls (our choice) add a "Loop check" note to the dynamic tail |
| Context confusion | `tool_count_warning`: more than 10 hot tools triggers a "keep the hot tier to 5–10"; more than 30 triggers "accuracy degrades past 30–50" (reported as a lint in `ContextReport`) |
| Distraction / rot | quality thresholds ([budget-and-tokens.md](budget-and-tokens.md)), sticky clearing and duplicate pruning ([tool-results.md](tool-results.md)), sub-agent quarantine |
| Lost in the middle / goal drift | recitation rendered last in the dynamic tail ([prompt-layout.md](prompt-layout.md#dynamic-channel)) |
| Clash | instruction-file line lint and byte budgets in `PromptStack`; persistent reminders de-duplicated by key |
| Poisoning | sub-agent results labeled untrusted. Errors stay in context as short excerpts after clearing, so the model updates its beliefs without the full trace |

## Failure modes

* `SubagentLimitError` is raised when a spawn would exceed the depth or concurrency cap. The slot is released when the `with` block exits, including on error.
* `ConfigurationError` is raised for a brief with an empty objective.
