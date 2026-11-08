# Prompt caching: `harness_context.cache`

**Purpose:** decide where breakpoints go and how long they live. Detect silent cache invalidators before they cost money, and verify hits from `usage`.

**Research basis:** reference report §3.3 techniques 5–7, §3.7 steps 1–3 and §3.8 "Cache breakers". The Anthropic cache is a byte-exact prefix match in the order `tools → system → messages`, with ≤4 breakpoints and a 20-block lookback. Agents are input-dominated: Manus measures ~100:1 input to output and treats the hit rate as "the single most important metric". Timestamps or ids in the prefix, unsorted JSON, varying tool sets and effort or thinking changes all invalidate the cache silently. Track "expected rebuilds" separately from unexpected misses.

## Breakpoint placement (done by the assembler)

| Priority | Breakpoint | Where |
|---|---|---|
| 1 | `system` | last system/project block; covers tools + system |
| 2 | `tail` | last eligible block of the last *durable* message (moves each turn; the dynamic tail sits after it) |
| 3 | `mid` | previous call's tail position, when ≥ 20 blocks were added since (lookback) |
| 4 | `compaction` | the client compaction marker or the server `compaction` block |
| 5 | `tools` | last non-deferred tool via `ToolSpec.metadata["cache_control"]` (off by default; the adapter must emit it) |

The assembler keeps the first `max_breakpoints` entries in this order. Only text and tool-result blocks, plus the server `compaction` block, can carry a breakpoint. Thinking blocks never do.

## API

```python
policy = CachePolicy(main_ttl="1h", subagent_ttl=None)       # ctx.depth > 0 → 5-minute default
policy.cache_control(ctx)                                     # {"type": "ephemeral", "ttl": "1h"}

checker = CacheStabilityChecker(strict=False, allow=["Policy version 2026-01-01"])
findings = checker.scan_request(request, prefix_tokens=n, min_cacheable_tokens=1024)
checker.enforce(findings)                                     # BreakpointLimitError always; CacheStabilityError if strict
breaks = CacheStabilityChecker.compare(prev_snapshot, PrefixSnapshot.of(request), expected_reason="compaction")

monitor = CacheMonitor()
monitor.observe(response.usage, expected_rebuild=None)        # True = unexpected miss
monitor.stats.hit_rate                                         # cache_read / (cache_read + cache_write + input)
```

### Detected invalidators (`VOLATILE_PATTERNS`)

| Kind | Example | Severity |
|---|---|---|
| `iso_timestamp` | `2026-11-05T10:00` | error |
| `uuid` | `3f2a9c1e-1111-4abc-…` | error |
| `clock_time` | `10:00:00` | error |
| `unix_epoch` | `1762336800` | error |
| `run_id` | `session_…`, `req_…`, `toolu_…`, `msg_…` | error |
| `date` | `2026-11-05` | warning |
| `relative_now` | "today is", "current date" | warning |

Other request-level findings are `duplicate_tool`, `deferred_tool_breakpoint` (deferred tools cannot carry `cache_control`), `breakpoint_limit`, `volatile_breakpoint` (a breakpoint on a message block that contains a timestamp means "a write every request and never a read") and `below_min_cacheable`.

`PrefixSnapshot` stores fingerprints of tools, system, *config* and every window message. Config covers `extra` and `max_tokens`, so an effort, thinking or `context_management` change counts as a break. `compare` reports a break in `tools`, `system` or `config`, or the first diverging message index. A break is marked `expected` when a rebuild reason was declared: clearing, compaction, server compaction or `declare_rebuild`.

## Defaults

| Setting | Default | Source |
|---|---|---|
| `main_ttl` | `"1h"` | recommended (interactive main thread) |
| `subagent_ttl` | `None` = 5 minutes | vendor default for sub-agents |
| `max_breakpoints` | 4 | vendor limit |
| `lookback_blocks` | 20 | vendor limit |
| `breakpoint_on_tools` | `False` (the system breakpoint already covers the tools prefix) | our choice |
| `min_cacheable_tokens` | 1,024 (model dependent: 512–4,096) | our choice; set per model |

## Invariants and failure modes

* More than 4 breakpoints raises `BreakpointLimitError` in every mode.
* The checker runs on every `prepare`. Unexpected prefix breaks emit `Event(type=CUSTOM, data={"kind": "cache_break", …})`.
* `observe_response` counts a response with zero cache reads as an unexpected miss *unless* it is the session's first response in this process or follows a declared rebuild. Each unexpected miss emits a CUSTOM `cache_miss` event. Treat these as regressions.
* The heuristics produce false positives, such as a pinned version date. Use `allow=[...]` instead of disabling the checker.
