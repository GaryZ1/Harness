# Server-side context management: `harness_context.server`

**Purpose:** let the provider edit the window instead of the client. The same policies are emitted as Anthropic `context_management` edits in `ModelRequest.extra`, and server compaction blocks are handled correctly on later turns.

**Research basis:** reference report §3.3 techniques 14–16, §3.5, and the API facts for this build. The research prefers provider mechanisms where available, because the server reduces the window without the client rewriting history. Editing earlier turns on the client invalidates thinking signatures on the newest models.

## API

```python
cfg = AssemblerConfig(
    clearing=None,                                   # client clearing off (mutually exclusive with clear_tool_uses)
    server=ServerContextManagement(clear_tool_uses=True, clear_thinking=True, compact=True, pause_after_compaction=True),
    request_extra={"output_config": {"effort": config.effort}},
)
asm = LayeredContextAssembler(config=cfg, reinjection=builder)    # no client compactor with server compaction
req = asm.prepare(system, log, tools, ctx)
req.extra["context_management"]   # {"edits": [clear_thinking…, clear_tool_uses…, compact…]}
req.extra["betas"]                # sorted union with any caller betas
```

The emitted edits are:

| Edit | Emitted fields | Defaults | Source |
|---|---|---|---|
| `clear_thinking_20251015` (always listed first) | `keep`: `"all"` or `{"type": "thinking_turns", "value": N}` | `"all"` | vendor rule (list first) |
| `clear_tool_uses_20250919` | `trigger` `{"type": "input_tokens", "value": 100000}`, `keep` `{"type": "tool_uses", "value": 3}`, `clear_at_least` 5,000, `exclude_tools` (memory/todo/plan), `clear_tool_inputs` only if true | 100k / 3 | vendor defaults; `clear_at_least` and exclusions are our choice |
| `compact_20260112` | `trigger` `{"type": "input_tokens", "value": 150000}` (≥ 50,000 enforced), `pause_after_compaction` if true, `instructions` = `SUMMARY_PROMPT` | 150k | vendor default; instructions are our choice |

The betas are `context-management-2025-06-27` (for the clearing edits) and `compact-2026-01-12` (for compaction).

**Integrator contract:** the adapter forwards `extra["betas"]` as beta headers and `extra["context_management"]` as the request field. Other `extra` keys pass through unchanged.

## Server compaction blocks

* The response's `compaction` block is an `OpaqueBlock`. The loop appends the full `response.content` to the log, and this package never edits it.
* On the next `prepare`, the assembler finds the newest assistant message that holds a `compaction` block and does the following:
  * it emits a COMPACTION event (`server_compaction`) and resets the pre-compaction warning;
  * it marks the next prefix diff as an expected rebuild;
  * it appends the re-injection bundle as a `kind="reinjection"` user message, but only when the log does not end with an unanswered `tool_use` (otherwise it retries on the next call).
* The bundle is appended at most once per compaction block, even after a restart, because the assembler checks the log for an existing bundle that refers to that block.
* Messages before the block are still sent verbatim, which keeps history append-only, but they are left out of the token estimate because the API drops them.
* The compaction message gets a cache breakpoint. When the message contains only that block, the breakpoint goes on the block itself. The durable copy is never touched.

## Invariants and failure modes

* `ConfigurationError` is raised for a compaction trigger under 50,000, for server `clear_tool_uses` combined with client `clearing`, and for server `compact` combined with client auto-compaction.
* `apply()` returns a new dict and leaves the caller's `extra` unchanged.
* Unknown providers ignore beta headers. For example, the DeepSeek Anthropic-compatible endpoint ignores them. Use the client-side ladder there.
