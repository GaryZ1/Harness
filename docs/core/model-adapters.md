# Model adapters and test doubles: `anthropic_adapter`, `testing`

**Purpose:** implement the provider-neutral `ModelClient` protocol for the Anthropic Messages API,
and give every package a deterministic fake for offline tests.

## Research basis

* Checklist §2.7 step 6: one reasoning adapter per provider; Anthropic signed thinking blocks must be returned unmodified; store reasoning verbatim; never edit prior turns; never change effort or thinking mid-session.
* Current API facts: `thinking={"type": "adaptive"}` plus `output_config={"effort": …}`; `budget_tokens`, `temperature`, `top_p`, `top_k` and assistant prefill are rejected on current models; forced `tool_choice` (`any`/`tool`) returns 400 on the newest models; context editing and server compaction are betas on `client.beta.messages.create`; never defer all tools.
* Pitfall §1.8: "Assuming a compatible endpoint supports everything" — keep provider specifics inside the adapter.

## `AnthropicModelClient`

```python
import os
from harness_core.anthropic_adapter import AnthropicModelClient

model = AnthropicModelClient(
    os.environ["HARNESS_MODEL"],     # required; never defaulted
    effort="medium",                 # output_config.effort; None = model default
    betas=[],                        # sent on every call
    client_kwargs={"base_url": os.environ.get("ANTHROPIC_BASE_URL")},   # e.g. a compatible endpoint
)
```

`anthropic` is imported only when no `client` is passed (`ImportError` with install instructions
otherwise); the SDK client is created with `max_retries=0` so the loop's `RetryPolicy` is the only
retry layer.

| Concern | Behaviour |
|---|---|
| Request | `model`, `max_tokens`, `messages` (`Message.to_api()`, including mid-conversation `system` role messages), `system` (string or block list) if non-empty, `tools` (`ToolSpec.to_api()`, provider-defined tools such as `memory_20250818` pass through), `thinking`, `output_config` |
| Per-request extras | `ModelRequest.extra` merged over constructor `extra`; `output_config` dicts are merged; `betas` lists are concatenated |
| Betas | Any beta → `client.beta.messages.create(betas=[…])`. `context_management.edits` types add their header: `clear_tool_uses_20250919`/`clear_thinking_20251015` → `context-management-2025-06-27`, `compact_20260112` → `compact-2026-01-12` |
| Refused up front (`ConfigError`) | empty model id, unknown effort, `temperature`/`top_p`/`top_k`/`budget_tokens`, `thinking.type == "enabled"`, `tool_choice` `any`/`tool`, every tool `defer_loading` |
| Response | `block_from_api` per block: text, `tool_use`, `tool_result`; everything else (thinking, redacted thinking, compaction, server tool use and results) is an `OpaqueBlock` kept verbatim; `stop_reason` via `StopReason.parse` (unknown → `other`); usage with `None` → 0 |
| Errors | `classify_exception`: 429 / `RateLimitError` → `RateLimitError(retry_after from the header)`; connection/timeout and 408/409/5xx/529 → `TransientModelError`; 400/413 mentioning "prompt is too long" / context window → `ContextWindowExceededError`; other statuses and unknown exceptions → `TerminalModelError` |

Prompt caching is left to the context assembler: `cache_control` on `TextBlock`,
`ToolResultBlock` and system blocks passes through unchanged.

## `harness_core.testing`

```python
from harness_core.testing import ScriptedModelClient, text_response, tool_response, make_response, FakeMonotonic

model = ScriptedModelClient([
    tool_response(("read", {"path": "a"}, "tu_1"), ("read", {"path": "b"})),   # id optional
    TransientModelError("overloaded"),                                       # raised on that call
    lambda request: text_response(f"{len(request.messages)} messages"),     # computed from the request
    text_response("done"),
], default=None)            # default=<item> answers forever once the script is used up
result = AgentLoop(model, tools).run("go")
model.requests[1].messages  # snapshots of every request received
```

| Helper | Purpose |
|---|---|
| `ScriptedModelClient` | Replays responses / exceptions (any `BaseException`, so tests can simulate a process crash) / callables; records request snapshots; `ScriptExhaustedError` when out of script |
| `text_response`, `tool_response`, `tool_use`, `make_response` | Build `ModelResponse`s (default usage 100 in / 20 out, model `"test-model"`) |
| `FakeMonotonic` | Inject as `AgentLoop(monotonic=…)`; `advance(seconds)` drives wall-clock budgets |
| `RecordingTracer`, `RecordingSink` | Capture spans (with attributes) and events |

## Failure modes

* **SDK not installed:** `ImportError` naming the package, only when the adapter builds its own client.
* **Forbidden or obsolete parameters:** `ConfigError` before any network call, instead of a 400 mid-run.
* **Endpoint ignores betas** (compatible third-party endpoints): feature-detect at configuration time; the adapter cannot tell.
