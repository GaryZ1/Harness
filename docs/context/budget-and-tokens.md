# Tokens and budget: `harness_context.tokens`, `harness_context.budget`

**Purpose:** estimate window size behind a pluggable protocol, and decide when to warn, clear, compact or refuse.

**Research basis:** reference report §3.3 techniques 2, 4 and 25, §3.7 steps 1, 11 ("Guards") and 15, and §3.5. Claude token counts must come from `client.messages.count_tokens` and never from tiktoken. Total input is `input + cache_read + cache_creation`. Trigger compaction for quality, not overflow: the Claude default is 150k (minimum 50k for the server edit) and Manus uses a "pre-rot" 128K–200K. The hard ceiling is `min(configured, 90% of window)` with a 95% effective window (Codex). Skip auto-compaction when the window is unknown (Cline). Give the model a turn-scoped "tokens used / remaining / next compaction at N" note.

## Token counters

```python
class TokenCounter(Protocol):
    def count_text(self, text: str) -> int: ...
    def count_request(self, request: ModelRequest) -> int: ...

HeuristicTokenCounter()                        # ceil(chars / 4) + 4/message + 3/block + 12/tool; images 1,600
AnthropicTokenCounter(model=config.model, client=anthropic_client)   # count_request → messages.count_tokens
```

`AnthropicTokenCounter` imports `anthropic` lazily and raises a clear `ImportError` if the package is missing. It falls back to the heuristic when the API call fails (`strict=True` re-raises), and always uses the heuristic for `count_text` and per-block estimates. The model id must be supplied by the caller. The assembler calls `count_request` once per window state, so roughly 1–3 times per `prepare` plus once for the prefix, and an exact counter therefore adds API calls.

## Budget policy

```python
policy = BudgetPolicy(window_tokens=200_000)       # trigger = min(150_000, 180_000) = 150_000
policy.classify(tokens, max_tokens=16_000)         # OK | WARN | COMPACT | OVERFLOW
policy.note(tokens)                                # "Context budget: ~X tokens in use of a … window (~Y remaining); next compaction at ~N."
```

| Setting | Default | Source |
|---|---|---|
| `window_tokens` | `None` (unknown: no auto-compaction, no overflow check) | Cline behaviour |
| `compaction_threshold_tokens` | 150,000 | vendor default (`compact_20260112`) |
| `max_threshold_fraction` | 0.90 | Codex |
| `effective_window_fraction` | 0.95 | Codex `effective_context_window_percent` |
| `warn_fraction` | 0.90 of the trigger | our choice |

The levels are:

* `WARN` adds `PRE_COMPACTION_WARNING` to the tail once per compaction cycle. The warning tells the model to save findings to memory or the progress log before results are cleared.
* `COMPACT` triggers L3.
* `OVERFLOW` is reached when `tokens + max_tokens > effective window`. It raises `ContextOverflowError`.

## Calibration

`UsageCalibration` keeps the ratio `usage.total_input / estimate` from the last response, clamped to [0.5, 3.0], and applies it to later estimates. The assembler stores the raw estimate of each request it builds. `observe_response(ctx, response)` feeds the ratio back, so heuristic decisions follow real token counts without an extra API call.

## Failure modes

* A ratio from a response whose request differed greatly in content (for example images) can mislead one decision. The clamp bounds the error.
* When the window is unknown, only clearing runs automatically. Use `assembler.compact(...)` explicitly.
