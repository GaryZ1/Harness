# Registry and execution: `harness_tools.registry`, `harness_tools.output`

**Purpose:** the single `ToolExecutor` the agent loop calls. It publishes a deterministic tool array (`specs()`) and executes calls (`execute`, `execute_batch`, `run_tool_uses`). Execution never raises; every failure becomes an `is_error` tool result the model can act on.

## Research basis

* **Errors as tool results** with a corrected-input example. Validation failures, unknown tools, permission denials and dead tools are returned to the model, never raised as transport errors (MCP SEP-1303; OpenAI Agents SDK `"return_error_to_model"`). The reference's §5.8 lists "opaque errors" as an anti-pattern.
* **Escalation**: 3 consecutive or 20 total denials go to a human (Claude Code auto mode). Tool errors and permission denials are counted separately.
* **Output budgets**: a 25,000-token cap per result (Claude Code). Truncation messages that coach narrower queries. Oversize results offloaded to a file, returning path + preview + size (deepagents `_blob_offload`).
* **Concurrency classes**: read-only tools in parallel, mutating tools sequentially (Claude Agent SDK), with a concurrency cap (`max_function_tool_concurrency`-style).
* **Parallel tool use**: one assistant turn may hold many `tool_use` blocks, and all `tool_result` blocks go back in ONE user message, failures included.
* **Cache stability**: deterministic (sorted) serialisation. The tool array is never mutated mid-session; deferral is used instead.
* **Telemetry per call** (checklist step 24): tool name, call id, args hash, result size and tokens, truncated flag, error class, duration and idempotency key.

## API

```python
from harness_tools import Tool, ToolRegistry, OutputPolicy, function_tool
from harness_core.types import RunContext, ToolCall, ToolSpec

reg = ToolRegistry(
    output_policy=OutputPolicy(max_tokens=25_000, offload_dir=None),  # register_kernel sets offload_dir
    max_concurrency=8,
    permission_policy=None,        # any harness_core PermissionPolicy
    approver=None,                 # (call, spec, ctx, reason) -> bool, for ASK decisions
    taint_tracker=None,            # e.g. harness_memory.TaintTracker
    on_record=None,                # ToolCallRecord callback (metrics, traces)
    deferral="reference",          # or "reveal" for providers without defer_loading
)

# 1. Decorated function: schema from the signature, description from the docstring.
@reg.tool(read_only=True, input_examples=[{"user_id": "USR-12345", "limit": 5}])
def recent_orders(user_id: str, limit: int = 10) -> list[dict]:
    """List a customer's most recent orders (newest first).

    Args:
        user_id: Customer ID in the form USR-<digits>.
        limit: Maximum orders to return.
    """

# 2. Raw handler with an explicit spec: handler(input: dict, invocation: ToolInvocation)
def fetch(inp, inv):
    if inv.cancelled.is_set(): ...
    return {"title": "...", "text": "..."}

reg.register(Tool(ToolSpec("web_fetch", "Fetch a URL and return its readable text.",
                           {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
                           read_only=True, open_world=True), fetch, timeout_s=30))

reg.freeze()                                       # session start: hot tool set locked
out = reg.execute(ToolCall("toolu_1", "recent_orders", {"user_id": "USR-1"}), RunContext("s1"))
reply = reg.run_tool_uses(model_response, ctx)     # Message.user([...all tool_result blocks...])
```

Handler return values are normalised as follows: a `ToolOutcome` passes through, `str` becomes text, a list of content blocks (`text`/`image`/`document`/`tool_reference`/`search_result`) is kept as blocks, other JSON values are serialised with `json.dumps(indent=2)`, and `None` becomes `"(no output)"`. Raise `ToolInputError(msg, example={...})` for a model-fixable problem.

`Tool` fields: `timeout_s` (60 s), `max_output_tokens` (per-tool cap below the global one), `concurrency_safe` (defaults to `spec.read_only`), `idempotent`, `destructive`, `input_examples`, `validate_input`, `truncation_hint` and `source`. All flags are copied into `ToolSpec.metadata`, where permission and hook packages can read them.

`api_tools(include_input_examples=True)` emits `input_examples` (beta field). `ToolSpec` has no such field, so the examples live in `spec.metadata["input_examples"]`.

## Defaults

| Setting | Default | Anchor |
|---|---|---|
| Result cap | 25,000 tokens (~100k chars at 4 chars/token) | Claude Code tool-response cap |
| Offload preview | 2,000 tokens, 60% head / 40% tail | our choice (errors and summaries sit at the end) |
| Offload location | `<workspace>/.harness/tool-outputs/<call_id>.txt` | our choice (deepagents pattern) |
| Per-tool timeout | 60 s (`bash`: 630 s, enforced by its runner at ≤ 600 s) | our choice |
| Max parallel calls | 8 | our choice (Agents SDK exposes the knob, no public default) |
| Escalation | 3 consecutive / 20 total, errors and denials counted separately | Claude Code auto mode |
| Tool name rule | `^[a-zA-Z0-9_-]{1,64}$` | Anthropic tool-name rule |
| Spec order | sorted by name; in `reveal` mode, discovered tools are appended | MCP 2026-07-28 "deterministic order" |

## Invariants (enforced in code)

* `execute` never raises for a tool failure. Unknown tool, malformed or invalid input, permission DENY or un-approved ASK, timeout, handler exception and provider-only tool all come back as `ToolOutcome(is_error=True)` with `metadata["error_class"]`.
* Input must be a JSON object. A JSON string is parsed, never string-matched. Schema validation runs before the handler. A policy's `updated_input` is validated again.
* Registration fails loudly on a bad name, a duplicate, an empty description, a malformed schema, a `strict` tool without `additionalProperties: false`, an `input_example` that violates the schema, or a deferred `tool_search`.
* After `freeze()`, only `defer_loading=True` tools can be added, and loaded tools cannot be removed.
* `freeze()` and `validate_tool_array()` require that at least one tool stays loaded, and that a non-deferred search tool exists whenever anything is deferred.
* `execute_batch` returns outcomes in call order. A mutating call starts only after every earlier call has finished, and nothing after it starts until it has finished.
* Successful output of an `open_world` tool, or any outcome flagged `metadata["untrusted"]`, sets `ctx.tainted = True` and feeds the taint tracker.
* `tool_results_message` refuses mismatched call and outcome counts and duplicate ids.

## Failure modes

| Failure | Behaviour |
|---|---|
| Handler hangs | After `timeout_s`, an error result is returned, `invocation.cancelled` is set and the daemon worker thread is abandoned. Python cannot kill threads, so long-running handlers should poll `cancelled` or `remaining()`. |
| Offload directory unwritable | Logged; the result falls back to in-place head+tail truncation. |
| `on_record` callback raises | Logged and ignored; telemetry never breaks the loop. |
| Tracer raises | Converted into an internal-error result. |
| Model loops on failing calls | `metadata["escalate"]` is set on the result that crosses the limit; the loop decides whether to hand control to a human. |
