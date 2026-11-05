# Tool results: `harness_context.offload`, `harness_context.clearing`

**Purpose:** keep tool output from flooding the window, using the cheapest and most reversible rungs of the compaction ladder. L0 caps output at ingest, L1 offloads large output to files, and L2 clears stale results with restorable placeholders.

**Research basis:**

* Reference report §3.3 techniques 10, 14, 21 and 26, and §3.7 step 6 plus the L0–L2 rungs of step 11.
* "Claude Code restricts tool responses to 25,000 tokens by default". Truncation must say what was cut and how to narrow the query.
* Large outputs go to the workspace as path + preview + stats (Manus treats the file system as "the ultimate context"; see also deepagents `_blob_offload.py`).
* `clear_tool_uses_20250919` semantics: trigger at 100k, keep the last 3, `clear_at_least`, `exclude_tools`, `clear_tool_inputs` off.
* Context editing improved an agentic-search eval by 29%, or 39% with memory, and cut tokens by 84% on a 100-turn eval.
* Pitfall: clearing without `clear_at_least`, or pruning on every turn, pays for a cache write that saves almost nothing.
* Keep errors in, but compactly.

## Ingest: cap and offload (L0/L1)

```python
offloader = ToolOutputOffloader(".harness/tool-outputs", policy=OffloadPolicy(), display_path=lambda p: str(p.relative_to(repo)))
executor = OffloadingToolExecutor(inner_executor, offloader)   # implements harness_core ToolExecutor
outcome = offloader.process(call, outcome, ctx)                # or call directly
offloader.restore(path)                                        # sandboxed to the offload root
```

* If the output is above `offload_threshold_tokens` and a root is configured, the full text is written atomically to `<root>/<session_id>/<tool>-<tool_use_id>.txt`, with names sanitized to `[A-Za-z0-9_-]`. The result becomes an `<offloaded-tool-output path=… bytes=… lines=… tokens=… sha256=…>` block that holds a head and tail preview and the hint "read with an offset/limit or line range, grep, head/tail". Metadata gains `offloaded_path` and `original_tokens`, and existing metadata such as `untrusted` is kept.
* If the output is above `max_result_tokens` and there is no root, it is truncated deterministically with "Narrow the request (filters, line ranges, pagination…)".
* Image and document results, and excluded tools (`memory`), pass through untouched.

## Clearing (L2)

```python
clearer = ToolResultClearer(ClearingPolicy(), counter)
plan = clearer.plan(window, already_cleared)     # ClearingPlan(cleared_ids, freed_tokens, skipped_reason, duplicate_of)
window2 = clearer.apply(window, cleared_ids, duplicate_of=plan.duplicate_of)
```

A placeholder looks like this:

```text
<cleared-tool-result tool="read_file" tokens="~3086">
This tool output was cleared from context to save space.
Locators: path=src/app.py. To restore, re-run read_file with the same input or open the locator directly.
</cleared-tool-result>
```

When the original output was offloaded, the placeholder points at the file instead. Error results add `error="true"` and keep a 300-char excerpt of the error. An earlier result that is byte-identical to a later one is cleared even inside the keep window, with "Identical to the later result of tool call …".

Clearing is **sticky** and runs only at clearing passes. The assembler keeps the cleared set per session, and persists it with `state_dir`, then applies it identically to every later window. Between passes the derived prefix therefore stays byte-stable. Each pass is logged as a COMPACTION event (`tool_result_clearing`) and treated as an expected cache rebuild.

| Setting | Default | Source |
|---|---|---|
| `max_result_tokens` | 25,000 | vendor default (Claude Code) |
| `offload_threshold_tokens` | 8,000 | our choice |
| preview | first 1,500 + last 500 chars | our choice |
| `trigger_tokens` | 100,000 | vendor default |
| `keep_recent` | 3 | vendor default |
| `clear_at_least_tokens` | 5,000 | our choice (amortize one cache write) |
| `exclude_tools` | `memory`, `todo`, `todo_write`, `TodoWrite`, `update_plan`, `plan` | research recommendation |
| `clear_tool_inputs` | `False` | vendor default |
| `min_result_tokens` | 50 | our choice |
| `dedupe` | `True` | our choice (near-duplicate pruning) |

## Invariants

* The durable log is never modified. `apply` returns copies, and untouched messages are reused as they are. `tool_use_id`, `is_error` and block order are kept, so tool pairing stays intact.
* `clear_tool_inputs` never edits an assistant turn that contains thinking blocks, because that would invalidate signatures.
* Placeholders contain no timestamps, so the same input always produces the same bytes.

## Failure modes

* `OffloadError` is raised on an I/O failure or when `restore` targets a path outside the root.
* A result the task depends on as evidence should be offloaded, or its tool excluded, before it ages out of the keep window.
