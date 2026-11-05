# Tool catalog: `harness_tools.catalog`

**Purpose:** keep 3–5 hot tools in the prompt and defer everything else (including every MCP tool) behind a `tool_search` tool that loads matches on demand.

## Research basis

* Definitions are a context cost. Five MCP servers with 58 tools cost about 55K tokens, and up to 134K has been observed. Claude's selection "degrades once you exceed 30–50 available tools".
* Deferred loading behind tool search cut definition tokens by more than 85% and raised MCP-eval accuracy from 49% to 74% on Opus 4 and from 79.5% to 88.1% on Opus 4.5 (Anthropic, *Advanced tool use*).
* When to adopt it: 10+ tools, more than 10K definition tokens, multiple servers, or selection errors. It is not worth it below 10 tools.
* API limits and conventions: `tool_search_tool_bm25_20251119` takes natural language up to 500 characters, and `tool_search_tool_regex_20251119` takes a Python regex up to 200 characters. Both search names, descriptions, argument names and argument descriptions. Default 5 results. A custom (client-side) search returns `tool_reference` blocks.
* Pitfalls: deferring the search tool itself, deferring every tool (the API returns 400), putting `cache_control` on deferred tools, and mutating the tool array mid-session.

## API

```python
from harness_tools import (enable_tool_search, apply_tiering, should_enable_tool_search,
                           server_tool_search_definition, server_tool_search_spec)

catalog = enable_tool_search(registry, default_mode="bm25", default_limit=5)   # registers "tool_search"
if should_enable_tool_search(registry.specs()):
    apply_tiering(registry, hot=["read", "edit", "bash", "grep", "glob"])      # defers everything else
registry.freeze()

catalog.search("create a jira ticket")                  # [SearchHit(name="jira_issue_create", score=...)]
catalog.search(r"github.*pull", mode="regex")

# Server-side alternative (Anthropic executes the search):
registry.register_spec(server_tool_search_spec("bm25"))  # {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}
```

### Two delivery modes (`ToolRegistry(deferral=...)`)

| Mode | Tool array | `tool_search` result | Use with |
|---|---|---|---|
| `reference` (default) | Every tool; deferred ones carry `defer_loading: true` | `[{"type": "tool_reference", "tool_name": ...}, ...]`, which the API expands | Anthropic |
| `reveal` | Hot tools only; discovered tools are appended in discovery order | Text with each matched tool's name, description and `input_schema` | Providers without `defer_loading` |

Discovery is registry state, so use one registry per session in `reveal` mode. `tool_search` is `read_only` but not `concurrency_safe`, so parallel searches cannot make discovery order (and therefore the tool array) depend on thread timing.

### Search quality

Text is normalised before BM25 indexing: identifiers are split (`mcp__github__createIssue` → `mcp github create issue`) and words get a light suffix stem (`requests` → `request`). The BM25 index is `harness_memory.retrieval.BM25Index` (k1 = 1.5, b = 0.75) and is rebuilt whenever the set of deferred tools changes. Regex mode ranks name matches above description matches, then sorts by name, so results are deterministic.

## Defaults

| Setting | Default | Anchor |
|---|---|---|
| Results per search | 5 (max 20 client-side) | Anthropic tool search default `limit` 5 |
| BM25 query length | ≤ 500 chars | `tool_search_tool_bm25_20251119` |
| Regex length | ≤ 200 chars | `tool_search_tool_regex_20251119` |
| Adoption trigger | ≥ 10 tools or > 10,000 definition tokens | Anthropic *Advanced tool use* |
| Hot set | 3–5 tools (caller supplies the names) | Anthropic guidance |
| MCP tools | deferred | Claude Code (MCP tools deferred by default) |

## Invariants (enforced in code)

* `tool_search` can never be registered with `defer_loading=True`. `apply_tiering` never defers it or a server-side `tool_search_tool_*`.
* `validate_tool_array()` (also run by `freeze()` and `apply_tiering`) fails when every tool is deferred, or when tools are deferred without a non-deferred search tool.
* `apply_tiering` only runs before `freeze()`, and strips `cache_control` from deferred specs' metadata.
* Search covers deferred tools only (pass `include_loaded=True` to change that) and never returns the search tool itself or provider-only specs.

## Failure modes

* Invalid regex or an over-long query: an error result telling the model the limit or how to escape.
* No matches: a hint to try other keywords or the other mode. Nothing is loaded.
* Weak descriptions make tools unfindable. Run `design.lint_tools`, write descriptions that say *what* and *when*, and include argument descriptions, which are indexed too.
