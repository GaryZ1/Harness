# Tool design hygiene: `harness_tools.design`

**Purpose:** keep the tool surface cheap and unambiguous. Measure definition tokens, lint descriptions and schemas in CI, and pin approved third-party descriptions so changes are caught.

## Research basis

* *Writing effective tools for AI agents* (Anthropic, Sep 2025):
  * build task-shaped tools, not 1:1 endpoint wrappers;
  * namespace by service and resource;
  * use unambiguous parameters (`user_id`, not `user`);
  * describe every parameter, writing "as if onboarding a new hire";
  * return semantic fields instead of UUIDs and MIME types;
  * offer a `response_format` enum for large payloads (206 vs 72 tokens in the Slack example);
  * make errors actionable.
* `input_examples` (1–5 realistic calls) raised complex-parameter accuracy from 72% to 90%.
* The evidence-based tool-count triggers are more than 10K definition tokens, 10+ tools, or multiple servers, plus Claude's 30–50-tool degradation point. The folk "<20 tools" rule is not one of them.
* MCP hygiene: pin server versions, hash approved descriptions, and diff and alert on change (tool poisoning, rug pulls).
* Descriptions are versioned files next to the code, with their hash recorded in traces (Crush `.md.tpl`; checklist step 8). The kernel stores `description_sha256` in spec metadata.
* Iterate descriptions against a held-out eval set. A rewritten MCP description cut later agents' completion time by 40%, and overfitting without a held-out set is a listed pitfall.

## API

```python
from harness_tools import lint_tool, lint_tools, fingerprint, DescriptionPins, definition_tokens
from harness_tools.design import errors

findings = lint_tools(registry.specs(), target="anthropic")   # or "openai_strict" / "deepseek_strict"
assert not errors(findings), "\n".join(map(str, findings))    # CI gate; warnings/info are advisory

definition_tokens(registry.specs())                           # ~tokens the tool array costs

pins = DescriptionPins(path=".harness/mcp-pins.json")
pins.status("tracker/search_issues", fingerprint(raw_mcp_tool))   # "approved" | "new" | "changed"
pins.approve("tracker/search_issues", fingerprint(raw_mcp_tool))  # atomic JSON write
```

### Lint rules

| Severity | Rule |
|---|---|
| error | name does not match `^[a-zA-Z0-9_-]{1,64}$`; missing description; duplicate names; schema violates the target's strict rules (`additionalProperties`, all-`required`, DeepSeek-unsupported keywords) |
| warning | description under 40 chars; a single definition over ~1,000 tokens; parameter without a description (enums excepted); ≥ 10 loaded tools or > 10K definition tokens with nothing deferred; more than 30 tools loaded |
| info | ambiguous parameter names (`user`, `id`, `data`, ...); path parameters that do not say "absolute"; nested or formatted parameters without `input_examples`; keywords the client-side validator ignores; nearly identical tool names (overlap); unsorted tool array |

## Defaults

| Setting | Default | Anchor |
|---|---|---|
| Tool-search trigger | 10 tools / 10,000 tokens | Anthropic *Advanced tool use* |
| Loaded-tool warning | > 30 | lower edge of Claude's 30–50 degradation range |
| Min description | 40 chars | our choice |
| Max definition | ~1,000 tokens | our choice |
| Token estimate | 4 chars per token over the JSON definition | our choice (provider-neutral heuristic) |

## Invariants

* Lint never mutates specs and never blocks registration by itself. The registry enforces only the hard rules: name, description, schema, strict, and search-tool deferral.
* `fingerprint` covers exactly the model-visible definition (name, description, input schema, annotations), so a cosmetic server restart does not trigger it and a poisoned description does.
* `DescriptionPins` writes are atomic and thread-safe. A file path makes approvals durable across sessions.

## Failure modes

* Token estimates are heuristics (±20% vs real tokenizers). Use the provider's token-count endpoint when exact numbers matter.
* Lint cannot judge whether a description is *good*; only the tool-eval loop can (checklist step 9).
