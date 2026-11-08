# Definitions and briefs: `harness_multiagent.definitions` + `harness_multiagent.brief`

## Purpose

A **definition** says *who* a subagent is: name, routing description, system prompt, tools, model role, caps, write access and memory access. A **delegation brief** says *what* one run must do. For a fresh subagent, the brief is the only information it receives from its parent.

## Research basis

* Claude Code stores subagents as Markdown with YAML frontmatter. The fields include `name`, `description`, `tools`, `disallowedTools`, `model`, `maxTurns`, `memory`, `effort` and `isolation`, and scope priority runs managed > CLI > project > user > plugin (reference §8.5). Routing reads `description`. Combined descriptions above ~15k tokens trigger a startup warning.
* A fresh subagent "does **not** inherit conversation history, files already read, or invoked skills", and main-conversation auto memory is not loaded into it (§8.6).
* Vague briefs ("research the semiconductor shortage") produced duplicated work and gaps. Every brief needs "an objective, an output format, guidance on the tools and sources to use, and clear task boundaries" (Anthropic research system). The checklist's brief template (§8.7 item 3) adds why/context, inputs, non-goals, a budget and a write permission.
* Least privilege: "readers and critics get read-only tools" (item 11). Model tiering is configuration: frontier lead and reviewer, cheaper workers (item 15).

## API

```python
from harness_multiagent import AgentRegistry, SubagentDefinition, DelegationBrief, WriteAccess, select_tools

reg = AgentRegistry.from_project(".", user_dir="~/.harness/agents")   # project overrides user
reg.load_dirs([managed_dir, cli_dir], strict=False)                    # highest priority first; first name wins
reg.get("code-explorer"); reg.names(); reg.catalog()                   # catalog = sorted "- name (read-only): description"

d = SubagentDefinition.from_markdown(text, path)       # or .load(path), .from_mapping(dict, prompt)
d.read_only, d.fingerprint()                           # fingerprint feeds workflow replay keys

brief = DelegationBrief(
    objective="Find every caller of parse_config",     # required, one line
    why_context="We are renaming it; keep behaviour identical.",
    inputs=("src/config.py", "error: KeyError 'db'"),
    scope_in=("src/",), out_of_scope=("tests/ (sibling owns)",),
    sources_guidance="grep first, then read only matching files",
    max_tool_calls=15, max_turns=30, output_max_tokens=2000,
)
brief = DelegationBrief.from_mapping(tool_input)       # also accepts the checklist's nested scope/budget/output layout
brief.check_against(d, hard_output_cap=2000)           # raises if the brief would widen privileges
task_message = brief.render()                          # deterministic, self-contained task text
```

`select_tools(definition, available_specs, allow_writes=..., narrow_to=...)` returns the offered tools in the parent's order. Frontmatter accepts snake_case or Claude Code camelCase (`maxTurns`, `disallowedTools`, `ownedFiles`, `canSpawn`, ...). `model:` maps to `model_role`, and `isolation: worktree` maps to `write_access: own_worktree`. Unknown keys are kept in `extra`.

## Defaults

| Field | Our default | Vendor default (Claude Code) | Why ours differs |
|---|---|---|---|
| `tools` | `None` = parent's pool, privilege-filtered | inherit all tools | Same, but mutating tools are stripped unless writer |
| `write_access` | `none` | (all tools) | Least privilege (item 11) |
| `memory` | `read_only` | no memory unless `memory:` set | Subagents may read but never write main memory |
| `context` | `fresh` | fresh (fork opt-in) | Same |
| `can_spawn` | `false` | nesting allowed up to depth 3 | Over-delegation pitfall; nesting is opt-in **and** depth-capped |
| `model_role` | `worker` | `inherit` | Ids come only from the caller's role map; `inherit` is supported |
| `max_turns` | 30 | no limit | Checklist example value |
| `max_tokens` | manager default 200,000 | no limit | "Setting a budget is a good default" |
| `timeout_s` | manager default 600 s | none | Hung calls must end |
| `max_output_tokens` | 16,000 per call | — | `ModelRequest` default |
| brief `output_max_tokens` | 2,000 | ~2k guidance | Condensed returns |

## Invariants

* Names match `^[a-z0-9][a-z0-9_-]{0,63}$`. The description is required, one line, at most 1,024 characters. The system prompt is required.
* `owned_files` ⇔ `write_access: own_files`. `memory: read_write` requires a writer. A tool cannot be both allowed and disallowed.
* A definition never contains a model id, only a role. `ModelRouter` raises for an unknown role.
* A brief may only **narrow** privileges. It can set `write_permission` only to `none` or the definition's own value, and `tools_allowed` must be a subset of the definition's tools. `owned_files` must lie inside the definition's ownership, and `output_max_tokens` must be at most the manager's hard cap.
* With `allow_writes=False`, `select_tools` never returns a tool whose `read_only` is false, whatever the allowlist says.
* `render()` is deterministic: identical briefs give byte-identical task messages, which replay keys and cache-friendly fan-out depend on.

## Failure modes

| Failure | Behaviour |
|---|---|
| Malformed frontmatter / missing fields | `DefinitionError` (with `strict=False`, a `RuntimeWarning` and the file is skipped) |
| Duplicate name in `register` | `DefinitionError`, unless `override=True` |
| Catalog > 15k tokens | `RuntimeWarning` from `load_dirs` |
| Brief widens privileges / bad values | `BriefValidationError` (the spawn tool returns it as an `is_error` result) |
| Unknown agent | `UnknownAgentError` |
