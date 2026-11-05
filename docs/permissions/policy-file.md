# Policy file reference (`harness-policy.toml`)

## Purpose

One declarative file per settings layer configures every gate of the permission pipeline. The schema is the one the research reference proposes for our harness ([`reports/SOTA agentic harness reference.md` §7.7](../../reports/SOTA%20agentic%20harness%20reference.md), "Example of our own harness policy file": `[defaults]`, `[denials]`, `[rules]`, `[classifier]`, `[taint]`, `[mcp]`, `[hooks]`). This package adds `[workspace]`, `[network]`, `[exec_policy]`, `[subagents]`, `[approvals]`, `[fatigue]` and `[high_assurance]`, which you need to configure the rest of the pipeline. Every key is declared once in `harness_permissions.config.SCHEMA`. A test checks that this page documents every key.

Module: `config.py`. Entry points: `load_policy_file(path, layer)`, `parse_policy(mapping, layer)`, `resolve(documents)`, `load_layers(managed=…, project=…, user=…, plugins=…, session=…)`, `discover_policy_files(root)` and `default_policy()`.

## Formats

| Suffix | Parser | Notes |
|---|---|---|
| `.toml` | `tomllib` (Python 3.11+), else the optional `tomli` package | If neither is installed, loading raises `PolicyLoadError`. The message tells you to `pip install tomli` or to write the same schema as YAML or JSON. |
| `.yaml` / `.yml` | `yaml.safe_load` (PyYAML) | Same keys, same nesting. |
| `.json` | `json` | Same keys, same nesting. |

Relative `workspace.roots` and `workspace.read_roots` resolve against the directory that holds the policy file.

## Layers and precedence

Precedence, from most to least authority, follows checklist step 1: **managed/admin > session flags > project > user > plugins**. "A deny at any layer wins, and lower layers can only narrow."

| Layer | Conventional location (`discover_policy_files`) | May set |
|---|---|---|
| `managed` | `$HARNESS_MANAGED_POLICY`, else `/etc/harness/harness-policy.{toml,yaml,yml,json}` | everything; the only layer that may set `allow_bypass` / `allow_managed_rules_only` to true or turn off a default-on protection |
| `session` | in-memory flags (`load_layers(session={...})`) | everything except managed-only keys |
| `project` | `<repo>/harness-policy.*`, `<repo>/.harness/harness-policy.*` | everything except managed-only keys |
| `user` | `~/.harness/harness-policy.*` | everything except managed-only keys |
| `plugin` | passed explicitly | restrictions only: `rules.deny`, `rules.ask`, non-allow `exec_policy.prefix_rules`, `workspace.protected_paths`, `workspace.credential_paths`, `taint.*`, `classifier.block_rules`, `classifier.block_groups`, `version` |

Every loaded policy file is added to the workspace's **protected paths** automatically, so the agent cannot edit its own policy. Plugins cannot either ("letting the agent or plugins edit their own policy" is a §7.8 pitfall).

### Rules merging

* `rules.deny` and `rules.ask` are the union of all layers. Because matching is deny > ask > allow, a managed deny can never be overridden by a project, user or session allow, nor by an "always allow" a human grants during the session.
* `rules.allow` is the union of all layers **except plugins**. When managed sets `defaults.allow_managed_rules_only = true`, only managed allows count.
* Every allow rule is linted at load time. Blanket shell (`Bash`, `Bash(*)`), interpreter or wrapper wildcards (`python *`, `node *`, `bash *`, `sudo *`, `env *`, `xargs *`), package-runner wildcards (`npm run *`, `npx *`, `uv run *`, `yarn *`…), a wildcard in the executable position, `*`/`mcp__*` tool globs, and blanket egress (`WebFetch`, `WebFetch(domain:*)`) are **errors**, not warnings.

### Scalar merging

The "Merge across layers" column in the tables below gives each key's strategy. Numeric limits take the minimum, enumerations the strictest value, and modes the least permissive. Booleans whose `true` value is the strict one combine with OR, and only managed may turn a default-on protection off. Keys that widen permissions (`network.allowed_domains`, `classifier.allow_exceptions`, `mcp.vetted_servers`, `workspace.roots`) come only from the most authoritative layer that sets them, so a lower layer cannot add to them.

## Full example (every section)

```toml
version = 1

[defaults]
interactive_mode = "auto"            # plan | dont-ask | default | accept-edits | auto | bypass
headless_mode    = "dont-ask"
sandbox          = "workspace-write" # none | read-only | workspace-write | external
deny_wins        = true              # must be true

[denials]
escalate_consecutive = 3
escalate_total       = 20
headless_on_limit    = "terminate"

[rules]
allow = ["Read(*)", "Grep(*)", "Glob(*)", "Bash(git status*)", "Bash(npm test*)", "mcp__github__get_*"]
ask   = ["Bash(terraform apply*)"]
deny  = ["Bash(git push --force*)", "Edit(.git/hooks/**)", "Read(~/.ssh/**)", "Read(**/.env)"]

[classifier]
enabled          = true
stages           = ["single_token_block_biased", "cot_on_flag"]
inputs           = ["user_messages", "tool_calls"]   # tool_results / assistant text are rejected
environment      = ["current_git_repo"]
block_groups     = ["destroy_or_exfiltrate", "degrade_security_posture", "cross_trust_boundary", "bypass_review_or_affect_others"]
block_rules      = ["Never modify files under infra/prod/"]
allow_exceptions = ["Pushing to the session branch"]

[taint]
sources        = ["web_fetch", "browser", "mcp:openWorld", "email", "out_of_repo_files", "subagent_report"]
strict_actions = ["network_write", "message_send", "git_push_non_session_branch", "new_egress_domain", "memory_write"]
on_strict      = "ask"               # or "strict_review"

[mcp]
pin_versions           = true
hash_tool_descriptions = true
on_description_change  = "block_and_ask"   # block | block_and_ask | warn
trust_annotations_from = ["vetted"]
vetted_servers         = ["github"]

[hooks]
on_failure = "block"
timeouts_s = { command = 600, http = 600, mcp_tool = 600, prompt = 30, agent = 60 }

[[hooks.pre_tool_use]]
name    = "no-main-push"
command = ["python3", "hooks/no_main_push.py"]   # argv list, never a shell string
matcher = "Bash"

[workspace]
roots            = [".."]            # relative to this file
read_roots       = []
temp_roots       = ["/tmp"]
protected_paths  = ["deploy/secrets/**"]
credential_paths = ["~/.config/acme/token"]
session_branch   = "claude/session-123"

[network]
allowed_domains = ["github.com", "*.github.com", "pypi.org", "files.pythonhosted.org"]
block_private   = true

[[exec_policy.prefix_rules]]
pattern       = ["git", ["status", "diff", "log"]]
decision      = "allow"
justification = "read-only repository inspection"
match         = ["git status", "git log --oneline"]
not_match     = ["git push"]

[[exec_policy.prefix_rules]]
pattern       = ["git", "push"]
decision      = "prompt"
justification = "network write; must target the session branch"

[[exec_policy.host_executables]]
name  = "git"
paths = ["/usr/bin/git"]

[subagents]
max_depth      = 3
max_concurrent = 20
on_ask         = "escalate"

[approvals]
timeout_s  = 300
on_timeout = "deny"

[fatigue]
window              = 50
max_ask_rate        = 0.25
rubber_stamp_rate   = 0.93
min_human_decisions = 20
fast_approval_s     = 2.0
suggest_after       = 3

[high_assurance]
plan_then_execute = false
```

The same document in YAML:

```yaml
defaults: {interactive_mode: auto, headless_mode: dont-ask}
rules:
  allow: ["Bash(npm test*)"]
  deny: ["Read(**/.env)"]
network: {allowed_domains: ["github.com"]}
```

## Key reference

`version` is the only top-level key. Every other key belongs to a section. "Default source" marks whether the default comes from the research schema example, a vendor default (Claude Code, Codex or MCP), or is our choice.

### Top level

| Key | Type / values | Default | Merge across layers | Default source | Meaning |
|---|---|---|---|---|---|
| `version` | int ≥ 1 | `1` | most authoritative layer that sets it (plugin-settable) | our choice | Schema version. Only 1 exists. |

### `[defaults]`

| Key | Type / values | Default | Merge across layers | Default source | Meaning |
|---|---|---|---|---|---|
| `defaults.interactive_mode` | mode | `"auto"` | least permissive mode | research schema | Mode for interactive sessions: plan, dont-ask, default, accept-edits, auto or bypass. |
| `defaults.headless_mode` | mode | `"dont-ask"` | least permissive mode | research schema | Mode for headless runs (no human attached). |
| `defaults.sandbox` | one of `none`, `read-only`, `workspace-write`, `external` | `"workspace-write"` | most authoritative layer that sets it | research schema | OS sandbox the host runs tools in. Mutating shell commands are boundary-auto-allowed only under workspace-write. |
| `defaults.deny_wins` | bool (must be true) | `true` | most authoritative layer that sets it | research schema | Deny > ask > allow. Must be true. |
| `defaults.allow_bypass` | bool | `false` | managed layer only | our choice | Admin consent for bypass mode (honoured only from the managed layer). |
| `defaults.allow_managed_rules_only` | bool | `false` | managed layer only | our choice | Ignore allow rules from every layer except managed. |

### `[denials]`

| Key | Type / values | Default | Merge across layers | Default source | Meaning |
|---|---|---|---|---|---|
| `denials.escalate_consecutive` | int ≥ 1 | `3` | minimum | vendor default | Consecutive denials before escalating to the human (auto mode: 3). |
| `denials.escalate_total` | int ≥ 1 | `20` | minimum | vendor default | Total denials before escalating to the human (auto mode: 20). |
| `denials.headless_on_limit` | one of `terminate`, `deny` | `"terminate"` | strictest value | research schema | Headless behaviour at the denial limit: terminate the run (claude -p) or keep denying. |

### `[rules]`

| Key | Type / values | Default | Merge across layers | Default source | Meaning |
|---|---|---|---|---|---|
| `rules.allow` | list[str] | `[]` | see Rules merging | research schema | Allow rules (Tool(pattern)). Blanket/interpreter/package-runner wildcards are rejected. |
| `rules.ask` | list[str] | `[]` | see Rules merging (plugin-settable) | our choice | Ask rules: always require a human, even in bypass mode. |
| `rules.deny` | list[str] | `[]` | see Rules merging (plugin-settable) | research schema | Deny rules. Union of every layer; matched fuzzily. |

### `[classifier]`

| Key | Type / values | Default | Merge across layers | Default source | Meaning |
|---|---|---|---|---|---|
| `classifier.enabled` | bool | `true` | most authoritative layer that sets it | our choice | Use the auto-reviewer in auto mode when one is configured. |
| `classifier.stages` | list (first must be `single_token_block_biased`) | `["single_token_block_biased", "cot_on_flag"]` | most authoritative layer that sets it | research schema | Reviewer stages. Stage 2 runs only on a stage-1 flag. |
| `classifier.inputs` | list: `user_messages`, `tool_calls` | `["user_messages", "tool_calls"]` | most authoritative layer that sets it | research schema | Reviewer inputs. tool_results / assistant text are rejected. |
| `classifier.environment` | list[str] | `["current_git_repo"]` | most authoritative layer that sets it | research schema | Environment slot: the trusted boundary described to the reviewer. |
| `classifier.block_groups` | list of `destroy_or_exfiltrate`, `degrade_security_posture`, `cross_trust_boundary`, `bypass_review_or_affect_others` | `["destroy_or_exfiltrate", "degrade_security_posture", "cross_trust_boundary", "bypass_review_or_affect_others"]` | union (base: managed list, else default) (plugin-settable) | research schema | Default block-rule groups enabled. |
| `classifier.block_rules` | list[str] | `[]` | union (base: managed list, else default) (plugin-settable) | our choice | Extra block rules (group 'custom'). |
| `classifier.allow_exceptions` | list[str] | `[]` | most authoritative layer that sets it | vendor default | Allow-exceptions slot (widening, so only the highest layer that sets it counts). |

### `[taint]`

| Key | Type / values | Default | Merge across layers | Default source | Meaning |
|---|---|---|---|---|---|
| `taint.sources` | list of `web_fetch`, `browser`, `mcp:openWorld`, `email`, `out_of_repo_files`, `subagent_report`, `memory_read` | `["web_fetch", "browser", "mcp:openWorld", "email", "out_of_repo_files", "subagent_report"]` | union (base: managed list, else default) (plugin-settable) | research schema | Tool categories whose output taints the session. |
| `taint.strict_actions` | list of `read_outside_workspace`, `write_outside_workspace`, `network_egress`, `new_egress_domain`, `external_tool`, `subagent_spawn`, `credential_access`, `environment_dump`, `secret_in_input`, `network_write`, `message_send`, `irreversible`, `catastrophic`, `private_network`, `git_push_session_branch`, `git_push_non_session_branch`, `deploy`, `memory_write`, `protected_path`, `privilege_escalation`, `degrade_security_posture`, `self_harm`, `opaque_execution`, `code_execution`, `dangerous_env`, `unknown_effects`, `suspicious_input` | `["network_write", "message_send", "git_push_non_session_branch", "new_egress_domain", "memory_write", "deploy", "irreversible"]` | union (base: managed list, else default) (plugin-settable) | research schema | Crossings that need a human (or strict review) while tainted. |
| `taint.on_strict` | one of `ask`, `strict_review` | `"ask"` | strictest value | our choice | Tainted + strict action: always ask, or allow the stage-1-only reviewer to approve. |

### `[mcp]`

| Key | Type / values | Default | Merge across layers | Default source | Meaning |
|---|---|---|---|---|---|
| `mcp.pin_versions` | bool | `true` | true if any layer; only managed may turn a default-on flag off | research schema | Include the MCP server version in the pinned description hash. |
| `mcp.hash_tool_descriptions` | bool | `true` | true if any layer; only managed may turn a default-on flag off | research schema | Hash tool descriptions and detect changes (rug pulls). |
| `mcp.on_description_change` | one of `block_and_ask`, `block`, `warn` | `"block_and_ask"` | strictest value | research schema | What a changed description does: ask a human, deny, or only warn. |
| `mcp.trust_annotations_from` | list of `vetted`, `none` | `["vetted"]` | most authoritative layer that sets it | research schema | Whose MCP annotations are trusted: vetted servers, or none. |
| `mcp.vetted_servers` | list[str] | `[]` | most authoritative layer that sets it | our choice | MCP server names whose annotations are trusted. |

### `[hooks]`

| Key | Type / values | Default | Merge across layers | Default source | Meaning |
|---|---|---|---|---|---|
| `hooks.on_failure` | one of `block`, `ignore` | `"block"` | strictest value | research schema | Hook failure/timeout handling. block = fail closed. |
| `hooks.timeouts_s` | table kind → seconds | `{command = 600, http = 600, mcp_tool = 600, prompt = 30, agent = 60, callable = 600}` | most authoritative layer that sets it | vendor default | Per-kind hook timeouts in seconds. |
| `hooks.pre_tool_use` | enum | `[]` | concatenated | our choice | Command hooks: [[hooks.pre_tool_use]] name, command (argv list), matcher (regex), timeout_s, events. |

### `[workspace]`

| Key | Type / values | Default | Merge across layers | Default source | Meaning |
|---|---|---|---|---|---|
| `workspace.roots` | list[str] | `[]` | most authoritative layer that sets it | our choice | Writable roots (empty = enclosing git repo of the engine's cwd). Relative paths resolve against the policy file's directory. |
| `workspace.read_roots` | list[str] | `[]` | most authoritative layer that sets it | our choice | Extra read-only roots inside the boundary. |
| `workspace.temp_roots` | list[str] | `["/tmp"]` | most authoritative layer that sets it | vendor default | Temp dirs treated as inside the boundary (Codex workspace-write includes /tmp). |
| `workspace.protected_paths` | list[str] | `[]` | union (base: managed list, else default) (plugin-settable) | our choice | Extra protected write paths (added to the built-ins). |
| `workspace.credential_paths` | list[str] | `[]` | union (base: managed list, else default) (plugin-settable) | our choice | Extra credential paths (read-denied, added to the built-ins). |
| `workspace.session_branch` | string | `null` | most authoritative layer that sets it | our choice | Branch the session may push to without the git_push_non_session_branch crossing. |

### `[network]`

| Key | Type / values | Default | Merge across layers | Default source | Meaning |
|---|---|---|---|---|---|
| `network.allowed_domains` | list[str] | `[]` | most authoritative layer that sets it | our choice | Egress allowlist (example.com exact, *.example.com subdomains). |
| `network.block_private` | bool | `true` | true if any layer; only managed may turn a default-on flag off | vendor default | Hard-deny loopback/private/link-local/metadata destinations. |

### `[exec_policy]`

| Key | Type / values | Default | Merge across layers | Default source | Meaning |
|---|---|---|---|---|---|
| `exec_policy.prefix_rules` | enum | `[]` | concatenated (plugin-settable) | vendor default | Codex-style [[exec_policy.prefix_rules]]: pattern, decision, justification, match, not_match. |
| `exec_policy.host_executables` | enum | `[]` | concatenated | vendor default | [[exec_policy.host_executables]]: name, paths (pinned absolute binaries). |

### `[subagents]`

| Key | Type / values | Default | Merge across layers | Default source | Meaning |
|---|---|---|---|---|---|
| `subagents.max_depth` | int ≥ 1 | `3` | minimum | vendor default | Maximum sub-agent nesting depth (CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH). |
| `subagents.max_concurrent` | int ≥ 1 | `20` | minimum | vendor default | Maximum concurrently running sub-agents (CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS). |
| `subagents.on_ask` | one of `escalate`, `deny` | `"escalate"` | strictest value | our choice | A sub-agent call that needs a human: escalate to the broker or deny. |

### `[approvals]`

| Key | Type / values | Default | Merge across layers | Default source | Meaning |
|---|---|---|---|---|---|
| `approvals.timeout_s` | number > 0 | `300` | minimum | our choice | Seconds a pending approval waits before it is denied. |
| `approvals.on_timeout` | one of `deny` | `"deny"` | most authoritative layer that sets it | our choice | Timeout outcome. Only deny is allowed. |

### `[fatigue]`

| Key | Type / values | Default | Merge across layers | Default source | Meaning |
|---|---|---|---|---|---|
| `fatigue.window` | int ≥ 1 | `50` | most authoritative layer that sets it | our choice | Decisions in the ask-rate window. |
| `fatigue.max_ask_rate` | 0–1 | `0.25` | most authoritative layer that sets it | our choice | Ask rate above which fatigue is reported. |
| `fatigue.rubber_stamp_rate` | 0–1 | `0.93` | most authoritative layer that sets it | our choice | Human approval rate treated as rubber-stamping (93% observed by Anthropic). |
| `fatigue.min_human_decisions` | int ≥ 1 | `20` | most authoritative layer that sets it | our choice | Human decisions needed before approval-rate/latency checks apply. |
| `fatigue.fast_approval_s` | number > 0 | `2.0` | most authoritative layer that sets it | our choice | Median approval latency below which approvals look reflexive. |
| `fatigue.suggest_after` | int ≥ 1 | `3` | most authoritative layer that sets it | our choice | Approvals of the same pattern before a narrow allow rule is suggested. |

### `[high_assurance]`

| Key | Type / values | Default | Merge across layers | Default source | Meaning |
|---|---|---|---|---|---|
| `high_assurance.plan_then_execute` | bool | `false` | true if any layer; only managed may turn a default-on flag off | our choice | Once tainted, deny calls outside the locked plan (deny all mutations if none is locked). |

## Validation and failure modes

| Problem | Behaviour |
|---|---|
| Unknown section or key, wrong type, bad enum value | `PolicyValidationError`. `problems` lists **every** issue, each prefixed with its file path and dotted key. |
| `defaults.deny_wins = false` | rejected: deny always wins |
| Blanket, interpreter or package-runner allow rule | rejected (lint) |
| `classifier.inputs` containing `tool_results` or assistant text | rejected: the reviewer is reasoning-blind by design |
| Exec prefix rule whose `match`/`not_match` examples fail, or with no `justification` | rejected (the examples run as unit tests at load) |
| `allow_bypass` / `allow_managed_rules_only` set to true outside the managed layer | rejected |
| A plugin document setting anything except restrictions | rejected |
| Unreadable file, invalid TOML/YAML/JSON, unsupported suffix, no TOML parser | `PolicyLoadError` |
| `interactive_mode`/`headless_mode = "bypass"` without managed `allow_bypass` and an explicit `SandboxAttestation` | the `PolicyEngine` constructor raises `BypassRefusedError` |

## Invariants

* A deny at any layer wins. No lower layer, session approval, hook or reviewer can turn it into an allow.
* Plugins can only add restrictions.
* Policy files that were loaded are protected write paths.
* Defaults with security weight (`block_private`, description hashing, taint sources) can be relaxed only by the managed layer.
