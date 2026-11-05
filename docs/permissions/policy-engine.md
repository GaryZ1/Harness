# Policy engine, modes and rules

Modules: `engine.py` (`PolicyEngine`, `DecisionRecord`, `SessionState`, `suggest_rules`), `modes.py` (`PermissionMode`, `SandboxAttestation`) and `rules.py` (`Rule`, `lint_rule`, matchers).

## Purpose

`PolicyEngine` implements `harness_core.types.PermissionPolicy`. Every tool call goes through the same ordered gates: direct calls, calls from code mode and calls from sub-agents alike. The result is `ALLOW`, `DENY` (returned to the model as a tool result) or `ASK` (resolved by the [approval broker](approvals.md)). `evaluate()` returns the full `DecisionRecord`: the deciding layer, the rule, the `ActionProfile`, taint state, `human_only`, `escalated`, `terminate` and the suggested session rules.

## Research basis

* **Layered pipeline (§7.3 technique 1, checklist step 2).** The order is: deny rules, then sandbox-boundary auto-allow, then narrow allow rules, then hooks/exec policy, then taint/annotation policy, then the reasoning-blind reviewer, then the human. "The order matters, because deny rules must not be overridable by a later allow."
* **Modes (technique 3, checklist step 3).** Claude Code has `default`, `acceptEdits`, `plan`, `dontAsk`, `auto` and `bypassPermissions`. Bypass is "only in a container, VM or sandbox", "cannot be used as root", and Codex `--yolo` is "solely for … externally sandboxed" environments. On entering auto mode, blanket shell, interpreter and package-manager-run rules are dropped.
* **Rules (technique 2, checklist step 4).** Use Claude Code `Tool(pattern)` strings and lint out `Bash(*)`, `python *`, `node *` and `npm run *`.
* **Denials (technique 9, checklist step 8).** A denial is a tool result. Three consecutive or 20 total denials escalate to the human, and headless `claude -p` terminates.
* **Multi-agent trust (technique 14, checklist step 16).** Sub-agents inherit at most the parent's permissions, the harness enforces depth (3) and concurrency (20), and reports carry no authority.

## Pipeline

| Layer id (`DecisionRecord.layer`) | Check | Outcome |
|---|---|---|
| `terminated` | the session already hit the headless denial limit | deny |
| `L0_integrity` | MCP description hash differs from the pinned/approved one | ask (human only), or deny with `on_description_change = "block"` |
| `L1_hard_deny` | `protected_path`, `credential_access`, `private_network`, `catastrophic`, `self_harm` crossings | deny, in every mode including bypass |
| `L1_deny_rule` | a deny rule from any layer matches (fuzzy) | deny |
| `L1_exec_policy` | an exec prefix rule says `forbidden` (deny) or `prompt` (ask) | deny / ask |
| `L1_subagent_limit` | an `Agent`/`Task` call beyond `subagents.max_depth`, or the parent already runs `max_concurrent` children | deny |
| `L1_plan_lock` | the call is outside a locked Plan-Then-Execute plan | deny |
| `L2_mode` | `plan` mode and the call mutates | deny |
| `L1_ask_rule` | an ask rule matches (deny > ask > allow; asks even in bypass) | ask (human only) |
| `bypass` | session in bypass mode (see below) | allow |
| `L3_boundary` | the action stays inside the boundary and is not taint-strict (table below); a sub-agent launch outside auto mode | allow |
| `L4_rule` | **every** sub-command is covered by an allow rule or exec `allow` (in-boundary sub-commands count as covered) and the session is not taint-strict for it | allow |
| `L5_hook` | PreToolUse hooks: deny / ask / allow. A rewrite (`updatedInput`) is re-checked by L0–L4 first. | deny / ask / allow |
| `L6_taint` | tainted session plus a `taint.strict_actions` crossing | ask (human only), or stage-1-only review with `on_strict = "strict_review"` |
| `L7_classifier` | `auto` mode with a reviewer configured | allow / deny with guidance; failure asks |
| `L8_human` | everything else | ask |
| `dont_ask`, `headless`, `subagent` | post-processing: ASK in `dont-ask` mode, in a headless session, or in a sub-agent when `subagents.on_ask = "deny"` | deny |
| `L8_escalation` | a reviewer denial at the 3/20 limit in an interactive session | ask |

### What counts as "inside the boundary" (L3)

| Tool kind | Auto-allowed when there are no crossings and… |
|---|---|
| read / search / internal (`TodoWrite`) / memory view | always (Tier 1) |
| web fetch / search | the destination is allowlisted (`network.allowed_domains` or approved this session) |
| edit / write | mode is `accept-edits` or `auto` (Tier 2) |
| shell | every sub-command is read-only (`ls`, `cat`, `grep`, `git status/diff/log`…), **or** the mode is `auto`/`accept-edits` and `defaults.sandbox = "workspace-write"`, **or** the mode is `accept-edits` and the commands are `mkdir`/`touch`/`mv`/`cp` |
| unknown tool | the harness-defined `ToolSpec.read_only` is true |

Under `workspace-write`, the `code_execution` crossing (for example `python script.py`) is boundary-neutral because the OS sandbox contains it. Opaque execution (`curl … | sh`, `eval "$X"`, inline `python -c`) is never boundary-neutral.

## Modes

| Mode (aliases) | Behaviour | Source |
|---|---|---|
| `plan` (`read-only`) | reads and read-only shell only; mutations denied; ends in a plan approval (`broker.request_plan_approval`) that changes the mode but approves no action | Claude Code / OpenCode / Gemini CLI |
| `dont-ask` (`dontAsk`, `non-interactive`) | never prompts; anything not pre-approved is denied; **default for headless** | Claude Code |
| `default` (`manual`) | ask for anything that is not in-boundary or pre-approved | Claude Code |
| `accept-edits` (`acceptEdits`) | adds in-workspace edits and `mkdir`/`touch`/`mv`/`cp` | Claude Code |
| `auto` | adds the auto-reviewer for crossings; blanket allow rules ignored; **default for interactive** | Claude Code v2.1.283+ |
| `bypass` (`bypassPermissions`, `yolo`) | allows everything except L0/L1 hard denies, deny rules and ask rules | Claude Code / Codex |

Bypass needs **all** of the following, otherwise `BypassRefusedError` is raised by `set_mode()` or by the constructor:

* managed-layer `defaults.allow_bypass = true`;
* an explicit `SandboxAttestation(externally_sandboxed=True)` passed by the host, or `HARNESS_EXTERNAL_SANDBOX=1` read by `SandboxAttestation.detect()`. Container markers (`/.dockerenv`) are recorded as evidence but are never sufficient on their own;
* `running_as_root` false.

`least_permissive()` orders modes `plan < dont-ask < default < accept-edits < auto < bypass`. Modes from settings layers and sub-agent modes are clamped with it.

## Rules

| Rule | Matches |
|---|---|
| `Bash(npm test)` | exactly `npm test` |
| `Bash(npm test *)` or `Bash(npm test:*)` | `npm test` with or without arguments (word boundary) |
| `Bash(ls*)` | raw prefix (`lsof` too) |
| `Read(src/**)`, `Edit(./x.py)`, `Edit(/etc/hosts)`, `Edit(//etc/**)`, `Read(~/.ssh/**)`, `Read(**/.env)` | canonical paths (see [boundary-and-taint.md](boundary-and-taint.md)) |
| `WebFetch(domain:example.com)`, `WebFetch(domain:*.example.com)` | the request host (exact / strict subdomains) |
| `mcp__github`, `mcp__github__*`, `mcp__github__get_*` | MCP tools by server or tool glob |
| `mcp__slack__post(channel:#eng*)` | a top-level input field glob |
| `Agent(Explore)` | sub-agent type |

`Read` rules cover every reading tool and `Edit` rules every editing tool. **Our extension:** `Read`/`Edit` *deny and ask* rules also apply to the paths a shell command reads or writes (`cat .env`, `echo x > .git/hooks/pre-commit`), and `WebFetch(domain:…)` deny rules apply to shell network targets.

### Matching asymmetry

| | deny / ask (fuzzy) | allow (exact) |
|---|---|---|
| Text normalisation | NFKC, casefold, zero-width/bidi characters stripped | none (case-sensitive) |
| Executable | basename (`/bin/rm` = `rm`) | as written; a path-invoked binary (`./git`) only matches when pinned with `host_executable` or spelled out in the rule |
| Wrappers / env | ignored (`sudo rm -rf` matches `rm -rf*`) | wrappers must appear in the rule (`sudo git status` is not `git status`); dangerous env vars (`PATH`, `LD_*`, `GIT_*`, `PYTHONPATH`…) can never be covered |
| Flags | any order; short clusters expanded (`rm -rf` = `rm -r -f` = `rm --recursive --force`) | literal |
| Compound commands | any matching sub-command denies | every sub-command must be covered |
| Wildcards | — | a wildcard rule never covers `write_outside_workspace`, `privilege_escalation`, `degrade_security_posture`, `secret_in_input` or opaque sub-commands; an exact rule can (except opaque, dangerous-env and suspicious-input) |
| Unparsable command | a deny rule's literal text appearing anywhere in the raw command denies | never allowed by a rule |

## Sub-agents

```python
child_state = engine.register_subagent(parent_ctx, RunContext("child", depth=1, agent_name="explorer"), requested_mode="auto")
# child mode = least_permissive(parent mode, requested) ; inherits session rules, approved domains, MCP approvals, plan lock, taint
engine.set_mode("child", "auto")          # ModeError if above the parent's mode
labelled, probe = engine.subagent_finished("child", report_text)   # scanned + labelled "carries no authority"; taints the parent
```

A context with `depth > 0` and `metadata["parent_session_id"]` is registered automatically. A depth > 0 context with an unknown parent is clamped to `default` or stricter. Launching a sub-agent is allowed outside auto mode, because its own calls are checked. In auto mode the reviewer checks it at delegation (tier 3). Depth and concurrency caps are enforced at the `Agent` call (deny) and at registration (`SubagentLimitError`).

## Session grants and other API

| Call | Effect |
|---|---|
| `add_session_rule(sid, "Bash(make build *)")` | session-scoped allow (linted; deny rules still win) |
| `approve_domain(sid, host)` | allows egress to `host` for this session (human approvals only) |
| `approve_mcp_description(sid, spec)` | re-pins a changed MCP description |
| `lock_plan(sid, [rules])` | Plan-Then-Execute: denies calls outside the plan |
| `clear_taint(sid, confirmed_by_user=True, ctx=ctx)` | explicit user action only |
| `record_result(call, spec, ctx, outcome)` | PostToolUse: taint, probe and annotate (see [injection-defense.md](injection-defense.md)) |
| `record_sandbox_violation(ctx, detail)` | counts a sandbox denial into the same counters |
| `must_terminate(sid)`, `fatigue(sid)`, `record_for(sid, call_id)` | loop integration and monitoring |
| `listeners.append(fn(event, record))` | `PermissionRequest` / `PermissionDenied` notifications |

## Audit trail

Each decision emits `Event(type=EventType.PERMISSION, data={"phase": "decision", tool, call_id, behavior, layer, rule, reason, mode, tainted, depth, agent, crossings, human_only, escalated, terminate, input_sha256, input_preview})`. `input_preview` is redacted by `harness_memory.SecretRedactor` and truncated. Other phases are `result`, `mode_change`, `auto_mode_dropped_rules`, `session_rule_added`, `egress_domain_approved`, `mcp_description_approved`, `mcp_description_changed`, `plan_locked`, `taint_cleared`, `subagent_start`, `subagent_stop`, `sandbox_violation`, `approval_requested` and `approval_resolved`. The sink assigns `seq`. `MemoryEventSink` is provided for tests. A failing sink never breaks a decision.

## Defaults

| Setting | Default | Source |
|---|---|---|
| Interactive / headless mode | `auto` / `dont-ask` | research schema (Claude Code default) |
| Escalation | 3 consecutive / 20 total | vendor (auto mode) |
| Headless at limit | terminate | vendor (`claude -p`) |
| Sub-agent depth / concurrency | 3 / 20 | vendor (Claude Code env vars) |
| Sub-agent ASK | escalate to the broker | our choice |
| Ask rules in bypass | still ask | our choice |
| Sub-agent launch outside auto | allow (calls checked individually) | vendor (Claude Code) |

## Invariants

* Deny > ask > allow. Nothing after L1 can allow an action that L1 denies: no session grant, hook, reviewer or bypass.
* Every sub-command of a compound command must be allowed. An unparsable command is never rule-allowed.
* Allow matching is exact. Deny matching is fuzzy.
* A hook's rewritten input goes back through L0–L4 before anything else.
* Taint only adds checks. A new egress domain in a tainted session needs a human (`human_only`).
* Every denial carries `DENIAL_GUIDANCE` and counts toward the escalation counters.
* Thread safety: session state is guarded by per-session locks, and the session table and record cache by an engine lock. One engine can serve many concurrent sessions.

## Failure modes

| Failure | Behaviour |
|---|---|
| Reviewer raises or answers garbage | `ClassifierError`, and the engine asks (interactive) or denies (headless). It never allows. |
| Hook crashes, times out, or prints invalid JSON | deny (`hooks.on_failure = "block"`) |
| Shell command unparsable | `opaque_execution` crossing; the reviewer or a human decides |
| Unknown tool without a spec | `unknown_effects` crossing; asks |
| Audit sink raises | swallowed; the decision is still returned |
