# Exec policy, hooks and MCP description pinning

Modules: `execpolicy.py` (`prefix_rule`, `host_executable`, `ExecPolicy`), `hooks.py` (`HookRunner`, `CallableHook`, `CommandHook`, `HookInput`, `HookOutput`) and `mcp.py` (`ToolPinStore`, `description_hash`).

## Exec policy (Codex-style prefix rules)

**Research basis.** Codex `prefix_rule(pattern, decision="allow"|"prompt"|"forbidden", justification, match, not_match)`: "the `match`/`not_match` examples are validated at load time as unit tests". `host_executable(name, paths)` "pins which absolute binaries may match basename rules", which prevents the repo-local-shim pitfall. Our checklist makes `justification` mandatory.

```python
from harness_permissions import ExecPolicy, host_executable, prefix_rule

ep = ExecPolicy()
ep.add(prefix_rule(["git", ["status", "diff", "log"]], "allow", "read-only inspection",
                   match=["git status", ["git", "log", "-1"]], not_match=["git push"]))   # examples run now
ep.add(prefix_rule(["git", "push"], "prompt", "network write; must target the session branch"))
ep.add(prefix_rule(["rm", "-rf"], "forbidden", "destructive"))
ep.pin(host_executable("git", ["/usr/bin/git"]))
ep.check("git diff && rm -rf build")   # [("git diff", allow), ("rm -rf build", forbidden)]
```

Semantics:

* A rule matches when argv starts with the pattern. A token may be a list of alternatives. The first token is compared by basename.
* When several rules match, the strictest decision wins: forbidden > prompt > allow.
* A command invoked by path (`./git`, `/opt/git`) matches `allow`/`prompt` only if that path is pinned. `forbidden` always matches by basename.
* In the engine, `forbidden` is a deny at L1, `prompt` an ask at L1b, and `allow` covers one sub-command at L4. Allow does not cover sub-commands with `write_outside_workspace`, `privilege_escalation`, opaque or dangerous-env crossings.
* In the policy file these are `[[exec_policy.prefix_rules]]` and `[[exec_policy.host_executables]]` (see [policy-file.md](policy-file.md)). Plugins may add only `prompt`/`forbidden` rules.

## Hooks (PreToolUse / PermissionRequest)

**Research basis.** Claude Code and Codex share the event names and the `hookSpecificOutput.permissionDecision ∈ {allow, deny, ask, defer}` shape, plus `updatedInput`. Exit code 2 blocks, with stderr as the reason. `onFailure: "block"` fails closed. Default timeouts are 600 s for command/http/mcp_tool, 30 s for prompt and 60 s for agent. `prompt`/`agent` hooks are LLM judgements: "do not treat them as hard guarantees".

```python
from harness_permissions import CallableHook, CommandHook, HookRunner, PolicyEngine

hooks = HookRunner([
    CommandHook("no-main-push", ["python3", "hooks/no_main_push.py"], matcher="Bash"),   # argv list only
    CallableHook("rewrite", lambda hi: {"hookSpecificOutput": {"permissionDecision": "allow",
                                                               "updatedInput": {**hi.tool_input, "timeout": 60}}}),
    CallableHook("llm-judge", judge_fn, kind="prompt"),          # advisory: its allow is ignored
    CallableHook("approver", fn, events=("PermissionRequest",)), # runs when the engine would ASK
], on_failure="block")
engine = PolicyEngine(policy, hooks=hooks)   # default: built from [hooks] + [[hooks.pre_tool_use]] in the policy
```

| Hook result | Engine behaviour |
|---|---|
| `deny` (or command exit 2) | deny at L5, with the hook's reason |
| failure (exception, timeout, non-zero exit, invalid JSON) | deny when `on_failure = "block"` (default), otherwise ignored |
| `ask` | ask |
| `allow` from a non-advisory hook | allow, unless the action is taint-strict, outward (`network_write`, `message_send`, push, `deploy`, `irreversible`) or a new egress domain |
| `updatedInput` | the rewritten call is re-run through L0–L4 first; a rewrite into a denied command is denied |
| `defer` / no output | no opinion |
| `PermissionRequest` deny / allow | turns an ASK into deny / allow, except `human_only` asks (ask rules, taint, rug pull) |

`HookInput` carries `hook_event_name`, `session_id`, `tool_name`, `tool_input`, `tool_use_id`, `cwd`, `permission_mode`, `agent_name`, `depth` and `tainted`, as JSON on stdin for command hooks. Hooks run in a daemon thread with a timeout. A timed-out in-process hook keeps running in the background (Python threads cannot be killed), but its result is ignored and the call is denied.

## MCP description pinning (rug-pull detection)

**Research basis.** Tool poisoning hides instructions in descriptions. A rug pull changes them after approval (Invariant Labs). The checklist says to "hash approved tool descriptions; re-hash on every `tools/list` and block on change" and to key sticky approvals by server, tool and description hash. Annotations from unvetted servers are hints only.

```python
from harness_permissions import ToolPinStore
pins = ToolPinStore(".harness/mcp-pins.json")     # atomic JSON persistence; in-memory if no path
pins.check(spec)                                   # NEW (pinned now, trust on first use) | MATCH | CHANGED
engine = PolicyEngine(policy, pins=pins)
```

The hash covers name, description, `input_schema`, `metadata["annotations"]` and `metadata["mcp_server_version"]`. On `CHANGED`, the engine asks a human at L0 (`human_only`), or denies with `mcp.on_description_change = "block"`, before the tool runs. Approving with session scope, or calling `engine.approve_mcp_description()`, re-pins the tool. Annotations (`readOnlyHint`, `destructiveHint`, `openWorldHint`) are honoured only for servers in `mcp.vetted_servers`. For any other server the MCP defaults apply (`destructiveHint: true`, `openWorldHint: true`), so the tool counts as irreversible and as a taint source.

## Defaults

| Setting | Default | Source |
|---|---|---|
| Exec decision precedence | forbidden > prompt > allow | vendor (Codex) |
| `justification` | required | our checklist |
| Hook timeouts | 600 / 600 / 600 / 30 / 60 s | vendor (Claude Code) |
| `hooks.on_failure` | `block` | vendor option, our default |
| Advisory hook kinds | `prompt`, `agent` | research guidance |
| MCP first sight | pin (TOFU) | our choice |
| `mcp.on_description_change` | `block_and_ask` | research schema |
| Annotation trust | vetted servers only | research schema |

## Invariants

* An exec rule whose examples fail never loads.
* A hook can never weaken L0–L4: deny rules and hard denies run before hooks, and rewrites are re-checked.
* Hook failure is a denial, unless the policy explicitly sets `on_failure = "ignore"`.
* A changed MCP description is never used without a human decision (or a deny).

## Failure modes

* A command hook that hangs is killed by `subprocess.run(timeout=…)` and counted as a failure.
* A corrupted pin file is ignored (treated as empty) and re-pinned on first use. Keep it under protected paths.
