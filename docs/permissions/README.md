# Permissions, human-in-the-loop and security (`harness_permissions`)

`harness_permissions` is a production Python implementation of the permission and injection-defence design in [`reports/SOTA agentic harness reference.md` §7](../../reports/SOTA%20agentic%20harness%20reference.md). It also draws on the permissions and security parts of [`research_notes/.../tools_sandboxing_safety.md`](../../research_notes/SOTA%20agentic%20harness%20build%20plan/tools_sandboxing_safety.md) and the build-plan section "Permissions and injection: auto-allow inside the boundary, classify every crossing". `PolicyEngine` implements the shared `harness_core.types.PermissionPolicy` protocol: `check(call, spec, ctx) -> PermissionDecision(ALLOW | DENY | ASK, reason, updated_input)`.

| Component | What it does | Module(s) | Doc |
|---|---|---|---|
| **Policy engine** | Runs the layered L0–L8 pipeline. Owns modes, rules, sub-agent limits and denial escalation. | `engine.py`, `modes.py`, `rules.py` | [policy-engine.md](policy-engine.md) |
| **Shell analyzer** | Splits compound commands into every simple command that would run. | `shell.py` | [shell-analysis.md](shell-analysis.md) |
| **Boundary and taint** | Canonicalises paths and network targets, classifies boundary crossings, tracks session taint. | `boundary.py`, `paths.py`, `network.py`, `taint.py` | [boundary-and-taint.md](boundary-and-taint.md) |
| **Injection defence** | Heuristic and optional model probes on untrusted tool output. Annotates, never drops. | `injection.py` | [injection-defense.md](injection-defense.md) |
| **Auto-reviewer** | Two-stage, reasoning-blind transcript classifier for `auto` mode. | `reviewer.py` | [auto-reviewer.md](auto-reviewer.md) |
| **Approval broker** | Human-in-the-loop queue: once or session approvals, timeouts that deny, approval-fatigue tracking. | `broker.py`, `denials.py` | [approvals.md](approvals.md) |
| **Exec policy, hooks, MCP pins** | Tested prefix rules with pinned binaries, fail-closed PreToolUse hooks, rug-pull detection. | `execpolicy.py`, `hooks.py`, `mcp.py` | [exec-policy-hooks-mcp.md](exec-policy-hooks-mcp.md) |
| **Policy file** | `harness-policy.toml` (or YAML/JSON) loader, validator and layer merge. | `config.py` | [policy-file.md](policy-file.md) |
| Audit | Records every decision and approval as an `EventType.PERMISSION` event, with inputs hashed and redacted. | `audit.py` | [policy-engine.md](policy-engine.md#audit-trail) |

## Why this shape

Each bullet below is a research finding that this package enforces in code rather than in prompts. "Policy in CLAUDE.md is context, not enforced configuration."

1. **Approval fatigue makes per-action prompts a weak control.** Users "accept 93% of them anyway", and sandboxing removed 84% of prompts. The engine therefore auto-allows everything that stays inside the workspace boundary and spends human attention only on crossings. `FatigueMonitor` watches for rubber-stamping.
2. **Layered pipeline, cheap and deterministic first, human last** (Claude Code, Codex and Gemini CLI all converged on this). Hard denies run before any allow, so "deny rules must not be overridable by a later allow". A managed deny beats every lower layer.
3. **Prefix rules cannot see through `bash -c`, `eval`, pipes or substitutions.** Every sub-command is parsed and must be allowed on its own, so `Bash(git status*)` never matches `git status && rm -rf /`. An unparsable command falls through to the reviewer or a human.
4. **Never blanket-allow interpreters or package runners.** `Bash(*)`, `python *` and `npm run *` are rejected at load time. Auto mode drops them on entry, as Claude Code does.
5. **Basename-only exec rules are a pitfall.** Allow rules never match a repo-local `./git` shim unless the binary is pinned with `host_executable`.
6. **Prompt injection is unsolved at the model level.** Probes alone moved Opus 4.5 only from 17.6% to 16.7% ASR, and catching "95%" of attacks "is a failing grade". The design is therefore architectural: taint the session on untrusted input, then require a human for exfiltration-capable actions (lethal trifecta). Probes annotate content and can only tighten policy.
7. **The reviewer must be reasoning-blind.** It sees user messages and tool calls only, stripped in code, so injected text "cannot argue with the reviewer". It is not a boundary (17% FNR on real overeager actions), so it sits behind the hard-deny layer.
8. **Denials are tool results, and they are counted.** Every denial carries "find a safer way… do not route around". Three consecutive or 20 total denials escalate to the human, and headless runs terminate.
9. **Sub-agents and teammates carry no user authority.** Children inherit at most the parent's mode and grants. Their reports are scanned and labelled, and only an identified human can approve.
10. **Rug pulls are real.** MCP tool descriptions are hashed on first sight, and any change asks a human before the tool runs again.

## Architecture

```text
 tool_use ──▶ PolicyEngine.check(call, spec, ctx)
               │ normalize: classify() ── shell.analyze() (every sub-command) ── paths.canonicalize() ── network.host_from_url()
               │            ActionProfile{kind, crossings, paths, domains, segments, mcp}
               ▼
   L0 integrity ── MCP description hash changed? ───────────────────────────────▶ ASK (human only) / DENY
   L1 hard deny ── protected/credential paths, private net, catastrophic, self-harm,
                   deny rules (all layers, fuzzy), exec "forbidden", sub-agent caps, plan lock ─▶ DENY
   L2 mode ─────── plan + mutation ─────────────────────────────────────────────▶ DENY
   L1b ask ─────── ask rules, exec "prompt" ────────────────────────────────────▶ ASK
   bypass ──────── attested external sandbox + managed consent ────────────────▶ ALLOW
   L3 boundary ─── in-workspace, allowlisted egress, not taint-strict ─────────▶ ALLOW   (the "84%")
   L4 rules ────── every sub-command covered by a narrow rule / exec allow ────▶ ALLOW
   L5 hooks ────── PreToolUse deny/ask/allow, updatedInput re-checked, fail closed
   L6 taint ────── tainted + strict action (network_write, push off-branch, new domain…) ─▶ ASK (human)
   L7 reviewer ─── auto mode: stage 1 (block-biased) → stage 2 CoT on flag ────▶ ALLOW / DENY
   L8 human ─────────────────────────────────────────────────────────────────────▶ ASK
               │ post: dont-ask/headless ASK→DENY · denial counters (3 / 20) → escalate or terminate
               │       PermissionRequest hooks · audit Event(PERMISSION)
               ▼
 ASK ──▶ ApprovalBroker (durable queue, human-only approvers, once|session scope, timeout ⇒ deny)
 DENY ─▶ denial_outcome() → tool_result "Permission denied: … do not route around"
 after execution ─▶ engine.record_result(): taint sources → InjectionProbe → annotate output → taint session
```

## Quick start

```python
from harness_core.types import RunContext, ToolCall
from harness_permissions import (ApprovalBroker, Approver, HumanInTheLoopPolicy, PolicyEngine,
                                 TwoStageClassifier, denial_outcome, discover_policy_files, load_layers)

found = discover_policy_files(".")
policy = load_layers(managed=found[Layer.MANAGED], project=found[Layer.PROJECT], user=found[Layer.USER])
reviewer = TwoStageClassifier(model_client, model=os.environ["HARNESS_REVIEWER_MODEL"])  # model id from config
engine = PolicyEngine(policy, cwd=".", reviewer=reviewer, sink=event_log)
broker = ApprovalBroker(engine, state_path=".harness/approvals.json")

ctx = RunContext(session_id="s_1", cwd=".")
decision = engine.check(ToolCall("t1", "Bash", {"command": "npm test && git push origin feature"}), None, ctx)
if decision.behavior.value == "ask":
    decision = broker.request(call, spec, ctx, decision)      # UI calls broker.approve(id, Approver.human("alice"))
if not decision.allowed:
    result = denial_outcome(decision)                          # return to the model as the tool result
else:
    outcome = executor.execute(call, ctx)
    outcome = engine.record_result(call, spec, ctx, outcome)   # taint + injection probe (annotates)

# or, for a loop that does not understand ASK:
policy_for_loop = HumanInTheLoopPolicy(engine, broker)
```

Run the tests with `python3 -m pytest -p no:cacheprovider tests/permissions`.

## Scope notes

The OS sandbox and egress proxy (DNS-rebinding protection, credential injection), checkpoints and rewind, and MCP transport auth (RFC 8707, RFC 9207, `Origin` checks) belong to sibling packages. This package *classifies* those concerns and consumes them through `defaults.sandbox`, `SandboxAttestation` and `record_sandbox_violation()`. It does not implement them.
