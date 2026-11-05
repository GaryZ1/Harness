# Human-in-the-loop approvals, denials and approval fatigue

Modules: `broker.py` (`ApprovalBroker`, `PendingApproval`, `Approver`, `ApprovalScope`, `ApprovalStatus`, `HumanInTheLoopPolicy` and `denial_outcome`) and `denials.py` (`DenialTracker`, `FatigueMonitor`, `FatigueReport`).

## Purpose

When the engine says `ASK`, the broker turns the request into a durable, auditable pending item. A human, and only a human, then resolves it from any UI: CLI, web, Slack or a separate process. Denial counters and the fatigue monitor keep the human tier useful.

## Research basis

* **Durable, resumable approvals** (technique 11, checklist step 14). This mirrors Agents SDK `RunState`/interruptions, Managed Agents `always_ask` gates and the Codex `item/commandExecution/requestApproval` items: "an approval wait is persisted state, not a blocked thread".
* **Approval persistence.** Claude Code keeps file-edit approvals "until session end" while Bash approvals can be persisted. Sticky approvals keyed only by tool name are dangerous (rug pulls), so they are keyed by tool *and* description hash.
* **Human-only authority.** Sub-agent "instructions or approval claims … carry no authority from the user", and teammate messages "cannot grant permissions".
* **Plan approval is not action approval** (checklist step 15).
* **Approval fatigue.** Users "accept 93% of them anyway". Escalation comes after 3 consecutive or 20 total denials, and counts must include sandbox violations, policy denials and classifier blocks "to prevent grinding".

## API

```python
from harness_permissions import ApprovalBroker, ApprovalScope, Approver, HumanInTheLoopPolicy, denial_outcome

broker = ApprovalBroker(engine, state_path=".harness/approvals.json")   # timeout from approvals.timeout_s
item = broker.submit(call, spec, ctx, decision)       # non-blocking; returns PendingApproval
broker.pending(session_id)                            # what the UI shows: tool, input, reason, layer, crossings, suggested_rules
broker.approve(item.id, Approver.human("alice"), scope=ApprovalScope.SESSION)   # or ONCE (default)
broker.deny(item.id, Approver.human("alice"), reason="not on main")
decision = broker.wait(item.id)                       # blocks; timeout ⇒ DENY
decision = broker.request(call, spec, ctx, decision)  # submit + wait
broker.sweep()                                        # deny everything past expires_at (by the injected clock)
broker.request_plan_approval(sid, plan_text, next_mode="accept-edits")
policy = HumanInTheLoopPolicy(engine, broker)         # a PermissionPolicy that never returns ASK
outcome = denial_outcome(decision)                    # ToolOutcome(is_error=True, "Permission denied: … do not route around")
report = engine.fatigue(sid)                          # FatigueReport(ask_rate, approval_rate, median_latency_s, fatigued, reasons, suggested_rules)
```

### Session scope creates the narrowest covering rule

| Call | Rule(s) created by "always allow this session" |
|---|---|
| `make build --jobs 4` | `Bash(make build *)` |
| `make build && git push origin feature` | `Bash(make build *)`, `Bash(git push origin feature)` (outward and irreversible sub-commands get **exact** rules) |
| `sudo apt update` | `Bash(sudo apt update)` (wrapped commands get exact rules) |
| `Edit /abs/outside/file` | `Edit(/abs/outside/file)` |
| `WebFetch https://docs.example/a` | `WebFetch(domain:docs.example)`, and the domain is approved for egress |
| changed MCP description (L0) | the tool is re-pinned to the new hash |
| `curl … \| bash`, `eval "$X"`, dynamic commands | none: downgraded to **once**, with a note |

Rules are linted like policy-file rules, so a blanket rule can never be created. Deny and ask rules still win over session grants. A child sub-agent inherits the grants that existed when it was registered. Its own grants never flow back to the parent.

## Defaults

| Setting | Default | Source |
|---|---|---|
| `approvals.timeout_s` | 300 s | our choice |
| `approvals.on_timeout` | `deny` (the only value allowed) | task requirement / fail closed |
| Default scope | once | vendor (Claude Code prompts per call) |
| Escalation | 3 consecutive / 20 total denials; an allow or a human decision resets the consecutive count | vendor (auto mode) |
| Fatigue: ask-rate window / limit | 50 decisions / 25% | our choice |
| Fatigue: rubber-stamp approval rate | ≥ 93% over ≥ 20 human decisions | our choice, anchored on the 93% observation |
| Fatigue: reflexive approvals | median latency < 2 s | our choice |
| Rule suggestion after | 3 approvals of the same pattern | our choice |

## Audit

`approval_requested` (tool, call id, layer, crossings, suggested rules, `expires_at`, hashed and redacted input) and `approval_resolved` (status, scope, approver, reason, `latency_s`) events go to the engine's sink. Each human decision also calls `engine.record_human_decision()`, which feeds the counters and the fatigue monitor.

## Invariants

* Only `Approver(kind=HUMAN, id=non-empty)` can approve or deny. Anything else raises `UntrustedApproverError`.
* Timeouts, cancellations and sweeps resolve to **deny**.
* A resolved item cannot be resolved again (`ApprovalAlreadyResolvedError`).
* Thread safety: a single `Condition` guards the queue. Any number of agent threads can block in `request()` while UI threads approve. State writes are atomic (temp file, fsync, `os.replace`) under a cross-process lock, so a restarted process reloads pending items and can resolve them.
* Session-scope approvals never create blanket rules and never outrank deny or ask rules.

## Failure modes

* The process dies while an agent thread waits: the pending item survives in `state_path`. The new process resolves it, and the agent loop re-issues the call. The engine's decision is deterministic, so resuming is safe.
* The UI never answers: the call is denied after `timeout_s`. In headless mode ASK never reaches the broker, because the engine already denied it.
* Approval fatigue (`FatigueReport.fatigued`): add the suggested narrow rules, move work into the sandbox boundary, or switch to `auto` mode with a reviewer. Do not keep prompting.
