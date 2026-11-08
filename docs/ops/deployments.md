# Rainbow deployments and per-model feature flags

## Purpose

This component ships new harness builds without breaking agents that are already running. Old and new versions run side by
side, new sessions shift over gradually, every session finishes on the version it started on, and any build can be drained or
rolled back.

## Research basis

* **Rainbow deployments** (Anthropic multi-agent research system, 2025-06-13; reference §10.3; checklist item 17). Traffic shifts gradually while both versions run, so in-flight agents are not broken. Version prompts, tools and workflows together, stamp the version and its SHA on every root span, pin in-flight sessions, and "never hot-patch a running session's prompt or tools".
* **Cache keys**: model, effort and tool set are part of the prompt-cache key, and mid-session changes are a listed pitfall. Per-session flags are therefore frozen at session start.
* **Per-model profiles** (build plan "Co-design"): the harness loads a per-model profile rather than one universal configuration, for example edit-tool format or todo tools on or off.

## API

```python
from harness_ops import DeploymentRegistry, prompts_tools_sha, FeatureFlags

reg = DeploymentRegistry("state/deploy.db")
reg.register("v41", prompts_tools_sha(SYSTEM_V41, tool_defs_v41), weight=100)
reg.register("v42", prompts_tools_sha(SYSTEM_V42, tool_defs_v42))

reg.shift("v42", 5)            # 5% of NEW sessions; others scaled proportionally
pin = reg.pin(session_id)      # idempotent: returns the existing pin for a known session
version = reg.version(pin.version_id)
# ... root span: inst.invoke_agent(..., version=version) stamps harness.version + harness.prompts_tools_sha
reg.finish(session_id)         # at session end

reg.shift("v42", 50); reg.shift("v42", 100)   # v41 becomes "draining"; its sessions finish there
if reg.is_drained("v41"):
    reg.retire("v41")
reg.rollback("v42", "v41")     # bad build: new traffic back to v41; in-flight v42 sessions finish on v42
reg.rollback("v42", "v41", abort_inflight=True)   # or end them: their next pin() raises SessionAbortedError
```

* `route(routing_key)` is a pure function of the key and the current weights: `sha256(salt:key)` mapped to 10,000 buckets. The **newest** version owns the low end of the range, so ramping it up from 10% to 50% only *adds* sessions to it. Pass `routing_key=user_id` to `pin` for user-sticky assignment.
* `set_weights({...})` replaces the split (weights are relative). Versions left out become `draining`. Retired and rolled-back versions cannot take traffic.
* `inflight(version)`, `inflight_sessions(version)` and `pinned_version(session_id)` are for inspection. `drain(version)` refuses to drain the only version taking traffic.

### Feature flags

```yaml
flags:
  todo_tool:
    default: true
    models: {"test-model-new*": false}    # first matching glob wins; exact ids beat globs
  parallel_tools:
    default: true
    rollout_percent: 25                   # deterministic per routing key
  apply_patch:
    default: true
    versions: ["v42"]                     # off on other harness versions
```

```python
flags = FeatureFlags.load("flags.yaml")
snap = flags.snapshot(model, routing_key=session_id, version=pin.version_id)   # resolve once at session start
reg.pin(session_id, snapshot=snap.to_dict())     # frozen with the pin (only stored when the pin is created)
if snap.enabled("todo_tool"): ...
```

## Defaults

| Setting | Default | Anchor |
|---|---|---|
| Hash buckets / salt | 10,000 / `"harness-rainbow"` | our choice |
| New version weight | 0 (opt-in ramp) | our choice |
| In-flight sessions on rollback | keep running on their version | vendor practice (rainbow: never break in-flight agents) |
| Flag resolution | exact model id → first matching glob → rollout → default | our choice |
| Model ids | none hard-coded; patterns come from your config | requirement |

## Invariants

* A pinned session's version never changes. `pin` is idempotent and race-free (`BEGIN IMMEDIATE`). Concurrent pins of one session agree.
* Versions are immutable: re-registering an id with a different `prompts_tools_sha` raises `DeployError`.
* `retire` is refused while any session pinned to the version is still active.
* At least one active version carries traffic. Weights cannot be negative or all zero.

## Failure modes and runbook

* **Regression after a ramp.** Filter traces by `harness.version`, compare against the previous version's dashboards, then `rollback(bad, good)`. Use `abort_inflight=True` only when the in-flight sessions are unsafe, such as a security bug.
* **A version never drains.** Sessions that never call `finish` keep it pinned. Find them with `inflight_sessions(version)` and finish the ones that are actually dead (check `StepJournal.resumable_runs()`).
* **Users bouncing between versions.** Route with a stable `routing_key` (a user id) instead of the session id.
* **A flag change has no visible effect.** By design, running sessions keep their snapshot. Only new sessions see the change.
