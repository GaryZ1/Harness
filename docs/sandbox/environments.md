# Environments, detection and model-facing description

Modules: `harness_sandbox.detect`, `harness_sandbox.environment`, `harness_sandbox.describe`.

## Purpose

* **Detection** reports which isolation backends the host offers and picks the strongest one for a policy, stating honestly whether a real boundary exists.
* **Environments** put one interface (`provision` / `execute` / `teardown`) over local, OS-sandboxed, container and remote sandboxes. They provision lazily, survive sandbox death, and support warm pools.
* **Description** renders the stable, model-facing summary of the active sandbox, plus the `_meta` payload for sandbox-aware MCP servers.

## Research basis

* Codex resolves `bwrap` only outside the CWD, detects user namespaces and refuses WSL1. In containers it uses `external-sandbox` "rather than switching to full access" (§6.2, pitfalls §6.8). Our checklist adds: detect the environment "rather than trusting a flag" (§6.3 technique 7).
* Managed Agents splits session, harness and sandbox (`provision({resources})`, `execute`). Lazy provisioning cut p50 TTFT by about 60% and p95 by more than 90%. "A dead sandbox becomes a tool-call error the model can retry". Sandboxes are "cattle, not pets" (§6.2, §6.3 techniques 11 and 12).
* Side-effecting tools carry an idempotency key `hash(session_id, step_idx, tool_call_id)` (§6.3 technique 12).
* Warm pools recycle "only within a tenant and image" (§6.3 technique 17).
* Codex's per-mode prompt templates and `codex/sandbox-state` `_meta` (§6.3 techniques 8 and 19). The description is context, never enforcement.

## API

```python
from harness_sandbox import (detect_backends, choose_backend, create_runner, RunnerEnvironment,
                             RemoteEnvironment, EnvironmentPool, idempotency_key,
                             describe_sandbox, sandbox_state_meta, prepare_mcp_server)

report = detect_backends(workspace="/repo")   # BackendReport(bwrap=..., bwrap_usable=..., unshare_netns=..., container=..., wsl=...)
report.available()                            # e.g. ['bubblewrap', 'container', 'local+netns'], strongest first
choice = choose_backend(policy, report)       # BackendChoice(backend, mode, isolated, boundary, reason, warnings)
choice.auto_allow_in_boundary                 # feed into the permission layer
runner = create_runner(policy, "/repo", report=report, require_isolation=False)

env = RunnerEnvironment(lambda: create_runner(policy, "/repo"))  # nothing is created yet
res = env.execute("pytest -q", timeout=300)                      # provisions on first call

class E2BBackend:                               # any RemoteBackend: provision / execute / destroy
    def provision(self, resources): ...
    def execute(self, sandbox_id, argv, *, cwd, env, timeout, stdin): ...
    def destroy(self, sandbox_id): ...
remote = RemoteEnvironment(E2BBackend(), resources={"image": "py312", "cpu": 2})

pool = EnvironmentPool(lambda tenant, image: RemoteEnvironment(E2BBackend(), resources={"image": image}))
e = pool.acquire("tenant-a", "py312"); ...; pool.release(e, reset=lambda env: True)

key = idempotency_key(session_id, step_idx, tool_call.id)
system_prompt += describe_sandbox(policy, workspace="/repo", choice=choice)
mcp_meta = sandbox_state_meta(policy, choice=choice)   # {"harness/sandbox-state": {...}}
```

### Backend choice

| Condition (first match wins) | Backend | Mode | `isolated` | `boundary` |
|---|---|---|---|---|
| `mode: full-access` (admin) | local | full-access | no | none |
| `mode: external-sandbox` | local | external-sandbox | yes if a container is detected | external / none |
| Linux, `bwrap` usable, not WSL1 | bubblewrap | policy mode | yes | os |
| macOS with `/usr/bin/sandbox-exec` | seatbelt | policy mode | yes | os |
| inside a detected container | local | **external-sandbox** (proxy kept) | yes | external |
| otherwise, `require_isolation=True` | raises `SandboxUnavailableError` | | | |
| otherwise | local (`weak`, netns if available) | policy mode | **no**, plus a WARNING | none |

Containers are never auto-chosen, because they need an image. Construct `ContainerRunner(policy, ws, config=ContainerConfig(image=...))` explicitly; `report.available()` lists `container` when an engine is on PATH.

## Defaults

| Setting | Default | Source |
|---|---|---|
| Strongest-first order | bubblewrap > seatbelt > external container > local+netns > local | Codex and `srt` platform choices; the order is ours |
| `bwrap` usability probe | `bwrap --ro-bind / / --unshare-user --unshare-net --die-with-parent true` | Codex warns without user namespaces |
| Container detection | `/.dockerenv`, `/run/.containerenv`, `$container`, `/proc/1/cgroup` markers | our choice |
| VM detection | not used for auto-selection | our choice (a developer laptop VM must not drop to external-sandbox) |
| `require_isolation` | false (warns) | our choice; set true for multi-tenant or untrusted work |
| Provisioning | lazy, on first `execute` | Managed Agents |
| Sandbox error exit code | 125 with a `sandbox_error` violation | our choice |
| Pool `max_idle_per_key` | 4 | our choice |
| Remote tier for untrusted code | Firecracker microVM; gVisor where GPUs are needed; Docker only for trusted/eval | checklist item 7 (backend adapters are the integrator's) |

## Invariants

* The environment object, and the session log, live in the harness, outside the box. A command cannot kill its own harness.
* After a sandbox error the dead box is torn down and the next `execute` provisions a new one. State inside the old box is assumed lost.
* `EnvironmentPool` only hands an environment back to the exact `(tenant, image)` it came from, and tears it down if `reset` fails or raises.
* `describe_sandbox` output depends only on the policy, the workspace and the choice. It is sorted and has no timestamps, so it is prompt-cache stable.

## Failure modes

| Situation | Behaviour |
|---|---|
| `bwrap` installed but user namespaces disabled | `bwrap_usable=False`, a note in `report.notes`, fallback to the next backend |
| WSL1 | bubblewrap refused, note added |
| Attacker drops `./bin/bwrap` in the repo and prepends it to PATH | ignored: workspace, CWD and relative PATH entries are skipped |
| Remote backend raises `SandboxCrashedError`, `SandboxError`, `OSError` or `ConnectionError` | exit 125 result, box destroyed, re-provisioned on the next call |
| Windows host | only the local fallback is reported; a note says the Windows sandbox (dedicated user + WFP) is not implemented |

## Platform support

| Feature | Linux | macOS | Windows |
|---|---|---|---|
| `detect_backends` | full (bwrap, userns, unshare, WSL, container, engines) | seatbelt, engines | notes only |
| Environments, pool, idempotency key, description | yes | yes | yes (with the local runner's limits) |
