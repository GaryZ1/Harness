# Execution environment and sandboxing (`harness_sandbox`)

Production Python implementation of the execution-environment design in
[`reports/SOTA agentic harness reference.md` §6](../../reports/SOTA%20agentic%20harness%20reference.md)
(build checklist §6.7, pitfalls §6.8), the build plan section
"Execution: sandbox the whole process tree and proxy every outbound byte", and
[`research_notes/.../tools_sandboxing_safety.md` §5](../../research_notes/SOTA%20agentic%20harness%20build%20plan/tools_sandboxing_safety.md).
Dependencies are the stdlib and PyYAML, plus `harness_memory` (secret redactor, path sandbox, locks) and
`harness_core.types` (events).

| Component | What it does | Module | Doc |
|---|---|---|---|
| **Sandbox policy** | Declarative modes, filesystem read/write rules, network mode and allowlist, resource limits, env allowlist, admin-only escape hatches; YAML/JSON loading; `check_read` / `check_write` for file tools | `policy.py`, `environment_vars.py` | [policy.md](policy.md) |
| **Runners** | The `CommandRunner` interface; bubblewrap (Linux), Seatbelt (macOS), container (Docker/Podman) and the hardened local fallback; the in-sandbox launcher (rlimits, no_new_privs, seccomp, proxy bridge) | `runners/`, `_launcher.py`, `seccomp.py`, `violations.py` | [runners.md](runners.md) |
| **Egress proxy** | HTTP forward + `CONNECT` + SOCKS5 on one loopback/Unix-socket listener; domain allowlist, private-IP blocking after DNS, audit events, byte counters | `proxy.py` | [egress-proxy.md](egress-proxy.md) |
| **Environments and detection** | Backend detection and strongest-backend choice; lazy local/container/remote environments; sandbox death as a tool error; warm pools per tenant; idempotency keys; model-facing description and MCP `_meta` | `detect.py`, `environment.py`, `describe.py` | [environments.md](environments.md) |
| **Snapshots** | Content-addressed workspace checkpoints at turn boundaries: snapshot, diff, restore, prune; secrets excluded; symlink-safe | `snapshot.py` | [snapshots.md](snapshots.md) |

## Why this shape

Every bullet is enforced in code, not in prompt text.

1. **Sandbox the shell, not each tool.** Bash plus a filesystem is the universal tool, so one OS sandbox around every command *and its whole process tree* is the single choke point (Claude Code, Codex, `srt`). Runners start each command in a new session, kill the process group on timeout and after exit, and add PID namespaces where the OS allows.
2. **Both layers are required.** "Without network isolation, a compromised agent can exfiltrate SSH keys. Without filesystem isolation, it can escape the sandbox and regain network access" (Anthropic). Filesystem rules and the egress proxy ship together, and the network is denied by default.
3. **Copy the proven policy semantics.** Codex's four modes (`read-only`, `workspace-write` default, `external-sandbox`, `full-access`) and protected subpaths (`.git`, the resolved gitdir, our `.harness`); `srt`'s precedence (reads allowed with deny-then-allow where a more specific deny such as `**/.env` stays denied; writes allow-only with `deny_write` winning).
4. **Hostname allowlists are not enough.** The proxy resolves DNS itself, blocks private, loopback, link-local and metadata (`169.254.169.254`) answers even for allowlisted names (Codex), and connects to the exact address it vetted.
5. **Weakening is an admin decision.** Every escape hatch, and `full-access`, needs `admin=True` at load time. That flag is never read from the policy file, so an agent that edits the file cannot weaken its own box. Active weakenings are logged at WARNING.
6. **Do not mistake a sandbox API for isolation.** The local fallback is marked `weak` and `isolated=False` unless the harness sits inside a detected container. The permission layer should auto-allow in-boundary actions (Anthropic's 84% prompt reduction) only when `BackendChoice.auto_allow_in_boundary` is true.
7. **Blocked operations teach the model.** Denials come back as `<sandbox_violations>` records with command ID, operation, rule and hint (`srt`), not as opaque failures.
8. **Sandboxes are cattle.** Environments are provisioned lazily (Managed Agents: p50 TTFT about −60%), a dead sandbox is a tool error, and the harness never runs inside the box it controls.
9. **Rewind must cover Bash.** SDK checkpoints miss shell side effects, so snapshots cover the whole workspace, minus secrets.

## Architecture

```text
 harness process (outside every sandbox)                                host
 ┌───────────────────────────────────────────────────────────────┐
 │ tool call ─▶ CommandRunner.run(argv, cwd, env, timeout, stdin)│
 │   SandboxPolicy ─compile(workspace)─▶ CompiledPolicy          │      ┌─────────────────────────┐
 │   build_env(): allowlist, scrub secrets, inject proxy vars    │      │ EgressProxy (threaded)  │
 │   backend argv:                                               │      │  HTTP / CONNECT / SOCKS5│
 │     bwrap  --ro-bind / / --bind ws ws --ro-bind ws/.git ...   │      │  allowlist + DNS check  │
 │            --tmpfs ~/.ssh --unshare-net --die-with-parent     │      │  private-IP block       │
 │     sandbox-exec -p <generated SBPL>                          │      │  audit events, bytes    │
 │     docker run --network none --read-only --cap-drop ALL ...  │      └──────────▲──────────────┘
 │     [unshare --net]  (local fallback, weak)                   │                 │ Unix socket
 └──────────────┬────────────────────────────────────────────────┘                 │ (bind-mounted)
                ▼  new session / process group, capped stdout+stderr               │
 ┌─────────────────────────── sandbox (whole process tree) ──────────────────────────┼──┐
 │ _launcher.py: rlimits ─▶ PR_SET_NO_NEW_PRIVS ─▶ seccomp ─▶ exec(argv)              │  │
 │   (allowlist mode) bridge 127.0.0.1:3128 ─────────────────────────────────────────┘  │
 │ command ─▶ children ─▶ ...   HTTP(S)_PROXY=http://<command_id>:sandbox@127.0.0.1:3128 │
 └──────────────────────────────────────────────────────────────────────────────────────┘
                ▼
   CommandResult(exit_code, stdout, stderr, timed_out, truncated, duration_s,
                 violations=<sandbox_violations>, command_id, sandbox) ──▶ tool result
   WorkspaceSnapshotter.snapshot() at turn boundaries ──▶ restore / diff on rewind
```

## Quick start

```bash
pip install -e ".[dev]"
python3 -m pytest -p no:cacheprovider tests/sandbox
```

```python
from harness_sandbox import (EgressProxy, SandboxPolicy, WorkspaceSnapshotter, create_runner,
                             describe_sandbox, detect_backends, load_policy)

policy = load_policy("sandbox.yaml")                 # or SandboxPolicy() for the defaults
report = detect_backends(workspace="/repo")          # what this host offers
runner = create_runner(policy, "/repo", report=report, session_id="s_42")
system_prompt += describe_sandbox(policy, workspace="/repo", choice=runner.choice)

snaps = WorkspaceSnapshotter.for_policy(policy, "/repo", "~/.harness/snapshots/s_42")
snaps.snapshot("turn-1")
result = runner.run("pytest -q", timeout=300)
tool_result_text = result.model_text()               # stdout/stderr + <sandbox_violations> + exit code
if not runner.choice.auto_allow_in_boundary:
    ...                                              # no real boundary: keep asking for permission
runner.close()
```

`sandbox.yaml` (deny-type lists extend the built-in defaults; `allowed_domains` replaces them):

```yaml
mode: workspace-write
network:
  mode: allowlist
  allowed_domains: [pypi.org, files.pythonhosted.org, "*.githubusercontent.com"]
limits:
  default_timeout_s: 120
  memory_bytes: 4294967296
env:
  set: {CI: "1"}
```

## Integration notes

* **`CommandRunner`.** `run(argv, *, cwd, env, timeout, stdin) -> CommandResult`. The first six fields match the sibling `harness_tools` protocol in order (`runners.base.RESULT_FIELDS`). Extra fields have defaults.
* **Permissions.** Use `CompiledPolicy.check_read` / `check_write` in file tools (read, edit, write, glob). Auto-allow in-boundary actions only when `runner.choice.auto_allow_in_boundary` is true.
* **Events.** Pass an `EventSink`. Runners emit `EventType.CUSTOM` events with `data.kind == "sandbox.exec"`; the proxy emits `"sandbox.egress"`. Neither carries output or environment values.
* **Long-lived processes** (stdio MCP servers, PTY shells): `runner.prepare(...)` or `prepare_mcp_server(...)` give the same sandboxed argv and env for your own `Popen`. Restart them when `policy.fingerprint()` changes.
* **Not here:** git worktree management for parallel writers and the credential-injecting git proxy belong to sibling packages. This package provides the per-writer environment and the egress choke point they plug into.
