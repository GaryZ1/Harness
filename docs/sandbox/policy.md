# Sandbox policy

Module: `harness_sandbox.policy` (plus `harness_sandbox.environment_vars` for the environment).

## Purpose

`SandboxPolicy` is one declarative, frozen description of what sandboxed commands may do: the mode, filesystem read and write rules, network mode and domain lists, resource limits, environment allowlist and escape hatches. It loads from YAML or JSON. `policy.compile(workspace)` resolves it against one workspace, home directory and session tmp into a `CompiledPolicy`. Every runner builds its argv or profile from that, and file tools call `check_read` / `check_write` on it.

## Research basis

* **Codex `SandboxPolicy`** gives the four modes and `read_only_subpaths`, which keep `.git`, `.git/hooks` and the resolved gitdir read-only "to prevent privilege escalation" (§6.2, §6.3 technique 2).
* **`srt` precedence rules** (§6.2): reads are allowed by default with deny-then-allow, and "a more specific `denyRead` (for example `**/.env`) stays denied". Writes are allow-only, and `denyWrite` beats `allowWrite`. Network is denied by default.
* **Checklist §6.7 item 3** sets the filesystem defaults: writable workspace plus per-session tmp; `denyRead` for `~/.ssh`, `**/.env` and the harness secret store; cloud-credential directories "as our own addition"; and file tools that canonicalise and reject traversal, including `%2e%2e%2f`.
* **Checklist item 18**: weakening knobs are off by default, settable only by admin policy, and visible when on.
* **Pitfall "secrets inside the box"** and checklist item 9: no tokens in sandbox environment variables. Model API keys are injected harness-side.

## API

```python
from harness_sandbox import SandboxPolicy, FilesystemPolicy, NetworkPolicy, NetworkMode, load_policy

policy = SandboxPolicy(
    filesystem=FilesystemPolicy(allow_read=("~/.aws/config",)),      # re-allow one file under ~/.aws
    network=NetworkPolicy(mode=NetworkMode.ALLOWLIST, allowed_domains=("pypi.org", "*.pythonhosted.org")),
).validate()                                                         # raises PolicyError / EscapeHatchError

c = policy.compile("/repo", session_tmp="/tmp/harness-sbx-tmp-x")
c.check_read("~/.ssh/id_ed25519")      # PathDecision(allowed=False, rule="deny_read:~/.ssh", ...)
c.check_write(".git/hooks/pre-commit")  # PathDecision(allowed=False, rule="read_only_subpaths:/repo/.git")
c.check_write("src/app.py")             # PathDecision(allowed=True, rule="writable-root")
c.writable_roots, c.read_only_paths     # what runners mount read-write / read-only

admin_policy = load_policy("/etc/harness/sandbox.yaml", admin=True)  # only managed settings pass admin=True
policy.fingerprint()                    # stable hash; restart PTYs and MCP servers when it changes
policy.to_dict()                        # JSON-safe; round-trips through from_dict
```

`build_env(policy, host_env=..., extra=..., sandbox_vars=...)` returns `EnvBuild(env, scrubbed)`. Precedence, later winning: host variables matching `env.allow`, then the caller's `env`, then `env.set`, then sandbox-controlled variables (proxy, `TMPDIR`, `HARNESS_SANDBOX*`). Finally, every name matching `env.deny`, and every value the `harness_memory` `SecretRedactor` flags, is dropped. The only exception is a name an admin listed in `escape_hatches.env_secret_passthrough`.

### Path rules

* A pattern is an absolute path, a `~` path, a workspace-relative path, or a glob (`*`, `?`, `[...]`, `**`). A relative pattern starting with `**/` matches anywhere.
* A literal rule matches its path and everything beneath it. A glob matches a node or any of its ancestors.
* **Specificity** is the number of path components of the rule (literal) or of the matched node (glob). A read is denied when a deny matches and no allow is *strictly* more specific; a tie goes to deny. So `allow_read: [~/.aws/config]` beats `deny_read: ~/.aws`, but nothing short of an exact deeper allow beats `**/.env`.
* Both the lexical path and the symlink-resolved path must pass. A workspace symlink to `~/.ssh` is read-denied, and a write through a symlink pointing outside the workspace is refused.
* Write checks: the path must sit under a writable root (or match an `allow_write` glob); no `deny_write`, protected subpath or read-deny may match. Secret paths are neither readable nor writable.

## Defaults

| Setting | Default | Source |
|---|---|---|
| `mode` | `workspace-write` | Codex (default for coding) |
| Writable roots | workspace + per-session tmp | checklist item 3 |
| `read_only_subpaths` | `.git`, `.harness`, plus the resolved gitdir and commondir of worktrees | Codex (`.git`, `.git/hooks`, gitdir, `.codex`); `.harness` is our config dir |
| `deny_read` | `~/.ssh`, `**/.env` | `srt` example config |
| `deny_read` (cont.) | `**/.env.*`, `~/.aws`, `~/.azure`, `~/.config/gcloud`, `~/.config/gh`, `~/.kube`, `~/.docker`, `~/.gnupg`, `~/.netrc`, `~/.git-credentials`, `~/.npmrc`, `~/.pypirc`, `~/.cargo/credentials[.toml]`, `~/.harness/secrets` | our choice (the checklist asks for cloud credentials) |
| `deny_read` sockets | `/var/run/docker.sock`, `/run/docker.sock`, `/run/podman`, `/run/containerd`, `/run/user` | our choice (`srt` gates Unix sockets via `allowUnixSockets`) |
| `network.mode` | `none` | `srt`: "By default, all network access is denied" |
| `allowed_domains` (used in `allowlist`/`limited`) | pypi.org, files.pythonhosted.org, registry.npmjs.org, registry.yarnpkg.com, crates.io, static.crates.io, index.crates.io, proxy.golang.org, sum.golang.org, docs.python.org | our choice (the checklist asks for "a small default allowlist"); push-capable code hosts are left out on purpose |
| `allowed_ports` | 80, 443 | our choice |
| `proxy_port_in_sandbox` | 3128 | Codex HTTP proxy default |
| `ask_on_unknown` | false (headless denies) | checklist item 4 |
| `default_timeout_s` / `max_timeout_s` | 120 / 600 | Claude Code Bash tool (2 min default, 10 min max) |
| `cpu_seconds` | 600 (SIGXCPU, SIGKILL 5 s later) | our choice |
| `memory_bytes` | 4 GiB via `RLIMIT_DATA` | our choice (`RLIMIT_AS` breaks V8, Go and wasm reservations) |
| `max_processes` | 1024 (`RLIMIT_NPROC`, per real uid; `--pids-limit` in containers) | our choice |
| `file_size_bytes` / `open_files` | 1 GiB / 4096 | our choice |
| `core_dumps` | off (`RLIMIT_CORE=0`) | our choice (no memory images with secrets on disk) |
| `max_output_bytes` | 1 MiB per stream, head + tail kept | our choice |
| `env.allow` | PATH, HOME, USER, LOGNAME, SHELL, LANG, LANGUAGE, LC_*, TERM, TZ, COLORTERM, NO_COLOR, FORCE_COLOR, PYTHONIOENCODING, PYTHONUTF8, SOURCE_DATE_EPOCH | our choice |
| `env.deny` | `*TOKEN*`, `*SECRET*`, `*PASSWORD*`, `*API_KEY*`, `*ACCESS_KEY*`, `*CREDENTIAL*`, `AWS_*`, `ANTHROPIC_*`, `OPENAI_*`, `GH_*`, `GITHUB_*`, `SSH_AUTH_SOCK`, `*_PROXY`, ... | checklist item 9; list is ours |
| `redact_output` | true | our choice (reuses `harness_memory.SecretRedactor`) |
| Escape hatches | all off | `srt`, Codex |

### Escape hatches (admin only)

| Hatch | Effect | Vendor analogue |
|---|---|---|
| `mode: full-access` | no sandbox at all | Codex `danger-full-access`, `--yolo` |
| `allow_non_loopback_proxy` | proxy may bind beyond loopback | Codex `dangerously_allow_non_loopback_proxy` |
| `allowed_private_cidrs` | proxy may reach these private ranges | (none; Codex blocks private IPs) |
| `allow_unix_sockets` | re-allow specific socket paths under the default masks | `srt` `allowUnixSockets` |
| `allow_nested_namespaces` | seccomp stops blocking `unshare`/`setns`/`CLONE_NEW*`; no `--disable-userns` | `srt` `allowNestedUserNamespaces` |
| `disable_seccomp` | no seccomp filter | `srt` `enableWeakerNestedSandbox` (closest) |
| `env_secret_passthrough` | named secret-looking variables pass into the box | (none) |

## Invariants

* The default `deny_read` entries, protected subpaths and `env.deny` entries **cannot be removed**, however the policy is built (`__post_init__` merges them back). A more specific `allow_read` is the only way to carve an exception.
* `validate(admin=False)` raises `EscapeHatchError` if any weakening is active. The admin flag is a function argument, never a policy key.
* Unknown keys, wrong types, bad CIDRs, schemes or paths in domain patterns, mid-pattern wildcards (`py*.org`) and non-positive limits raise `PolicyError`.
* Timeouts are clamped to `max_timeout_s`.
* Percent-encoded traversal and NUL bytes in tool paths raise `PolicyError`. Plain `..` is resolved first and the result is then checked.

## Failure modes

| Situation | Behaviour |
|---|---|
| Policy file edited by the agent to add a hatch | `load_policy` without `admin=True` raises `EscapeHatchError`; the session should refuse to start |
| `network.mode: open` with workspace-write | allowed, logged at WARNING (filesystem isolation alone does not stop exfiltration) |
| Glob deny on a file created later inside the sandbox | OS masks are computed per command, so the next command masks it. Within a command the creator already knows the content |
| Secret in an env var name the deny list misses | the value scan (`SecretRedactor`) still drops recognisable tokens; extend `env.deny` for organisation-specific names |

## Platform support

| Feature | Linux | macOS | Windows |
|---|---|---|---|
| Policy load, validation, `check_read` / `check_write` | yes | yes | yes (paths are POSIX-style; not exercised in CI) |
| Environment scrubbing | yes | yes | yes |
