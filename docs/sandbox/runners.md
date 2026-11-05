# Runners

Modules: `harness_sandbox.runners` (`base`, `bubblewrap`, `seatbelt`, `container`, `local`), `harness_sandbox._launcher`, `harness_sandbox.seccomp`, `harness_sandbox.violations`.

## Purpose

A runner executes one command under a `SandboxPolicy`, confining the command **and every process it spawns**, and returns a `CommandResult`. All runners share one interface, one launcher and one violation-feedback path. They differ only in how they wrap the argv.

## Research basis

* **"Sandbox the shell, not each tool"** (§6.1). The restrictions "cover spawned subprocesses" (Anthropic).
* **Linux**: bubblewrap for filesystem and namespaces, plus seccomp and `PR_SET_NO_NEW_PRIVS` (Codex `codex-linux-sandbox`; Landlock is legacy). The network namespace is removed and traffic goes through host proxies on bind-mounted Unix sockets (`srt`). A system `bwrap` is resolved only from outside the workspace, and WSL1 is refused (Codex).
* **macOS**: Seatbelt via `sandbox-exec` with generated profiles; one localhost proxy port (`srt`).
* **Containers**: what the C-compiler swarm, mini-swe-agent and the Agents SDK `DockerSandboxClient` use. The checklist limits them to trusted and eval workloads (§6.3 technique 9).
* **Local fallback**: the Agents SDK's Unix-local client "adds no OS-level confinement"; smolagents' local executor "is not a security sandbox" (§6.8). Ours is hardened but explicitly labelled weak.
* **Violation feedback** (§6.3 technique 6) and **resource limits plus monitoring** for model-written code (§6.3 technique 15, checklist item 11).
* **Stateless actions** (mini-swe-agent) by default, with `prepare()` for PTYs and stdio MCP servers (checklist items 6 and 14).

## The `CommandRunner` interface

```python
class CommandRunner(Protocol):
    def run(self, argv: list[str] | str, *, cwd: str | None = None, env: dict | None = None,
            timeout: float | None = None, stdin: str | None = None) -> CommandResult: ...

@dataclass(frozen=True)
class CommandResult:
    exit_code: int; stdout: str; stderr: str; timed_out: bool = False; truncated: bool = False; duration_s: float = 0.0
    signal: int | None = None          # fatal signal; exit_code is then 128 + signal
    command_id: str = ""; sandbox: str = ""; violations: tuple[Violation, ...] = ()
    stdout_bytes: int = 0; stderr_bytes: int = 0; scrubbed_env: tuple[str, ...] = ()
    def model_text(self) -> str: ...   # stdout, stderr, [timed out], <sandbox_violations>, [exit code N]
```

The first six fields match the sibling `harness_tools.CommandResult` in name and order (`RESULT_FIELDS`), so an adapter is `ToolsResult(**{k: getattr(r, k) for k in RESULT_FIELDS})`. A string `argv` runs under `/bin/bash -c` (or `/bin/sh -c` where bash is missing). `cwd` defaults to the workspace and must be readable under the policy; otherwise the result is exit 126 with a violation. `timeout` defaults to `limits.default_timeout_s` and is clamped to `max_timeout_s`.

```python
from harness_sandbox import BubblewrapRunner, LocalRunner, SandboxPolicy

with BubblewrapRunner(SandboxPolicy(), "/repo", event_sink=sink, session_id="s1") as r:
    res = r.run(["pytest", "-q"], timeout=300)
    res.model_text()

    p = r.prepare(["node", "mcp-server.js"])           # long-lived: same sandbox, your Popen
    proc = subprocess.Popen(p.argv, stdin=PIPE, stdout=PIPE, **p.popen_kwargs())
```

## Shared behaviour (`BaseRunner`)

* **Process tree.** `Popen(start_new_session=True)`. On timeout the process group gets SIGTERM, then SIGKILL after `kill_grace_s`, and container runners also run `docker kill <name>`. After *any* exit the group is SIGKILLed, so background children (`sleep 60 &`) die with the call. Bubblewrap adds a PID namespace and `--die-with-parent`, which also catches processes that call `setsid`.
* **Capture.** stdout and stderr are read in threads into head+tail buffers of `max_output_bytes` each. `truncated` reports a cut, and the cut is marked in the text.
* **Environment.** `build_env` (see [policy.md](policy.md)) runs per command. Values go through `Popen(env=...)`, never through argv.
* **Redaction.** When `redact_output` is set, captured output goes through `SecretRedactor`.
* **Violations.** Proxy denials attributed to this `command_id`; `timeout`; `SIGXCPU` (CPU limit); `SIGXFSZ` or "File too large" (file size); `MemoryError` or "Cannot allocate memory" (memory); and EROFS/EACCES lines in stderr (filesystem).
* **Events.** One `EventType.CUSTOM` event with `data.kind = "sandbox.exec"`, carrying sandbox name, isolation level, `weak`, command ID, exit code, duration, violations and policy fingerprint. It never carries output or environment values.
* **Lazy resources.** The session tmp dir and an owned egress proxy are created on first use and removed by `close()`.

## The launcher (`_launcher.py`)

The launcher runs inside the sandbox as `python -I -S _launcher.py --spec JSON -- argv`. It is stdlib-only, so it never imports the package inside the box. It applies, in order:

1. `umask 022` and the rlimits (`CPU`, `DATA` or `AS`, `NPROC`, `FSIZE`, `NOFILE`, `CORE`). It never raises a hard limit, and a failure aborts with exit 126 (fail closed).
2. `PR_SET_NO_NEW_PRIVS`.
3. A classic-BPF **seccomp filter** (x86_64 and aarch64):
   * Wrong architecture: kill the process.
   * x32 ABI: `EPERM`.
   * `EPERM` for mount-family calls (`mount`, `umount2`, `pivot_root`, `open_tree`, `move_mount`, `fs*`, `mount_setattr`), `ptrace`, `process_vm_*`, `bpf`, `perf_event_open`, `userfaultfd`, `keyctl`/`add_key`/`request_key`, `kexec_*`, module loading, `swapon`/`swapoff`, `reboot`, `open_by_handle_at` and `io_uring_*`.
   * Unless `allow_nested_namespaces` is set: `unshare` and `setns` get `EPERM`, `clone` with any `CLONE_NEW*` flag gets `EPERM`, and `clone3` gets `ENOSYS` (libc then falls back to `clone`).
4. `exec` of the command. In proxy mode the launcher instead stays resident: it brings `lo` up, serves the bridge `127.0.0.1:<port> -> proxy Unix socket`, runs the command through a bridge-less copy of itself, forwards SIGTERM, SIGINT and SIGHUP, and re-raises the child's fatal signal.

## Backends

| Runner | `isolation` | `weak` | How it wraps |
|---|---|---|---|
| `BubblewrapRunner` | `os` | no | `bwrap --new-session --die-with-parent --unshare-user/pid/ipc/uts/cgroup-try [--unshare-net] [--disable-userns] --cap-drop ALL --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp`, then mounts sorted shallowest-first: `--bind` writable roots; `--ro-bind` protected subpaths; `--tmpfs` (directory) or `--ro-bind /dev/null` (file or socket) masks for read-denies, including concrete matches of glob denies; `--ro-bind` for `allow_read` exceptions beneath a mask; `--bind` for the proxy socket directory; then `--chdir`. `build_bwrap_argv()` is pure. |
| `SeatbeltRunner` | `os` | no | `sandbox-exec -p <profile>`. `build_seatbelt_profile()` emits `(deny default)`, a fixed mach-lookup list, `file-read*` everywhere `require-not` each deny (globs become POSIX regexes), re-allowed subtrees that still exclude more specific denies, `file-write*` under writable roots minus protected and denied paths, and `/dev` sinks. Network: no rule for `none`, exactly one `(remote ip "localhost:<port>")` for proxy modes, `(allow network*)` for `open`. |
| `ContainerRunner` | `container` | no | `docker|podman run --rm -i --init --name harness-<cmd> --network none|bridge --read-only --cap-drop ALL --security-opt no-new-privileges --pids-limit --memory --memory-swap --ulimit nofile/fsize/cpu/core --user uid:gid --tmpfs /tmp -v ws:/workspace[:ro] -v ws/.git:/workspace/.git:ro -v /dev/null:<.env>:ro --env-file <0600 file> --workdir`. In proxy modes the socket directory and the launcher are mounted, and `python3` must be present in the image. `build_container_argv()` is pure. |
| `LocalRunner` | `process` / `process+netns` | **yes** | launcher only, prefixed by `unshare --net [--map-root-user]` when that works (root or unprivileged user namespaces). Then network `none` is real and proxy modes use the same bridge as bwrap. Without `unshare`, proxy variables are advisory. **Reads and writes are not confined.** |

## Defaults

| Setting | Default | Source |
|---|---|---|
| Linux backend | bubblewrap + seccomp + no_new_privs | Codex |
| `bwrap` resolution | first PATH entry that is absolute and outside the workspace and CWD | Codex |
| WSL1 | bubblewrap refused | Codex |
| `/tmp` in bwrap | fresh tmpfs; session tmp bound at its host path, `TMPDIR` points there | our choice |
| Proxy port inside the sandbox | 3128 | Codex |
| Kill grace | 2 s SIGTERM to SIGKILL | our choice |
| Exit code on signal | `128 + signal` (shell convention), `signal` field set | our choice |
| Container `--pids-limit` / `--memory` | from `limits` (1024 / 4 GiB) | our choice |
| Container user | host `uid:gid` | our choice (files keep the host owner) |

## Invariants

* The harness process is never inside the box it controls. Runners only spawn children.
* Every command starts in a new session, and its process group is killed after it returns.
* The launcher fails closed: if rlimits, no_new_privs or seccomp cannot be applied, the command does not run (exit 126).
* No environment values in argv (bwrap uses `Popen(env)`; containers use a 0600 `--env-file` that is deleted after the run).
* The bwrap binary, the container engine and `unshare` are resolved outside the workspace.
* `BubblewrapRunner` refuses to start if the Python interpreter or the launcher would be read-denied inside the sandbox.

## Failure modes

| Failure | Result |
|---|---|
| Command not found | exit 127 (launcher message on stderr) |
| `cwd` read-denied or missing | exit 126, `filesystem` violation (`deny_read:...` or `cwd-missing`) |
| Timeout | `timed_out=True`, `timeout` violation, process group killed |
| CPU / file-size / memory limit | `SIGXCPU` exit 152 / EFBIG or `SIGXFSZ` / `MemoryError`, each with a `resource` violation |
| Write to `.git/hooks` or outside roots (bwrap) | EROFS from the kernel, `filesystem` violation parsed from stderr |
| Egress denied | proxy 403 or SOCKS refusal, `network` violation with the proxy rule |
| Wrapper cannot start (missing bwrap or python) | `SandboxError`. `RunnerEnvironment` turns it into exit 125 `sandbox_error` |
| `SeatbeltRunner.run` off macOS | `SandboxUnavailableError` (argv and profile building still work) |

## Platform support

| Runner | Linux x86_64 / aarch64 | Linux other arch | macOS | Windows |
|---|---|---|---|---|
| Bubblewrap | yes (needs `bwrap` + user namespaces; not WSL1) | yes, without seccomp | no | no |
| Seatbelt | profile and argv text only (tested in CI) | text only | yes | no |
| Container | yes (Docker or Podman; argv tested without an engine) | yes | yes (Docker Desktop) | not tested |
| Local | yes (seccomp, netns where available) | yes (no seccomp) | yes (rlimits only) | not supported (POSIX `resource` and process groups) |
| Windows sandbox user + WFP (checklist item 2) | — | — | — | **not implemented** (`srt` marks it alpha) |
