# Kernel tools: `harness_tools.kernel`, `harness_tools.workspace`, `harness_tools.runner`

**Purpose:** the always-loaded core: `read`, `write`, `edit`, `bash`, `grep`, `glob`. Each runs against one `Workspace` root, and `bash` goes through a pluggable `CommandRunner`.

## Research basis

* **The converged kernel** (§5.7 "Recommended core tool kernel"; build plan "five-tool kernel"): a shell, file read/search, a structured edit primitive and whole-file write. Claude Code ships `Bash`/`Read`/`Edit`/`Write`/`Glob`/`Grep`, Pi ships four tools, and mini-swe-agent only bash.
* **`str_replace` with strict uniqueness** (checklist step 3): error on zero or multiple matches, with an actionable message ("add more surrounding lines"). Silently replacing the first occurrence is a listed pitfall.
* **Absolute paths** (ACI appendix: "requiring absolute paths eliminated a class of errors after `cd`").
* **Line-numbered reads** in the memory `view` / `text_editor` format: 6-character right-aligned number, then a tab.
* **Shell** (checklist step 4): stateless `subprocess.run` mode for evals and RL. Log exit code, duration and output size. Sandbox the whole process tree.
* **Output budgets** (step 5): truncation hints and offload.
* **Edit loops** are a known failure (LangChain loop-detection middleware). **Bash edits bypass checkpoints**, so use git or worktree snapshots for those.

## API

```python
from harness_tools import ToolRegistry, Workspace, MemoryCheckpointer, LocalSubprocessRunner, register_kernel

ws = Workspace("/abs/repo", checkpointer=MemoryCheckpointer())     # require_absolute=True, .git write-protected
kernel = register_kernel(registry, ws,
                         runner=LocalSubprocessRunner(),           # or a sandbox package runner
                         profile="generic",                        # "anthropic" -> Read/Write/Edit/Bash/Grep/Glob
                         include=("read", "write", "edit", "bash", "grep", "glob"),
                         bash_open_world=False,                    # True if the shell has network egress
                         bash_env={"CI": "1"})
ws.checkpointer.rewind(1)                                          # undo the last write/edit
```

| Tool | Input | Output |
|---|---|---|
| `read` | `path` (absolute), `offset` (1-based, default 1), `limit` (default 2,000, max 10,000) | `     12\tline`; `[Showing lines a-b of N. Call read with offset=b+1 to continue.]`; images (`.png/.jpg/.gif/.webp`, ≤ 5 MB) as image blocks |
| `write` | `path`, `content` | `Created|Overwrote <path> (N lines, B bytes).` |
| `edit` | `path`, `old_string`, `new_string`, `replace_all=false` | Numbered snippet around the change; loop warning from edit #5 on the same file |
| `bash` | `command`, `timeout` (s, default 120, max 600), `description` | stdout, `[stderr]` section, `[exit code: N; 0.12s]`; `is_error` when exit ≠ 0 or timed out; metadata `exit_code`, `duration_ms`, `stdout_bytes`, `stderr_bytes`, `timed_out`, `sandbox_id`, `sandbox_violations` |
| `grep` | `pattern` (Python regex), `path`, `glob`, `output_mode` (`files_with_matches`/`content`/`count`), `case_insensitive`, `context` (0–10), `head_limit` (100), `offset` | path-sorted results, ripgrep-style `path:line:text` / `path-line-context`, pagination footer |
| `glob` | `pattern` (`**`, `*`, `?`, `[..]`, `{a,b}`), `path`, `sort` (`path`/`mtime`), `limit` (100), `offset` | absolute paths |

The descriptions live in `descriptions.py`, versioned next to the code (Crush pattern). The SHA-256 of each rendered description is stored in `spec.metadata["description_sha256"]` so traces record which version ran.

## Command runner

```python
class CommandRunner(Protocol):
    def run(self, request: CommandRequest) -> CommandResult: ...

CommandRequest(command, cwd, timeout_s=120.0, env=None, stdin=None, max_output_bytes=1_000_000)
CommandResult(exit_code, stdout, stderr, duration_s, timed_out, stdout_bytes, stderr_bytes,
              output_truncated, sandbox_id=None, violations=[])
```

A runner must not raise for a failing command: a non-zero exit or a timeout is a result. It raises `CommandRunnerError` only when it is itself broken. Hardened runners (seatbelt, bubblewrap, containers, egress proxy) belong to the sandbox package and report `sandbox_id` and `violations`. The kernel surfaces violations to the model in a `<sandbox_violations>` block (srt convention).

Runners with the keyword signature `run(argv, *, cwd, env, timeout, stdin)`, the shape the sandbox package's hardened runners use, are wrapped automatically by `as_command_runner` / `KeywordRunnerAdapter`. Their result fields (`exit_code`, `stdout`, `stderr`, `timed_out`, `truncated`, `duration_s`, byte counts, `sandbox`, `violations`) are read with `getattr`, so `harness_tools` never imports the sandbox package.

`LocalSubprocessRunner` provides **no OS-level confinement**. It runs `bash -c` (falling back to `sh -c`) in a new session/process group and kills the whole group on timeout. It drains stdout and stderr on threads with a per-stream capture cap, and passes only an allowlisted environment (`PATH HOME LANG LC_ALL LC_CTYPE TERM TMPDIR TZ USER LOGNAME SHELL`), so harness API keys never reach model-run commands. Set `inherit_env=True` to opt out.

## Defaults

| Setting | Default | Anchor |
|---|---|---|
| `read` page | 2,000 lines; lines cut at 2,000 chars; 100,000 chars per page | our choice (mirrors Claude Code `Read`; page size = the 25k-token cap) |
| Line-number format | `%6d\t` | `memory_20250818` / `text_editor` view |
| Max file size (read/write) | 10 MB | our choice |
| Image size | 5 MB | our choice (provider image limits) |
| `bash` timeout | 120 s default, 600 s max | our choice (mirrors Claude Code `Bash`) |
| `bash` output | 30,000 chars (≈ 7,500 tokens), head+tail, offloaded | our choice (stricter than the 25k-token global cap) |
| Runner capture cap | 1 MB per stream | our choice (memory bound) |
| `grep` / `glob` page | 100 results, `offset` pagination, max 1,000 | our choice |
| `grep` skips | files > 2 MB, binary (NUL in the first 8 KB), `.git node_modules __pycache__ .venv venv .mypy_cache .pytest_cache .ruff_cache .tox .harness` | our choice |
| Edit-loop warning | 5th edit of the same file in a session | our choice (loop-detection middleware precedent) |
| Write protection | `.git` | our choice (hooks and config there execute code) |
| Child environment | allowlist only | our choice |

## Invariants (enforced in code)

* Paths must be absolute. Relative paths are rejected with the corrected absolute path in the message.
* `..` segments, NUL bytes, `~`, and paths outside the root are rejected, and so is any symlink that resolves outside the root. Unlike the memory sandbox, paths are not URL-decoded, because real files contain `%`.
* `write` over an existing file and every `edit` require that the file was read in this session and that its mtime and size have not changed since. Writes record the new stamp, so chained edits need no re-read.
* Before every write, the `Checkpointer` receives the previous bytes (`None` for a new file). Writes are atomic (temp file, fsync, rename) and keep the file mode. A per-path lock serialises concurrent writers in the process.
* `edit` never applies a partial or ambiguous change: zero matches or several matches without `replace_all` leaves the file untouched.
* `grep` and `glob` never follow directory symlinks and never return files that resolve outside the root.

## Failure modes

| Failure | Result shown to the model |
|---|---|
| Relative path | `path must be absolute; got relative path 'src/a.py'. Did you mean /abs/repo/src/a.py ?` |
| `old_string` not found | Exact-match reminder, plus hints when a whitespace-trimmed match exists or line-number prefixes were pasted |
| `old_string` ambiguous | `matches N places ... Add more surrounding lines ... or set replace_all=true` |
| Stale file | `... was modified since you last read it ... Read it again, then retry.` |
| Binary / huge file | Suggests `bash` (`file`, `xxd`, `sed -n`) or `grep` |
| Command timeout | `[timed out after Xs; the process group was killed ...]`, `is_error` |
| Background job keeps the pipes open | After a 2 s grace the group is killed. The tool description tells the model to redirect background output to a file. |
| Shell edits (`sed -i`) | Not checkpointed. Pair with git or worktree snapshots (sandbox and ops packages). |
