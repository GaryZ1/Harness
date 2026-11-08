# Many readers, one writer: `leases`, `tools`, `worktree`, `coordination`

## Purpose

These modules enforce the invariant the research agrees on most firmly: **fan out reads, keep writes single-threaded**, unless the writes are strictly partitioned and merged through one integrator. Enforcement happens in code at three points: what tools are offered, what tool calls may execute, and what may be merged.

## Research basis

* "Enforce the one-writer rule. Readers and critics get read-only tools. Writes go to one integrator. Parallel writers are allowed only with (a) one git worktree or container per writer, (b) a file-ownership map or task locks, (c) a shared test oracle plus a CI gate before merge, and (d) a single integrator that owns the main branch" (§8.7 item 11).
* "Two teammates editing the same file leads to overwrites." Sixteen agents overwrote each other's fixes on one kernel bug until GCC was used as an oracle to partition the work. An imperfect test oracle makes agents "solve the wrong problem" (§8.8).
* Agent teams use a shared task list with **file-locked claiming**, dependencies and auto-unblocking. The compiler swarm claimed tasks with lock files arbitrated by git (item 12).

## Write modes (`WriteAccess`)

| Mode | Who | What the manager acquires | Parallel writers? |
|---|---|---|---|
| `none` (default) | Explorers, reviewers, critics | A read lease (unlimited, tracking only) | n/a: no mutating tools |
| `workspace` | The single writer of the shared checkout | Exclusive write lease on `ctx.cwd` | **No**: a second writer is refused `lease_held` |
| `own_files` | Partitioned writers in the shared checkout | Disjoint `FileOwnership` claim **and** a write lease per owned path | Yes, on disjoint paths |
| `own_worktree` | Parallel writers needing full isolation | A git worktree + lease `worktree:<path>` | Yes; merged by `Integrator` |

Leases on paths are hierarchical: a lease on `/repo` conflicts with `/repo/src` in both directions. So a `workspace` writer and an `own_files` writer can never overlap, and the integrator (which takes the `/repo` lease) waits for both.

## API

```python
lm = LeaseManager(lock_dir="~/.harness/state/leases")   # lock_dir adds cross-process exclusivity
with lm.acquire_write("/repo", "agent-7", timeout=0) as lease:   # LeaseHeldError if any overlapping writer
    lm.require_write("/repo/src/a.py", "agent-7")         # LeaseNotHeldError for anyone else
r = lm.acquire_read("/repo", "reader-1")                  # never blocks, never blocked
lm.writer_of("/repo"); lm.readers("/repo")

own = FileOwnership()
own.claim("run_a", ["src/auth"]); own.check("run_a", "src/auth/jwt.py"); own.owner_of(path); own.release("run_a")

scoped = ScopedToolExecutor(specs, handler, write_leases=[lease], ownership=own, owner="run_a", workspace="/repo")
scoped.execute(call, ctx)   # denial -> ToolOutcome(is_error=True, metadata={"denied": True})

wtm = WorktreeManager("/repo")                         # worktrees under /repo/.harness/worktrees (git-excluded)
wt = wtm.create("fixer", base="HEAD", owned_files=["src/auth"])
res = Integrator(wtm, lm, verify=lambda path: run_tests(path)).merge(wt)   # MergeResult
wtm.remove(wt)

board = TaskBoard(".harness/tasks.json", lease_seconds=1800, max_attempts=3)
board.add("api", "Build API", depends_on=["schema"], files=["src/api"])
task = board.claim("agent-2"); board.complete(task.id, "agent-2", "done")   # or fail(...), renew(...)
```

## Defaults

| Setting | Value | Note |
|---|---|---|
| `ManagerConfig.lease_wait_s` | 0 | Second writer refused immediately (ours) |
| `Integrator.lease_timeout` | 30 s | Waits for running workspace / own-files writers |
| `Integrator.merge(require_verifier=True)` | on | **Ours**: no merge without a test oracle (item 11c) |
| Worktree root | `<repo>/.harness/worktrees` | Excluded via `.git/info/exclude` |
| Worktree branch | `harness/<agent>-<id>` | |
| `TaskBoard.lease_seconds` / `max_attempts` | 1800 / 3 | Ours |
| `PATH_KEYS` checked for ownership | `path, file_path, filename, file, target, destination, old_path, new_path, paths, files` | |

## Invariants

* **Offer time.** A non-writer, or a writer without its lease, is never offered a tool with `ToolSpec.read_only == False`.
* **Execution time.** `ScopedToolExecutor` rejects a tool outside the agent's set, a mutating tool unless all of the agent's write leases are held, and, for `own_files`, any path argument outside the agent's claim (including absolute paths outside the workspace and `..`). The nested `spawn_subagent` route is exempt; the manager applies its own checks.
* At most one writer holds any path or an overlapping path. Readers are unlimited and never block the writer.
* Ownership claims are all-or-nothing and released when the run ends.
* Only the `Integrator` writes the target branch. Before merging it holds the workspace lease, commits the worktree, rejects changes outside `owned_files`, runs the verifier **inside the worktree**, and refuses to merge into a dirty main checkout. Conflicts abort the merge (`git merge --abort`) and raise `MergeConflictError`, leaving the branch untouched.
* `TaskBoard` writes atomically (temp file, fsync, `os.replace`) under a cross-process `fcntl` lock. A task is claimable only when all dependencies are completed and its `files` do not overlap an in-progress task. Expired claims return to pending. A failure after `max_attempts` attempts is final.

## Failure modes

| Failure | Behaviour |
|---|---|
| Crashed writer process | `flock` is released by the kernel, so the lease frees itself. The holder JSON file is diagnostic only |
| Crashed task claimant | Its lease expires after `lease_seconds` and the task returns to pending |
| Verifier raises | Treated as a failed gate (`merged=False`) |
| Out-of-ownership change in a worktree | `merged=False`, reason names the files |

Cross-process exclusivity (`lock_dir`) applies to the exact resource key. Overlap detection between *different* paths (parent/child) is in-process only, so in multi-process deployments use one canonical workspace key per process group.
