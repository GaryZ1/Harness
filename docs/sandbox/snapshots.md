# Workspace snapshots

Module: `harness_sandbox.snapshot`.

## Purpose

`WorkspaceSnapshotter` checkpoints the whole workspace, typically at every turn boundary, so a rewind also undoes what shell commands did. It supports `snapshot`, `diff` (between snapshots, or a snapshot and the live tree), `restore`, `delete`, `prune` and `destroy`. Storage is content-addressed: each distinct file body is stored once under `objects/<sha256>`, and a snapshot is a small JSON manifest. That makes consecutive snapshots of a mostly unchanged tree nearly free ("copy-on-write-ish").

## Research basis

* Claude Code's checkpoints are "separate from git", and SDK file checkpointing tracks only `Write`, `Edit` and `NotebookEdit`, **not Bash side effects**. "A robust harness adds git-level snapshots ... that capture shell changes too" (build plan; tools notes §6 inferences).
* Checklist §6.7 item 15: snapshot at turn boundaries (this also backs §7 rewind), scrub secrets from snapshots, and expire them with the session. §6.3 technique 16 warns that snapshots "can capture secrets that were present in the box".
* E2B pause/fork/snapshot, the Agents SDK `snapshot.py` and OpenCode `snapshot.ts` are the reference points for "fork to try an alternative" (§6.3 technique 16).

## API

```python
from harness_sandbox import WorkspaceSnapshotter

snaps = WorkspaceSnapshotter.for_policy(policy, "/repo", "~/.harness/snapshots/s_42", clock=clock)
s1 = snaps.snapshot("turn-7")             # Snapshot(id, label, created, seq, files, bytes, skipped)
runner.run("rm -rf build && make")
snaps.diff(s1.id)                         # SnapshotDiff(added, modified, deleted), relative paths
snaps.diff(s1.id, s2.id)                  # between two snapshots
snaps.restore(s1.id)                      # workspace matches turn-7; returns what restoring changed
snaps.list(); snaps.get(s1.id)
snaps.prune(keep_last=20, max_age=timedelta(hours=24))
snaps.destroy()                           # session end
```

`for_policy` adds the policy's relative glob read-denies (`**/.env`, `**/.env.*`, and any you add) to the exclude list.

## Defaults

| Setting | Default | Source |
|---|---|---|
| Granularity | whole workspace, caller decides when (turn boundaries recommended) | checklist item 15 |
| Excluded (never captured, never touched) | `.git`, `.env`, `.env.*`, `*.pem`, `*.key`, `*.p12`, `*.pfx`, `id_rsa*`, `id_ecdsa*`, `id_ed25519*`, `id_dsa*`, `.netrc`, `.npmrc`, `.pypirc`, `.git-credentials`, `.harness/secrets`, plus the policy's glob denies | checklist ("scrub secrets"); list is ours |
| `max_file_bytes` | 50 MiB; larger files are recorded as `skipped` and left alone by restore | our choice |
| Storage | content-addressed objects + JSON manifests, `0700` store dir | our choice ("copy-on-write-ish") |
| Expiry | explicit `prune` / `destroy` (call at session end) | checklist ("expire them with the session") |

`.git` is excluded because it is read-only inside the sandbox anyway (so shell commands cannot have changed it) and because VCS history has its own rollback.

## Invariants

* The store must live **outside the workspace** (`SnapshotError` otherwise). It should also sit outside every writable root, so the agent cannot edit its own rollback points.
* Scanning never follows symlinks. A link is recorded as a link (its target string) and restored as a link, so a planted `escape -> ~/.ssh` cannot pull secrets into the store.
* Restore resolves every parent directory inside the workspace (`harness_memory.PathSandbox`) and replaces symlinks and directories that sit where a file belongs, instead of writing through them.
* Manifests are written atomically (`atomic_write_text`) under a cross-process `file_lock`. Objects are written to a temp file and renamed.
* Excluded and oversized files are never deleted or overwritten by restore. Untracked empty directories are removed.
* `prune` and `delete` garbage-collect objects no manifest references.

## Failure modes

| Situation | Behaviour |
|---|---|
| Restore of a deleted or pruned snapshot | `SnapshotError("no such snapshot")` |
| Object missing (store tampered with) | `SnapshotError` naming the file; files restored before that point stay restored |
| Huge workspaces (`node_modules`) | captured unless excluded; pass `exclude=` to skip regenerable trees (restore then leaves them untouched) |
| Concurrent snapshot and restore from two processes | serialised by the store lock |
| Remote side effects (pushed commits, API calls) | not undoable; snapshots cover the local workspace only |

## Platform support

| Platform | Snapshot / diff / restore |
|---|---|
| Linux | yes (tested) |
| macOS | yes (POSIX `os.walk`, `symlink`, `flock`) |
| Windows | not tested (symlink restore needs privileges; `flock` falls back to a thread lock in `harness_memory.locking`) |
