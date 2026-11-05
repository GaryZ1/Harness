"""Workspace snapshots: content-addressed checkpoints with diff and restore.

Research basis (§6.3 technique 16, checklist §6.7 item 15, build plan "Recovery and parallelism"):

* Claude Code's checkpoints and the SDK's file checkpointing only track ``Write``/``Edit``/
  ``NotebookEdit``, **not Bash side effects**, so "a robust harness adds git-level snapshots ... that
  capture shell changes too". This module snapshots the *whole workspace* at turn boundaries, so a
  rewind also undoes what shell commands did.
* **Scrub secrets from snapshots and expire them with the session** (checklist 15; §6.3: "snapshots
  can capture secrets that were present in the box"). Secret-shaped files (``.env``, keys, ``.netrc``,
  the policy's read-deny globs) are excluded and left untouched by restore; :meth:`prune` and
  :meth:`destroy` expire snapshots.
* **Copy-on-write-ish storage**: file contents are stored once per SHA-256 under ``objects/``; a
  snapshot is a small JSON manifest, so consecutive snapshots of a mostly unchanged tree are cheap.
  Unreferenced objects are garbage-collected on delete/prune.

Safety invariants enforced in code:

* scanning never follows symlinks (a planted link cannot pull ``~/.ssh`` into a snapshot); links are
  recorded and restored as links;
* restore resolves every parent directory inside the workspace (``PathSandbox``) and replaces
  symlinks instead of writing through them;
* manifests are written atomically under a cross-process lock; objects are written temp-then-rename.

The store must live **outside** the workspace (and outside the sandbox's writable roots) so the agent
cannot tamper with its own rollback points.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import stat
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

from harness_core.types import Clock, new_id, utcnow
from harness_memory.locking import atomic_write_text, file_lock
from harness_memory.security import PathSandbox

from .errors import SnapshotError
from .policy import SandboxPolicy

#: Never captured, never touched by restore (secrets + VCS internals).
DEFAULT_SNAPSHOT_EXCLUDES: tuple[str, ...] = (
    ".git", ".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "id_rsa*", "id_ecdsa*", "id_ed25519*", "id_dsa*",
    ".netrc", ".npmrc", ".pypirc", ".git-credentials", ".harness/secrets",
)


@dataclass(frozen=True)
class Snapshot:
    id: str
    label: str
    created: str
    seq: int
    files: int
    bytes: int
    skipped: tuple[str, ...] = ()


@dataclass(frozen=True)
class SnapshotDiff:
    added: tuple[str, ...] = ()
    modified: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        return not (self.added or self.modified or self.deleted)

    def summary(self) -> str:
        return f"+{len(self.added)} ~{len(self.modified)} -{len(self.deleted)}"


@dataclass
class _Entry:
    kind: str                 # file | link | skipped
    sha256: str = ""
    mode: int = 0o644
    size: int = 0
    target: str = ""


@dataclass
class _Manifest:
    meta: Snapshot
    entries: dict[str, _Entry] = field(default_factory=dict)
    dirs: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({
            "id": self.meta.id, "label": self.meta.label, "created": self.meta.created, "seq": self.meta.seq,
            "files": self.meta.files, "bytes": self.meta.bytes, "skipped": list(self.meta.skipped),
            "dirs": self.dirs, "entries": {k: vars(v) for k, v in sorted(self.entries.items())},
        }, indent=1, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "_Manifest":
        d = json.loads(text)
        meta = Snapshot(d["id"], d["label"], d["created"], d["seq"], d["files"], d["bytes"], tuple(d.get("skipped", ())))
        return cls(meta, {k: _Entry(**v) for k, v in d["entries"].items()}, list(d.get("dirs", [])))


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class WorkspaceSnapshotter:
    def __init__(
        self,
        workspace: str | os.PathLike[str],
        store_dir: str | os.PathLike[str],
        *,
        exclude: Iterable[str] = DEFAULT_SNAPSHOT_EXCLUDES,
        max_file_bytes: int = 50 * 1024 * 1024,
        clock: Clock = utcnow,
    ) -> None:
        self.workspace = os.path.realpath(os.fspath(workspace))
        self.store = Path(os.path.realpath(os.fspath(store_dir)))
        if self.store == Path(self.workspace) or str(self.store).startswith(self.workspace.rstrip("/") + "/"):
            raise SnapshotError("snapshot store must live outside the workspace")
        self.exclude = tuple(p[3:] if p.startswith("**/") else p for p in exclude)
        self.max_file_bytes = max_file_bytes
        self.clock = clock
        self._sandbox = PathSandbox(self.workspace)
        (self.store / "objects").mkdir(parents=True, exist_ok=True)
        (self.store / "snapshots").mkdir(parents=True, exist_ok=True)
        os.chmod(self.store, 0o700)
        self._tlock = threading.RLock()

    @classmethod
    def for_policy(cls, policy: SandboxPolicy, workspace: str | os.PathLike[str], store_dir: str | os.PathLike[str], **kw) -> "WorkspaceSnapshotter":
        """Also exclude the policy's glob read-denies (``**/.env`` etc.)."""
        globs = [p for p in policy.filesystem.deny_read if any(c in p for c in "*?[") and not p.startswith(("/", "~"))]
        return cls(workspace, store_dir, exclude=(*DEFAULT_SNAPSHOT_EXCLUDES, *globs), **kw)

    # ------------------------------------------------------------------ internals
    def _lock(self):
        return file_lock(self.store / ".lock")

    def _excluded(self, rel: str) -> bool:
        parts = rel.split("/")
        for pat in self.exclude:
            if fnmatch.fnmatchcase(rel, pat) or any(fnmatch.fnmatchcase(p, pat) for p in parts):
                return True
            if "/" in pat and (rel == pat or rel.startswith(pat.rstrip("/") + "/")):
                return True
        return False

    def _obj_path(self, digest: str) -> Path:
        return self.store / "objects" / digest[:2] / digest[2:]

    def _scan(self) -> tuple[dict[str, _Entry], list[str], dict[str, str]]:
        """Walk the workspace without following symlinks. Returns entries, dirs and {rel: abs}."""
        entries: dict[str, _Entry] = {}
        dirs: list[str] = []
        paths: dict[str, str] = {}
        for dirpath, dirnames, filenames in os.walk(self.workspace, followlinks=False):
            rel_dir = os.path.relpath(dirpath, self.workspace)
            rel_dir = "" if rel_dir == "." else rel_dir
            keep = []
            for d in sorted(dirnames):
                rel = f"{rel_dir}/{d}" if rel_dir else d
                full = os.path.join(dirpath, d)
                if self._excluded(rel):
                    continue
                if os.path.islink(full):
                    entries[rel] = _Entry("link", target=os.readlink(full))
                    continue
                keep.append(d)
                dirs.append(rel)
            dirnames[:] = keep
            for name in sorted(filenames):
                rel = f"{rel_dir}/{name}" if rel_dir else name
                if self._excluded(rel):
                    continue
                full = os.path.join(dirpath, name)
                st = os.lstat(full)
                if stat.S_ISLNK(st.st_mode):
                    entries[rel] = _Entry("link", target=os.readlink(full))
                elif stat.S_ISREG(st.st_mode):
                    if st.st_size > self.max_file_bytes:
                        entries[rel] = _Entry("skipped", size=st.st_size)
                    else:
                        entries[rel] = _Entry("file", mode=stat.S_IMODE(st.st_mode), size=st.st_size)
                        paths[rel] = full
        return entries, dirs, paths

    def _manifest(self, snapshot_id: str) -> _Manifest:
        p = self.store / "snapshots" / f"{snapshot_id}.json"
        if not p.is_file():
            raise SnapshotError(f"no such snapshot: {snapshot_id}")
        return _Manifest.from_json(p.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ API
    def snapshot(self, label: str = "") -> Snapshot:
        with self._tlock, self._lock():
            entries, dirs, paths = self._scan()
            total = 0
            for rel, full in paths.items():
                digest = _sha256_file(full)
                entries[rel].sha256 = digest
                obj = self._obj_path(digest)
                if not obj.exists():
                    obj.parent.mkdir(parents=True, exist_ok=True)
                    tmp = obj.with_name(f".{obj.name}.tmp-{os.getpid()}-{threading.get_ident()}")
                    shutil.copyfile(full, tmp)
                    os.replace(tmp, obj)
                total += entries[rel].size
            seq = 1 + max((s.seq for s in self.list()), default=0)
            meta = Snapshot(new_id("snap"), label, self.clock().isoformat(), seq,
                            sum(1 for e in entries.values() if e.kind == "file"), total,
                            tuple(sorted(r for r, e in entries.items() if e.kind == "skipped")))
            atomic_write_text(self.store / "snapshots" / f"{meta.id}.json", _Manifest(meta, entries, dirs).to_json())
            return meta

    def list(self) -> list[Snapshot]:
        out = []
        for p in (self.store / "snapshots").glob("*.json"):
            try:
                out.append(_Manifest.from_json(p.read_text(encoding="utf-8")).meta)
            except (OSError, ValueError, KeyError):
                continue
        return sorted(out, key=lambda s: s.seq)

    def get(self, snapshot_id: str) -> Snapshot:
        return self._manifest(snapshot_id).meta

    def diff(self, base_id: str, target_id: str | None = None) -> SnapshotDiff:
        """Changes from snapshot ``base_id`` to ``target_id`` (or to the current workspace)."""
        base = self._manifest(base_id).entries
        if target_id is not None:
            target = self._manifest(target_id).entries
        else:
            target, _, paths = self._scan()
            for rel, full in paths.items():
                target[rel].sha256 = _sha256_file(full)
        def sig(e: _Entry) -> tuple:
            return (e.kind, e.sha256, e.target, e.mode if e.kind == "file" else 0, e.size if e.kind == "skipped" else 0)
        added = tuple(sorted(set(target) - set(base)))
        deleted = tuple(sorted(set(base) - set(target)))
        modified = tuple(sorted(r for r in set(base) & set(target) if sig(base[r]) != sig(target[r])))
        return SnapshotDiff(added, modified, deleted)

    def restore(self, snapshot_id: str) -> SnapshotDiff:
        """Make the workspace match the snapshot. Excluded and oversized files are left alone."""
        with self._tlock, self._lock():
            man = self._manifest(snapshot_id)
            change = self.diff(snapshot_id)  # base=snapshot -> current; invert for the report
            current, _, _ = self._scan()
            for rel in sorted(set(current) - set(man.entries), key=lambda r: -r.count("/")):
                if current[rel].kind == "skipped":
                    continue
                full = os.path.join(self.workspace, rel)
                if os.path.islink(full) or os.path.isfile(full):
                    os.unlink(full)
            for d in sorted(man.dirs, key=lambda r: r.count("/")):
                parent = self._sandbox.resolve(os.path.dirname(d)) if os.path.dirname(d) else Path(self.workspace)
                target = parent / os.path.basename(d)
                if target.is_symlink() or (target.exists() and not target.is_dir()):
                    target.unlink()
                target.mkdir(exist_ok=True)
            for rel, e in sorted(man.entries.items()):
                if e.kind == "skipped":
                    continue
                parent_rel = os.path.dirname(rel)
                parent = self._sandbox.resolve(parent_rel) if parent_rel else Path(self.workspace)
                parent.mkdir(parents=True, exist_ok=True)
                dst = parent / os.path.basename(rel)
                if e.kind == "link":
                    if dst.is_symlink() and os.readlink(dst) == e.target:
                        continue
                    if dst.is_symlink() or dst.is_file():
                        dst.unlink()
                    elif dst.is_dir():
                        shutil.rmtree(dst)
                    os.symlink(e.target, dst)
                    continue
                if dst.is_symlink():
                    dst.unlink()
                elif dst.is_dir():
                    shutil.rmtree(dst)
                if dst.is_file() and _sha256_file(str(dst)) == e.sha256:
                    os.chmod(dst, e.mode)
                    continue
                obj = self._obj_path(e.sha256)
                if not obj.is_file():
                    raise SnapshotError(f"snapshot {snapshot_id} is missing object {e.sha256[:12]} for {rel}")
                tmp = dst.with_name(f".{dst.name}.restore-{os.getpid()}")
                shutil.copyfile(obj, tmp)
                os.chmod(tmp, e.mode)
                os.replace(tmp, dst)
            keep = set(man.dirs)
            for dirpath, _dirnames, _ in os.walk(self.workspace, topdown=False, followlinks=False):
                rel = os.path.relpath(dirpath, self.workspace)
                if rel == "." or rel in keep or self._excluded(rel) or os.path.islink(dirpath):
                    continue
                try:
                    os.rmdir(dirpath)  # only empty, untracked directories
                except OSError:
                    pass
            return SnapshotDiff(added=change.deleted, modified=change.modified, deleted=change.added)

    def delete(self, snapshot_id: str) -> None:
        with self._tlock, self._lock():
            p = self.store / "snapshots" / f"{snapshot_id}.json"
            if not p.exists():
                raise SnapshotError(f"no such snapshot: {snapshot_id}")
            p.unlink()
            self._gc()

    def prune(self, *, keep_last: int | None = None, max_age: timedelta | None = None) -> list[str]:
        """Expire snapshots older than ``max_age`` and/or beyond the newest ``keep_last``."""
        with self._tlock, self._lock():
            snaps = self.list()
            doomed: set[str] = set()
            if keep_last is not None:
                if keep_last < 0:
                    raise SnapshotError("keep_last must be >= 0")
                doomed.update(s.id for s in (snaps[:-keep_last] if keep_last else snaps))
            if max_age is not None:
                now = self.clock()
                doomed.update(s.id for s in snaps if now - datetime.fromisoformat(s.created) > max_age)
            for sid in doomed:
                (self.store / "snapshots" / f"{sid}.json").unlink(missing_ok=True)
            self._gc()
            return sorted(doomed)

    def destroy(self) -> None:
        """Delete the whole store (session end)."""
        with self._tlock:
            shutil.rmtree(self.store, ignore_errors=True)

    def _gc(self) -> None:
        live: set[str] = set()
        for p in (self.store / "snapshots").glob("*.json"):
            live.update(e.sha256 for e in _Manifest.from_json(p.read_text(encoding="utf-8")).entries.values() if e.sha256)
        for obj in (self.store / "objects").glob("*/*"):
            if obj.parent.name + obj.name not in live:
                obj.unlink(missing_ok=True)
