"""File-backed memory store: the single storage abstraction every memory type writes through.

Guarantees (from the build checklist, section 4.7 steps 4 and 10):

* path-sandboxed (:class:`~harness_memory.security.PathSandbox`);
* per-file cap of 100 kB and per-store cap of 10,000 files (Managed Agents limits);
* secrets redacted on every write;
* ``read_only`` scopes enforced in code, not prompts;
* optimistic concurrency through a ``content_sha256`` precondition;
* atomic writes and a cross-process lock;
* ``modified`` (and ``tainted``) stamped on every frontmatter document write.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator

from ._time import Clock, to_iso, utcnow
from .errors import (
    AccessDeniedError,
    AlreadyExistsError,
    ConcurrencyError,
    LimitExceededError,
    NotFoundError,
    PathSecurityError,
    ValidationError,
)
from .frontmatter import MemoryDocument, Scope, join_frontmatter, sha256_text, split_frontmatter
from .locking import atomic_write_text, file_lock
from .security import PathSandbox, SecretRedactor
from .versioning import GitVersioner

#: Precondition value meaning "the file must not exist yet".
ABSENT = "absent"

_IGNORED_NAMES = {".git", ".lock", "node_modules", "__pycache__"}


class Access(str, Enum):
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"


@dataclass(frozen=True)
class StoreLimits:
    max_file_bytes: int = 100_000
    max_files: int = 10_000


@dataclass(frozen=True)
class StoredFile:
    path: str
    content: str
    sha256: str


class FileMemoryStore:
    """A directory of UTF-8 text memories with safety rails."""

    def __init__(
        self,
        root: str | Path,
        *,
        scope: Scope = Scope.PROJECT,
        access: Access = Access.READ_WRITE,
        limits: StoreLimits | None = None,
        redactor: SecretRedactor | None = None,
        clock: Clock = utcnow,
        versioner: GitVersioner | None = None,
        session_tainted: bool = False,
        create: bool = True,
    ) -> None:
        root = Path(root).expanduser()
        if create and access is Access.READ_WRITE:
            root.mkdir(parents=True, exist_ok=True)
        if not root.exists():
            raise NotFoundError(f"memory store root does not exist: {root}")
        self.sandbox = PathSandbox(root)
        self.root = self.sandbox.root
        self.scope = scope
        self.access = access
        self.limits = limits or StoreLimits()
        self.redactor = redactor or SecretRedactor()
        self.clock = clock
        self.versioner = versioner
        self.session_tainted = session_tainted
        self._lock_path = self.root / ".lock"

    # ------------------------------------------------------------------ helpers
    @property
    def writable(self) -> bool:
        return self.access is Access.READ_WRITE

    def _require_writable(self) -> None:
        if not self.writable:
            raise AccessDeniedError(f"{self.scope.value} memory at {self.root} is read-only")

    def _resolve(self, rel: str) -> Path:
        path = self.sandbox.resolve(rel)
        rel_parts = Path(self.sandbox.relative(path)).parts if path != self.root else ()
        if any(p in _IGNORED_NAMES for p in rel_parts):
            raise PathSecurityError(f"protected path: {rel!r}")
        return path

    def lock(self):
        """Exclusive store-wide lock (re-entrant within a thread)."""
        return file_lock(self._lock_path)

    # ------------------------------------------------------------------ reads
    def exists(self, rel: str) -> bool:
        return self._resolve(rel).exists()

    def is_dir(self, rel: str) -> bool:
        return self._resolve(rel).is_dir()

    def read(self, rel: str) -> StoredFile:
        path = self._resolve(rel)
        if not path.is_file():
            raise NotFoundError(f"no such memory file: {rel}")
        content = path.read_text(encoding="utf-8")
        return StoredFile(self.sandbox.relative(path), content, sha256_text(content))

    def read_document(self, rel: str) -> MemoryDocument:
        f = self.read(rel)
        return MemoryDocument.from_markdown(f.content, path=f.path)

    def iter_files(self, prefix: str = "", *, suffix: str | None = None) -> Iterator[str]:
        """Yield relative paths of files under ``prefix`` (recursive, sorted, hidden files skipped)."""
        base = self._resolve(prefix) if prefix else self.root
        if not base.exists():
            return
        if base.is_file():
            yield self.sandbox.relative(base)
            return
        found = []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in _IGNORED_NAMES]
            for name in filenames:
                if name.startswith(".") or (suffix and not name.endswith(suffix)):
                    continue
                found.append(self.sandbox.relative(Path(dirpath) / name))
        yield from sorted(found)

    def iter_documents(self, prefix: str = "") -> Iterator[MemoryDocument]:
        """Yield every parseable frontmatter document under ``prefix``; files without frontmatter are skipped."""
        for rel in self.iter_files(prefix, suffix=".md"):
            try:
                yield self.read_document(rel)
            except ValidationError:
                continue

    def count_files(self) -> int:
        return sum(1 for _ in self.iter_files())

    # ------------------------------------------------------------------ writes
    def write(self, rel: str, content: str, *, expected_sha256: str | None = None, redact: bool = True) -> StoredFile:
        """Create or replace ``rel``.

        ``expected_sha256`` implements optimistic concurrency: pass the hash you read, or
        :data:`ABSENT` to require that the file does not exist yet.
        """
        self._require_writable()
        path = self._resolve(rel)
        if path == self.root:
            raise PathSecurityError("cannot write to the memory root")
        if redact:
            content = self.redactor.redact(content)
        size = len(content.encode("utf-8"))
        if size > self.limits.max_file_bytes:
            raise LimitExceededError(f"{rel} is {size} bytes; the per-file limit is {self.limits.max_file_bytes}")
        with self.lock():
            exists = path.exists()
            if path.is_dir():
                raise AlreadyExistsError(f"{rel} is a directory")
            if expected_sha256 is not None:
                if expected_sha256 == ABSENT:
                    if exists:
                        raise ConcurrencyError(f"{rel} already exists")
                else:
                    current = sha256_text(path.read_text(encoding="utf-8")) if exists else None
                    if current != expected_sha256:
                        raise ConcurrencyError(f"{rel} changed since it was read (content_sha256 mismatch)")
            if not exists and self.count_files() >= self.limits.max_files:
                raise LimitExceededError(f"store holds {self.limits.max_files} files; prune before adding more")
            atomic_write_text(path, content)
        return StoredFile(self.sandbox.relative(path), content, sha256_text(content))

    def stamp(self, content: str, *, created_if_missing: bool = True) -> str:
        """Stamp ``modified`` (and ``tainted`` for tainted sessions) into a frontmatter document."""
        data, body = split_frontmatter(content)
        if data is None:
            return content
        now = self.clock()
        data["modified"] = to_iso(now)
        if created_if_missing and not data.get("created"):
            data["created"] = to_iso(now)
        if self.session_tainted:
            data["tainted"] = True
        return join_frontmatter(data, body)

    def write_document(self, doc: MemoryDocument, *, expected_sha256: str | None = None) -> StoredFile:
        if not doc.path:
            raise ValidationError("document has no path")
        now = self.clock()
        doc.meta.modified = now
        doc.meta.created = doc.meta.created or now
        if self.session_tainted:
            doc.meta.tainted = True
        return self.write(doc.path, doc.to_markdown(), expected_sha256=expected_sha256)

    def delete(self, rel: str) -> None:
        self._require_writable()
        path = self._resolve(rel)
        if path == self.root:
            raise PathSecurityError("refusing to delete the memory root")
        with self.lock():
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            else:
                raise NotFoundError(f"no such memory path: {rel}")

    def rename(self, old: str, new: str) -> None:
        self._require_writable()
        src, dst = self._resolve(old), self._resolve(new)
        if self.root in (src, dst):
            raise PathSecurityError("refusing to rename the memory root")
        with self.lock():
            if not src.exists():
                raise NotFoundError(f"no such memory path: {old}")
            if dst.exists():
                raise AlreadyExistsError(f"destination exists: {new}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(src, dst)

    def mkdir(self, rel: str) -> None:
        self._require_writable()
        self._resolve(rel).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ versioning
    def commit(self, message: str) -> str | None:
        if self.versioner is None or not self.writable:
            return None
        with self.lock():
            return self.versioner.commit_all(message)
