"""Workspace confinement for the file tools: absolute paths, no escapes, read-before-edit, checkpoints.

Research basis (``reports/SOTA agentic harness reference.md`` §5.7 steps 3 and the ``write`` row;
§5.8 "Relative paths and ambiguous params", "Bash edits bypass checkpoints"):

* **Poka-yoke paths.** File tools require *absolute* paths ("the classic SWE-bench failure was
  relative paths after ``cd``"; Anthropic's ACI appendix fixed it by requiring absolute paths).
  A relative path is rejected with the corrected absolute path in the message.
* **Confinement.** Every path must resolve inside the workspace root, symlinks included
  (``Path.resolve()`` + ``relative_to()``, the same rule as the memory tool's sandbox). ``..``
  segments and NUL bytes are rejected outright. Unlike :class:`harness_memory.PathSandbox` the
  workspace does *not* URL-decode paths, because real source trees contain file names with ``%``.
* **Protected paths.** Writes into ``.git`` are refused by default (hooks/config there execute code).
* **Read before overwrite.** Like Claude Code's ``Edit``/``Write``, modifying an existing file
  requires that this session read it first and that it has not changed on disk since; this
  prevents blind overwrites and lost concurrent edits.
* **Checkpoint before write.** A :class:`Checkpointer` sees every file's previous bytes before
  ``write``/``edit`` replace them (shell edits are not covered; use git/worktree snapshots).
* **Atomic writes.** Temp file in the same directory + ``fsync`` + ``os.replace``; file mode kept.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Iterable, Protocol, runtime_checkable

from .errors import FileStateError, WorkspaceError

DEFAULT_IGNORED_DIRS = frozenset({".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".harness"})
DEFAULT_PROTECTED_WRITE = frozenset({".git"})
DEFAULT_MAX_FILE_BYTES = 10 * 1024 * 1024


@runtime_checkable
class Checkpointer(Protocol):
    """Called with the previous bytes (``None`` = file did not exist) before a file tool writes."""

    def before_write(self, path: Path, previous: bytes | None) -> None: ...


@dataclass
class MemoryCheckpointer:
    """In-memory checkpoint stack with :meth:`rewind`. Bounded by ``max_bytes`` (oldest dropped)."""

    max_bytes: int = 50 * 1024 * 1024
    _stack: list[tuple[Path, bytes | None]] = field(default_factory=list)
    _bytes: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def before_write(self, path: Path, previous: bytes | None) -> None:
        with self._lock:
            self._stack.append((path, previous))
            self._bytes += len(previous or b"")
            while self._bytes > self.max_bytes and self._stack:
                _, old = self._stack.pop(0)
                self._bytes -= len(old or b"")

    def __len__(self) -> int:
        return len(self._stack)

    def rewind(self, steps: int | None = None) -> list[Path]:
        """Undo the last ``steps`` writes (all if ``None``), newest first. Returns the paths restored."""
        restored: list[Path] = []
        with self._lock:
            n = len(self._stack) if steps is None else min(steps, len(self._stack))
            for _ in range(n):
                path, previous = self._stack.pop()
                self._bytes -= len(previous or b"")
                if previous is None:
                    path.unlink(missing_ok=True)
                else:
                    atomic_write_bytes(path, previous)
                restored.append(path)
        return restored


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically replace ``path`` with ``data`` (temp + fsync + rename), preserving its mode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = None
    try:
        mode = path.stat().st_mode & 0o7777
    except FileNotFoundError:
        pass
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


@dataclass(frozen=True)
class FileStamp:
    mtime_ns: int
    size: int


class FileStateTracker:
    """Per-session record of which files were read/written and their on-disk stamp at that time."""

    def __init__(self) -> None:
        self._state: dict[tuple[str, str], FileStamp] = {}
        self._edits: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _stamp(path: Path) -> FileStamp:
        st = path.stat()
        return FileStamp(st.st_mtime_ns, st.st_size)

    def record(self, session_id: str, path: Path) -> None:
        with self._lock:
            self._state[(session_id, str(path))] = self._stamp(path)

    def check_fresh(self, session_id: str, path: Path, *, display: str) -> None:
        """Raise :class:`FileStateError` unless ``path`` was read in this session and is unchanged since."""
        with self._lock:
            seen = self._state.get((session_id, str(path)))
        if seen is None:
            raise FileStateError(f"{display} exists but has not been read in this session. Read it first, then retry the change.")
        if self._stamp(path) != seen:
            raise FileStateError(f"{display} was modified since you last read it (by the user, a linter or a command). Read it again, then retry.")

    def count_edit(self, session_id: str, path: Path) -> int:
        with self._lock:
            key = (session_id, str(path))
            self._edits[key] = self._edits.get(key, 0) + 1
            return self._edits[key]


class Workspace:
    """The directory the file tools and ``bash`` operate in.

    :param root: workspace root (resolved; must exist).
    :param require_absolute: reject relative paths (poka-yoke; default ``True``).
    :param protected_write: top-level-or-nested directory names that may never be written.
    :param ignored_dirs: directory names ``grep``/``glob`` skip by default.
    :param require_read_before_write: enforce read-before-overwrite/edit (default ``True``).
    :param checkpointer: receives previous bytes before every write.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        require_absolute: bool = True,
        protected_write: Iterable[str] = DEFAULT_PROTECTED_WRITE,
        ignored_dirs: Iterable[str] = DEFAULT_IGNORED_DIRS,
        require_read_before_write: bool = True,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        checkpointer: Checkpointer | None = None,
        offload_dirname: str = ".harness/tool-outputs",
    ) -> None:
        given = Path(root).expanduser()
        if not given.is_dir():
            raise WorkspaceError(f"workspace root does not exist or is not a directory: {given}")
        self._given_root = given.absolute()
        self.root = given.resolve()
        self.require_absolute = require_absolute
        self.protected_write = frozenset(protected_write)
        self.ignored_dirs = frozenset(ignored_dirs)
        self.require_read_before_write = require_read_before_write
        self.max_file_bytes = max_file_bytes
        self.checkpointer = checkpointer
        self.files = FileStateTracker()
        self.offload_dir = self.root / offload_dirname
        self._path_locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # ------------------------------------------------------------------ paths
    def resolve(self, path: str, *, for_write: bool = False, param: str = "path") -> Path:
        """Validate a model-supplied path and return the resolved absolute path inside the root."""
        if not isinstance(path, str) or not path:
            raise WorkspaceError(f"{param} must be a non-empty string")
        if "\x00" in path:
            raise WorkspaceError(f"{param} contains a NUL byte")
        pure = PurePath(path)
        if any(part == ".." for part in pure.parts):
            raise WorkspaceError(f"{param} must not contain '..' segments: {path!r}. Use the absolute path inside {self.root}.")
        if pure.is_absolute():
            rel = self._relative_to_root(pure)
            if rel is None:
                raise WorkspaceError(f"{path} is outside the workspace. Only paths under {self.root} are accessible.")
        else:
            if path.startswith("~"):
                raise WorkspaceError(f"{param} must be an absolute path under {self.root}; '~' is not expanded.")
            if self.require_absolute:
                suggestion = self.root / pure
                raise WorkspaceError(f"{param} must be absolute; got relative path {path!r}. Did you mean {suggestion} ?")
            rel = pure
        target = (self.root / rel).resolve()
        try:
            rel_resolved = target.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceError(f"{path} resolves outside the workspace (symlink escape) and is not accessible.") from exc
        if for_write and any(part in self.protected_write for part in rel_resolved.parts):
            raise WorkspaceError(f"{path} is inside a protected directory ({', '.join(sorted(self.protected_write))}) and cannot be written.")
        return target

    def _relative_to_root(self, pure: PurePath) -> PurePath | None:
        for base in (self.root, self._given_root):
            try:
                return pure.relative_to(base)
            except ValueError:
                continue
        return None

    def display(self, path: Path) -> str:
        """Absolute display form (what the model should send back)."""
        return str(path)

    def is_ignored(self, path: Path) -> bool:
        try:
            parts = path.relative_to(self.root).parts
        except ValueError:
            return True
        return any(p in self.ignored_dirs for p in parts[:-1]) or (path.is_dir() and parts and parts[-1] in self.ignored_dirs)

    # ------------------------------------------------------------------ io helpers
    def lock_for(self, path: Path) -> threading.Lock:
        with self._locks_guard:
            lock = self._path_locks.get(str(path))
            if lock is None:
                lock = self._path_locks[str(path)] = threading.Lock()
            return lock

    def check_writable(self, session_id: str, path: Path) -> None:
        """Enforce read-before-overwrite for an existing file."""
        if self.require_read_before_write and path.exists():
            self.files.check_fresh(session_id, path, display=self.display(path))

    def write_bytes(self, session_id: str, path: Path, data: bytes) -> None:
        """Checkpoint, then atomically write, then record the new stamp as "read" for this session."""
        if len(data) > self.max_file_bytes:
            raise WorkspaceError(f"refusing to write {len(data):,} bytes to {path}: limit is {self.max_file_bytes:,} bytes")
        if path.exists() and path.is_dir():
            raise WorkspaceError(f"{path} is a directory")
        if self.checkpointer is not None:
            previous = path.read_bytes() if path.exists() else None
            self.checkpointer.before_write(path, previous)
        atomic_write_bytes(path, data)
        self.files.record(session_id, path)


__all__ = [
    "Checkpointer",
    "DEFAULT_IGNORED_DIRS",
    "DEFAULT_MAX_FILE_BYTES",
    "DEFAULT_PROTECTED_WRITE",
    "FileStamp",
    "FileStateTracker",
    "MemoryCheckpointer",
    "Workspace",
    "atomic_write_bytes",
]
