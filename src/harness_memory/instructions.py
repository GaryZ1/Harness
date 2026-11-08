"""Instruction memory: human-written, always-loaded ``AGENTS.md`` / ``CLAUDE.md`` files.

Loading semantics follow Claude Code:

* files from the project root down to the working directory are concatenated root-first, so
  the closest file is read last (managed and user files come first); ``*.local.md`` is appended after its sibling at each level;
* ``AGENTS.md`` is used when no ``CLAUDE.md`` exists at a level (``mode="claude-md-or-agents-md"``);
* ``@path`` imports are expanded recursively up to **4 hops**; imports that leave the project
  require an approval callback; files over 4 MiB are skipped;
* files over 200 lines produce a warning (they "reduce adherence");
* subdirectory files are loaded lazily via :meth:`InstructionLoader.for_path`.

Instructions are *context, not enforced configuration* - enforce policy with hooks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

MAX_IMPORT_DEPTH = 4
MAX_FILE_BYTES = 4 * 1024 * 1024
RECOMMENDED_MAX_LINES = 200
_IMPORT = re.compile(r"(?<![\w`])@((?:~|\.{1,2})?/?[\w./\-]+\.(?:md|txt))")


@dataclass
class InstructionFile:
    path: Path
    content: str
    level: int
    imported_from: Path | None = None


@dataclass
class LoadedInstructions:
    files: list[InstructionFile] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def render(self) -> str:
        parts = []
        for f in self.files:
            parts.append(f'<instructions source="{f.path}">\n{f.content.strip()}\n</instructions>')
        return "\n\n".join(parts)


class InstructionLoader:
    def __init__(
        self,
        project_root: str | Path,
        *,
        names: tuple[str, ...] = ("CLAUDE.md", ".claude/CLAUDE.md"),
        fallback: str = "AGENTS.md",
        mode: str = "claude-md-or-agents-md",
        user_file: str | Path | None = None,
        managed_file: str | Path | None = None,
        approve_external: Callable[[Path], bool] | None = None,
    ) -> None:
        if mode not in ("claude-md-or-agents-md", "claude-md-and-agents-md", "agents-md-only"):
            raise ValueError(f"unknown mode {mode}")
        self.project_root = Path(project_root).resolve()
        self.names = names
        self.fallback = fallback
        self.mode = mode
        self.user_file = Path(user_file).expanduser() if user_file else None
        self.managed_file = Path(managed_file) if managed_file else None
        self.approve_external = approve_external or (lambda p: False)
        self._approved: dict[Path, bool] = {}

    def _files_at(self, directory: Path) -> list[Path]:
        primary = [directory / n for n in self.names if (directory / n).is_file()] if self.mode != "agents-md-only" else []
        agents = directory / self.fallback
        out = primary[:1]
        if agents.is_file() and (self.mode != "claude-md-or-agents-md" or not out):
            out.append(agents)
        for p in list(out):
            local = p.with_name(p.stem + ".local.md")
            if local.is_file():
                out.append(local)
        return out

    def load(self, cwd: str | Path | None = None) -> LoadedInstructions:
        cwd = Path(cwd or self.project_root).resolve()
        result = LoadedInstructions()
        chain: list[Path] = []
        for extra in (self.managed_file, self.user_file):
            if extra and extra.is_file():
                chain.append(extra)
        # Only directories from the project root down to cwd are searched (never above the project).
        if cwd == self.project_root or self.project_root in cwd.parents:
            dirs = [cwd, *[p for p in cwd.parents if p == self.project_root or self.project_root in p.parents]]
        else:
            dirs = [self.project_root]
        for d in reversed(dirs):
            chain.extend(self._files_at(d))
        seen: set[Path] = set()
        for level, path in enumerate(chain):
            self._load_file(path, level, None, 0, result, seen)
        return result

    def for_path(self, touched: str | Path) -> LoadedInstructions:
        """Lazily load instruction files in subdirectories between the project root and ``touched``."""
        target = Path(touched).resolve()
        directory = target if target.is_dir() else target.parent
        result = LoadedInstructions()
        seen: set[Path] = set()
        if self.project_root not in directory.parents:
            return result
        rel_dirs = []
        d = directory
        while d != self.project_root:
            rel_dirs.append(d)
            d = d.parent
        for level, sub in enumerate(reversed(rel_dirs)):
            for p in self._files_at(sub):
                self._load_file(p, level, None, 0, result, seen)
        return result

    def _load_file(self, path: Path, level: int, parent: Path | None, depth: int, result: LoadedInstructions, seen: set[Path]) -> None:
        path = path.resolve()
        if path in seen:
            return
        seen.add(path)
        try:
            size = path.stat().st_size
        except OSError:
            result.warnings.append(f"cannot read {path}")
            return
        if size > MAX_FILE_BYTES:
            result.warnings.append(f"skipped {path}: larger than 4 MiB")
            return
        text = path.read_text(encoding="utf-8", errors="replace")
        n_lines = len(text.splitlines())
        if n_lines > RECOMMENDED_MAX_LINES and parent is None:
            result.warnings.append(f"{path} has {n_lines} lines; keep instruction files under {RECOMMENDED_MAX_LINES} lines")
        result.files.append(InstructionFile(path, text, level, parent))
        if depth >= MAX_IMPORT_DEPTH:
            if _IMPORT.search(text):
                result.warnings.append(f"import depth limit ({MAX_IMPORT_DEPTH}) reached in {path}")
            return
        for m in _IMPORT.finditer(text):
            raw = m.group(1)
            target = Path(raw).expanduser() if raw.startswith("~") else (path.parent / raw)
            target = target.resolve()
            if not target.is_file():
                continue
            if self.project_root not in target.parents:
                approved = self._approved.get(target)
                if approved is None:
                    approved = self._approved[target] = bool(self.approve_external(target))
                if not approved:
                    result.warnings.append(f"external import {target} not approved")
                    continue
            self._load_file(target, level, path, depth + 1, result, seen)
