"""The tool kernel: ``read``, ``write``, ``edit``, ``bash``, ``grep``, ``glob``.

Research basis (``reports/SOTA agentic harness reference.md`` §5.7 "Recommended core tool
kernel", steps 3-5; build plan "a five-tool kernel"): every leading harness converged on a shell,
file read/search, one structured edit primitive and whole-file write; Pi ships four tools and
mini-swe-agent only bash. Defaults:

=========  ==========================================================================================
``read``   absolute ``path``, ``offset``/``limit`` (2,000 lines), 6-char right-aligned line numbers +
           tab (memory ``view`` / ``text_editor`` format), long lines cut at 2,000 chars, a
           100,000-char page with "call again with offset=N" continuation, images as image blocks.
``write``  create/overwrite a whole file; absolute path; read-before-overwrite; checkpoint; atomic.
``edit``   ``old_string`` -> ``new_string`` with **strict uniqueness**: zero matches and multiple
           matches are errors with actionable messages ("add more surrounding lines"); optional
           ``replace_all``; CRLF-tolerant; edit-loop warning after 5 edits to one file.
``bash``   stateless ``bash -c`` through a pluggable :class:`~harness_tools.runner.CommandRunner`;
           timeout 120 s default / 600 s max; output capped at 30,000 chars (head+tail, offloaded).
``grep``   Python-regex search, read-only, ``files_with_matches``/``content``/``count``,
           ``head_limit`` 100 with ``offset`` pagination, deterministic (path-sorted) results.
``glob``   file-pattern search (``**``, ``{a,b}``), read-only, ``limit`` 100 + ``offset``.
=========  ==========================================================================================

Read-only tools (``read``, ``grep``, ``glob``) are ``concurrency_safe`` and run in parallel; the
mutating ones run one at a time. Descriptions live in :mod:`harness_tools.descriptions` (versioned
next to the code; their SHA-256 is recorded in ``ToolSpec.metadata["description_sha256"]``).
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

from harness_core.types import ToolOutcome, ToolSpec

from . import descriptions as D
from .errors import ToolInputError, WorkspaceError
from .registry import Tool, ToolInvocation, ToolRegistry
from .runner import CommandRequest, CommandRunner, LocalSubprocessRunner, as_command_runner
from .workspace import Workspace

READ_DEFAULT_LIMIT = 2_000
READ_MAX_LINE_CHARS = 2_000
READ_PAGE_CHARS = 100_000
IMAGE_MAX_BYTES = 5 * 1024 * 1024
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
BASH_DEFAULT_TIMEOUT_S = 120.0
BASH_MAX_TIMEOUT_S = 600.0
BASH_MAX_OUTPUT_CHARS = 30_000
SEARCH_DEFAULT_LIMIT = 100
SEARCH_MAX_LIMIT = 1_000
GREP_MAX_FILE_BYTES = 2 * 1024 * 1024
GREP_MAX_PATTERN = 1_000
GREP_MAX_LINE_CHARS = 500
WALK_MAX_FILES = 100_000
EDIT_LOOP_THRESHOLD = 5

Profile = Literal["generic", "anthropic"]
#: Tool names per model profile (§5.7 step 1). ``anthropic`` uses Claude Code's names, which double
#: as permission-rule and hook-matcher keys.
PROFILE_NAMES: dict[str, dict[str, str]] = {
    "generic": {k: k for k in ("read", "write", "edit", "bash", "grep", "glob")},
    "anthropic": {"read": "Read", "write": "Write", "edit": "Edit", "bash": "Bash", "grep": "Grep", "glob": "Glob"},
}


def _is_binary(sample: bytes) -> bool:
    return b"\x00" in sample


def _split_lines(text: str) -> list[str]:
    lines = text.split("\n")
    if text.endswith("\n"):
        lines = lines[:-1]
    return [ln[:-1] if ln.endswith("\r") else ln for ln in lines]


def _number(lines: list[str], start: int) -> list[str]:
    out = []
    for i, line in enumerate(lines, start=start):
        if len(line) > READ_MAX_LINE_CHARS:
            line = line[:READ_MAX_LINE_CHARS] + f"... [line truncated, {len(line):,} chars]"
        out.append(f"{i:6d}\t{line}")
    return out


# ----------------------------------------------------------------------------- glob matching


def expand_braces(pattern: str) -> list[str]:
    """``"*.{ts,tsx}"`` -> ``["*.ts", "*.tsx"]`` (nested braces supported)."""
    m = re.search(r"\{([^{}]*)\}", pattern)
    if not m:
        return [pattern]
    out: list[str] = []
    for alt in m.group(1).split(","):
        out.extend(expand_braces(pattern[: m.start()] + alt + pattern[m.end() :]))
    return out


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a ``**``-aware glob (POSIX separators) into an anchored regex."""
    i, out = 0, []
    while i < len(pattern):
        c = pattern[i]
        if c == "*":
            if pattern[i : i + 3] == "**/":
                out.append("(?:.*/)?")
                i += 3
                continue
            if pattern[i : i + 2] == "**":
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = pattern.find("]", i + 1)
            if j == -1:
                out.append(re.escape(c))
            else:
                body = pattern[i + 1 : j].replace("\\", "\\\\")
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append(f"[{body}]")
                i = j
        else:
            out.append(re.escape(c))
        i += 1
    return re.compile("^" + "".join(out) + "$")


@dataclass(frozen=True)
class GlobMatcher:
    """Matches workspace-relative POSIX paths. Patterns without ``/`` match the basename."""

    patterns: tuple[str, ...]

    @classmethod
    def compile(cls, pattern: str) -> "GlobMatcher":
        return cls(tuple(expand_braces(pattern.strip())))

    def __post_init__(self) -> None:
        object.__setattr__(self, "_regexes", [(("/" in p), _glob_to_regex(p.lstrip("/"))) for p in self.patterns])

    def matches(self, rel: str) -> bool:
        base = rel.rsplit("/", 1)[-1]
        for has_slash, rx in self._regexes:  # type: ignore[attr-defined]
            if rx.match(rel if has_slash else base):
                return True
        return False


# ----------------------------------------------------------------------------- the kernel


class Kernel:
    """Implements the kernel tools against one :class:`Workspace` and one :class:`CommandRunner`."""

    def __init__(
        self,
        workspace: Workspace,
        runner: CommandRunner | None = None,
        *,
        profile: Profile = "generic",
        bash_open_world: bool = False,
        bash_env: dict[str, str] | None = None,
        edit_loop_threshold: int = EDIT_LOOP_THRESHOLD,
    ) -> None:
        if profile not in PROFILE_NAMES:
            raise ValueError(f"unknown profile {profile!r}; choose one of {sorted(PROFILE_NAMES)}")
        self.ws = workspace
        self.runner: CommandRunner = as_command_runner(runner) if runner is not None else LocalSubprocessRunner()
        self.names = PROFILE_NAMES[profile]
        self.bash_open_world = bash_open_world
        self.bash_env = dict(bash_env or {})
        self.edit_loop_threshold = edit_loop_threshold

    # ------------------------------------------------------------------ read
    def read(self, inp: dict[str, Any], inv: ToolInvocation) -> ToolOutcome:
        path = self.ws.resolve(inp["path"])
        shown = self.ws.display(path)
        if not path.exists():
            raise WorkspaceError(f"{shown} does not exist. Use glob to find the right path.")
        if path.is_dir():
            raise WorkspaceError(f"{shown} is a directory. Use glob (pattern '*', path '{shown}') to list it.")
        size = path.stat().st_size
        media = IMAGE_TYPES.get(path.suffix.lower())
        if media:
            if size > IMAGE_MAX_BYTES:
                raise ToolInputError(f"image {shown} is {size:,} bytes; the limit is {IMAGE_MAX_BYTES:,} bytes")
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            self.ws.files.record(inv.ctx.session_id, path)
            return ToolOutcome([{"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}], metadata={"bytes": size})
        if size > self.ws.max_file_bytes:
            raise ToolInputError(f"{shown} is {size:,} bytes (limit {self.ws.max_file_bytes:,}). Use grep to search it, or bash with head/tail/sed -n.")
        raw = path.read_bytes()
        if _is_binary(raw[:8192]):
            raise ToolInputError(f"{shown} looks like a binary file; read only shows text. Inspect it with bash (e.g. `file`, `xxd | head`).")
        self.ws.files.record(inv.ctx.session_id, path)
        lines = _split_lines(raw.decode("utf-8", errors="replace"))
        total = len(lines)
        if total == 0:
            return ToolOutcome(f"{shown} is empty (0 lines).")
        offset = int(inp.get("offset", 1))
        limit = int(inp.get("limit", READ_DEFAULT_LIMIT))
        if offset > total:
            raise ToolInputError(f"offset {offset} is past the end of {shown} ({total} lines). Use an offset between 1 and {total}.")
        window = lines[offset - 1 : offset - 1 + limit]
        numbered = _number(window, offset)
        out: list[str] = []
        used = 0
        for row in numbered:
            if used + len(row) + 1 > READ_PAGE_CHARS and out:
                break
            out.append(row)
            used += len(row) + 1
        last = offset + len(out) - 1
        text = "\n".join(out)
        if last < total:
            text += f"\n\n[Showing lines {offset}-{last} of {total}. Call {self.names['read']} with offset={last + 1} to continue.]"
        return ToolOutcome(text, metadata={"lines": len(out), "total_lines": total})

    # ------------------------------------------------------------------ write
    def write(self, inp: dict[str, Any], inv: ToolInvocation) -> str:
        path = self.ws.resolve(inp["path"], for_write=True)
        content: str = inp["content"]
        sid = inv.ctx.session_id
        with self.ws.lock_for(path):
            existed = path.exists()
            if existed and path.is_dir():
                raise WorkspaceError(f"{path} is a directory")
            self.ws.check_writable(sid, path)
            self.ws.write_bytes(sid, path, content.encode("utf-8"))
        n_lines = len(_split_lines(content))
        verb = "Overwrote" if existed else "Created"
        return f"{verb} {self.ws.display(path)} ({n_lines} lines, {len(content.encode('utf-8')):,} bytes)."

    # ------------------------------------------------------------------ edit
    def edit(self, inp: dict[str, Any], inv: ToolInvocation) -> str:
        path = self.ws.resolve(inp["path"], for_write=True)
        shown = self.ws.display(path)
        old: str = inp["old_string"]
        new: str = inp["new_string"]
        replace_all = bool(inp.get("replace_all", False))
        sid = inv.ctx.session_id
        if old == "":
            raise ToolInputError(f"old_string is empty. To create a file use {self.names['write']}; to edit, quote the exact existing text.")
        if old == new:
            raise ToolInputError("old_string and new_string are identical; nothing would change.")
        with self.ws.lock_for(path):
            if not path.exists():
                raise WorkspaceError(f"{shown} does not exist. Use {self.names['write']} to create it.")
            if path.is_dir():
                raise WorkspaceError(f"{shown} is a directory")
            self.ws.check_writable(sid, path)
            raw = path.read_bytes()
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ToolInputError(f"{shown} is not valid UTF-8 text and cannot be edited with {self.names['edit']}.") from exc
            count = content.count(old)
            if count == 0 and "\r\n" in content and "\n" in old:
                crlf_old, crlf_new = old.replace("\r\n", "\n").replace("\n", "\r\n"), new.replace("\r\n", "\n").replace("\n", "\r\n")
                if content.count(crlf_old):
                    old, new, count = crlf_old, crlf_new, content.count(crlf_old)
            if count == 0:
                hint = ""
                if old.strip() and old.strip() in content:
                    hint = " A match exists with different leading/trailing whitespace; copy the text exactly."
                elif re.search(r"^\s*\d+\t", old, re.M):
                    hint = f" old_string seems to include line-number prefixes from {self.names['read']}; remove them."
                raise ToolInputError(
                    f"old_string was not found in {shown}. It must match the file exactly, including whitespace and indentation.{hint} "
                    f"Read the file again to copy the current text."
                )
            if count > 1 and not replace_all:
                raise ToolInputError(
                    f"old_string matches {count} places in {shown}. Add more surrounding lines to make it unique, "
                    "or set replace_all=true to change every occurrence."
                )
            first = content.index(old)
            updated = content.replace(old, new) if replace_all else content.replace(old, new, 1)
            self.ws.write_bytes(sid, path, updated.encode("utf-8"))
        start_line = content.count("\n", 0, first) + 1
        new_lines = _split_lines(updated)
        span = max(new.count("\n") + 1, 1)
        lo, hi = max(start_line - 3, 1), min(start_line + span + 2, len(new_lines))
        snippet = "\n".join(_number(new_lines[lo - 1 : hi], lo))
        what = f"Replaced {count} occurrences" if replace_all and count > 1 else "Applied 1 edit"
        msg = f"{what} in {shown}. Snippet of the result:\n{snippet}"
        n = self.ws.files.count_edit(sid, path)
        if n >= self.edit_loop_threshold:
            msg += (
                f"\n\nNote: this is edit #{n} to {shown} in this session. If you are going back and forth, step back, "
                "re-read the file and reconsider the approach before editing again."
            )
        return msg

    # ------------------------------------------------------------------ bash
    def bash(self, inp: dict[str, Any], inv: ToolInvocation) -> ToolOutcome:
        command: str = inp["command"]
        if not command.strip():
            raise ToolInputError("command is empty")
        timeout = float(inp.get("timeout", BASH_DEFAULT_TIMEOUT_S))
        result = self.runner.run(
            CommandRequest(command=command, cwd=str(self.ws.root), timeout_s=timeout, env=self.bash_env or None)
        )
        parts: list[str] = []
        if result.stdout:
            parts.append(result.stdout.rstrip("\n"))
        if result.stderr:
            parts.append("[stderr]\n" + result.stderr.rstrip("\n"))
        if result.output_truncated:
            parts.append(f"[capture limit reached: {result.stdout_bytes + result.stderr_bytes:,} bytes produced; redirect to a file and grep it]")
        if result.violations:
            parts.append("<sandbox_violations>\n" + "\n".join(result.violations) + "\n</sandbox_violations>")
        if result.timed_out:
            parts.append(
                f"[timed out after {timeout:g}s; the process group was killed. Use a longer timeout (max {BASH_MAX_TIMEOUT_S:g}s), "
                "or start long jobs in the background with output redirected to a file]"
            )
        else:
            parts.append(f"[exit code: {result.exit_code}; {result.duration_s:.2f}s]")
        meta = {
            "exit_code": result.exit_code,
            "duration_ms": round(result.duration_s * 1000, 1),
            "stdout_bytes": result.stdout_bytes,
            "stderr_bytes": result.stderr_bytes,
            "timed_out": result.timed_out,
        }
        if result.sandbox_id:
            meta["sandbox_id"] = result.sandbox_id
        if result.violations:
            meta["sandbox_violations"] = list(result.violations)
        if self.bash_open_world:
            meta["untrusted"] = True
        return ToolOutcome("\n".join(parts), is_error=not result.ok, metadata=meta)

    # ------------------------------------------------------------------ search helpers
    def _search_root(self, inp: dict[str, Any]) -> Path:
        raw = inp.get("path")
        root = self.ws.resolve(raw) if raw else self.ws.root
        if not root.exists():
            raise WorkspaceError(f"{root} does not exist")
        return root

    def _walk(self, base: Path) -> Iterator[Path]:
        if base.is_file():
            yield base
            return
        seen = 0
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if d not in self.ws.ignored_dirs)
            for name in sorted(filenames):
                seen += 1
                if seen > WALK_MAX_FILES:
                    return
                yield Path(dirpath) / name

    def _inside(self, p: Path) -> bool:
        try:
            p.resolve().relative_to(self.ws.root)
            return True
        except (ValueError, OSError):
            return False

    @staticmethod
    def _page(entries: list[str], inp: dict[str, Any], what: str, narrow: str) -> str:
        offset = int(inp.get("offset", 0))
        limit = int(inp.get("head_limit", inp.get("limit", SEARCH_DEFAULT_LIMIT)))
        total = len(entries)
        if total == 0:
            return f"No {what} found."
        page = entries[offset : offset + limit]
        if not page:
            return f"No more {what}: offset {offset} is past the {total} available."
        text = "\n".join(page)
        end = offset + len(page)
        if end < total or offset:
            text += f"\n\n[Showing {what} {offset + 1}-{end} of {total}."
            text += f" Use offset={end} for the next page, or {narrow}.]" if end < total else "]"
        return text

    # ------------------------------------------------------------------ grep
    def grep(self, inp: dict[str, Any], inv: ToolInvocation) -> str:
        pattern: str = inp["pattern"]
        if len(pattern) > GREP_MAX_PATTERN:
            raise ToolInputError(f"pattern is {len(pattern)} characters; the limit is {GREP_MAX_PATTERN}")
        flags = re.IGNORECASE if inp.get("case_insensitive") else 0
        try:
            rx = re.compile(pattern, flags)
        except re.error as exc:
            raise ToolInputError(f"invalid regular expression {pattern!r}: {exc}. Escape literal characters such as ( ) [ ] {{ }} . * + ?") from exc
        mode = inp.get("output_mode", "files_with_matches")
        ctx_n = int(inp.get("context", 0))
        matcher = GlobMatcher.compile(inp["glob"]) if inp.get("glob") else None
        base = self._search_root(inp)
        entries: list[str] = []
        for f in self._walk(base):
            rel = f.relative_to(self.ws.root).as_posix() if self._inside(f) else None
            if rel is None or (matcher and not matcher.matches(rel)):
                continue
            try:
                if f.stat().st_size > GREP_MAX_FILE_BYTES:
                    continue
                raw = f.read_bytes()
            except OSError:
                continue
            if _is_binary(raw[:8192]):
                continue
            lines = _split_lines(raw.decode("utf-8", errors="replace"))
            hits = [i for i, line in enumerate(lines) if rx.search(line)]
            hit_set = set(hits)
            if not hits:
                continue
            shown = self.ws.display(f.resolve())
            if mode == "files_with_matches":
                entries.append(shown)
            elif mode == "count":
                entries.append(f"{shown}:{len(hits)}")
            else:
                for group in _context_groups(hits, ctx_n, len(lines)):
                    rows = []
                    for i in group:
                        sep = ":" if i in hit_set else "-"
                        line = lines[i] if len(lines[i]) <= GREP_MAX_LINE_CHARS else lines[i][:GREP_MAX_LINE_CHARS] + "..."
                        rows.append(f"{shown}{sep}{i + 1}{sep}{line}")
                    entries.append("\n".join(rows) if ctx_n == 0 else "\n".join(rows) + "\n--")
        what = {"files_with_matches": "matching files", "count": "file counts", "content": "matches"}[mode]
        return self._page(entries, inp, what, "narrow the pattern, path or glob")

    # ------------------------------------------------------------------ glob
    def glob(self, inp: dict[str, Any], inv: ToolInvocation) -> str:
        pattern: str = inp["pattern"]
        matcher = GlobMatcher.compile(pattern)
        base = self._search_root(inp)
        if not base.is_dir():
            raise WorkspaceError(f"{base} is not a directory")
        found: list[Path] = []
        for f in self._walk(base):
            rel_base = f.relative_to(base).as_posix()
            if matcher.matches(rel_base) and self._inside(f):
                found.append(f)
        if inp.get("sort", "path") == "mtime":
            found.sort(key=lambda p: (-_mtime(p), str(p)))
        entries = [self.ws.display(p.resolve()) for p in found]
        return self._page(entries, inp, "files", "use a more specific pattern or path")

    # ------------------------------------------------------------------ specs
    def tools(self, include: tuple[str, ...] = ("read", "write", "edit", "bash", "grep", "glob")) -> list[Tool]:
        root = str(self.ws.root)
        n = self.names
        abs_path = {"type": "string", "description": f"Absolute path, e.g. {root}/src/app.py"}
        defs: dict[str, Tool] = {
            "read": Tool(
                spec=ToolSpec(
                    n["read"],
                    D.READ.format(edit=n["edit"], limit=READ_DEFAULT_LIMIT),
                    {
                        "type": "object",
                        "properties": {
                            "path": abs_path,
                            "offset": {"type": "integer", "minimum": 1, "default": 1, "description": "1-based line number to start from."},
                            "limit": {
                                "type": "integer", "minimum": 1, "maximum": 10_000, "default": READ_DEFAULT_LIMIT,
                                "description": "Maximum number of lines to return.",
                            },
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                    read_only=True,
                ),
                handler=self.read,
                idempotent=True,
                truncation_hint=f"Call {n['read']} with offset/limit to page through the file.",
            ),
            "write": Tool(
                spec=ToolSpec(
                    n["write"],
                    D.WRITE.format(edit=n["edit"], read=n["read"]),
                    {
                        "type": "object",
                        "properties": {"path": abs_path, "content": {"type": "string", "description": "The complete new file content."}},
                        "required": ["path", "content"],
                        "additionalProperties": False,
                    },
                ),
                handler=self.write,
                idempotent=True,
                destructive=True,
            ),
            "edit": Tool(
                spec=ToolSpec(
                    n["edit"],
                    D.EDIT.format(read=n["read"], write=n["write"]),
                    {
                        "type": "object",
                        "properties": {
                            "path": abs_path,
                            "old_string": {"type": "string", "description": "Exact existing text to replace (no line-number prefixes)."},
                            "new_string": {"type": "string", "description": "Replacement text (must differ from old_string)."},
                            "replace_all": {"type": "boolean", "default": False, "description": "Replace every occurrence instead of exactly one."},
                        },
                        "required": ["path", "old_string", "new_string"],
                        "additionalProperties": False,
                    },
                ),
                handler=self.edit,
                destructive=True,
            ),
            "bash": Tool(
                spec=ToolSpec(
                    n["bash"],
                    D.BASH.format(root=root, read=n["read"], edit=n["edit"], grep=n["grep"], glob=n["glob"], max_timeout=int(BASH_MAX_TIMEOUT_S)),
                    {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "The shell command to run (bash -c)."},
                            "timeout": {
                                "type": "number", "exclusiveMinimum": 0, "maximum": BASH_MAX_TIMEOUT_S, "default": BASH_DEFAULT_TIMEOUT_S,
                                "description": f"Timeout in seconds (default {int(BASH_DEFAULT_TIMEOUT_S)}, max {int(BASH_MAX_TIMEOUT_S)}).",
                            },
                            "description": {"type": "string", "description": "5-10 word summary of what the command does, for the user."},
                        },
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                    open_world=self.bash_open_world,
                ),
                handler=self.bash,
                timeout_s=BASH_MAX_TIMEOUT_S + 30,  # the runner enforces the per-call timeout itself
                max_output_tokens=BASH_MAX_OUTPUT_CHARS // 4,
                destructive=True,
                truncation_hint="Re-run with a narrower command (| head, | tail, grep, or redirect to a file and grep it).",
            ),
            "grep": Tool(
                spec=ToolSpec(
                    n["grep"],
                    D.GREP.format(limit=SEARCH_DEFAULT_LIMIT, bash=n["bash"]),
                    {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "minLength": 1, "description": "Python regular expression, e.g. 'def \\w+_test' or 'TODO|FIXME'."},
                            "path": {**abs_path, "description": f"Absolute file or directory to search (default: {root})."},
                            "glob": {"type": "string", "description": "Only search files matching this glob, e.g. '*.py' or 'src/**/*.{ts,tsx}'."},
                            "output_mode": {
                                "type": "string", "enum": ["files_with_matches", "content", "count"], "default": "files_with_matches",
                                "description": "files_with_matches: paths only; content: matching lines as path:line:text; count: matches per file.",
                            },
                            "case_insensitive": {"type": "boolean", "default": False, "description": "Match regardless of letter case."},
                            "context": {"type": "integer", "minimum": 0, "maximum": 10, "default": 0, "description": "Lines of context around matches (content mode)."},
                            "head_limit": {
                                "type": "integer", "minimum": 1, "maximum": SEARCH_MAX_LIMIT, "default": SEARCH_DEFAULT_LIMIT,
                                "description": "Maximum results per page.",
                            },
                            "offset": {"type": "integer", "minimum": 0, "default": 0, "description": "Skip this many results (pagination)."},
                        },
                        "required": ["pattern"],
                        "additionalProperties": False,
                    },
                    read_only=True,
                ),
                handler=self.grep,
                idempotent=True,
                truncation_hint="Narrow the pattern, path or glob, or page with offset/head_limit.",
            ),
            "glob": Tool(
                spec=ToolSpec(
                    n["glob"],
                    D.GLOB.format(limit=SEARCH_DEFAULT_LIMIT, grep=n["grep"]),
                    {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "minLength": 1, "description": "Glob such as '**/*.py', 'src/*.{ts,tsx}' or 'README*'."},
                            "path": {**abs_path, "description": f"Absolute directory to search in (default: {root})."},
                            "sort": {"type": "string", "enum": ["path", "mtime"], "default": "path", "description": "path (alphabetical) or mtime (newest first)."},
                            "limit": {
                                "type": "integer", "minimum": 1, "maximum": SEARCH_MAX_LIMIT, "default": SEARCH_DEFAULT_LIMIT,
                                "description": "Maximum paths per page.",
                            },
                            "offset": {"type": "integer", "minimum": 0, "default": 0, "description": "Skip this many paths (pagination)."},
                        },
                        "required": ["pattern"],
                        "additionalProperties": False,
                    },
                    read_only=True,
                ),
                handler=self.glob,
                idempotent=True,
            ),
        }
        unknown = set(include) - set(defs)
        if unknown:
            raise ValueError(f"unknown kernel tools {sorted(unknown)}")
        out = []
        for key in include:
            t = defs[key]
            t.spec.metadata["description_sha256"] = hashlib.sha256(t.spec.description.encode("utf-8")).hexdigest()
            t.spec.metadata["kernel_tool"] = key
            out.append(t)
        return out


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _context_groups(hits: list[int], n: int, total: int) -> list[list[int]]:
    """Merge hit lines with ``n`` lines of context into contiguous groups."""
    groups: list[list[int]] = []
    for h in hits:
        lo, hi = max(h - n, 0), min(h + n, total - 1)
        if groups and lo <= groups[-1][-1] + 1:
            groups[-1].extend(range(groups[-1][-1] + 1, hi + 1))
        else:
            groups.append(list(range(lo, hi + 1)))
    return groups


def register_kernel(
    registry: ToolRegistry,
    workspace: Workspace,
    *,
    runner: CommandRunner | None = None,
    profile: Profile = "generic",
    include: tuple[str, ...] = ("read", "write", "edit", "bash", "grep", "glob"),
    offload: bool = True,
    bash_open_world: bool = False,
    bash_env: dict[str, str] | None = None,
) -> Kernel:
    """Register the kernel tools on ``registry``.

    ``offload=True`` points the registry's output policy at ``<workspace>/.harness/tool-outputs``
    (when it has no offload directory yet) so oversize results are saved and paged, not dropped.
    """
    kernel = Kernel(workspace, runner, profile=profile, bash_open_world=bash_open_world, bash_env=bash_env)
    for tool in kernel.tools(include):
        registry.register(tool)
    if offload and registry.output_policy.offload_dir is None:
        registry.output_policy = dataclasses.replace(registry.output_policy, offload_dir=workspace.offload_dir)
    return kernel


__all__ = [
    "BASH_DEFAULT_TIMEOUT_S",
    "BASH_MAX_OUTPUT_CHARS",
    "BASH_MAX_TIMEOUT_S",
    "GlobMatcher",
    "Kernel",
    "PROFILE_NAMES",
    "Profile",
    "READ_DEFAULT_LIMIT",
    "SEARCH_DEFAULT_LIMIT",
    "expand_braces",
    "register_kernel",
]
