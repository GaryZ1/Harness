"""Client-side handler for the ``memory_20250818`` tool (write channel A: the main agent).

Implements the six commands with the same parameter names as Anthropic's API memory tool
(``view``, ``create``, ``str_replace``, ``insert``, ``delete``, ``rename``) against a
:class:`~harness_memory.store.FileMemoryStore`. The ``/memories`` prefix the model sees is mapped
onto the store root.

Safety rails (all enforced in code):

* path sandbox, root protected from delete/rename;
* ``view`` output truncated at 16,000 characters; files over 999,999 lines rejected;
* per-file cap from the store (100 kB default); ``MEMORY.md`` cap of 200 lines / 25 KB
  (write accepted, error returned past the cap; reminder near it);
* secrets redacted, ``modified``/``tainted`` stamped on frontmatter files;
* optional *restricted mode* (Codex-style): only new notes under ``inbox/`` may be created.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .errors import (
    AccessDeniedError,
    AlreadyExistsError,
    MemoryError_,
    NotFoundError,
    PathSecurityError,
    ValidationError,
)
from .index import INDEX_FILE, IndexHealth, index_status
from .store import ABSENT, FileMemoryStore

MEMORY_PREFIX = "/memories"
VIEW_CHAR_LIMIT = 16_000
MAX_VIEW_LINES = 999_999
TOOL_DEFINITION: dict[str, str] = {"type": "memory_20250818", "name": "memory"}

#: Optional, opt-in guidance the docs suggest adding to the system prompt to reduce clutter.
ORGANISATION_GUIDANCE = (
    "Keep your memory directory up-to-date, coherent and organized. Do not create new files unless "
    "necessary. Keep MEMORY.md to one line per memory and put detail in topic files. Never store "
    "secrets, credentials or anything derivable from the codebase or git history."
)


@dataclass(frozen=True)
class ToolResult:
    content: str
    is_error: bool = False

    def to_block(self, tool_use_id: str) -> dict[str, Any]:
        """Render as an Anthropic ``tool_result`` content block."""
        block: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": self.content}
        if self.is_error:
            block["is_error"] = True
        return block


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "K", "M", "G"):
        if size < 1024 or unit == "G":
            return f"{int(size)}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{n}B"  # pragma: no cover


class MemoryToolHandler:
    """Execute ``memory`` tool calls. Thread-safe; one instance per (session, store)."""

    def __init__(
        self,
        store: FileMemoryStore,
        *,
        restricted: bool = False,
        create_overwrites: bool = False,
        on_write: Callable[[str, str], None] | None = None,
    ) -> None:
        """
        :param restricted: Codex-style mode - the agent may only *create* new ``inbox/*.md`` notes.
        :param create_overwrites: the reference implementation errors when ``create`` targets an
            existing file; models often assume it overwrites. Set ``True`` to overwrite.
        :param on_write: callback ``(command, path)`` after each successful mutation (telemetry, hooks).
        """
        self.store = store
        self.restricted = restricted
        self.create_overwrites = create_overwrites
        self.on_write = on_write

    # ------------------------------------------------------------------ entry point
    def handle(self, tool_input: dict[str, Any]) -> ToolResult:
        """Dispatch one ``tool_use.input`` payload and never raise (errors become ``is_error`` results)."""
        command = tool_input.get("command")
        handler = {
            "view": self._view,
            "create": self._create,
            "str_replace": self._str_replace,
            "insert": self._insert,
            "delete": self._delete,
            "rename": self._rename,
        }.get(command or "")
        if handler is None:
            return ToolResult(f"Error: unknown command {command!r}", True)
        try:
            return handler(tool_input)
        except KeyError as exc:
            return ToolResult(f"Error: missing required parameter {exc.args[0]!r} for {command}", True)
        except (MemoryError_, ValueError, TypeError) as exc:
            return ToolResult(f"Error: {exc}", True)

    # ------------------------------------------------------------------ path mapping
    def _rel(self, path: str) -> str:
        if not isinstance(path, str) or not path:
            raise ValidationError("path must be a non-empty string")
        p = path.strip()
        if p == MEMORY_PREFIX or p == MEMORY_PREFIX + "/":
            return ""
        if not p.startswith(MEMORY_PREFIX + "/"):
            raise PathSecurityError(f"path must start with {MEMORY_PREFIX}/: {path!r}")
        return self.store.sandbox.normalise(p[len(MEMORY_PREFIX) + 1 :])

    @staticmethod
    def _display(rel: str) -> str:
        return f"{MEMORY_PREFIX}/{rel}" if rel else MEMORY_PREFIX

    def _check_mutation(self, command: str, rel: str) -> None:
        if not self.store.writable:
            raise AccessDeniedError("this memory directory is mounted read-only")
        if self.restricted and not (command == "create" and rel.startswith("inbox/") and rel.endswith(".md")):
            raise AccessDeniedError(
                "memory is in restricted mode: you may only create new notes under /memories/inbox/ "
                "(e.g. /memories/inbox/<timestamp>-<slug>.md) when the user explicitly asks you to remember something"
            )

    def _write(self, rel: str, content: str, *, expected: str | None) -> str:
        """Write with stamping and return any index warning to append to the result."""
        content = self.store.stamp(content)
        self.store.write(rel, content, expected_sha256=expected)
        if rel == INDEX_FILE:
            status = index_status(content)
            if status.health is not IndexHealth.OK:
                return "\n" + (status.message or "")
        return ""

    def _notify(self, command: str, rel: str) -> None:
        if self.on_write:
            self.on_write(command, rel)

    # ------------------------------------------------------------------ commands
    def _view(self, inp: dict[str, Any]) -> ToolResult:
        rel = self._rel(inp["path"])
        display = self._display(rel)
        if not self.store.exists(rel):
            raise NotFoundError(f"The path {display} does not exist. Please provide a valid path.")
        if self.store.is_dir(rel):
            return ToolResult(self._truncate(self._listing(rel, display)))
        content = self.store.read(rel).content
        lines = content.split("\n")
        if content.endswith("\n"):
            lines = lines[:-1]
        if len(lines) > MAX_VIEW_LINES:
            raise ValidationError(f"File {display} exceeds maximum line limit of {MAX_VIEW_LINES} lines.")
        start = 1
        view_range = inp.get("view_range")
        if view_range is not None:
            if not (isinstance(view_range, (list, tuple)) and len(view_range) == 2):
                raise ValidationError("view_range must be [start, end]")
            start, end = int(view_range[0]), int(view_range[1])
            if start < 1 or start > max(len(lines), 1):
                raise ValidationError(f"view_range start {start} is out of range 1..{len(lines)}")
            end = len(lines) if end == -1 else end
            if end < start:
                raise ValidationError("view_range end must be >= start (or -1 for end of file)")
            lines = lines[start - 1 : end]
        numbered = "\n".join(f"{i:6d}\t{line}" for i, line in enumerate(lines, start=start))
        return ToolResult(self._truncate(f"Here's the content of {display} with line numbers:\n{numbered}"))

    def _listing(self, rel: str, display: str) -> str:
        base = self.store.sandbox.resolve(rel)
        rows: list[str] = []

        def size_of(p) -> int:
            if p.is_file():
                return p.stat().st_size
            return sum(
                c.stat().st_size for c in p.rglob("*") if c.is_file() and not any(part.startswith(".") for part in c.relative_to(p).parts)
            )

        rows.append(f"{_human_size(size_of(base))}\t{display}")

        def walk(path, depth: int) -> None:
            for child in sorted(path.iterdir()):
                if child.name.startswith(".") or child.name == "node_modules":
                    continue
                child_rel = self.store.sandbox.relative(child)
                rows.append(f"{_human_size(size_of(child))}\t{self._display(child_rel)}{'/' if child.is_dir() else ''}")
                if child.is_dir() and depth < 2:
                    walk(child, depth + 1)

        walk(base, 1)
        return (
            f"Here're the files and directories up to 2 levels deep in {display}, excluding hidden items and node_modules:\n"
            + "\n".join(rows)
        )

    @staticmethod
    def _truncate(text: str) -> str:
        if len(text) <= VIEW_CHAR_LIMIT:
            return text
        note = f"\n... [output truncated at {VIEW_CHAR_LIMIT} characters; use view_range to read the rest]"
        return text[: VIEW_CHAR_LIMIT - len(note)] + note

    def _create(self, inp: dict[str, Any]) -> ToolResult:
        rel = self._rel(inp["path"])
        text = inp["file_text"]
        if not isinstance(text, str):
            raise ValidationError("file_text must be a string")
        self._check_mutation("create", rel)
        if rel == "":
            raise PathSecurityError("cannot create the memory root")
        exists = self.store.exists(rel)
        if exists and (not self.create_overwrites or self.restricted):
            raise AlreadyExistsError(f"File {self._display(rel)} already exists")
        warning = self._write(rel, text, expected=None if exists else ABSENT)
        self._notify("create", rel)
        return ToolResult(f"File created successfully at: {self._display(rel)}{warning}", bool(warning) and "Error" in warning)

    def _str_replace(self, inp: dict[str, Any]) -> ToolResult:
        rel = self._rel(inp["path"])
        old = inp["old_str"]
        new = inp.get("new_str") or ""
        self._check_mutation("str_replace", rel)
        if not self.store.exists(rel) or self.store.is_dir(rel):
            raise NotFoundError(f"The path {self._display(rel)} does not exist. Please provide a valid path.")
        current = self.store.read(rel)
        count = current.content.count(old) if old else 0
        if count == 0:
            raise ValidationError(f"No replacement was performed, old_str `{old}` did not appear verbatim in {self._display(rel)}.")
        if count > 1:
            lines = [str(i + 1) for i, ln in enumerate(current.content.split("\n")) if old.split("\n")[0] in ln]
            raise ValidationError(
                f"No replacement was performed. Multiple occurrences of old_str `{old}` in lines: {', '.join(lines)}. Please ensure it is unique"
            )
        updated = current.content.replace(old, new, 1)
        warning = self._write(rel, updated, expected=current.sha256)
        self._notify("str_replace", rel)
        return ToolResult(f"The memory file has been edited.{warning}", "Error" in warning)

    def _insert(self, inp: dict[str, Any]) -> ToolResult:
        rel = self._rel(inp["path"])
        line_no = int(inp["insert_line"])
        text = inp["insert_text"]
        self._check_mutation("insert", rel)
        if not self.store.exists(rel) or self.store.is_dir(rel):
            raise NotFoundError(f"The path {self._display(rel)} does not exist")
        current = self.store.read(rel)
        lines = current.content.split("\n")
        trailing_newline = current.content.endswith("\n")
        if trailing_newline:
            lines = lines[:-1]
        if line_no < 0 or line_no > len(lines):
            raise ValidationError(
                f"Invalid `insert_line` parameter: {line_no}. It should be within the range of lines of the file: [0, {len(lines)}]"
            )
        new_lines = text.split("\n")
        if text.endswith("\n"):
            new_lines = new_lines[:-1]
        lines[line_no:line_no] = new_lines
        updated = "\n".join(lines) + ("\n" if trailing_newline or text.endswith("\n") else "")
        warning = self._write(rel, updated, expected=current.sha256)
        self._notify("insert", rel)
        return ToolResult(f"The file {self._display(rel)} has been edited.{warning}", "Error" in warning)

    def _delete(self, inp: dict[str, Any]) -> ToolResult:
        rel = self._rel(inp["path"])
        if rel == "":
            raise PathSecurityError(f"Cannot delete the {MEMORY_PREFIX} directory itself")
        self._check_mutation("delete", rel)
        if not self.store.exists(rel):
            raise NotFoundError(f"The path {self._display(rel)} does not exist")
        self.store.delete(rel)
        self._notify("delete", rel)
        return ToolResult(f"Successfully deleted {self._display(rel)}")

    def _rename(self, inp: dict[str, Any]) -> ToolResult:
        old_rel = self._rel(inp["old_path"])
        new_rel = self._rel(inp["new_path"])
        if old_rel == "" or new_rel == "":
            raise PathSecurityError(f"Cannot rename the {MEMORY_PREFIX} directory itself")
        self._check_mutation("rename", old_rel)
        if not self.store.exists(old_rel):
            raise NotFoundError(f"The path {self._display(old_rel)} does not exist")
        if self.store.exists(new_rel):
            raise AlreadyExistsError(f"The destination {self._display(new_rel)} already exists")
        self.store.rename(old_rel, new_rel)
        self._notify("rename", new_rel)
        return ToolResult(f"Successfully renamed {self._display(old_rel)} to {self._display(new_rel)}")
