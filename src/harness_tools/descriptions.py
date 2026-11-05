"""Model-facing tool descriptions, versioned next to the code (Crush ``.md.tpl`` pattern).

Written "as if onboarding a new hire" (Anthropic, *Writing effective tools for AI agents*): what
the tool does, when to use it instead of a neighbour, parameter conventions, limits and how to
recover from the common errors. Small description edits move eval scores, so change these only
against the tool-eval set; the SHA-256 of each rendered description is stored in
``ToolSpec.metadata["description_sha256"]`` so traces show which version a run used.

Templates use ``str.format`` placeholders for tool names (they differ per model profile).
"""

from __future__ import annotations

READ = """Read a text file (or an image) from the workspace.

- `path` must be an absolute path inside the workspace.
- Returns up to {limit} lines starting at `offset` (1-based). Each line is prefixed with its line number, right-aligned to 6 characters, then a tab. That prefix is NOT part of the file: never include it in {edit} old_string.
- Long files are paged: when the result says "Call ... with offset=N", do so to continue. Prefer grep to locate the region you need first.
- PNG/JPEG/GIF/WebP files are returned as images. Binary files are rejected; inspect them with bash.
- Reading a file is required before you edit or overwrite it."""

WRITE = """Create a new file or completely overwrite an existing one with `content`.

- `path` must be absolute; parent directories are created.
- To change part of an existing file use {edit} instead: it is safer and cheaper.
- Overwriting an existing file requires that you {read} it first in this session and that it has not changed since.
- Files inside .git cannot be written."""

EDIT = """Replace an exact string in a file: `old_string` -> `new_string`.

- `old_string` must match the current file content exactly (whitespace and indentation included) and must be unique in the file. If it occurs more than once, add surrounding lines until it is unique, or set `replace_all` to true to change every occurrence (e.g. renaming a variable).
- Copy text from {read} output WITHOUT the line-number prefix.
- You must {read} the file in this session before editing it; if the file changed on disk since, read it again.
- To create a new file use {write}."""

BASH = """Run a shell command (`bash -c`) in the workspace root {root} and return its stdout, stderr and exit code.

- Each call is a fresh shell: `cd` and shell variables do not persist between calls. Use absolute paths.
- Default timeout 120 s, maximum {max_timeout} s (`timeout` parameter). On timeout the whole process group is killed.
- Output beyond ~30,000 characters is truncated (head and tail kept) and the full output is saved to a file you can page through. Prefer narrow commands: `| head`, `| tail`, `grep -n`, `wc -l`.
- Use the dedicated tools instead of shell equivalents when they fit: {read} instead of cat/head, {edit} instead of sed -i, {grep} instead of grep/rg, {glob} instead of find/ls.
- For long-running jobs (servers, watchers) redirect their output to a file and run them in the background (`cmd > server.log 2>&1 &`); a background job that keeps stdout/stderr attached to the call is killed when the call ends.
- Give a short `description` (5-10 words) of what the command does."""

GREP = """Search file contents in the workspace with a regular expression (Python `re` syntax). Read-only and fast; prefer it over running grep through bash.

- `output_mode`: files_with_matches (default; just paths), content (path:line:text, optional `context` lines), count (matches per file).
- Filter files with `glob` (e.g. '*.py', 'src/**/*.{{ts,tsx}}') and/or `path` (absolute file or directory).
- Results are sorted by path and paged: {limit} per page by default (`head_limit`); use `offset` for the next page. If you get too many results, narrow the pattern or glob rather than paging through everything.
- Directories such as .git, node_modules and __pycache__ are skipped; binary files and files over 2 MB are ignored (use {bash} for those)."""

GLOB = """Find files by name pattern in the workspace. Read-only.

- Patterns: `*` (within one directory), `**` (any depth), `?`, `[abc]`, `{{a,b}}`. Examples: '**/*.py', 'tests/test_*.py', '*.{{yml,yaml}}'. Patterns without '/' match file names at any depth.
- `path` is the absolute directory to search (default: workspace root). Returns absolute paths, sorted by path (or newest first with sort='mtime'), {limit} per page; use `offset` for more.
- To search file contents use {grep}."""

TOOL_SEARCH = """Search the catalog of additional tools that are not loaded yet, and load the best matches so you can call them.

- Use it whenever you need a capability none of your current tools provides (e.g. an external service, issue tracker, browser, database).
- `query`: natural-language description of what you need (e.g. "create a jira ticket"), up to 500 characters. With `mode`="regex", a Python regular expression over tool names and descriptions (up to 200 characters), e.g. "github.*pull".
- Returns up to `limit` (default 5) tools; matched tools become callable immediately."""

__all__ = ["BASH", "EDIT", "GLOB", "GREP", "READ", "TOOL_SEARCH", "WRITE"]
