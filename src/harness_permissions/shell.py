"""Shell command analyzer: split compound commands into every simple command that would run.

Prefix rules "cannot see through ``bash -c``, ``eval``, pipes into interpreters, or scripts written
to disk" (research §7.3 technique 5), so the checklist (§7.7 step 5) requires parsing compound
commands "with a shell AST parser, as OpenHands does, and evaluate every segment; if parsing
fails, fall through to the classifier or the human". This module is that parser. It is a
conservative, dependency-free subset of POSIX/bash grammar:

* operators ``&&`` ``||`` ``;`` ``|`` ``|&`` ``&`` and newlines split commands;
* ``( ... )`` subshells, ``$( ... )`` and backtick command substitution (also inside double
  quotes and unquoted heredocs) and ``<( ... )`` / ``>( ... )`` process substitution are parsed
  recursively;
* quotes (``'...'``, ``"..."``, ``$'...'`` with escape decoding) and backslashes are removed, so
  ``'r'm``, ``\\rm`` and ``$'\\x72m'`` all resolve to ``rm``;
* redirections (``>``, ``>>``, ``&>``, ``>|``, ``<>``, ``<``, heredocs, here-strings) are recorded
  with their targets;
* leading ``NAME=value`` assignments are recorded separately from argv;
* wrappers (``sudo``, ``doas``, ``su -c``, ``env``, ``nohup``, ``time``, ``nice``, ``ionice``,
  ``timeout``, ``stdbuf``, ``command``, ``builtin``, ``exec``, ``xargs``, ``watch``) are peeled off
  and remembered; ``sh -c``/``bash -c``, ``eval``, ``env -S`` and ``find -exec`` payloads are parsed
  recursively;
* anything the analyzer cannot resolve statically (dynamic command names, ``eval $x``, inline
  interpreter code, piping into a shell or interpreter, unknown grammar such as ``case``) is
  reported in ``opaque_reasons`` or ``errors`` so the policy can refuse to auto-allow it.

The analyzer never executes anything and never expands variables.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field

from ._util import has_invisible, non_ascii_lookalikes

MAX_COMMAND_CHARS = 100_000
MAX_NESTING = 8

SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh", "mksh", "ash", "fish", "csh", "tcsh", "busybox"})
_INTERP_INLINE_FLAGS: dict[str, frozenset[str]] = {
    "python": frozenset({"-c"}),
    "node": frozenset({"-e", "--eval", "-p", "--print"}),
    "deno": frozenset({"eval"}),
    "bun": frozenset({"-e", "--eval"}),
    "perl": frozenset({"-e", "-E"}),
    "ruby": frozenset({"-e"}),
    "php": frozenset({"-r"}),
    "lua": frozenset({"-e"}),
    "Rscript": frozenset({"-e"}),
    "osascript": frozenset({"-e"}),
    "pwsh": frozenset({"-c", "-Command", "-command"}),
    "powershell": frozenset({"-c", "-Command", "-command"}),
}
_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\+?=")
_REDIR = re.compile(r"(\d*)(&>>|&>|>>|>&|>\||<<<|<<-|<<|<>|<&|>|<)")
_RESERVED_SKIP = frozenset({"{", "}", "!", "then", "else", "do", "done", "fi", "if", "elif", "while", "until"})
_UNSUPPORTED = frozenset({"case", "esac", "function", "select", "coproc"})
_PYTHON = re.compile(r"^python(\d+(\.\d+)?)?$")


def interpreter_family(exe: str) -> str | None:
    """Map an executable basename to its interpreter family (``python3.12`` -> ``python``)."""
    if _PYTHON.match(exe) or exe in {"pypy", "pypy3"}:
        return "python"
    if exe in {"nodejs"}:
        return "node"
    return exe if exe in _INTERP_INLINE_FLAGS else None


# ----------------------------------------------------------------------------- data model


@dataclass(frozen=True)
class Word:
    """One shell word after quote removal. Expansions (``$X``, ``$(...)``) are kept literally."""

    text: str
    raw: str
    dynamic: bool = False   # contains $-expansion or substitution outside single quotes
    globbed: bool = False   # contains unquoted *, ?, [ or brace expansion
    quoted: bool = False


@dataclass(frozen=True)
class Redirect:
    op: str
    fd: str = ""
    target: Word | None = None

    @property
    def writes(self) -> bool:
        if self.op not in {">", ">>", "&>", "&>>", ">|", "<>", ">&"}:
            return False
        if self.target is None:
            return False
        t = self.target.text
        # fd duplication (2>&1, >&-) and the null/std devices are not file writes.
        return not (re.fullmatch(r"\d+|-", t) or t in {"/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty"})

    @property
    def reads(self) -> bool:
        return self.op in {"<", "<>"} and self.target is not None


@dataclass
class SimpleCommand:
    """A single command that would be executed, after wrappers are peeled off."""

    words: list[Word]
    env: list[tuple[str, Word]] = field(default_factory=list)
    redirects: list[Redirect] = field(default_factory=list)
    wrappers: list[str] = field(default_factory=list)
    origin: str = "top"
    piped_input: bool = False
    background: bool = False
    opaque: list[str] = field(default_factory=list)
    code_execution: bool = False
    dynamic_args: bool = False
    transparent: bool = False   # a wrapper (sh -c, eval, su -c, env -S, watch) whose payload was parsed


    @property
    def argv(self) -> tuple[str, ...]:
        return tuple(w.text for w in self.words)

    @property
    def executable(self) -> str:
        """Basename of argv[0] ('' for a pure assignment / redirect-only command)."""
        return posixpath.basename(self.words[0].text) if self.words else ""

    @property
    def executable_path(self) -> str | None:
        """argv[0] when it is a path (``./git``, ``/usr/bin/git``), else ``None``."""
        if self.words and "/" in self.words[0].text:
            return self.words[0].text
        return None

    @property
    def dynamic_command(self) -> bool:
        return bool(self.words) and (self.words[0].dynamic or self.words[0].globbed)

    @property
    def text(self) -> str:
        """Space-joined argv: the string ``Tool(pattern)`` rules are matched against."""
        return " ".join(self.argv)

    @property
    def privileged(self) -> bool:
        return any(w in {"sudo", "doas", "su", "pkexec", "run0"} for w in self.wrappers) or self.executable in {"sudo", "doas", "su", "pkexec", "run0"}


@dataclass
class ShellAnalysis:
    source: str
    commands: list[SimpleCommand] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    suspicious: list[str] = field(default_factory=list)
    substitutions: int = 0
    operators: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def opaque_reasons(self) -> list[str]:
        out: list[str] = []
        for c in self.commands:
            out.extend(c.opaque)
            if c.dynamic_command:
                out.append(f"dynamic command name {c.words[0].raw!r}")
        return out

    @property
    def fully_resolved(self) -> bool:
        """True when every command that would run is statically known."""
        return self.ok and not self.opaque_reasons

    @property
    def is_compound(self) -> bool:
        return len(self.commands) > 1


class ShellParseError(Exception):
    pass


# ----------------------------------------------------------------------------- tokenizer

_TOKEN_WORD, _TOKEN_OP, _TOKEN_REDIR = "word", "op", "redir"
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "a": "\a", "b": "\b", "e": "\x1b", "E": "\x1b", "f": "\f", "v": "\v", "\\": "\\", "'": "'", '"': '"', "?": "?"}


def _decode_ansi_c(body: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch != "\\" or i + 1 >= len(body):
            out.append(ch)
            i += 1
            continue
        nxt = body[i + 1]
        if nxt in _ESCAPES:
            out.append(_ESCAPES[nxt])
            i += 2
        elif nxt == "x":
            m = re.match(r"[0-9A-Fa-f]{1,2}", body[i + 2:])
            out.append(chr(int(m.group(0), 16)) if m else "\\x")
            i += 2 + (len(m.group(0)) if m else 0)
        elif nxt in "uU":
            m = re.match(r"[0-9A-Fa-f]{1,8}" if nxt == "U" else r"[0-9A-Fa-f]{1,4}", body[i + 2:])
            try:
                out.append(chr(int(m.group(0), 16)) if m else "\\" + nxt)
            except (ValueError, OverflowError):
                out.append("?")
            i += 2 + (len(m.group(0)) if m else 0)
        elif nxt in "01234567":
            m = re.match(r"[0-7]{1,3}", body[i + 1:])
            assert m is not None
            out.append(chr(int(m.group(0), 8) & 0xFF))
            i += 1 + len(m.group(0))
        else:
            out.append("\\" + nxt)
            i += 2
    return "".join(out)


class _Tokenizer:
    def __init__(self, src: str) -> None:
        self.s = src
        self.n = len(src)
        self.i = 0
        self.tokens: list[tuple[str, object]] = []
        self.subs: list[tuple[str, str]] = []  # (origin, inner source)
        self.heredocs: list[tuple[str, bool, bool]] = []  # (delimiter, quoted, strip_tabs)
        self.expect_heredoc_delim: list[bool] = []  # strip_tabs flags for pending delimiters

    # -- balanced scanners ---------------------------------------------------------------
    def _scan_single(self, i: int) -> int:
        j = self.s.find("'", i)
        if j < 0:
            raise ShellParseError("unterminated single quote")
        return j + 1

    def _scan_backtick(self, i: int) -> tuple[str, int]:
        buf: list[str] = []
        while i < self.n:
            ch = self.s[i]
            if ch == "\\" and i + 1 < self.n:
                buf.append(self.s[i + 1] if self.s[i + 1] in "`\\$" else ch + self.s[i + 1])
                i += 2
                continue
            if ch == "`":
                return "".join(buf), i + 1
            buf.append(ch)
            i += 1
        raise ShellParseError("unterminated backtick substitution")

    def _scan_double_inner(self, i: int) -> int:
        """Skip a double-quoted string starting after its opening quote (used inside $(...))."""
        while i < self.n:
            ch = self.s[i]
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                return i + 1
            if ch == "$" and self.s.startswith("$(", i):
                _, i = self._scan_paren(i + 2)
                continue
            if ch == "`":
                _, i = self._scan_backtick(i + 1)
                continue
            i += 1
        raise ShellParseError("unterminated double quote")

    def _scan_paren(self, i: int) -> tuple[str, int]:
        """Return (inner, index after the matching ')') for a '(' that ended at i-1."""
        start, depth = i, 1
        while i < self.n:
            ch = self.s[i]
            if ch == "\\":
                i += 2
                continue
            if ch == "'":
                i = self._scan_single(i + 1)
                continue
            if ch == '"':
                i = self._scan_double_inner(i + 1)
                continue
            if ch == "`":
                _, i = self._scan_backtick(i + 1)
                continue
            if ch == "#" and (i == start or self.s[i - 1] in " \t\n;"):
                nl = self.s.find("\n", i)
                i = self.n if nl < 0 else nl
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return self.s[start:i], i + 1
            i += 1
        raise ShellParseError("unterminated '(' / '$('")

    def _scan_brace(self, i: int) -> int:
        depth = 1
        while i < self.n:
            ch = self.s[i]
            if ch == "\\":
                i += 2
                continue
            if ch == "'":
                i = self._scan_single(i + 1)
                continue
            if ch == '"':
                i = self._scan_double_inner(i + 1)
                continue
            if ch == "$" and self.s.startswith("$(", i):
                inner, i = self._scan_paren(i + 2)
                self.subs.append(("command_substitution", inner))
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        raise ShellParseError("unterminated '${'")

    # -- dollar forms --------------------------------------------------------------------
    def _dollar(self, i: int) -> tuple[str, int, bool]:
        """Handle '$...' at i. Returns (literal text, new index, is_dynamic)."""
        s = self.s
        if s.startswith("$((", i):
            inner, j = self._scan_paren(i + 3)
            # inner ends with the first ')' of '))'
            if j < self.n and s[j] == ")":
                j += 1
            self._collect_nested_subs(inner)
            return s[i:j], j, True
        if s.startswith("$(", i):
            inner, j = self._scan_paren(i + 2)
            self.subs.append(("command_substitution", inner))
            return s[i:j], j, True
        if s.startswith("${", i):
            j = self._scan_brace(i + 2)
            return s[i:j], j, True
        m = re.match(r"\$([A-Za-z_][A-Za-z0-9_]*|[0-9@*#?$!-])", s[i:])
        if m:
            return m.group(0), i + len(m.group(0)), True
        return "$", i + 1, False

    def _collect_nested_subs(self, text: str) -> None:
        for m in re.finditer(r"\$\(|`", text):
            if m.group(0) == "`":
                end = text.find("`", m.end())
                if end > 0:
                    self.subs.append(("command_substitution", text[m.end():end]))
            else:
                sub = _Tokenizer(text)
                try:
                    inner, _ = sub._scan_paren(m.end())
                    self.subs.append(("command_substitution", inner))
                except ShellParseError:
                    pass

    # -- main loop -----------------------------------------------------------------------
    def run(self) -> None:
        s = self.s
        while self.i < self.n:
            ch = s[self.i]
            if ch == "\\" and self.i + 1 < self.n and s[self.i + 1] == "\n":
                self.i += 2
                continue
            if ch in " \t\r":
                self.i += 1
                continue
            if ch == "\n":
                self.tokens.append((_TOKEN_OP, "\n"))
                self.i += 1
                self._consume_heredocs()
                continue
            if ch == "#":
                nl = s.find("\n", self.i)
                self.i = self.n if nl < 0 else nl
                continue
            if ch in "<>" and self.i + 1 < self.n and s[self.i + 1] == "(":
                inner, j = self._scan_paren(self.i + 2)
                self.subs.append(("process_substitution", inner))
                self.tokens.append((_TOKEN_WORD, Word(text=s[self.i:j], raw=s[self.i:j], dynamic=True)))
                self.i = j
                continue
            m = _REDIR.match(s, self.i)
            if m:
                op = m.group(2)
                if op in {"<<", "<<-"}:
                    self.expect_heredoc_delim.append(op == "<<-")
                self.tokens.append((_TOKEN_REDIR, (m.group(1), op)))
                self.i = m.end()
                continue
            two = s[self.i:self.i + 2]
            if two in {"&&", "||", "|&", ";;"}:
                if two == ";;":
                    raise ShellParseError("case statements are not supported")
                self.tokens.append((_TOKEN_OP, two))
                self.i += 2
                continue
            if ch in ";|&()":
                self.tokens.append((_TOKEN_OP, ch))
                self.i += 1
                continue
            word = self._read_word()
            if self.expect_heredoc_delim and self.tokens and self.tokens[-1][0] == _TOKEN_REDIR and self.tokens[-1][1][1] in {"<<", "<<-"}:  # type: ignore[index]
                strip = self.expect_heredoc_delim.pop(0)
                self.heredocs.append((word.text, word.quoted, strip))
            self.tokens.append((_TOKEN_WORD, word))
        if self.heredocs:
            self._consume_heredocs()

    def _consume_heredocs(self) -> None:
        while self.heredocs:
            delim, quoted, strip = self.heredocs.pop(0)
            body: list[str] = []
            while self.i < self.n:
                nl = self.s.find("\n", self.i)
                line = self.s[self.i:] if nl < 0 else self.s[self.i:nl]
                self.i = self.n if nl < 0 else nl + 1
                check = line.lstrip("\t") if strip else line
                if check == delim:
                    break
                body.append(line)
            if not quoted:
                self._collect_nested_subs("\n".join(body))

    def _read_word(self) -> Word:
        s = self.s
        buf: list[str] = []
        raw_start = self.i
        dynamic = globbed = quoted = False
        while self.i < self.n:
            ch = s[self.i]
            if ch in " \t\r\n;|&()<>":
                break
            if ch == "\\":
                if self.i + 1 < self.n:
                    nxt = s[self.i + 1]
                    if nxt != "\n":
                        buf.append(nxt)
                    quoted = True
                    self.i += 2
                else:
                    self.i += 1
                continue
            if ch == "'":
                j = self._scan_single(self.i + 1)
                buf.append(s[self.i + 1:j - 1])
                quoted = True
                self.i = j
                continue
            if ch == "$" and self.i + 1 < self.n and s[self.i + 1] == "'":
                j = self.i + 2
                while j < self.n and s[j] != "'":
                    j += 2 if s[j] == "\\" else 1
                if j >= self.n:
                    raise ShellParseError("unterminated $'...' quote")
                buf.append(_decode_ansi_c(s[self.i + 2:j]))
                quoted = True
                self.i = j + 1
                continue
            if ch == "$" and self.i + 1 < self.n and s[self.i + 1] == '"':
                self.i += 1
                continue
            if ch == '"':
                text, dyn = self._read_double()
                buf.append(text)
                dynamic = dynamic or dyn
                quoted = True
                continue
            if ch == "$":
                text, self.i, dyn = self._dollar(self.i)
                buf.append(text)
                dynamic = dynamic or dyn
                continue
            if ch == "`":
                inner, self.i = self._scan_backtick(self.i + 1)
                self.subs.append(("command_substitution", inner))
                buf.append("`" + inner + "`")
                dynamic = True
                continue
            if ch in "*?[":
                globbed = True
            if ch == "{":
                close = s.find("}", self.i)
                seg = s[self.i:close] if close > 0 else ""
                if "," in seg or ".." in seg:
                    globbed = True
            buf.append(ch)
            self.i += 1
        return Word(text="".join(buf), raw=s[raw_start:self.i], dynamic=dynamic, globbed=globbed, quoted=quoted)

    def _read_double(self) -> tuple[str, bool]:
        s = self.s
        self.i += 1  # opening quote
        buf: list[str] = []
        dynamic = False
        while self.i < self.n:
            ch = s[self.i]
            if ch == "\\" and self.i + 1 < self.n:
                nxt = s[self.i + 1]
                if nxt in '$`"\\':
                    buf.append(nxt)
                elif nxt == "\n":
                    pass
                else:
                    buf.append(ch + nxt)
                self.i += 2
                continue
            if ch == '"':
                self.i += 1
                return "".join(buf), dynamic
            if ch == "$":
                text, self.i, dyn = self._dollar(self.i)
                buf.append(text)
                dynamic = dynamic or dyn
                continue
            if ch == "`":
                inner, self.i = self._scan_backtick(self.i + 1)
                self.subs.append(("command_substitution", inner))
                buf.append("`" + inner + "`")
                dynamic = True
                continue
            buf.append(ch)
            self.i += 1
        raise ShellParseError("unterminated double quote")


# ----------------------------------------------------------------------------- parser


def _split_commands(tokens: list[tuple[str, object]], origin: str, out: ShellAnalysis, depth: int) -> list[SimpleCommand]:
    commands: list[SimpleCommand] = []
    cur = SimpleCommand(words=[], origin=origin)
    pending_redirect: tuple[str, str] | None = None
    piped = False
    after_group = False
    i = 0

    def finish(op: str | None) -> None:
        nonlocal cur, piped, after_group
        if pending_redirect is not None:
            raise ShellParseError(f"redirection {pending_redirect[1]!r} without a target")
        if cur.words or cur.env or cur.redirects:
            cur.piped_input = piped
            cur.background = op == "&"
            commands.append(cur)
        elif op in {"|", "|&", "&&", "||"} and not after_group:
            raise ShellParseError(f"syntax error near {op!r}")
        piped = op in {"|", "|&"}
        after_group = False
        cur = SimpleCommand(words=[], origin=origin)

    while i < len(tokens):
        kind, val = tokens[i]
        if kind == _TOKEN_REDIR:
            if pending_redirect is not None:
                raise ShellParseError("consecutive redirections")
            pending_redirect = val  # type: ignore[assignment]
            i += 1
            continue
        if kind == _TOKEN_WORD:
            word: Word = val  # type: ignore[assignment]
            if pending_redirect is not None:
                cur.redirects.append(Redirect(op=pending_redirect[1], fd=pending_redirect[0], target=word))
                pending_redirect = None
            elif not cur.words and _ASSIGN.match(word.raw):
                name, _, value = word.text.partition("=")
                cur.env.append((name.rstrip("+"), Word(text=value, raw=word.raw, dynamic=word.dynamic, globbed=word.globbed, quoted=word.quoted)))
            elif not cur.words and word.text in _RESERVED_SKIP and not word.quoted:
                pass
            elif not cur.words and word.text in _UNSUPPORTED and not word.quoted:
                raise ShellParseError(f"unsupported shell construct {word.text!r}")
            else:
                cur.words.append(word)
            i += 1
            continue
        op: str = val  # type: ignore[assignment]
        if op == "(":
            if cur.words:
                raise ShellParseError("unexpected '(' (function definitions are not supported)")
            level, j = 1, i + 1
            while j < len(tokens):
                if tokens[j] == (_TOKEN_OP, "("):
                    level += 1
                elif tokens[j] == (_TOKEN_OP, ")"):
                    level -= 1
                    if level == 0:
                        break
                j += 1
            if level:
                raise ShellParseError("unbalanced '('")
            if depth >= MAX_NESTING:
                raise ShellParseError("nesting too deep")
            inner = _split_commands(tokens[i + 1:j], "subshell", out, depth + 1)
            if piped and inner:
                inner[0].piped_input = True
            commands.extend(inner)
            piped = False
            after_group = True
            i = j + 1
            continue
        if op == ")":
            raise ShellParseError("unbalanced ')'")
        out.operators.append(op)
        finish(op)
        i += 1
    finish(None)
    return commands


# ----------------------------------------------------------------------------- wrappers

_SUDO_ARG_OPTS = frozenset({"-u", "-g", "-C", "-D", "-h", "-p", "-r", "-t", "-U", "-T", "--user", "--group", "--chdir", "--prompt", "--role", "--type", "--host", "--close-from"})
_TIMEOUT_ARG_OPTS = frozenset({"-s", "-k", "--signal", "--kill-after"})
_XARGS_ARG_OPTS = frozenset({"-I", "-i", "-n", "-P", "-d", "-L", "-l", "-s", "-E", "-e", "-a", "--max-args", "--max-procs", "--delimiter", "--arg-file", "--replace", "--max-lines", "--max-chars", "--eof", "--process-slot-var"})


def _skip_options(words: list[Word], i: int, arg_opts: frozenset[str]) -> int:
    while i < len(words):
        t = words[i].text
        if t == "--":
            return i + 1
        if not t.startswith("-") or t == "-":
            return i
        if t in arg_opts:
            i += 2
            continue
        if "=" in t and t.split("=", 1)[0] in arg_opts:
            i += 1
            continue
        # combined short flags with a trailing arg-taking option, e.g. sudo -Eu root
        if len(t) > 2 and t[1] != "-" and f"-{t[-1]}" in arg_opts:
            i += 2
            continue
        i += 1
    return i


def _unwrap(cmd: SimpleCommand, analysis: ShellAnalysis, depth: int) -> list[SimpleCommand]:
    """Peel wrappers off ``cmd`` in place and return any nested commands found in payloads."""
    nested: list[SimpleCommand] = []
    guard = 0
    while cmd.words and guard < 16:
        guard += 1
        exe = posixpath.basename(cmd.words[0].text)
        words = cmd.words
        if exe in {"sudo", "doas", "pkexec", "run0"}:
            cmd.wrappers.append(exe)
            j = _skip_options(words, 1, _SUDO_ARG_OPTS)
            flags = {w.text for w in words[1:j]}
            if j >= len(words):
                if flags & {"-i", "-s", "--login", "--shell"}:
                    cmd.opaque.append(f"{exe} opens an interactive root shell")
                cmd.words = words[:1]
                return nested
            cmd.words = words[j:]
            continue
        if exe == "su":
            cmd.wrappers.append("su")
            payload = None
            for k, w in enumerate(words[1:], start=1):
                if w.text in {"-c", "--command"} and k + 1 < len(words):
                    payload = words[k + 1]
                    break
            if payload is None:
                cmd.opaque.append("su opens an interactive shell")
                return nested
            cmd.words = words[:1]
            nested.extend(_nested_source(payload, "su -c", cmd, analysis, depth))
            cmd.transparent = not cmd.opaque
            return nested
        if exe == "env":
            cmd.wrappers.append("env")
            j = 1
            while j < len(words):
                t = words[j].text
                if t in {"-u", "--unset", "-C", "--chdir"}:
                    j += 2
                    continue
                if t in {"-S", "--split-string"} and j + 1 < len(words):
                    cmd.words = words[:1]
                    nested.extend(_nested_source(words[j + 1], "env -S", cmd, analysis, depth))
                    cmd.transparent = not cmd.opaque
                    return nested
                if t.startswith("-S") and len(t) > 2:
                    cmd.words = words[:1]
                    nested.extend(_nested_source(Word(t[2:], t[2:], words[j].dynamic), "env -S", cmd, analysis, depth))
                    cmd.transparent = not cmd.opaque
                    return nested
                if t.startswith("-") and t != "-":
                    j += 1
                    continue
                if _ASSIGN.match(words[j].raw) or _ASSIGN.match(t):
                    name, _, value = t.partition("=")
                    cmd.env.append((name, Word(value, words[j].raw, words[j].dynamic)))
                    j += 1
                    continue
                break
            if j >= len(words):
                cmd.words = words[:1]  # bare `env` prints the environment
                cmd.wrappers.pop()
                return nested
            cmd.words = words[j:]
            continue
        if exe in {"nohup", "builtin", "exec", "unbuffer", "time"}:
            cmd.wrappers.append(exe)
            j = 1
            while exe == "time" and j < len(words) and words[j].text.startswith("-"):
                j += 1
            if j >= len(words):
                return nested
            cmd.words = words[j:]
            continue
        if exe == "command":
            j = 1
            lookup = False
            while j < len(words) and words[j].text.startswith("-"):
                lookup = lookup or words[j].text in {"-v", "-V"}
                j += 1
            if lookup or j >= len(words):
                return nested  # `command -v git` only looks the name up
            cmd.wrappers.append(exe)
            cmd.words = words[j:]
            continue
        if exe in {"nice", "ionice", "stdbuf", "chrt", "taskset"}:
            cmd.wrappers.append(exe)
            j = 1
            while j < len(words) and words[j].text.startswith("-"):
                t = words[j].text
                j += 2 if t in {"-n", "-c", "-p", "-i", "-o", "-e", "--adjustment", "--class", "--classdata"} else 1
            if exe in {"chrt", "taskset"} and j < len(words):
                j += 1  # priority / cpu mask
            if j >= len(words):
                return nested
            cmd.words = words[j:]
            continue
        if exe == "timeout":
            cmd.wrappers.append(exe)
            j = _skip_options(words, 1, _TIMEOUT_ARG_OPTS) + 1  # + DURATION
            if j >= len(words):
                return nested
            cmd.words = words[j:]
            continue
        if exe == "xargs":
            cmd.wrappers.append(exe)
            j = _skip_options(words, 1, _XARGS_ARG_OPTS)
            cmd.dynamic_args = True
            cmd.words = words[j:] if j < len(words) else [Word("echo", "echo")]
            continue
        if exe == "watch":
            cmd.wrappers.append(exe)
            j = _skip_options(words, 1, frozenset({"-n", "--interval", "-d", "--differences"}))
            if j < len(words):
                payload = " ".join(w.raw for w in words[j:])
                cmd.words = words[:1]
                nested.extend(_nested_source(Word(payload, payload, any(w.dynamic for w in words[j:])), "watch", cmd, analysis, depth))
                cmd.transparent = not cmd.opaque
            return nested
        break

    if not cmd.words:
        return nested
    exe = cmd.executable
    args = cmd.words[1:]
    if exe in SHELLS:
        payload_idx = None
        for k, w in enumerate(args):
            t = w.text
            if t == "-c" or (re.fullmatch(r"-[a-zA-Z]*c[a-zA-Z]*", t) and not t.startswith("--")):
                payload_idx = k + 1
                break
            if not t.startswith("-"):
                break
        if payload_idx is not None:
            if payload_idx >= len(args):
                cmd.opaque.append(f"{exe} -c without a command string")
            else:
                nested.extend(_nested_source(args[payload_idx], f"{exe} -c", cmd, analysis, depth))
                cmd.transparent = not cmd.opaque
        else:
            script = next((w for w in args if not w.text.startswith("-")), None)
            if script is None:
                cmd.opaque.append(f"{'pipe into ' if cmd.piped_input else ''}{exe} reads commands from stdin")
            else:
                cmd.code_execution = True
        return nested
    if exe in {"eval"}:
        if not args:
            return nested
        if any(w.dynamic for w in args):
            cmd.opaque.append("eval of a dynamic string")
            return nested
        joined = " ".join(w.text for w in args)
        nested.extend(_nested_source(Word(joined, joined), "eval", cmd, analysis, depth))
        cmd.transparent = not cmd.opaque
        return nested
    if exe in {"source", "."}:
        cmd.code_execution = True
        if any(w.dynamic for w in args):
            cmd.opaque.append(f"{exe} of a dynamic path")
        return nested
    family = interpreter_family(exe)
    if family is not None:
        inline = _INTERP_INLINE_FLAGS[family]
        if any(w.text in inline or any(w.text.startswith(f) and len(f) == 2 and len(w.text) > 2 and not w.text.startswith("--") for f in inline) for w in args):
            cmd.opaque.append(f"inline {family} code")
        elif not any(not w.text.startswith("-") for w in args):
            if cmd.piped_input:
                cmd.opaque.append(f"pipe into {family} interpreter")
            else:
                cmd.opaque.append(f"{family} reads code from stdin")
        cmd.code_execution = True
        return nested
    if exe == "find":
        k = 0
        while k < len(args):
            t = args[k].text
            if t in {"-exec", "-execdir", "-ok", "-okdir"}:
                end = k + 1
                while end < len(args) and args[end].text not in {";", "+"}:
                    end += 1
                inner_words = args[k + 1:end]
                if inner_words:
                    if depth >= MAX_NESTING:
                        raise ShellParseError("nesting too deep")
                    sub = SimpleCommand(words=list(inner_words), origin="find -exec", dynamic_args=True)
                    nested.append(sub)
                    nested.extend(_unwrap(sub, analysis, depth + 1))
                k = end + 1
                continue
            k += 1
        return nested
    if exe == "git":
        k = 0
        while k < len(args):
            t = args[k].text
            if t in {"-c", "--config-env"} or t.startswith("--config-env="):
                cmd.opaque.append("git -c/--config-env can inject config that executes code (alias, core.sshCommand, core.pager...)")
                break
            if not t.startswith("-"):
                break
            k += 2 if t in {"-C", "--git-dir", "--work-tree", "--namespace", "--exec-path"} else 1
    return nested


def _nested_source(payload: Word, origin: str, parent: SimpleCommand, analysis: ShellAnalysis, depth: int) -> list[SimpleCommand]:
    if payload.dynamic:
        parent.opaque.append(f"{origin} with a dynamic command string")
        return []
    if depth >= MAX_NESTING:
        raise ShellParseError("nesting too deep")
    inner = _analyze(payload.text, origin, analysis, depth + 1)
    for c in inner:
        c.wrappers = list(parent.wrappers) + c.wrappers
    return inner


def _analyze(source: str, origin: str, analysis: ShellAnalysis, depth: int) -> list[SimpleCommand]:
    tok = _Tokenizer(source)
    tok.run()
    commands = _split_commands(tok.tokens, origin, analysis, depth)
    out: list[SimpleCommand] = []
    for cmd in commands:
        out.append(cmd)
        out.extend(_unwrap(cmd, analysis, depth))
    for sub_origin, inner in tok.subs:
        analysis.substitutions += 1
        if depth >= MAX_NESTING:
            raise ShellParseError("nesting too deep")
        out.extend(_analyze(inner, sub_origin, analysis, depth + 1))
    return out


_FORK_BOMB = re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:")


def analyze(command: str) -> ShellAnalysis:
    """Parse ``command`` into every simple command it would run. Never raises."""
    analysis = ShellAnalysis(source=command)
    if not isinstance(command, str):
        analysis.errors.append("command is not a string")
        return analysis
    if len(command) > MAX_COMMAND_CHARS:
        analysis.errors.append(f"command longer than {MAX_COMMAND_CHARS} characters")
        return analysis
    if "\x00" in command:
        analysis.errors.append("command contains a NUL byte")
        return analysis
    if has_invisible(command):
        analysis.suspicious.append("invisible (zero-width or bidi) characters")
    looks = non_ascii_lookalikes(command)
    if looks:
        analysis.suspicious.append(f"look-alike characters {''.join(looks)!r}")
    if "\r" in command:
        analysis.suspicious.append("carriage return")
    if "\x1b" in command:
        analysis.suspicious.append("terminal escape sequence")
    if _FORK_BOMB.search(command):
        analysis.errors.append("fork bomb")
        return analysis
    try:
        analysis.commands = _analyze(command, "top", analysis, 0)
    except ShellParseError as exc:
        analysis.errors.append(str(exc))
    except RecursionError:  # pragma: no cover - defensive
        analysis.errors.append("nesting too deep")
    return analysis
