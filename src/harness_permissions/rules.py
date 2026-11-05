"""``Tool(pattern)`` permission rules: parsing, linting and matching.

Syntax follows Claude Code "for ecosystem compatibility" (checklist §7.7 step 4):

=====================================  ===================================================
``Bash``                               every call of the tool (rejected in ``allow``)
``Bash(npm test)``                     exact command
``Bash(npm test *)`` / ``Bash(git push:*)``  prefix with word boundary (``:*`` is the legacy form)
``Bash(ls*)``                          raw prefix (also matches ``lsof``)
``Read(src/**)`` / ``Edit(~/.ssh/**)``  path globs (see :mod:`harness_permissions.paths`)
``WebFetch(domain:example.com)``       network destination (``*.example.com`` = subdomains)
``mcp__github`` / ``mcp__github__*``   every tool of an MCP server
``mcp__slack__post(channel:#eng*)``    ``field:glob`` on a top-level input field
``Agent(Explore)``                     sub-agent type
=====================================  ===================================================

``Read`` rules also cover every reading tool (Grep, Glob, LS...) and ``Edit`` rules every editing
tool (Write, MultiEdit, NotebookEdit...), as in Claude Code. Our extension: ``Read``/``Edit``
*deny and ask* rules also apply to the paths a shell command reads or writes, so
``Read(**/.env)`` stops ``cat .env`` as well as ``Read(.env)``.

Matching is deliberately asymmetric:

* **deny/ask (fuzzy)** - NFKC + casefold + invisible-character stripping, executable *basename*,
  wrappers and env prefixes ignored, flags matched in any order and short-flag clusters
  expanded (``rm -rf`` matches ``rm -r -f``, ``rm -fr`` and ``rm --recursive --force``). A false
  positive only denies a harmless look-alike.
* **allow (exact)** - case-sensitive, per simple command, and the command must be fully resolved
  (no dynamic command name, no opaque payload). ``Bash(git status*)`` therefore never matches
  ``git status && rm -rf /``.

Lint (checklist step 4 and §7.8): ``allow`` rules may not be blanket shell (``Bash``,
``Bash(*)``), interpreter/wrapper wildcards (``python *``, ``node *``, ``bash *``, ``sudo *``,
``env *``, ``xargs *``) or package-runner wildcards (``npm run *``, ``npx *``, ``uv run *``...),
nor blanket tool/egress globs (``*``, ``mcp__*``, ``WebFetch``, ``WebFetch(domain:*)``).
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ._util import fold
from .boundary import ActionProfile, ToolKind
from .errors import RuleSyntaxError
from .network import domain_matches
from .paths import Workspace
from .shell import SimpleCommand, interpreter_family


class RuleEffect(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


_RULE = re.compile(r"^\s*([A-Za-z0-9_\-*?.:]+)\s*(?:\((.*)\))?\s*$", re.DOTALL)
_SHELL_TOOLS = {"Bash", "bash", "Shell", "shell"}
_READ_FAMILY = {"Read"}
_EDIT_FAMILY = {"Edit", "Write"}
_PATH_TOOLS = {"Read", "Edit", "Write", "MultiEdit", "NotebookEdit", "Glob", "Grep", "LS", "NotebookRead"}

_INTERPRETERS_AND_WRAPPERS = frozenset({
    "python", "python2", "python3", "pypy", "pypy3", "node", "nodejs", "deno", "bun", "ruby", "perl", "php", "lua", "Rscript",
    "osascript", "pwsh", "powershell", "bash", "sh", "zsh", "dash", "ksh", "fish", "csh", "tcsh", "eval", "exec", "source", ".",
    "sudo", "doas", "su", "env", "xargs", "nohup", "timeout", "command", "builtin", "time", "nice", "watch", "npx", "bunx",
    "pnpx", "uvx", "pipx", "busybox", "docker", "podman", "ssh", "find",
})
_PACKAGE_RUNNERS = frozenset({
    ("npm", "run"), ("npm", "exec"), ("npm", "x"), ("yarn", "run"), ("yarn", "exec"), ("yarn", "dlx"), ("pnpm", "run"),
    ("pnpm", "exec"), ("pnpm", "dlx"), ("bun", "run"), ("bun", "x"), ("uv", "run"), ("uv", "tool"), ("poetry", "run"),
    ("pipenv", "run"), ("cargo", "run"), ("go", "run"), ("deno", "run"), ("deno", "task"), ("pdm", "run"), ("hatch", "run"),
    ("rye", "run"), ("conda", "run"), ("make", "-f"), ("just", "--justfile"),
})
_SCRIPT_RUNNER_HEADS = frozenset({"yarn", "pnpm", "bun"})  # `yarn <script>` runs package.json scripts
_LONG_TO_SHORT = {"--recursive": "r", "--force": "f", "--interactive": "i", "--verbose": "v", "--all": "a", "--delete": "d"}


@dataclass(frozen=True)
class Rule:
    """One parsed rule. ``layer`` records which settings layer it came from (for audit)."""

    source: str
    tool: str
    specifier: str | None
    effect: RuleEffect
    layer: str = "session"
    origin: str = ""

    @classmethod
    def parse(cls, text: str, effect: RuleEffect | str, *, layer: str = "session", origin: str = "") -> "Rule":
        if not isinstance(text, str) or not text.strip():
            raise RuleSyntaxError(f"rule must be a non-empty string, got {text!r}")
        m = _RULE.match(text)
        if not m:
            raise RuleSyntaxError(f"malformed rule {text!r}: expected Tool or Tool(pattern)")
        tool, spec = m.group(1), m.group(2)
        if spec is not None:
            spec = spec.strip()
            if spec == "":
                raise RuleSyntaxError(f"empty pattern in rule {text!r}")
            if spec.count("(") != spec.count(")") and tool not in _SHELL_TOOLS:
                raise RuleSyntaxError(f"unbalanced parentheses in rule {text!r}")
        return cls(source=text.strip(), tool=tool, specifier=spec, effect=RuleEffect(effect), layer=layer, origin=origin)

    @property
    def id(self) -> str:
        return f"{self.layer}:{self.effect.value}:{self.source}"

    @property
    def is_shell(self) -> bool:
        return self.tool in _SHELL_TOOLS

    @property
    def has_wildcard(self) -> bool:
        return self.specifier is None or "*" in self.specifier or "?" in self.specifier

    # -- tool-name matching ----------------------------------------------------------
    def matches_tool(self, name: str, kind: ToolKind, *, fuzzy: bool) -> bool:
        rt, nm = (fold(self.tool), fold(name)) if fuzzy else (self.tool, name)
        if rt.startswith("mcp__") and "__" not in rt[5:] and "*" not in rt:
            if nm == rt or nm.startswith(rt + "__"):
                return True
        if "*" in rt or "?" in rt:
            if fnmatch.fnmatchcase(nm, rt):
                return True
        elif rt == nm:
            return True
        if self.tool in _READ_FAMILY and kind in {ToolKind.READ, ToolKind.SEARCH}:
            return True
        if self.tool in _EDIT_FAMILY and kind is ToolKind.EDIT:
            return True
        if self.is_shell and kind is ToolKind.SHELL:
            return True
        return False


# ----------------------------------------------------------------------------- linting


def _normalize_shell_spec(spec: str) -> str:
    s = spec.strip()
    if s.endswith(":*"):
        s = s[:-2].rstrip() + " *"
    return s


def lint_rule(rule: Rule) -> list[str]:
    """Problems that make ``rule`` unacceptable as an allow rule (empty list = fine)."""
    if rule.effect is not RuleEffect.ALLOW:
        return []
    problems: list[str] = []
    tool = rule.tool
    if tool in {"*", "mcp__*", "mcp__*__*"} or (tool.startswith("*")):
        problems.append(f"{rule.source!r}: blanket tool wildcard in an allow rule")
    if rule.is_shell:
        if rule.specifier is None or _normalize_shell_spec(rule.specifier) in {"*", ""}:
            problems.append(f"{rule.source!r}: blanket shell allow rule (equivalent to Bash(*))")
        else:
            problems.extend(_lint_shell_spec(rule))
    if tool == "WebFetch" and (rule.specifier is None or rule.specifier in {"*", "domain:*"}):
        problems.append(f"{rule.source!r}: blanket egress allow rule defeats the domain allowlist")
    return problems


def _lint_shell_spec(rule: Rule) -> list[str]:
    spec = _normalize_shell_spec(rule.specifier or "")
    tokens = spec.split()
    if not tokens:
        return [f"{rule.source!r}: empty shell pattern"]
    wildcard = "*" in spec or "?" in spec
    head = tokens[0].rsplit("/", 1)[-1]
    if "*" in head or "?" in head:
        return [f"{rule.source!r}: wildcard in the executable position matches any program"]
    if not wildcard:
        return []
    if head in _INTERPRETERS_AND_WRAPPERS or interpreter_family(head):
        return [f"{rule.source!r}: interpreter/wrapper wildcard is arbitrary code execution"]
    second = tokens[1] if len(tokens) > 1 else ""
    if (head, second) in _PACKAGE_RUNNERS or (head in _SCRIPT_RUNNER_HEADS and ("*" in second or not second)):
        return [f"{rule.source!r}: package-manager script-runner wildcard runs arbitrary scripts"]
    return []


def is_blanket(rule: Rule) -> bool:
    """True for rules auto mode drops on entry (blanket shell, interpreter, package-runner wildcards)."""
    return bool(lint_rule(Rule(rule.source, rule.tool, rule.specifier, RuleEffect.ALLOW, rule.layer, rule.origin)))


# ----------------------------------------------------------------------------- shell matching


def _command_regex(spec: str, fuzzy: bool) -> re.Pattern[str]:
    s = _normalize_shell_spec(spec)
    if fuzzy:
        s = fold(s)
    trailing = s.endswith(" *")
    body = s[:-2] if trailing else s
    regex = ".*".join(re.escape(part) for part in body.split("*"))
    regex = regex.replace(r"\?", ".")
    if trailing:
        regex += r"(?: .*)?"
    return re.compile(regex + r"\Z", re.DOTALL)


def _token_match(pat: str, tok: str) -> bool:
    if "*" in pat or "?" in pat:
        return fnmatch.fnmatchcase(tok, pat)
    return pat == tok


def _deny_token_match(pattern: str, argv: list[str]) -> bool:
    """Flag-order-insensitive matching for deny/ask rules (all inputs already folded)."""
    ptoks = _normalize_shell_spec(pattern).split()
    if ptoks and ptoks[-1] == "*":
        ptoks = ptoks[:-1]
    lead: list[str] = []
    for t in ptoks:
        if t.startswith("-") or "*" in t:
            break
        lead.append(t)
    rest_p = ptoks[len(lead):]
    if len(argv) < len(lead):
        return False
    # leading literal tokens must appear in order, options may be interleaved before them
    positional = [a for a in argv[1:] if not a.startswith("-")]
    head_ok = bool(lead) and _token_match(lead[0], argv[0]) and all(
        i < len(positional) and _token_match(p, positional[i]) for i, p in enumerate(lead[1:])
    )
    if not head_ok:
        return False
    remaining = argv[1:]
    short = set()
    for a in remaining:
        if a.startswith("-") and not a.startswith("--") and a[1:].isalpha():
            short.update(a[1:])
        if a in _LONG_TO_SHORT:
            short.add(_LONG_TO_SHORT[a])
    for p in rest_p:
        if re.fullmatch(r"-[a-z]{2,}", p, re.IGNORECASE) and "*" not in p:
            if not set(p[1:]) <= short:
                return False
            continue
        if re.fullmatch(r"-[a-z]", p, re.IGNORECASE):
            if p[1] not in short:
                return False
            continue
        if not any(_token_match(p, a) for a in remaining):
            return False
    return True


def match_command(rule: Rule, cmd: SimpleCommand, *, fuzzy: bool) -> bool:
    """Does shell rule ``rule`` match simple command ``cmd``?"""
    if rule.specifier is None:
        return True
    spec = rule.specifier
    if not fuzzy:
        return bool(_command_regex(spec, False).match(cmd.text))
    argv = [fold(a) for a in cmd.argv]
    if not argv:
        return False
    argv[0] = argv[0].rsplit("/", 1)[-1]
    joined = " ".join(argv)
    full = " ".join([*(fold(w) for w in cmd.wrappers), *argv])
    rx = _command_regex(spec, True)
    if rx.match(joined) or rx.match(full):
        return True
    return _deny_token_match(fold(spec), argv)


# ----------------------------------------------------------------------------- other specifiers


def match_path_rule(rule: Rule, path: str, workspace: Workspace, *, fuzzy: bool) -> bool:
    if rule.specifier is None or rule.specifier == "*":
        return True
    spec = rule.specifier
    if spec.startswith("domain:"):
        return False
    return workspace.pattern(spec).matches(path, fuzzy=fuzzy)


def match_field_rule(spec: str, inp: dict[str, Any], *, fuzzy: bool) -> bool:
    key, sep, pattern = spec.partition(":")
    if not sep:
        return False
    value = inp.get(key)
    if not isinstance(value, (str, int, float)):
        return False
    text, pat = (fold(str(value)), fold(pattern)) if fuzzy else (str(value), pattern)
    return fnmatch.fnmatchcase(text, pat)


def match_non_shell(rule: Rule, profile: ActionProfile, workspace: Workspace, *, fuzzy: bool) -> bool:
    """Match ``rule`` against a non-shell action (path, domain, field or sub-agent specifier)."""
    if not rule.matches_tool(profile.tool, profile.kind, fuzzy=fuzzy):
        return False
    spec = rule.specifier
    if spec is None or spec == "*":
        return True
    if spec.startswith("domain:"):
        if not profile.domains:
            return False
        hits = [domain_matches(d, spec) for d in profile.domains]
        return any(hits) if fuzzy else all(hits)
    if profile.kind is ToolKind.SUBAGENT:
        inp = profile.call.input
        agent = str(inp.get("subagent_type") or inp.get("agent") or inp.get("name") or "")
        return fnmatch.fnmatchcase(fold(agent), fold(spec)) if fuzzy else fnmatch.fnmatchcase(agent, spec)
    if profile.paths and (rule.tool in _PATH_TOOLS or profile.kind in {ToolKind.READ, ToolKind.SEARCH, ToolKind.EDIT}):
        hits = [match_path_rule(rule, p.path, workspace, fuzzy=fuzzy) for p in profile.paths if p.path]
        if not hits:
            return False
        return any(hits) if fuzzy else all(hits) and len(hits) == len(profile.paths)
    return match_field_rule(spec, profile.call.input, fuzzy=fuzzy)


def match_shell_paths(rule: Rule, profile: ActionProfile, workspace: Workspace) -> list[str]:
    """Our extension: Read/Edit deny/ask rules applied to the paths a shell command touches."""
    if rule.tool not in _READ_FAMILY | _EDIT_FAMILY or rule.specifier is None:
        return []
    want_write = rule.tool in _EDIT_FAMILY
    hits = []
    for p in profile.paths:
        if not p.path:
            continue
        if want_write and not p.write:
            continue
        if match_path_rule(rule, p.path, workspace, fuzzy=True):
            hits.append(p.path)
    return hits
