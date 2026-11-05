"""Codex-style exec policy: tested prefix rules with pinned executables.

Mirrors ``codex-rs/execpolicy`` (research §7.3 technique 5, checklist §7.7 step 5):

* ``prefix_rule(pattern, decision="allow"|"prompt"|"forbidden", justification, match, not_match)``.
  ``pattern`` is a list of tokens; a token may itself be a list of alternatives
  (``["git", ["status", "diff", "log"]]``). A rule matches when argv *starts with* the pattern.
* ``justification`` is mandatory (our checklist), and the ``match`` / ``not_match`` examples are
  executed as unit tests at load time - a rule whose examples fail is rejected with
  :class:`~harness_permissions.errors.PolicyValidationError`.
* ``host_executable(name, paths)`` pins which absolute binaries may match a basename rule:
  "pinning executables prevents a repo-local ``git`` shim from matching an 'allow git' rule".
  A command invoked by path (``./git``, ``/tmp/git``) matches ``allow``/``prompt`` rules only if
  the path is pinned; ``forbidden`` rules always match by basename (fail closed).

When several rules match, the strictest decision wins (forbidden > prompt > allow).
"""

from __future__ import annotations

import posixpath
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence, Union

from .errors import PolicyValidationError
from .shell import SimpleCommand, analyze

Token = Union[str, Sequence[str]]


class ExecDecision(str, Enum):
    ALLOW = "allow"
    PROMPT = "prompt"
    FORBIDDEN = "forbidden"

    @property
    def severity(self) -> int:
        return {"allow": 0, "prompt": 1, "forbidden": 2}[self.value]


@dataclass(frozen=True)
class PrefixRule:
    pattern: tuple[tuple[str, ...], ...]
    decision: ExecDecision
    justification: str
    match: tuple[tuple[str, ...], ...] = ()
    not_match: tuple[tuple[str, ...], ...] = ()

    @property
    def id(self) -> str:
        return "exec:" + " ".join("|".join(alts) if len(alts) > 1 else alts[0] for alts in self.pattern)

    def matches_argv(self, argv: Sequence[str]) -> bool:
        if len(argv) < len(self.pattern):
            return False
        first = posixpath.basename(argv[0])
        for i, alts in enumerate(self.pattern):
            tok = first if i == 0 else argv[i]
            if tok not in alts:
                return False
        return True


@dataclass(frozen=True)
class HostExecutable:
    name: str
    paths: tuple[str, ...]


@dataclass(frozen=True)
class ExecMatch:
    decision: ExecDecision
    rule: PrefixRule


def _tokens(pattern: Sequence[Token]) -> tuple[tuple[str, ...], ...]:
    out: list[tuple[str, ...]] = []
    for t in pattern:
        if isinstance(t, str):
            out.append((t,))
        elif isinstance(t, (list, tuple)) and t and all(isinstance(x, str) for x in t):
            out.append(tuple(t))
        else:
            raise PolicyValidationError(f"exec pattern token must be a string or non-empty list of strings, got {t!r}")
    if not out:
        raise PolicyValidationError("exec pattern must not be empty")
    return tuple(out)


def _example(ex: Any) -> tuple[str, ...]:
    if isinstance(ex, str):
        return tuple(shlex.split(ex))
    if isinstance(ex, (list, tuple)) and all(isinstance(x, str) for x in ex):
        return tuple(ex)
    raise PolicyValidationError(f"exec example must be a string or list of strings, got {ex!r}")


def prefix_rule(
    pattern: Sequence[Token],
    decision: str | ExecDecision = "allow",
    justification: str = "",
    match: Sequence[Any] = (),
    not_match: Sequence[Any] = (),
) -> PrefixRule:
    """Build and *self-test* a prefix rule (examples run immediately)."""
    if not justification or not justification.strip():
        raise PolicyValidationError(f"exec rule {list(pattern)!r}: justification is required")
    try:
        dec = ExecDecision(decision)
    except ValueError as exc:
        raise PolicyValidationError(f"exec rule {list(pattern)!r}: decision must be allow|prompt|forbidden, got {decision!r}") from exc
    rule = PrefixRule(
        pattern=_tokens(pattern),
        decision=dec,
        justification=justification.strip(),
        match=tuple(_example(m) for m in match),
        not_match=tuple(_example(m) for m in not_match),
    )
    problems = [f"{rule.id}: example {' '.join(m)!r} should match" for m in rule.match if not rule.matches_argv(m)]
    problems += [f"{rule.id}: example {' '.join(m)!r} should not match" for m in rule.not_match if rule.matches_argv(m)]
    if problems:
        raise PolicyValidationError(problems)
    return rule


def host_executable(name: str, paths: Sequence[str]) -> HostExecutable:
    if not name or "/" in name:
        raise PolicyValidationError(f"host_executable name must be a bare basename, got {name!r}")
    bad = [p for p in paths if not p.startswith("/")]
    if bad or not paths:
        raise PolicyValidationError(f"host_executable {name!r}: paths must be non-empty absolute paths, got {list(paths)!r}")
    return HostExecutable(name=name, paths=tuple(paths))


@dataclass
class ExecPolicy:
    rules: list[PrefixRule] = field(default_factory=list)
    host_executables: dict[str, HostExecutable] = field(default_factory=dict)

    def add(self, rule: PrefixRule) -> None:
        self.rules.append(rule)

    def pin(self, exe: HostExecutable) -> None:
        self.host_executables[exe.name] = exe

    def _path_ok(self, cmd_path: str | None, name: str) -> bool:
        if cmd_path is None:
            return True  # resolved through PATH; the sandbox controls PATH
        pinned = self.host_executables.get(name)
        return pinned is not None and cmd_path in pinned.paths

    def evaluate_argv(self, argv: Sequence[str], *, executable_path: str | None = None) -> ExecMatch | None:
        best: ExecMatch | None = None
        if not argv:
            return None
        name = posixpath.basename(argv[0])
        for rule in self.rules:
            if not rule.matches_argv(argv):
                continue
            if rule.decision is not ExecDecision.FORBIDDEN and not self._path_ok(executable_path, name):
                continue
            if best is None or rule.decision.severity > best.decision.severity:
                best = ExecMatch(rule.decision, rule)
        return best

    def evaluate(self, cmd: SimpleCommand) -> ExecMatch | None:
        return self.evaluate_argv(cmd.argv, executable_path=cmd.executable_path)

    def check(self, command: str) -> list[tuple[str, ExecMatch | None]]:
        """``codex execpolicy check`` equivalent: decision per simple command of ``command``."""
        return [(c.text, self.evaluate(c)) for c in analyze(command).commands]
