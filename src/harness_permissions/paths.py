"""Workspace boundary, path canonicalisation and gitignore-style path globs.

"Auto-allow inside the boundary, classify every crossing" (build plan): the boundary is the
workspace (by default the current git repo - the auto-mode "Environment" slot). Every path a
tool would touch is canonicalised with ``realpath`` (``..`` segments and symlinks resolved) before
it is compared with the boundary or with a rule, so ``src/../../etc/passwd`` and a symlink that
points outside both count as *outside*.

Two path classes are hard-denied regardless of mode (research §7.7 steps 1 and 12, §6 protected
subpaths):

* **protected write paths** - policy files, hook config, ``.git`` internals (``.git/hooks`` is the
  classic privilege-escalation vector Codex keeps read-only), harness/agent config dirs;
* **credential paths** - ``~/.ssh``, ``**/.env``, cloud credential stores (srt's read-deny example).

Rule path syntax (Claude Code compatible, plus plain absolute paths):

==================  ==========================================================
``src/**``          relative to the workspace root
``./src/*.py``      same
``/etc/**``         absolute (``//etc/**`` also accepted, Claude Code style)
``~/.ssh/**``       relative to the home directory
``**/.env``         any directory depth, anywhere
==================  ==========================================================

``**`` matches any number of path segments, ``*`` and ``?`` stay within one segment. A pattern
ending in ``/**`` also matches the directory itself.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from ._util import fold

DEFAULT_PROTECTED_PATHS: tuple[str, ...] = (
    ".git/**",
    ".harness/**",
    ".claude/settings.json",
    ".claude/settings.local.json",
    ".claude/hooks/**",
    ".codex/**",
    "harness-policy.toml",
    "harness-policy.yaml",
    "harness-policy.yml",
    "harness-policy.json",
    "~/.harness/**",
    "~/.claude/settings.json",
    "~/.codex/**",
    "/etc/harness/**",
)

DEFAULT_CREDENTIAL_PATHS: tuple[str, ...] = (
    "~/.ssh/**",
    "~/.gnupg/**",
    "~/.aws/**",
    "~/.azure/**",
    "~/.config/gcloud/**",
    "~/.kube/config",
    "~/.docker/config.json",
    "~/.netrc",
    "~/.git-credentials",
    "~/.npmrc",
    "~/.pypirc",
    "~/.config/gh/hosts.yml",
    "**/.env",
    "**/.env.local",
    "**/.env.*.local",
    "**/.env.production",
    "**/.env.development",
    "**/id_rsa",
    "**/id_ed25519",
    "**/*.pem",
    "**/*.key",
    "/etc/shadow",
    "/etc/sudoers",
    "/proc/*/environ",
)


def _glob_to_regex(pattern: str) -> str:
    out: list[str] = []
    i = 0
    n = len(pattern)
    while i < n:
        ch = pattern[i]
        if ch == "*":
            if pattern.startswith("**", i):
                # '**/' -> zero or more segments; trailing '**' -> anything
                if pattern.startswith("**/", i):
                    out.append("(?:.*/)?")
                    i += 3
                else:
                    out.append(".*")
                    i += 2
                continue
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        elif ch == "[":
            j = pattern.find("]", i + 1)
            if j < 0:
                out.append(re.escape(ch))
            else:
                body = pattern[i + 1:j]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append("[" + body.replace("\\", "\\\\") + "]")
                i = j
        else:
            out.append(re.escape(ch))
        i += 1
    regex = "".join(out)
    if pattern.endswith("/**"):
        # also match the directory itself
        regex = regex[: -len("/.*")] + "(?:/.*)?" if regex.endswith("/.*") else regex
    return regex


@lru_cache(maxsize=2048)
def _compile(pattern: str, fuzzy: bool) -> re.Pattern[str]:
    return re.compile(_glob_to_regex(fold(pattern) if fuzzy else pattern) + r"\Z", re.DOTALL)


@dataclass(frozen=True)
class PathPattern:
    """A path glob anchored to an absolute base (or floating, for ``**/x``)."""

    source: str
    absolute: str          # absolute glob, or a floating glob starting with '**/'

    @classmethod
    def parse(cls, pattern: str, *, root: str, home: str | None = None) -> "PathPattern":
        home = home or os.path.expanduser("~")
        p = pattern.strip()
        if not p:
            raise ValueError("empty path pattern")
        if p.startswith("//"):
            absolute = "/" + p.lstrip("/")
        elif p.startswith("/"):
            absolute = p
        elif p == "~" or p.startswith("~/"):
            absolute = home.rstrip("/") + p[1:]
        elif p.startswith("**/") or p == "**":
            absolute = p
        else:
            rel = p[2:] if p.startswith("./") else p
            absolute = root.rstrip("/") + "/" + rel
        if not absolute.startswith("**"):
            absolute = _normpath_glob(absolute)
        return cls(source=pattern, absolute=absolute)

    def matches(self, path: str, *, fuzzy: bool = False) -> bool:
        """Match a canonical absolute path. ``fuzzy`` (deny side) folds case and Unicode."""
        candidate = fold(path) if fuzzy else path
        if self.absolute.startswith("**"):
            pat = self.absolute[3:] if self.absolute.startswith("**/") else ""
            if not pat:
                return True
            return bool(_compile("**/" + pat, fuzzy).match(candidate.lstrip("/")))
        return bool(_compile(self.absolute, fuzzy).match(candidate))


def _normpath_glob(p: str) -> str:
    """Collapse '.', '..' and duplicate slashes in a glob without touching glob characters."""
    parts: list[str] = []
    for seg in p.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if parts:
                parts.pop()
            continue
        parts.append(seg)
    return "/" + "/".join(parts)


def canonicalize(path: str, *, cwd: str, home: str | None = None) -> str:
    """Absolute, symlink-resolved, normalised path for ``path`` as seen from ``cwd``.

    Paths that do not exist yet are resolved as far as possible (``realpath`` resolves the existing
    prefix), so a *new* file under a symlinked directory still lands where the symlink points.
    """
    if "\x00" in path:
        raise ValueError("path contains a NUL byte")
    home = home or os.path.expanduser("~")
    p = path.strip()
    if p == "~" or p.startswith("~/"):
        p = home + p[1:]
    if not os.path.isabs(p):
        p = os.path.join(cwd, p)
    return os.path.realpath(p)


def _within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


@dataclass
class Workspace:
    """The trust boundary: writable roots plus extra read-only roots and protected sub-paths."""

    roots: list[str]
    read_roots: list[str] = field(default_factory=list)
    protected_paths: list[str] = field(default_factory=lambda: list(DEFAULT_PROTECTED_PATHS))
    credential_paths: list[str] = field(default_factory=lambda: list(DEFAULT_CREDENTIAL_PATHS))
    home: str = field(default_factory=lambda: os.path.expanduser("~"))
    temp_roots: list[str] = field(default_factory=lambda: ["/tmp"])

    def __post_init__(self) -> None:
        if not self.roots:
            raise ValueError("workspace needs at least one root")
        self.roots = [os.path.realpath(os.path.expanduser(r)) for r in self.roots]
        self.read_roots = [os.path.realpath(os.path.expanduser(r)) for r in self.read_roots]
        self.temp_roots = [os.path.realpath(r) for r in self.temp_roots]
        self._protected = [PathPattern.parse(p, root=self.root, home=self.home) for p in self.protected_paths]
        self._credentials = [PathPattern.parse(p, root=self.root, home=self.home) for p in self.credential_paths]

    @property
    def root(self) -> str:
        return self.roots[0]

    def protect(self, paths: Iterable[str]) -> None:
        """Add protected write paths (e.g. the policy files that were actually loaded)."""
        for p in paths:
            self.protected_paths.append(p)
            self._protected.append(PathPattern.parse(p, root=self.root, home=self.home))

    def canonical(self, path: str, cwd: str | None = None) -> str:
        return canonicalize(path, cwd=cwd or self.root, home=self.home)

    def pattern(self, pattern: str) -> PathPattern:
        return PathPattern.parse(pattern, root=self.root, home=self.home)

    def inside(self, canonical_path: str, *, for_write: bool = False) -> bool:
        roots = self.roots + self.temp_roots if for_write else self.roots + self.read_roots + self.temp_roots
        return any(_within(canonical_path, r) for r in roots)

    def is_protected(self, canonical_path: str) -> str | None:
        """The protected pattern ``canonical_path`` falls under (fuzzy match), or ``None``."""
        for pat in self._protected:
            if pat.matches(canonical_path, fuzzy=True):
                return pat.source
        return None

    def is_credential(self, canonical_path: str) -> str | None:
        name = os.path.basename(canonical_path)
        if name.lower() in {".env.example", ".env.sample", ".env.template", ".env.dist"}:
            return None
        for pat in self._credentials:
            if pat.matches(canonical_path, fuzzy=True):
                return pat.source
        return None

    @classmethod
    def from_cwd(cls, cwd: str | Path = ".") -> "Workspace":
        """Default Environment slot: the enclosing git repository, else ``cwd`` itself."""
        start = Path(cwd).resolve()
        for candidate in [start, *start.parents]:
            if (candidate / ".git").exists():
                return cls(roots=[str(candidate)])
        return cls(roots=[str(start)])
