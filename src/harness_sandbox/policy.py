"""Declarative sandbox policy: filesystem, network, resource limits, environment and escape hatches.

The shape follows the two open implementations the research recommends copying
(``reports/SOTA agentic harness reference.md`` §6.2, §6.5, checklist §6.7 items 1, 3, 4, 18):

* **One mode enum mirroring Codex** ``SandboxPolicy``: ``read-only``, ``workspace-write`` (default),
  ``external-sandbox`` (already inside a container/VM: skip the OS sandbox, keep policy and proxy) and
  ``full-access`` (never default; admin only; logged loudly).
* **srt precedence rules.** Reads are allowed by default with deny-then-allow: ``allow_read`` beats
  ``deny_read`` *unless the deny is more specific* (``**/.env`` stays denied under an allowed tree).
  Writes are denied by default and allow-only; ``deny_write`` beats ``allow_write``.
* **Codex protected subpaths.** ``.git`` (hence ``.git/hooks``), the resolved gitdir of a worktree and
  our own ``.harness`` config directory stay read-only even inside writable roots, because they
  execute later, outside the sandbox (privilege escalation via a planted hook).
* **Secrets denied by default.** ``~/.ssh`` (srt example), ``**/.env`` (srt), plus cloud-credential
  directories and container-runtime sockets (our addition, per the checklist).
* **Network denied by default** (srt: "By default, all network access is denied"). ``allowlist`` mode
  routes through the egress proxy; ``limited`` is Codex's read-only network mode.
* **Escape hatches are off by default and need admin approval** (checklist item 18). Admin approval is
  a parameter of :meth:`SandboxPolicy.validate` / :func:`load_policy`, never a key in the policy file,
  so an agent that can edit the file still cannot weaken its own sandbox.

The policy is pure data. :meth:`SandboxPolicy.compile` resolves it against a concrete workspace into a
:class:`CompiledPolicy` that answers ``check_read`` / ``check_write`` questions (used by file tools and
by every runner when it builds its argv or profile).
"""

from __future__ import annotations

import fnmatch
import hashlib
import ipaddress
import json
import logging
import os
import re
import urllib.parse
from dataclasses import asdict, dataclass, field, fields, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import EscapeHatchError, PolicyError

log = logging.getLogger("harness_sandbox")


# ----------------------------------------------------------------------------- enums


class SandboxMode(str, Enum):
    """Codex ``SandboxPolicy`` variants (checklist item 1)."""

    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    EXTERNAL_SANDBOX = "external-sandbox"
    FULL_ACCESS = "full-access"

    @classmethod
    def parse(cls, value: str | "SandboxMode") -> "SandboxMode":
        if isinstance(value, SandboxMode):
            return value
        norm = str(value).strip().lower().replace("_", "-")
        aliases = {"danger-full-access": "full-access", "readonly": "read-only"}
        try:
            return cls(aliases.get(norm, norm))
        except ValueError as exc:
            raise PolicyError(f"unknown sandbox mode {value!r}; expected one of {[m.value for m in cls]}") from exc


class NetworkMode(str, Enum):
    """``none``: no network at all. ``allowlist``: egress only via the proxy, domain allowlist.
    ``limited``: like allowlist but read-only (GET/HEAD/OPTIONS; see proxy docs). ``open``: host network."""

    NONE = "none"
    ALLOWLIST = "allowlist"
    LIMITED = "limited"
    OPEN = "open"

    @classmethod
    def parse(cls, value: str | "NetworkMode") -> "NetworkMode":
        if isinstance(value, NetworkMode):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise PolicyError(f"unknown network mode {value!r}; expected one of {[m.value for m in cls]}") from exc

    @property
    def uses_proxy(self) -> bool:
        return self in (NetworkMode.ALLOWLIST, NetworkMode.LIMITED)


# ----------------------------------------------------------------------------- defaults

#: Credential locations read-denied by default. ``~/.ssh`` and ``**/.env`` come from srt's example
#: config; cloud/CLI credential stores and runtime sockets are our addition (checklist item 3).
DEFAULT_DENY_READ: tuple[str, ...] = (
    "~/.ssh",
    "~/.aws",
    "~/.azure",
    "~/.config/gcloud",
    "~/.config/gh",
    "~/.kube",
    "~/.docker",
    "~/.gnupg",
    "~/.netrc",
    "~/.git-credentials",
    "~/.npmrc",
    "~/.pypirc",
    "~/.cargo/credentials",
    "~/.cargo/credentials.toml",
    "~/.harness/secrets",
    "**/.env",
    "**/.env.*",
    # Unix sockets are an escape path (Docker socket = root on the host); see srt allowUnixSockets.
    "/var/run/docker.sock",
    "/run/docker.sock",
    "/run/podman",
    "/run/containerd",
    "/run/user",
)

#: Relative to every writable root (Codex ``read_only_subpaths`` plus our harness config dir).
DEFAULT_READ_ONLY_SUBPATHS: tuple[str, ...] = (".git", ".harness")

#: Small default allowlist (checklist item 4: "package registries, docs hosts"). The list is ours.
#: Code hosts that accept pushes (github.com...) are deliberately absent: they are exfiltration
#: channels for any token an injection smuggles in. Add them per project.
DEFAULT_ALLOWED_DOMAINS: tuple[str, ...] = (
    "pypi.org",
    "files.pythonhosted.org",
    "registry.npmjs.org",
    "registry.yarnpkg.com",
    "crates.io",
    "static.crates.io",
    "index.crates.io",
    "proxy.golang.org",
    "sum.golang.org",
    "docs.python.org",
)

#: Environment variables passed through from the host (names or fnmatch globs).
DEFAULT_ENV_ALLOW: tuple[str, ...] = (
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LANGUAGE", "LC_*", "TERM", "TZ",
    "COLORTERM", "NO_COLOR", "FORCE_COLOR", "PYTHONIOENCODING", "PYTHONUTF8", "SOURCE_DATE_EPOCH",
)

#: Names that are always scrubbed, even if allowlisted or passed explicitly (checklist item 9:
#: "nothing long-lived inside the sandbox"; model API keys are injected harness-side).
DEFAULT_ENV_DENY: tuple[str, ...] = (
    "*TOKEN*", "*SECRET*", "*PASSWORD*", "*PASSWD*", "*API_KEY*", "*APIKEY*", "*ACCESS_KEY*",
    "*PRIVATE_KEY*", "*CREDENTIAL*", "*_AUTH", "*_AUTH_*", "*SESSION_KEY*", "AWS_*", "AZURE_*",
    "GOOGLE_APPLICATION_CREDENTIALS", "ANTHROPIC_*", "OPENAI_*", "GH_*", "GITHUB_*", "SSH_AUTH_SOCK",
    "GPG_AGENT_INFO", "KUBECONFIG", "DOCKER_HOST", "DOCKER_CONFIG", "VAULT_*", "NPM_CONFIG__AUTH*",
    "*_PROXY",  # host proxy settings may embed credentials; ours are injected instead
)

KIB = 1024
MIB = 1024 * KIB
GIB = 1024 * MIB


# ----------------------------------------------------------------------------- sub-policies


@dataclass(frozen=True)
class FilesystemPolicy:
    """Read/write rules. Patterns: absolute paths, ``~`` paths, workspace-relative paths, or globs
    (``*``, ``?``, ``[]``, ``**``). A relative pattern starting with ``**/`` matches anywhere."""

    allow_read: tuple[str, ...] = ()
    deny_read: tuple[str, ...] = DEFAULT_DENY_READ
    allow_write: tuple[str, ...] = ()            # extra writable roots beyond workspace + session tmp
    deny_write: tuple[str, ...] = ()
    read_only_subpaths: tuple[str, ...] = DEFAULT_READ_ONLY_SUBPATHS
    workspace_writable: bool = True              # ignored in read-only mode
    session_tmp: bool = True                     # per-session writable tmp directory

    def __post_init__(self) -> None:
        # Invariant: the default secret denies and protected subpaths can never be dropped, however
        # the policy is built; a more specific ``allow_read`` is the only way to carve an exception.
        object.__setattr__(self, "deny_read", _dedupe((*DEFAULT_DENY_READ, *self.deny_read)))
        object.__setattr__(self, "read_only_subpaths", _dedupe((*DEFAULT_READ_ONLY_SUBPATHS, *self.read_only_subpaths)))


@dataclass(frozen=True)
class NetworkPolicy:
    mode: NetworkMode = NetworkMode.NONE
    allowed_domains: tuple[str, ...] = DEFAULT_ALLOWED_DOMAINS
    denied_domains: tuple[str, ...] = ()
    allowed_ports: tuple[int, ...] = (80, 443)
    #: Interactive sessions may ask the user about unknown domains; headless sessions deny.
    ask_on_unknown: bool = False
    proxy_port_in_sandbox: int = 3128            # Codex default HTTP proxy port

    def domain_decision(self, host: str) -> tuple[bool | None, str]:
        """``(True, rule)`` allowlisted, ``(False, rule)`` denylisted, ``(None, "")`` unknown."""
        h = normalise_host(host)
        for pat in self.denied_domains:
            if domain_matches(pat, h):
                return False, f"denied_domains:{pat}"
        for pat in self.allowed_domains:
            if domain_matches(pat, h):
                return True, f"allowed_domains:{pat}"
        return None, ""


@dataclass(frozen=True)
class ResourceLimits:
    """Per-command limits (checklist item 11: CPU, memory, wall clock, processes, plus monitoring).

    ``memory_bytes`` is enforced with ``RLIMIT_DATA`` by default (Linux >= 4.7 counts private writable
    mappings), because ``RLIMIT_AS`` also counts address-space *reservations* and breaks V8/Go/wasm.
    """

    cpu_seconds: int | None = 600
    memory_bytes: int | None = 4 * GIB
    memory_limit_kind: str = "data"              # "data" (RLIMIT_DATA) or "as" (RLIMIT_AS)
    max_processes: int | None = 1024             # RLIMIT_NPROC (per real uid!) / --pids-limit
    file_size_bytes: int | None = 1 * GIB        # RLIMIT_FSIZE
    open_files: int | None = 4096                # RLIMIT_NOFILE
    core_dumps: bool = False                     # RLIMIT_CORE=0: never dump process memory to disk
    default_timeout_s: float = 120.0             # Claude Code Bash tool default (2 min)
    max_timeout_s: float = 600.0                 # Claude Code Bash tool maximum (10 min)
    max_output_bytes: int = 1 * MIB              # per stream; head + tail kept
    kill_grace_s: float = 2.0                    # SIGTERM -> SIGKILL grace on timeout

    def effective_timeout(self, requested: float | None) -> float:
        t = self.default_timeout_s if requested is None else float(requested)
        if t <= 0:
            raise PolicyError("timeout must be positive")
        return min(t, self.max_timeout_s)


@dataclass(frozen=True)
class EnvPolicy:
    allow: tuple[str, ...] = DEFAULT_ENV_ALLOW
    deny: tuple[str, ...] = DEFAULT_ENV_DENY
    set: Mapping[str, str] = field(default_factory=dict)
    scrub_secret_values: bool = True             # drop any value the secret redactor flags

    def __post_init__(self) -> None:
        object.__setattr__(self, "deny", _dedupe((*DEFAULT_ENV_DENY, *self.deny)))


@dataclass(frozen=True)
class EscapeHatches:
    """Weakening knobs (srt ``enableWeaker*``, ``allowUnixSockets``; Codex
    ``dangerously_allow_non_loopback_proxy``). Every one needs admin approval."""

    allow_non_loopback_proxy: bool = False
    allow_nested_namespaces: bool = False        # disables the seccomp namespace block
    allowed_private_cidrs: tuple[str, ...] = ()  # private/loopback ranges the proxy may reach
    allow_unix_sockets: tuple[str, ...] = ()     # socket paths re-allowed through the default masks
    env_secret_passthrough: tuple[str, ...] = () # env names exempted from secret scrubbing
    disable_seccomp: bool = False

    def active(self) -> list[str]:
        return [f.name for f in fields(self) if getattr(self, f.name)]


# ----------------------------------------------------------------------------- top-level policy


@dataclass(frozen=True)
class SandboxPolicy:
    mode: SandboxMode = SandboxMode.WORKSPACE_WRITE
    filesystem: FilesystemPolicy = field(default_factory=FilesystemPolicy)
    network: NetworkPolicy = field(default_factory=NetworkPolicy)
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    env: EnvPolicy = field(default_factory=EnvPolicy)
    escape_hatches: EscapeHatches = field(default_factory=EscapeHatches)
    redact_output: bool = True                   # scrub secrets from captured stdout/stderr

    # -------------------------------------------------------------- validation
    def weakenings(self) -> list[str]:
        """Names of every active weakening (escape hatches plus ``full-access`` mode)."""
        out = [f"escape_hatches.{n}" for n in self.escape_hatches.active()]
        if self.mode is SandboxMode.FULL_ACCESS:
            out.insert(0, "mode=full-access")
        return out

    def validate(self, *, admin: bool = False) -> "SandboxPolicy":
        """Check invariants; raise :class:`EscapeHatchError` for weakenings without ``admin=True``."""
        lim = self.limits
        for name in ("cpu_seconds", "memory_bytes", "max_processes", "file_size_bytes", "open_files"):
            v = getattr(lim, name)
            if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v <= 0):
                raise PolicyError(f"limits.{name} must be a positive integer or null, got {v!r}")
        if lim.memory_limit_kind not in ("data", "as"):
            raise PolicyError("limits.memory_limit_kind must be 'data' or 'as'")
        if not (0 < lim.default_timeout_s <= lim.max_timeout_s):
            raise PolicyError("limits require 0 < default_timeout_s <= max_timeout_s")
        if lim.max_output_bytes < 1024:
            raise PolicyError("limits.max_output_bytes must be >= 1024")
        for pat in (*self.network.allowed_domains, *self.network.denied_domains):
            validate_domain_pattern(pat)
        for port in self.network.allowed_ports:
            if not (isinstance(port, int) and 0 < port < 65536):
                raise PolicyError(f"invalid port {port!r}")
        if not (0 < self.network.proxy_port_in_sandbox < 65536):
            raise PolicyError("network.proxy_port_in_sandbox out of range")
        for cidr in self.escape_hatches.allowed_private_cidrs:
            try:
                ipaddress.ip_network(cidr, strict=False)
            except ValueError as exc:
                raise PolicyError(f"invalid CIDR {cidr!r}") from exc
        for k, v in self.env.set.items():
            if not isinstance(k, str) or not isinstance(v, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
                raise PolicyError(f"env.set entries must be NAME: string, got {k!r}")
        for pat in (*self.filesystem.allow_read, *self.filesystem.deny_read, *self.filesystem.allow_write,
                    *self.filesystem.deny_write, *self.filesystem.read_only_subpaths):
            if not isinstance(pat, str) or not pat or "\x00" in pat:
                raise PolicyError(f"invalid path pattern {pat!r}")
        weak = self.weakenings()
        if weak:
            if not admin:
                raise EscapeHatchError(f"policy weakens the sandbox ({', '.join(weak)}); requires admin approval")
            log.warning("SANDBOX WEAKENED by admin policy: %s", ", ".join(weak))
        if self.network.mode is NetworkMode.OPEN and self.mode in (SandboxMode.READ_ONLY, SandboxMode.WORKSPACE_WRITE):
            log.warning("sandbox network mode is 'open': filesystem isolation alone does not stop exfiltration")
        return self

    # -------------------------------------------------------------- (de)serialisation
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mode"] = self.mode.value
        d["network"]["mode"] = self.network.mode.value
        d["env"]["set"] = dict(self.env.set)
        return _lists(d)

    def fingerprint(self) -> str:
        """Stable hash; long-lived shells/PTYs must be restarted when it changes (checklist 14)."""
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, admin: bool = False) -> "SandboxPolicy":
        """Build from a mapping. Deny-type lists **extend** the defaults (enforced by the dataclasses;
        defaults can only be overridden by a more specific ``allow_read``); ``allowed_domains``
        replaces the default list."""
        if not isinstance(data, Mapping):
            raise PolicyError("policy must be a mapping")
        _reject_unknown(data, {"mode", "filesystem", "network", "limits", "env", "escape_hatches", "redact_output"}, "policy")
        kw: dict[str, Any] = {}
        if "mode" in data:
            kw["mode"] = SandboxMode.parse(data["mode"])
        if "redact_output" in data:
            kw["redact_output"] = _bool(data["redact_output"], "redact_output")
        if "filesystem" in data:
            kw["filesystem"] = FilesystemPolicy(**_section(data["filesystem"], FilesystemPolicy, "filesystem"))
        if "network" in data:
            net = _section(data["network"], NetworkPolicy, "network")
            if "mode" in net:
                net["mode"] = NetworkMode.parse(net["mode"])
            if "allowed_ports" in net:
                net["allowed_ports"] = tuple(int(p) for p in net["allowed_ports"])
            kw["network"] = NetworkPolicy(**net)
        if "limits" in data:
            kw["limits"] = ResourceLimits(**_section(data["limits"], ResourceLimits, "limits"))
        if "env" in data:
            env = _section(data["env"], EnvPolicy, "env")
            if "set" in env:
                if not isinstance(env["set"], Mapping):
                    raise PolicyError("env.set must be a mapping")
                env["set"] = {str(k): str(v) for k, v in env["set"].items()}
            kw["env"] = EnvPolicy(**env)
        if "escape_hatches" in data:
            kw["escape_hatches"] = EscapeHatches(**_section(data["escape_hatches"], EscapeHatches, "escape_hatches"))
        return cls(**kw).validate(admin=admin)

    # -------------------------------------------------------------- compilation
    def compile(self, workspace: str | os.PathLike[str], *, home: str | None = None,
                session_tmp: str | os.PathLike[str] | None = None) -> "CompiledPolicy":
        return CompiledPolicy(self, Path(workspace), home=home, session_tmp=Path(session_tmp) if session_tmp else None)

    def with_mode(self, mode: SandboxMode | str) -> "SandboxPolicy":
        return replace(self, mode=SandboxMode.parse(mode))


def load_policy(path: str | os.PathLike[str], *, admin: bool = False) -> SandboxPolicy:
    """Load a policy from ``.yaml``/``.yml``/``.json`` (YAML parsing also accepts JSON)."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PolicyError(f"{p}: invalid JSON: {exc}") from exc
    else:
        import yaml

        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise PolicyError(f"{p}: invalid YAML: {exc}") from exc
    return SandboxPolicy.from_dict(data or {}, admin=admin)


# ----------------------------------------------------------------------------- helpers (parsing)


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise PolicyError(f"unknown key(s) in {where}: {sorted(unknown)}")


def _section(raw: Any, cls: type, where: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise PolicyError(f"{where} must be a mapping")
    names = {f.name: f for f in fields(cls)}
    _reject_unknown(raw, set(names), where)
    out: dict[str, Any] = {}
    for k, v in raw.items():
        default = getattr(cls(), k)
        if isinstance(default, tuple):
            if isinstance(v, str) or not isinstance(v, Iterable):
                raise PolicyError(f"{where}.{k} must be a list")
            out[k] = tuple(v)
        elif isinstance(default, bool):
            out[k] = _bool(v, f"{where}.{k}")
        elif isinstance(default, float) and isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _bool(v: Any, where: str) -> bool:
    if not isinstance(v, bool):
        raise PolicyError(f"{where} must be true/false")
    return v


def _dedupe(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def _lists(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _lists(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_lists(v) for v in obj]
    return obj


# ----------------------------------------------------------------------------- domains


def normalise_host(host: str) -> str:
    h = host.strip().rstrip(".").lower()
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]
    try:
        return h.encode("idna").decode("ascii") if h and not _is_ip(h) else h
    except UnicodeError:
        return h


def _is_ip(h: str) -> bool:
    try:
        ipaddress.ip_address(h)
        return True
    except ValueError:
        return False


def validate_domain_pattern(pat: str) -> None:
    if not isinstance(pat, str) or not pat:
        raise PolicyError(f"invalid domain pattern {pat!r}")
    if "://" in pat or "/" in pat or (":" in pat and not _is_ip(pat)):
        raise PolicyError(f"domain pattern must be a bare host, got {pat!r}")
    body = pat[2:] if pat.startswith("*.") else pat
    if "*" in body or not body:
        raise PolicyError(f"wildcards are only allowed as a leading '*.': {pat!r}")


def domain_matches(pattern: str, host: str) -> bool:
    """Exact match, or ``*.example.com`` matching any subdomain (not the apex)."""
    p = normalise_host(pattern[2:] if pattern.startswith("*.") else pattern)
    h = normalise_host(host)
    if pattern.startswith("*."):
        return h.endswith("." + p)
    return h == p


# ----------------------------------------------------------------------------- paths


@dataclass(frozen=True)
class PathDecision:
    allowed: bool
    path: str
    operation: str          # "read" | "write"
    rule: str = ""          # the rule that decided (for violation feedback)
    reason: str = ""

    def __bool__(self) -> bool:
        return self.allowed


@dataclass(frozen=True)
class PathRule:
    """One compiled pattern. ``literal`` rules match the path and everything beneath it."""

    source: str
    list_name: str
    literal: str | None = None
    regex: re.Pattern[str] | None = None

    def match(self, path: str) -> int:
        """Specificity of the match (number of path components), or ``-1`` for no match."""
        if self.literal is not None:
            if path == self.literal or path.startswith(self.literal.rstrip("/") + "/"):
                return _depth(self.literal)
            return -1
        assert self.regex is not None
        cand = path
        while True:
            if self.regex.fullmatch(cand):
                return _depth(cand)  # a glob matching a node is as specific as that node
            if cand == "/":
                return -1
            cand = os.path.dirname(cand)

    @property
    def is_glob(self) -> bool:
        return self.regex is not None


def _depth(p: str) -> int:
    return len([x for x in p.split("/") if x])


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a path glob: ``**`` crosses directories, ``*``/``?`` do not, ``[..]`` classes."""
    i, out = 0, []
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "[":
            j = pattern.find("]", i + 1)
            if j == -1:
                out.append(re.escape(c))
                i += 1
            else:
                body = pattern[i + 1:j].replace("\\", "\\\\")
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append(f"[{body}]")
                i = j + 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("".join(out))


def _has_glob(p: str) -> bool:
    return any(ch in p for ch in "*?[")


def canonicalize(path: str | os.PathLike[str], base: str | os.PathLike[str]) -> str:
    """Absolute, normalised path; rejects NUL bytes and percent-encoded traversal.

    Plain ``..`` is resolved lexically and the *result* is checked against the rules
    (resolve-then-check, as the Anthropic memory-tool guidance prescribes). Percent-encoded
    traversal (``%2e%2e%2f``, double-encoded) is never legitimate in a tool path and is rejected.
    """
    s = os.fspath(path)
    if "\x00" in s:
        raise PolicyError("path contains a NUL byte")
    decoded = s
    for _ in range(3):
        nxt = urllib.parse.unquote(decoded)
        if nxt == decoded:
            break
        decoded = nxt
    if decoded != s and re.search(r"(^|[\\/])\.\.([\\/]|$)", decoded):
        raise PolicyError(f"encoded path traversal rejected: {s!r}")
    s = os.path.expanduser(s)
    if not os.path.isabs(s):
        s = os.path.join(os.fspath(base), s)
    return os.path.normpath(s)


class CompiledPolicy:
    """A :class:`SandboxPolicy` resolved against one workspace, home directory and session tmp."""

    def __init__(self, policy: SandboxPolicy, workspace: Path, *, home: str | None = None, session_tmp: Path | None = None):
        self.policy = policy
        self.home = os.path.normpath(home or os.path.expanduser("~"))
        self.workspace = os.path.realpath(os.fspath(workspace))
        self.session_tmp = os.path.realpath(os.fspath(session_tmp)) if session_tmp else None
        fs = policy.filesystem
        hatches = policy.escape_hatches

        roots: list[str] = []
        if policy.mode in (SandboxMode.WORKSPACE_WRITE, SandboxMode.EXTERNAL_SANDBOX, SandboxMode.FULL_ACCESS):
            if fs.workspace_writable:
                roots.append(self.workspace)
            roots.extend(self._literal(p) for p in fs.allow_write if not _has_glob(p))
        if self.session_tmp and fs.session_tmp:
            roots.append(self.session_tmp)
        self.writable_roots: tuple[str, ...] = tuple(dict.fromkeys(roots))

        self.allow_read = self._rules((*fs.allow_read, *hatches.allow_unix_sockets), "allow_read")
        self.deny_read = self._rules(fs.deny_read, "deny_read")
        self.allow_write_globs = tuple(r for r in self._rules(fs.allow_write, "allow_write") if r.is_glob)

        protected: list[str] = []
        for root in self.writable_roots:
            for sub in fs.read_only_subpaths:
                if os.path.isabs(os.path.expanduser(sub)):
                    continue
                protected.append(os.path.join(root, sub))
            protected.extend(_resolved_gitdirs(root))
        protected.extend(self._literal(s) for s in fs.read_only_subpaths if os.path.isabs(os.path.expanduser(s)))
        self.read_only_paths: tuple[str, ...] = tuple(dict.fromkeys(os.path.normpath(p) for p in protected))
        self.deny_write = (
            *self._rules(fs.deny_write, "deny_write"),
            *(PathRule(p, "read_only_subpaths", literal=p) for p in self.read_only_paths),
        )

    # -------------------------------------------------------------- compile helpers
    def _literal(self, pattern: str) -> str:
        p = self.home + pattern[1:] if (pattern == "~" or pattern.startswith("~/")) else pattern
        if not os.path.isabs(p):
            p = os.path.join(self.workspace, p)
        return os.path.normpath(p)

    def _rules(self, patterns: Iterable[str], list_name: str) -> tuple[PathRule, ...]:
        out: list[PathRule] = []
        for pat in patterns:
            if _has_glob(pat):
                if pat.startswith("**"):
                    g = pat
                else:
                    g = self._literal(pat)
                out.append(PathRule(pat, list_name, regex=glob_to_regex(g if g.startswith(("/", "**")) else "/" + g)))
            else:
                lit = self._literal(pat)
                out.append(PathRule(pat, list_name, literal=lit))
                real = os.path.realpath(lit)
                if real != lit:
                    out.append(PathRule(pat, list_name, literal=real))
        return tuple(out)

    # -------------------------------------------------------------- evaluation
    @staticmethod
    def _best(rules: Iterable[PathRule], path: str) -> tuple[int, PathRule | None]:
        best, which = -1, None
        for r in rules:
            s = r.match(path)
            if s > best:
                best, which = s, r
        return best, which

    def _candidates(self, path: str | os.PathLike[str], cwd: str | None) -> list[str]:
        lexical = canonicalize(path, cwd or self.workspace)
        real = os.path.realpath(lexical)
        return list(dict.fromkeys([lexical, real]))

    def check_read(self, path: str | os.PathLike[str], *, cwd: str | None = None) -> PathDecision:
        """srt read semantics: allowed by default; deny wins unless an allow is strictly more specific.
        Both the lexical and the symlink-resolved path must pass."""
        if self.policy.mode is SandboxMode.FULL_ACCESS:
            return PathDecision(True, os.fspath(path), "read", "mode=full-access")
        for cand in self._candidates(path, cwd):
            d_spec, d_rule = self._best(self.deny_read, cand)
            if d_rule is None:
                continue
            a_spec, a_rule = self._best(self.allow_read, cand)
            if a_rule is None or d_spec >= a_spec:
                return PathDecision(False, cand, "read", f"deny_read:{d_rule.source}",
                                    "path holds credentials or secrets and is read-denied by sandbox policy")
        return PathDecision(True, self._candidates(path, cwd)[-1], "read", "default-allow")

    def check_write(self, path: str | os.PathLike[str], *, cwd: str | None = None) -> PathDecision:
        """Write-allow-only: the path must sit under a writable root, and no ``deny_write``,
        protected subpath or read-deny may match. Both lexical and resolved paths must pass, which
        closes symlink escapes out of the workspace."""
        if self.policy.mode is SandboxMode.FULL_ACCESS:
            return PathDecision(True, os.fspath(path), "write", "mode=full-access")
        cands = self._candidates(path, cwd)
        for cand in cands:
            root_rule = next((r for r in self.writable_roots if cand == r or cand.startswith(r.rstrip("/") + "/")), None)
            glob_rule = next((r for r in self.allow_write_globs if r.match(cand) >= 0), None)
            if root_rule is None and glob_rule is None:
                return PathDecision(False, cand, "write", "write-allow-only",
                                    f"writes are only allowed under: {', '.join(self.writable_roots) or '(nothing: read-only mode)'}")
            _, deny = self._best(self.deny_write, cand)
            if deny is not None:
                return PathDecision(False, cand, "write", f"{deny.list_name}:{deny.source}",
                                    "protected path (executes or is read outside the sandbox) is read-only")
            read = self.check_read(cand)
            if not read.allowed:
                return PathDecision(False, cand, "write", read.rule, "secret paths are neither readable nor writable")
        return PathDecision(True, cands[-1], "write", "writable-root")

    # -------------------------------------------------------------- helpers for runners
    def literal_deny_reads(self) -> list[str]:
        return sorted({r.literal for r in self.deny_read if r.literal})

    def glob_deny_reads(self) -> list[PathRule]:
        return [r for r in self.deny_read if r.is_glob]

    def literal_allow_reads(self) -> list[str]:
        return sorted({r.literal for r in self.allow_read if r.literal})

    def literal_deny_writes(self) -> list[str]:
        return sorted({r.literal for r in self.deny_write if r.literal})

    def glob_deny_writes(self) -> list[PathRule]:
        return [r for r in self.deny_write if r.is_glob]

    def find_glob_matches(self, rules: Iterable[PathRule], *, max_entries: int = 20_000, max_depth: int = 12) -> list[str]:
        """Expand glob rules to concrete existing paths under the writable roots and workspace.

        OS mounts cannot express ``**/.env``; runners mask each concrete match instead (bwrap,
        containers). Files the sandboxed process creates later are its own and need no mask.
        """
        rules = list(rules)
        if not rules:
            return []
        bases = list(dict.fromkeys([self.workspace, *self.writable_roots]))
        found: set[str] = set()
        seen = 0
        for base in bases:
            if not os.path.isdir(base):
                continue
            base_depth = _depth(base)
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = [d for d in dirnames if d not in ("node_modules", ".git", "__pycache__", ".venv")]
                if _depth(dirpath) - base_depth >= max_depth:
                    dirnames[:] = []
                for name in (*dirnames, *filenames):
                    seen += 1
                    full = os.path.join(dirpath, name)
                    if any(r.regex is not None and r.regex.fullmatch(full) for r in rules):
                        found.add(full)
                if seen > max_entries:
                    log.warning("glob expansion stopped after %d entries under %s", max_entries, base)
                    break
        return sorted(found)


def _resolved_gitdirs(root: str) -> list[str]:
    """Codex keeps the *resolved* gitdir read-only: a worktree's ``.git`` file points elsewhere."""
    dotgit = os.path.join(root, ".git")
    out: list[str] = []
    if os.path.isfile(dotgit):
        try:
            text = Path(dotgit).read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return out
        if text.startswith("gitdir:"):
            gd = text.split(":", 1)[1].strip()
            gd = os.path.normpath(gd if os.path.isabs(gd) else os.path.join(root, gd))
            out.append(gd)
            common = os.path.join(gd, "commondir")
            if os.path.isfile(common):
                c = Path(common).read_text(encoding="utf-8", errors="replace").strip()
                out.append(os.path.normpath(c if os.path.isabs(c) else os.path.join(gd, c)))
    elif os.path.isdir(dotgit):
        out.append(os.path.realpath(dotgit))
    return out


def env_name_matches(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(name.upper(), p.upper()) for p in patterns)
