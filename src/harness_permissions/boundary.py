"""Boundary-crossing classification: "auto-allow inside the boundary, classify every crossing".

The build plan's permission section rests on one observation: sandboxing "safely reduces
permission prompts by 84%" because actions confined to the workspace need no human, so humans
(and the auto-reviewer) are reserved for *crossings*. :func:`classify` turns a tool call into an
:class:`ActionProfile` - the ``normalize(call)`` step of the checklist pipeline (§7.7 step 2):
tool kind, every touched path (canonicalised), every network target, the parsed shell command,
MCP server/annotations, and the set of :class:`Crossing` values with human-readable reasons.

Crossing groups mirror auto mode's four default block-rule groups (destroy/exfiltrate, degrade
security posture, cross trust boundaries, bypass review/affect others) plus the taint-strict
action names from our ``harness-policy.toml`` schema (``network_write``, ``message_send``,
``git_push_non_session_branch``, ``new_egress_domain``, ``memory_write``).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from harness_core.types import RunContext, ToolCall, ToolSpec
from harness_memory.security import SecretRedactor

from ._util import canonical_json, has_invisible
from .network import any_domain_matches, extract_urls, host_from_url, is_private_host, looks_like_host, normalize_host
from .paths import Workspace
from .shell import SHELLS, ShellAnalysis, SimpleCommand, analyze, interpreter_family


class ToolKind(str, Enum):
    SHELL = "shell"
    READ = "read"
    SEARCH = "search"
    EDIT = "edit"
    WEB_FETCH = "web_fetch"
    WEB_SEARCH = "web_search"
    BROWSER = "browser"
    SUBAGENT = "subagent"
    MEMORY = "memory"
    MESSAGE = "message"
    INTERNAL = "internal"
    MCP = "mcp"
    UNKNOWN = "unknown"


class Crossing(str, Enum):
    # cross trust boundaries
    READ_OUTSIDE_WORKSPACE = "read_outside_workspace"
    WRITE_OUTSIDE_WORKSPACE = "write_outside_workspace"
    NETWORK_EGRESS = "network_egress"
    NEW_EGRESS_DOMAIN = "new_egress_domain"
    EXTERNAL_TOOL = "external_tool"
    SUBAGENT_SPAWN = "subagent_spawn"
    # destroy or exfiltrate
    CREDENTIAL_ACCESS = "credential_access"
    ENVIRONMENT_DUMP = "environment_dump"
    SECRET_IN_INPUT = "secret_in_input"
    NETWORK_WRITE = "network_write"
    MESSAGE_SEND = "message_send"
    IRREVERSIBLE = "irreversible"
    CATASTROPHIC = "catastrophic"
    PRIVATE_NETWORK = "private_network"
    # bypass review / affect others
    GIT_PUSH_SESSION_BRANCH = "git_push_session_branch"
    GIT_PUSH_NON_SESSION_BRANCH = "git_push_non_session_branch"
    DEPLOY = "deploy"
    MEMORY_WRITE = "memory_write"
    # degrade security posture
    PROTECTED_PATH = "protected_path"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SECURITY_POSTURE = "degrade_security_posture"
    SELF_HARM = "self_harm"
    # cannot be analysed statically
    OPAQUE_EXECUTION = "opaque_execution"
    CODE_EXECUTION = "code_execution"
    DANGEROUS_ENV = "dangerous_env"
    UNKNOWN_EFFECTS = "unknown_effects"
    SUSPICIOUS_INPUT = "suspicious_input"


#: Denied in every mode, before any allow rule or human is consulted.
HARD_DENY: frozenset[Crossing] = frozenset({
    Crossing.PROTECTED_PATH, Crossing.CREDENTIAL_ACCESS, Crossing.PRIVATE_NETWORK, Crossing.CATASTROPHIC, Crossing.SELF_HARM,
})
#: Do not by themselves take an action outside the boundary (they only matter while tainted).
BOUNDARY_NEUTRAL: frozenset[Crossing] = frozenset({Crossing.MEMORY_WRITE})
#: Shell crossings that a *wildcard* allow rule can never authorise (an exact rule can).
WILDCARD_UNCOVERABLE: frozenset[Crossing] = frozenset({
    Crossing.WRITE_OUTSIDE_WORKSPACE, Crossing.PRIVILEGE_ESCALATION, Crossing.OPAQUE_EXECUTION, Crossing.DANGEROUS_ENV,
    Crossing.SUSPICIOUS_INPUT, Crossing.SECRET_IN_INPUT, Crossing.SECURITY_POSTURE,
})
#: Outward / irreversible effects that checkpoints cannot undo (research §7.3 technique 12).
OUTWARD: frozenset[Crossing] = frozenset({
    Crossing.NETWORK_WRITE, Crossing.MESSAGE_SEND, Crossing.GIT_PUSH_SESSION_BRANCH, Crossing.GIT_PUSH_NON_SESSION_BRANCH,
    Crossing.DEPLOY, Crossing.IRREVERSIBLE,
})

# ----------------------------------------------------------------------------- tool kinds

_KIND_BY_NAME: dict[str, ToolKind] = {}
for _kind, _names in {
    ToolKind.SHELL: ("bash", "shell", "run_shell", "exec_command", "run_command", "terminal", "bash_20250124", "local_shell"),
    ToolKind.READ: ("read", "read_file", "view", "view_file", "cat", "open_file", "notebookread"),
    ToolKind.SEARCH: ("glob", "grep", "ls", "list_dir", "list_files", "search", "find_files", "file_search"),
    ToolKind.EDIT: ("edit", "multiedit", "write", "write_file", "create_file", "str_replace", "str_replace_editor",
                    "str_replace_based_edit_tool", "apply_patch", "notebookedit", "insert"),
    ToolKind.WEB_FETCH: ("webfetch", "web_fetch", "fetch", "http_get", "fetch_url"),
    ToolKind.WEB_SEARCH: ("websearch", "web_search", "search_web"),
    ToolKind.BROWSER: ("browser", "computer", "navigate", "browser_navigate"),
    ToolKind.SUBAGENT: ("agent", "task", "spawn_agent", "subagent", "delegate"),
    ToolKind.MEMORY: ("memory",),
    ToolKind.INTERNAL: ("todowrite", "todo", "update_plan", "think", "todoread", "exitplanmode", "enter_plan_mode", "exit_plan_mode"),
}.items():
    for _n in _names:
        _KIND_BY_NAME[_n] = _kind

_MESSAGE_NAME = re.compile(r"(?i)(send|post|reply|forward|publish|notify|tweet|comment|email|mail|message|dm|invite|share)")
_DEFAULT_PATH_ARGS = ("file_path", "path", "notebook_path", "filename", "file", "target_file", "directory", "dir", "paths", "files")
_PATCH_FILE = re.compile(r"(?m)^\*\*\* (?:Add|Update|Delete) File: (.+)$|^\+\+\+ (?:b/)?(\S+)")


def tool_kind(name: str, spec: ToolSpec | None) -> ToolKind:
    """Classify a tool by explicit metadata, provider type, MCP prefix or well-known name."""
    if spec is not None:
        declared = spec.metadata.get("permission_kind")
        if declared:
            return ToolKind(declared)
        if spec.provider_type and spec.provider_type.startswith("memory"):
            return ToolKind.MEMORY
        if spec.provider_type and spec.provider_type.startswith("bash"):
            return ToolKind.SHELL
    if name.startswith("mcp__"):
        return ToolKind.MCP
    kind = _KIND_BY_NAME.get(name.lower())
    if kind is not None:
        return kind
    if _MESSAGE_NAME.search(name) and not (spec and spec.read_only):
        return ToolKind.MESSAGE
    return ToolKind.UNKNOWN


@dataclass(frozen=True)
class McpRef:
    server: str
    tool: str
    vetted: bool
    read_only: bool
    destructive: bool
    open_world: bool


def parse_mcp(name: str, spec: ToolSpec | None, vetted_servers: Iterable[str]) -> McpRef | None:
    """MCP tools are namespaced ``mcp__<server>__<tool>``; annotations are trusted only from vetted servers.

    Absent fields take the MCP defaults (``readOnlyHint=false``, ``destructiveHint=true``,
    ``openWorldHint=true``) - "remember the defaults when a field is absent" (§7.8).
    """
    if not name.startswith("mcp__"):
        return None
    rest = name[len("mcp__"):]
    server, _, tool = rest.partition("__")
    ann: dict[str, Any] = dict((spec.metadata.get("annotations") if spec else None) or {})
    vetted = server in set(vetted_servers) or bool(spec and spec.metadata.get("mcp_vetted") is True and server in set(vetted_servers))
    read_hint = bool(ann.get("readOnlyHint", False)) or bool(spec and spec.read_only)
    destructive_hint = bool(ann.get("destructiveHint", True))
    open_world = bool(ann.get("openWorldHint", True)) or bool(spec and spec.open_world)
    return McpRef(
        server=server,
        tool=tool,
        vetted=vetted,
        read_only=vetted and read_hint,
        destructive=(not vetted) or (destructive_hint and not read_hint),
        open_world=open_world if vetted else True,
    )


# ----------------------------------------------------------------------------- profile


@dataclass(frozen=True)
class PathAccess:
    raw: str
    path: str           # canonical absolute path ('' when unresolvable)
    write: bool
    source: str = ""


@dataclass
class SegmentProfile:
    """Classification of one simple command inside a shell invocation."""

    command: SimpleCommand
    crossings: dict[Crossing, list[str]] = field(default_factory=dict)
    paths: list[PathAccess] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    read_only: bool = False
    fs_common: bool = False      # mkdir/touch/mv/cp - what acceptEdits auto-runs

    def add(self, crossing: Crossing, reason: str) -> None:
        self.crossings.setdefault(crossing, []).append(reason)


@dataclass
class ActionProfile:
    call: ToolCall
    spec: ToolSpec | None
    kind: ToolKind
    crossings: dict[Crossing, list[str]] = field(default_factory=dict)
    paths: list[PathAccess] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    shell: ShellAnalysis | None = None
    segments: list[SegmentProfile] = field(default_factory=list)
    mcp: McpRef | None = None
    read_only: bool = False

    @property
    def tool(self) -> str:
        return self.call.name

    def add(self, crossing: Crossing, reason: str) -> None:
        reasons = self.crossings.setdefault(crossing, [])
        if reason not in reasons:
            reasons.append(reason)

    def has(self, *crossings: Crossing) -> bool:
        return any(c in self.crossings for c in crossings)

    @property
    def boundary_crossings(self) -> set[Crossing]:
        return set(self.crossings) - BOUNDARY_NEUTRAL

    @property
    def in_boundary(self) -> bool:
        return not self.boundary_crossings

    @property
    def hard_denials(self) -> list[tuple[Crossing, str]]:
        return [(c, r[0] if r else c.value) for c, r in self.crossings.items() if c in HARD_DENY]

    @property
    def mutating(self) -> bool:
        return not self.read_only

    def summary(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "kind": self.kind.value,
            "crossings": {c.value: r for c, r in sorted(self.crossings.items(), key=lambda kv: kv[0].value)},
            "paths": [{"path": p.path, "write": p.write} for p in self.paths][:20],
            "domains": self.domains[:20],
            "read_only": self.read_only,
            "mcp_server": self.mcp.server if self.mcp else None,
        }


@dataclass
class BoundaryConfig:
    allowed_domains: list[str] = field(default_factory=list)
    approved_domains: set[str] = field(default_factory=set)   # per-session human approvals
    block_private_network: bool = True
    vetted_mcp_servers: list[str] = field(default_factory=list)
    session_branch: str | None = None
    detect_secrets: bool = True

    def egress_allowed(self, host: str) -> bool:
        return any_domain_matches(host, self.allowed_domains) or normalize_host(host) in self.approved_domains


# ----------------------------------------------------------------------------- shell vocabularies

READ_ONLY_COMMANDS = frozenset({
    "ls", "cat", "head", "tail", "less", "more", "wc", "grep", "egrep", "fgrep", "rg", "ag", "ack", "pwd", "echo", "printf",
    "true", "false", "which", "whereis", "type", "file", "stat", "du", "df", "tree", "sort", "uniq", "cut", "tr", "diff",
    "cmp", "basename", "dirname", "realpath", "readlink", "date", "uname", "whoami", "id", "hostname", "jq", "nl", "tac",
    "column", "fold", "fmt", "comm", "join", "paste", "expand", "unexpand", "od", "xxd", "hexdump", "strings", "md5sum",
    "sha1sum", "sha256sum", "sha512sum", "cksum", "b2sum", "test", "[", "[[", "seq", "yes", "sleep", "cd", "pushd", "popd",
    "bat", "fd", "locale", "tty", "groups", "nproc", "arch", "lsb_release", "uptime", "free", "ps", "pgrep", "lsof",
    "man", "help", "info", "tldr", "cal", "base64", "shasum", "command", "hash", "read",
})
_NO_PATH_ARGS = frozenset({
    "echo", "printf", "true", "false", "sleep", "yes", "seq", "date", "uname", "whoami", "id", "hostname", "test", "[", "[[",
    "which", "whereis", "type", "command", "hash", "read", "locale", "tty", "groups", "nproc", "arch", "uptime", "free", "ps",
    "pgrep", "man", "help", "info", "tldr", "cal", "lsb_release", "popd",
})
_PATH_ARG_READ = frozenset({
    "cat", "head", "tail", "less", "more", "wc", "file", "stat", "xxd", "od", "hexdump", "strings", "base64", "md5sum", "sha1sum",
    "sha256sum", "sha512sum", "shasum", "cksum", "b2sum", "diff", "cmp", "ls", "tree", "du", "nl", "tac", "sort", "uniq", "cut",
    "jq", "realpath", "readlink", "source", ".", "bat", "cd", "pushd", "comm", "join", "paste",
})
_PATH_ARG_WRITE = frozenset({"rm", "rmdir", "shred", "unlink", "truncate", "touch", "mkdir", "chmod", "chown", "chgrp", "ln", "tee", "install", "mkfifo"})
_SEARCH_WITH_PATTERN = frozenset({"grep", "egrep", "fgrep", "rg", "ag", "ack"})
_FS_COMMON = frozenset({"mkdir", "touch", "mv", "cp"})
_GIT_READ = frozenset({
    "status", "diff", "log", "show", "rev-parse", "ls-files", "blame", "describe", "grep", "shortlog", "ls-tree", "cat-file",
    "merge-base", "reflog", "whatchanged", "name-rev", "rev-list", "for-each-ref", "show-ref", "count-objects", "version", "help",
    "check-ignore", "var",
})
_GIT_NETWORK_READ = frozenset({"fetch", "pull", "clone", "ls-remote", "remote-update"})
_NET_TOOLS = frozenset({"curl", "wget", "http", "https", "xh", "aria2c", "httpie", "lynx", "w3m"})
_NET_CHANNELS = frozenset({"nc", "ncat", "netcat", "socat", "telnet", "ssh", "scp", "sftp", "ftp", "rsync", "rclone", "sshpass"})
_NET_PROBES = frozenset({"ping", "dig", "nslookup", "host", "whois", "traceroute", "mtr"})
_PKG_INSTALL = {
    "pip": {"install", "download"}, "pip3": {"install", "download"}, "npm": {"install", "i", "ci", "add", "update"},
    "yarn": {"add", "install"}, "pnpm": {"add", "install", "i"}, "bun": {"add", "install", "i"}, "cargo": {"install", "add", "fetch"},
    "go": {"get", "install", "mod"}, "gem": {"install"}, "apt": {"install"}, "apt-get": {"install"}, "yum": {"install"},
    "dnf": {"install"}, "brew": {"install"}, "apk": {"add"}, "uv": {"pip", "add", "sync"}, "poetry": {"add", "install"},
    "composer": {"install", "require"}, "conda": {"install"},
}
_PUBLISH = {
    "npm": {"publish", "unpublish"}, "yarn": {"publish"}, "pnpm": {"publish"}, "twine": {"upload"}, "cargo": {"publish", "yank"},
    "gem": {"push", "yank"}, "docker": {"push"}, "podman": {"push"}, "helm": {"push", "install", "upgrade", "uninstall", "rollback", "delete"},
    "kubectl": {"apply", "create", "delete", "replace", "patch", "scale", "rollout", "edit", "drain", "cordon", "exec", "set", "label", "annotate", "taint"},
    "terraform": {"apply", "destroy", "import", "taint", "state"}, "tofu": {"apply", "destroy", "import"}, "pulumi": {"up", "destroy", "import"},
    "fly": {"deploy", "destroy", "scale", "secrets"}, "flyctl": {"deploy", "destroy", "scale", "secrets"}, "vercel": {"deploy", "--prod", "rm", "remove"},
    "netlify": {"deploy"}, "firebase": {"deploy"}, "serverless": {"deploy", "remove"}, "sls": {"deploy", "remove"},
    "heroku": {"releases:rollback", "ps:scale", "config:set", "apps:destroy"}, "poetry": {"publish"}, "uv": {"publish"},
}
_CLOUD_WRITE_VERBS = re.compile(r"^(delete|create|deploy|update|put|rm|cp|mv|sync|terminate|run-instances|apply|remove|set|start|stop|reboot|modify|attach|detach|invoke|publish|send)")
_GH_WRITE = {"create", "merge", "close", "delete", "comment", "edit", "review", "reopen", "ready", "upload", "fork", "transfer", "archive", "rename", "set", "add", "remove", "lock"}
_MAIL_TOOLS = frozenset({"mail", "mailx", "sendmail", "mutt", "msmtp", "swaks", "s-nail"})
_SECURITY_TOOLS = frozenset({"iptables", "ip6tables", "nft", "ufw", "setenforce", "auditctl", "aa-disable", "csrutil", "spctl", "firewall-cmd"})
_SYSTEM_HALT = frozenset({"shutdown", "reboot", "halt", "poweroff"})
_DISK_DESTROY = re.compile(r"^(mkfs(\..+)?|fdisk|sfdisk|parted|wipefs|mkswap|blkdiscard)$")
_SELF_KILL_TARGETS = re.compile(r"^(bash|sh|zsh|dash|python\d*(\.\d+)?|node|claude|codex|harness.*|init|systemd|sshd|-1|1)$")
DANGEROUS_ENV = re.compile(
    r"^(PATH|LD_.*|DYLD_.*|PYTHON(PATH|STARTUP|HOME|INSPECT)|NODE_OPTIONS|NODE_PATH|PERL5.*|PERLLIB|RUBY(OPT|LIB)|BASH_ENV|ENV|"
    r"PROMPT_COMMAND|IFS|SHELLOPTS|BASHOPTS|PS4|GIT_.*|GCONV_PATH|HOSTALIASES|LESSOPEN|LESSCLOSE|EDITOR|VISUAL|PAGER|"
    r"MANPAGER|SSH_ASKPASS|SUDO_ASKPASS|HISTFILE|JAVA_TOOL_OPTIONS|_JAVA_OPTIONS|CLASSPATH|DOTNET_.*|NPM_CONFIG_.*|PIP_.*)$"
)
_CATASTROPHIC_TARGETS = {"/", "/*", "~", "~/", "~/*", "$HOME", "${HOME}", "/bin", "/boot", "/dev", "/etc", "/home", "/lib", "/lib64",
                         "/opt", "/proc", "/root", "/sbin", "/sys", "/usr", "/var", "/System", "/Users", "/Library", "/Applications", ".", "..", "*"}


def _short_flags(args: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for a in args:
        if a.startswith("-") and not a.startswith("--") and len(a) > 1 and a[1:].isalpha():
            out.update(a[1:])
    return out


def _non_options(args: Iterable[str]) -> list[str]:
    out, after_dd = [], False
    for a in args:
        if a == "--" and not after_dd:
            after_dd = True
            continue
        if not after_dd and a.startswith("-") and a != "-":
            continue
        out.append(a)
    return out


def _looks_like_path(token: str) -> bool:
    return "/" in token or token.startswith(("~", ".")) or token in {"*"}


_UNSET = object()


class _Classifier:
    def __init__(self, workspace: Workspace, config: BoundaryConfig, ctx: RunContext) -> None:
        self.ws = workspace
        self.cfg = config
        self.ctx = ctx
        self.home = workspace.home
        cwd = ctx.cwd if ctx.cwd and ctx.cwd != "." else workspace.root
        self.cwd: str | None = os.path.realpath(os.path.join(workspace.root, cwd)) if not os.path.isabs(cwd) else os.path.realpath(cwd)

    # -- paths -----------------------------------------------------------------------
    def resolve(self, word_text: str, dynamic: bool = False, cwd: object = _UNSET) -> str:
        base = self.cwd if cwd is _UNSET else cwd
        assert base is None or isinstance(base, str)
        text = word_text
        for var in ("${HOME}", "$HOME"):
            if text.startswith(var):
                text = self.home + text[len(var):]
                dynamic = "$" in text or "`" in text
        if dynamic or "$" in text or "`" in text:
            return ""
        if any(ch in text for ch in "*?[") or ("{" in text and "}" in text):
            cut = min(i for i in (text.find(c) for c in "*?[{") if i >= 0)
            prefix = text[:cut]
            text = prefix if prefix.endswith("/") or not prefix else os.path.dirname(prefix) or "."
        if not os.path.isabs(text) and not text.startswith("~") and base is None:
            return ""
        try:
            return self.ws.canonical(text, cwd=base or self.ws.root)
        except ValueError:
            return ""

    def access(self, target: SegmentProfile | ActionProfile, raw: str, canonical: str, write: bool, source: str) -> None:
        target.paths.append(PathAccess(raw=raw, path=canonical, write=write, source=source))
        if not canonical:
            target.add(Crossing.WRITE_OUTSIDE_WORKSPACE if write else Crossing.READ_OUTSIDE_WORKSPACE, f"unresolvable path {raw!r}")
            return
        cred = self.ws.is_credential(canonical)
        if cred:
            target.add(Crossing.CREDENTIAL_ACCESS, f"{canonical} matches credential path {cred!r}")
        if write:
            prot = self.ws.is_protected(canonical)
            if prot:
                target.add(Crossing.PROTECTED_PATH, f"{canonical} matches protected path {prot!r}")
            if not self.ws.inside(canonical, for_write=True):
                target.add(Crossing.WRITE_OUTSIDE_WORKSPACE, f"write to {canonical}")
        elif not self.ws.inside(canonical):
            target.add(Crossing.READ_OUTSIDE_WORKSPACE, f"read of {canonical}")

    # -- network ---------------------------------------------------------------------
    def network_target(self, target: SegmentProfile | ActionProfile, host: str, *, write: bool = False) -> None:
        h = normalize_host(host)
        if not h:
            target.add(Crossing.NETWORK_EGRESS, "unknown network destination")
            return
        target.domains.append(h)
        if self.cfg.block_private_network and is_private_host(h):
            target.add(Crossing.PRIVATE_NETWORK, f"private/loopback/metadata destination {h}")
            return
        if not self.cfg.egress_allowed(h):
            target.add(Crossing.NEW_EGRESS_DOMAIN, f"{h} is not on the egress allowlist")
            target.add(Crossing.NETWORK_EGRESS, f"egress to {h}")
        if write:
            target.add(Crossing.NETWORK_WRITE, f"data sent to {h}")

    # -- shell -----------------------------------------------------------------------
    def shell(self, profile: ActionProfile, command: str) -> None:
        analysis = analyze(command)
        profile.shell = analysis
        if analysis.errors:
            profile.add(Crossing.OPAQUE_EXECUTION, "unparsable command: " + "; ".join(analysis.errors))
        for s in analysis.suspicious:
            profile.add(Crossing.SUSPICIOUS_INPUT, s)
        if analysis.errors and "fork bomb" in analysis.errors:
            profile.add(Crossing.CATASTROPHIC, "fork bomb")
        cwd = self.cwd
        for cmd in analysis.commands:
            seg = SegmentProfile(command=cmd)
            self._segment(seg, cmd, cwd)
            if cmd.origin == "top" and cmd.executable in {"cd", "pushd"} and not cmd.background:
                dest = _non_options(cmd.argv[1:])
                if not dest:
                    cwd = self.home
                else:
                    resolved = self.resolve(dest[0], cmd.words[1].dynamic if len(cmd.words) > 1 else False, cwd)
                    cwd = resolved or None
            profile.segments.append(seg)
            for c, reasons in seg.crossings.items():
                for r in reasons:
                    profile.add(c, r)
            profile.paths.extend(seg.paths)
            profile.domains.extend(d for d in seg.domains if d not in profile.domains)
        profile.read_only = bool(profile.segments) and all(s.read_only for s in profile.segments) and not analysis.errors

    def _segment(self, seg: SegmentProfile, cmd: SimpleCommand, cwd: str | None) -> None:
        if cmd.privileged:
            seg.add(Crossing.PRIVILEGE_ESCALATION, f"runs via {'/'.join(w for w in cmd.wrappers if w in {'sudo', 'doas', 'su', 'pkexec', 'run0'}) or cmd.executable}")
        for name, _value in cmd.env:
            if DANGEROUS_ENV.match(name):
                seg.add(Crossing.DANGEROUS_ENV, f"sets {name}")
        for reason in cmd.opaque:
            seg.add(Crossing.OPAQUE_EXECUTION, reason)
        if cmd.dynamic_command:
            seg.add(Crossing.OPAQUE_EXECUTION, f"dynamic command name {cmd.words[0].raw!r}")
        for r in cmd.redirects:
            if r.target is None:
                continue
            if r.writes:
                self.access(seg, r.target.text, self.resolve(r.target.text, r.target.dynamic, cwd), True, f"redirect {r.op}")
            elif r.reads:
                self.access(seg, r.target.text, self.resolve(r.target.text, r.target.dynamic, cwd), False, f"redirect {r.op}")
        if cmd.transparent or not cmd.words:
            seg.read_only = not seg.crossings
            return
        if cmd.code_execution and not cmd.opaque:
            seg.add(Crossing.CODE_EXECUTION, f"{cmd.executable} executes code")
        exe = cmd.executable
        words = cmd.words[1:]
        args = [w.text for w in words]
        dyn = {w.text: w.dynamic for w in words}

        def paths(tokens: Iterable[str], write: bool, src: str, only_pathlike: bool = False) -> None:
            for t in tokens:
                if only_pathlike and not _looks_like_path(t):
                    continue
                self.access(seg, t, self.resolve(t, dyn.get(t, False), cwd), write, src)

        if exe in {"env", "printenv", "set", "export", "declare"} and not _non_options(args):
            seg.add(Crossing.ENVIRONMENT_DUMP, f"{exe} prints environment variables (may expose secrets)")
            return
        if exe == "git":
            self._git(seg, args, cwd)
            return
        if exe in _NET_TOOLS:
            self._http_client(seg, exe, args, cwd)
            return
        if exe in _NET_CHANNELS:
            self._channel(seg, exe, args)
            return
        if exe in _NET_PROBES:
            hosts = [a for a in _non_options(args) if looks_like_host(a) or is_private_host(a)]
            for h in hosts:
                self.network_target(seg, h)
            if not hosts:
                seg.add(Crossing.NETWORK_EGRESS, f"{exe} contacts the network")
            return
        if exe in _MAIL_TOOLS:
            seg.add(Crossing.MESSAGE_SEND, f"{exe} sends email")
            seg.add(Crossing.NETWORK_WRITE, f"{exe} sends email")
            return
        if exe == "gh":
            sub = _non_options(args)
            verb = sub[1] if len(sub) > 1 else (sub[0] if sub else "")
            if sub and sub[0] == "api":
                method = next((args[i + 1] for i, a in enumerate(args[:-1]) if a in {"-X", "--method"}), "GET")
                if method.upper() != "GET" or any(a in {"-f", "-F", "--field", "--raw-field", "--input"} for a in args):
                    seg.add(Crossing.NETWORK_WRITE, "gh api write")
                else:
                    seg.add(Crossing.NETWORK_EGRESS, "gh api read")
            elif verb in _GH_WRITE:
                seg.add(Crossing.NETWORK_WRITE, f"gh {' '.join(sub[:2])}")
                if verb in {"comment", "create", "review"}:
                    seg.add(Crossing.MESSAGE_SEND, f"gh {' '.join(sub[:2])} posts content others see")
                if verb in {"merge", "delete", "transfer", "archive"}:
                    seg.add(Crossing.IRREVERSIBLE, f"gh {' '.join(sub[:2])}")
            else:
                seg.add(Crossing.NETWORK_EGRESS, f"gh {' '.join(sub[:2])}")
            return
        if exe in {"aws", "gcloud", "az", "gsutil", "doctl"}:
            verbs = [a for a in _non_options(args)]
            if any(_CLOUD_WRITE_VERBS.match(v) for v in verbs[1:4]):
                seg.add(Crossing.DEPLOY, f"{exe} {' '.join(verbs[:3])}")
                seg.add(Crossing.NETWORK_WRITE, f"{exe} cloud API write")
            else:
                seg.add(Crossing.NETWORK_EGRESS, f"{exe} cloud API")
            return
        sub0 = next(iter(_non_options(args)), "")
        if exe in _PUBLISH and (sub0 in _PUBLISH[exe] or any(a in _PUBLISH[exe] for a in args)):
            seg.add(Crossing.DEPLOY, f"{exe} {sub0}".strip())
            seg.add(Crossing.NETWORK_WRITE, f"{exe} {sub0} changes remote state".strip())
            return
        if exe in _PKG_INSTALL and sub0 in _PKG_INSTALL[exe]:
            seg.add(Crossing.NETWORK_EGRESS, f"{exe} {sub0} downloads packages")
            seg.add(Crossing.CODE_EXECUTION, f"{exe} {sub0} may run install scripts")
            return
        if exe in _SECURITY_TOOLS:
            seg.add(Crossing.SECURITY_POSTURE, f"{exe} changes security controls")
            return
        if exe in _SYSTEM_HALT or (exe in {"init", "telinit"} and args[:1] in (["0"], ["6"])):
            seg.add(Crossing.SELF_HARM, f"{exe} stops the machine")
            return
        if _DISK_DESTROY.match(exe):
            seg.add(Crossing.CATASTROPHIC, f"{exe} destroys a filesystem")
            return
        if exe in {"kill", "pkill", "killall"}:
            targets = _non_options(args)
            if exe == "kill":
                targets = [t for t in args if not re.fullmatch(r"-(\d+|[A-Z]+|s|n)", t)]
            if any(_SELF_KILL_TARGETS.match(t) or t in {"$$", "$PPID", "${PPID}"} for t in targets) or "-1" in args[:2] and exe == "kill":
                seg.add(Crossing.SELF_HARM, f"{exe} {' '.join(args)} can kill the harness or the machine")
            return
        if exe == "crontab" and "-r" in args:
            seg.add(Crossing.IRREVERSIBLE, "crontab -r removes all cron jobs")
            return
        if exe in {"history"} and "-c" in args or exe == "unset" and "HISTFILE" in args:
            seg.add(Crossing.SECURITY_POSTURE, "erases shell history")
            return
        if exe == "dd":
            for a in args:
                if a.startswith("of="):
                    target = a[3:]
                    if target.startswith("/dev/") and target not in {"/dev/null", "/dev/stdout", "/dev/stderr"}:
                        seg.add(Crossing.CATASTROPHIC, f"dd writes raw device {target}")
                    else:
                        paths([target], True, "dd of=")
                elif a.startswith("if="):
                    paths([a[3:]], False, "dd if=")
            return
        if exe in {"rm", "rmdir", "shred", "unlink"}:
            flags = _short_flags(args) | ({"r"} if any(a in {"--recursive"} for a in args) else set())
            targets = _non_options(args)
            recursive = bool(flags & {"r", "R"}) or exe == "shred"
            for t in targets:
                literal = t.rstrip("/") or "/"
                resolved = self.resolve(t, dyn.get(t, False), cwd)
                ws_ancestor = bool(resolved) and any(r == resolved or r.startswith(resolved.rstrip("/") + "/") for r in self.ws.roots)
                if recursive and (literal in _CATASTROPHIC_TARGETS and literal not in {".", "..", "*"} or resolved in {"/", self.home} or (ws_ancestor and resolved != self.ws.root)):
                    seg.add(Crossing.CATASTROPHIC, f"recursive delete of {t!r}")
            if recursive:
                seg.add(Crossing.IRREVERSIBLE, f"{exe} -r deletes trees (not covered by checkpoints)")
            elif exe == "shred":
                seg.add(Crossing.IRREVERSIBLE, "shred destroys file contents")
            paths(targets, True, exe)
            return
        if exe == "find":
            roots: list[str] = []
            for a in args:
                if a.startswith(("-", "(", "!")) or a in {")"}:
                    break
                roots.append(a)
            if "-delete" in args:
                seg.add(Crossing.IRREVERSIBLE, "find -delete")
                paths(roots or ["."], True, "find -delete")
            else:
                paths(roots or ["."], False, "find")
            for k, a in enumerate(args):
                if a in {"-fprint", "-fprint0", "-fprintf", "-fls"} and k + 1 < len(args):
                    paths([args[k + 1]], True, f"find {a}")
            seg.read_only = not seg.crossings and "-delete" not in args and not any(a in {"-exec", "-execdir", "-ok", "-okdir"} for a in args)
            return
        if exe in {"cp", "mv"}:
            targets = _non_options(args)
            t_flag = next((args[i + 1] for i, a in enumerate(args[:-1]) if a in {"-t", "--target-directory"}), None)
            if t_flag:
                dest, sources = [t_flag], [t for t in targets if t != t_flag]
            else:
                dest, sources = targets[-1:], targets[:-1]
            paths(sources, exe == "mv", exe)
            paths(dest, True, exe)
            seg.fs_common = True
            return
        if exe in _PATH_ARG_WRITE:
            targets = _non_options(args)
            if exe in {"chmod", "chown", "chgrp"} and targets:
                mode = targets[0]
                if exe == "chmod" and (re.search(r"[ug]?\+[rwx]*s|^[2467][0-7]{3}$", mode) or mode in {"777", "a+rwx", "o+w"}):
                    seg.add(Crossing.SECURITY_POSTURE, f"chmod {mode} weakens permissions")
                if exe == "chown" and mode.split(":")[0] in {"root", "0"}:
                    seg.add(Crossing.PRIVILEGE_ESCALATION, "chown to root")
                targets = targets[1:]
            if exe == "truncate":
                seg.add(Crossing.IRREVERSIBLE, "truncate discards file contents")
            paths(targets, True, exe)
            seg.fs_common = exe in _FS_COMMON
            return
        if exe in {"sed", "perl"} and any(a == "-i" or a.startswith("-i") or a == "--in-place" for a in args):
            files = _non_options(args)[1:]
            paths(files, True, f"{exe} -i")
            return
        if exe in _SEARCH_WITH_PATTERN:
            ops = _non_options(args)
            has_e = any(a in {"-e", "-f", "--regexp", "--file"} for a in args)
            paths(ops if has_e else ops[1:], False, exe)
            seg.read_only = not seg.crossings
            return
        if exe in READ_ONLY_COMMANDS:
            if exe in _NO_PATH_ARGS:
                seg.read_only = not seg.crossings
                return
            if exe in _PATH_ARG_READ:
                paths(_non_options(args), False, exe)
            else:
                paths(_non_options(args), False, exe, only_pathlike=True)
            for a in args:
                if "=" in a and a.startswith("--"):
                    v = a.split("=", 1)[1]
                    if _looks_like_path(v):
                        paths([v], False, exe)
            seg.read_only = not seg.crossings
            return
        # unknown executable / interpreter / build tool: inspect path-like args conservatively
        if exe in SHELLS or interpreter_family(exe):
            script = next(iter(_non_options(args)), None)
            if script:
                paths([script], False, exe)
        paths([a for a in _non_options(args)], False, exe, only_pathlike=True)
        for u in extract_urls(" ".join(args)):
            self.network_target(seg, host_from_url(u))
        if cmd.executable_path:
            self.access(seg, cmd.words[0].text, self.resolve(cmd.words[0].text, False, cwd), False, "executable")

    def _git(self, seg: SegmentProfile, args: list[str], cwd: str | None) -> None:
        k = 0
        while k < len(args) and args[k].startswith("-"):
            k += 2 if args[k] in {"-C", "--git-dir", "--work-tree", "--namespace", "-c", "--exec-path"} else 1
        sub = args[k] if k < len(args) else ""
        rest = args[k + 1:]
        if sub in _GIT_READ or sub == "":
            seg.read_only = not seg.crossings
            return
        if sub == "branch" and not ({"-d", "-D", "-m", "-M", "-c", "-C", "--delete", "--move", "--force", "-f"} & set(rest)) and not _non_options(rest):
            seg.read_only = not seg.crossings
            return
        if sub == "remote" and (not rest or rest[0] in {"-v", "show", "get-url"}):
            seg.read_only = not seg.crossings
            return
        if sub == "config" and any(a in {"--get", "--list", "-l", "--get-all", "--get-regexp"} for a in rest):
            seg.read_only = not seg.crossings
            return
        if sub == "config":
            keys = _non_options(rest)
            if any(k_.lower().startswith(("core.hookspath", "core.sshcommand", "core.pager", "core.editor", "core.fsmonitor", "alias.", "credential.", "url.", "http.")) for k_ in keys) or "--global" in rest or "--system" in rest:
                seg.add(Crossing.SECURITY_POSTURE, f"git config {' '.join(rest)} can make git run code or change global behaviour")
            return
        if sub in _GIT_NETWORK_READ:
            remotes = [a for a in _non_options(rest) if "://" in a or "@" in a]
            for r in remotes:
                host = host_from_url(r) if "://" in r else normalize_host(r.split("@", 1)[-1].split(":", 1)[0])
                self.network_target(seg, host)
            if not remotes:
                seg.add(Crossing.NETWORK_EGRESS, f"git {sub} contacts a remote")
            return
        if sub == "push":
            self._git_push(seg, rest)
            return
        if sub in {"reset"} and "--hard" in rest or sub == "clean" and ({"f"} & _short_flags(rest) or "--force" in rest):
            seg.add(Crossing.IRREVERSIBLE, f"git {sub} {' '.join(rest)} discards uncommitted work")
        if sub in {"checkout", "restore"} and ("--" in rest or "." in rest or "-f" in rest or "--force" in rest):
            seg.add(Crossing.IRREVERSIBLE, f"git {sub} {' '.join(rest)} can discard uncommitted work")
        if sub == "branch" and ({"-D"} & set(rest) or "--force" in rest):
            seg.add(Crossing.IRREVERSIBLE, "git branch -D deletes unmerged work")
        if sub in {"filter-branch", "filter-repo", "gc"} and ("--prune=now" in rest or sub != "gc"):
            seg.add(Crossing.IRREVERSIBLE, f"git {sub} rewrites history")
        if sub == "stash" and rest[:1] in (["drop"], ["clear"]):
            seg.add(Crossing.IRREVERSIBLE, f"git stash {rest[0]}")

    def _git_push(self, seg: SegmentProfile, rest: list[str]) -> None:
        flags = set(rest)
        positional = _non_options(rest)
        force = bool({"-f", "--force", "--mirror", "--delete", "-d", "--prune"} & flags) or any(a.startswith("--force-with-lease") or a.startswith("--force-if-includes") for a in rest) or "f" in _short_flags(rest)
        refspecs = positional[1:]
        if any(r.startswith("+") or r.startswith(":") for r in refspecs):
            force = True
        if force:
            seg.add(Crossing.IRREVERSIBLE, "force/delete push rewrites or removes remote history")
        branch = self.cfg.session_branch or self.ctx.metadata.get("session_branch")
        dests = [r.lstrip("+").split(":")[-1].removeprefix("refs/heads/") for r in refspecs]
        if not dests:
            current = self.ctx.metadata.get("git_branch")
            dests = [current] if current else ["<current branch>"]
        if branch and all(d == branch for d in dests) and not force:
            seg.add(Crossing.GIT_PUSH_SESSION_BRANCH, f"push to session branch {branch}")
        else:
            seg.add(Crossing.GIT_PUSH_NON_SESSION_BRANCH, f"push to {', '.join(dests)}" + (f" (session branch is {branch})" if branch else ""))
        if positional and ("://" in positional[0] or "@" in positional[0]):
            r = positional[0]
            host = host_from_url(r) if "://" in r else normalize_host(r.split("@", 1)[-1].split(":", 1)[0])
            self.network_target(seg, host)

    def _http_client(self, seg: SegmentProfile, exe: str, args: list[str], cwd: str | None) -> None:
        write_flags = {"-d", "--data", "--data-raw", "--data-binary", "--data-urlencode", "-F", "--form", "--form-string", "-T",
                       "--upload-file", "--json", "--post-data", "--post-file", "--body-data", "--body-file"}
        write = any(a in write_flags or any(a.startswith(f + "=") for f in write_flags if f.startswith("--")) for a in args)
        for i, a in enumerate(args):
            if a in {"-X", "--request", "--method"} and i + 1 < len(args) and args[i + 1].upper() != "GET" and args[i + 1].upper() != "HEAD":
                write = True
            if a.startswith(("--request=", "--method=")) and a.split("=", 1)[1].upper() not in {"GET", "HEAD"}:
                write = True
            if exe in {"http", "https", "xh", "httpie"} and a.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
                write = True
            if a in {"-o", "--output", "-O", "--output-document"} and i + 1 < len(args) and a not in {"-O"}:
                self.access(seg, args[i + 1], self.resolve(args[i + 1], False, cwd), True, f"{exe} {a}")
            if a.startswith("@") and len(a) > 1:
                self.access(seg, a[1:], self.resolve(a[1:], False, cwd), False, f"{exe} upload")
        if exe in {"http", "https", "xh", "httpie"} and any(re.match(r"^[A-Za-z0-9_-]+(:=|=|@)", a) for a in _non_options(args)[1:]):
            write = True
        urls = extract_urls(" ".join(args))
        hosts = [host_from_url(u) for u in urls]
        if not hosts:
            hosts = [host_from_url(a) for a in _non_options(args) if looks_like_host(a) or is_private_host(a.split("/")[0].split(":")[0])]
        if not hosts:
            seg.add(Crossing.NETWORK_EGRESS, f"{exe} to an unresolvable destination")
            if write:
                seg.add(Crossing.NETWORK_WRITE, f"{exe} sends data")
        for h in hosts:
            self.network_target(seg, h, write=write)

    def _channel(self, seg: SegmentProfile, exe: str, args: list[str]) -> None:
        hosts: list[str] = []
        for a in _non_options(args):
            if "://" in a:
                hosts.append(host_from_url(a))
            elif "@" in a or (":" in a and exe in {"scp", "rsync", "sftp", "rclone"}):
                hosts.append(normalize_host(a.split("@", 1)[-1].split(":", 1)[0]))
            elif exe in {"nc", "ncat", "netcat", "telnet", "ssh", "ftp", "socat"} and (looks_like_host(a) or is_private_host(a)):
                hosts.append(normalize_host(a))
        local_rsync = exe == "rsync" and not hosts
        if local_rsync:
            ops = _non_options(args)
            if ops:
                self.access(seg, ops[-1], self.resolve(ops[-1]), True, "rsync")
            return
        for h in hosts:
            self.network_target(seg, h, write=True)
        if not hosts:
            seg.add(Crossing.NETWORK_EGRESS, f"{exe} opens a network channel")
            seg.add(Crossing.NETWORK_WRITE, f"{exe} can send data")


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


_REDACTOR = SecretRedactor()


def classify(call: ToolCall, spec: ToolSpec | None, ctx: RunContext, workspace: Workspace, config: BoundaryConfig | None = None) -> ActionProfile:
    """Build the :class:`ActionProfile` for ``call``. Pure: never executes or touches the network."""
    cfg = config or BoundaryConfig()
    kind = tool_kind(call.name, spec)
    profile = ActionProfile(call=call, spec=spec, kind=kind)
    c = _Classifier(workspace, cfg, ctx)
    inp = call.input or {}
    meta = spec.metadata if spec else {}

    if any(has_invisible(s) for s in _strings(inp)) and kind in {ToolKind.SHELL, ToolKind.EDIT, ToolKind.READ, ToolKind.SEARCH, ToolKind.WEB_FETCH}:
        profile.add(Crossing.SUSPICIOUS_INPUT, "input contains invisible (zero-width or bidi) characters")
    if cfg.detect_secrets and kind not in {ToolKind.READ, ToolKind.SEARCH, ToolKind.INTERNAL} and _REDACTOR.contains_secret(canonical_json(inp)):
        profile.add(Crossing.SECRET_IN_INPUT, "input contains a credential-like string")

    if kind is ToolKind.SHELL:
        cmd = inp.get(meta.get("command_arg", "command"))
        if not isinstance(cmd, str):
            profile.add(Crossing.OPAQUE_EXECUTION, "shell call without a string 'command'")
        else:
            c.shell(profile, cmd)
        return profile

    path_keys = tuple(meta.get("path_args", _DEFAULT_PATH_ARGS))
    raw_paths: list[str] = []
    for key in path_keys:
        v = inp.get(key)
        if isinstance(v, str) and v:
            raw_paths.append(v)
        elif isinstance(v, list):
            raw_paths.extend(x for x in v if isinstance(x, str) and x)

    if kind in {ToolKind.READ, ToolKind.SEARCH}:
        pattern = inp.get("pattern") if kind is ToolKind.SEARCH and call.name.lower() == "glob" else None
        if isinstance(pattern, str) and (pattern.startswith(("/", "~")) or ".." in pattern):
            raw_paths.append(pattern)
        for p in raw_paths or ["."]:
            c.access(profile, p, c.resolve(p), False, call.name)
        profile.read_only = True
        return profile

    if kind is ToolKind.EDIT:
        for text in (inp.get("patch"), inp.get("input")):
            if isinstance(text, str):
                for m in _PATCH_FILE.finditer(text):
                    raw_paths.append((m.group(1) or m.group(2)).strip())
        if not raw_paths:
            profile.add(Crossing.UNKNOWN_EFFECTS, "edit without a recognisable target path")
        for p in raw_paths:
            c.access(profile, p, c.resolve(p), True, call.name)
        return profile

    if kind in {ToolKind.WEB_FETCH, ToolKind.BROWSER}:
        url = next((inp.get(k) for k in ("url", "uri", "href", "address") if isinstance(inp.get(k), str)), None)
        method = str(inp.get("method", "GET")).upper()
        write = method not in {"GET", "HEAD"} or any(k in inp for k in ("body", "data", "json", "form"))
        if url:
            c.network_target(profile, host_from_url(url), write=write)
        else:
            profile.add(Crossing.NETWORK_EGRESS, f"{call.name} without a URL")
        if kind is ToolKind.BROWSER and not url:
            profile.add(Crossing.UNKNOWN_EFFECTS, "browser action")
        profile.read_only = not write and kind is ToolKind.WEB_FETCH
        return profile

    if kind is ToolKind.WEB_SEARCH:
        profile.add(Crossing.NETWORK_EGRESS, "web search query leaves the machine")
        profile.read_only = True
        return profile

    if kind is ToolKind.SUBAGENT:
        profile.add(Crossing.SUBAGENT_SPAWN, f"spawns a sub-agent ({inp.get('subagent_type') or inp.get('agent') or 'general'})")
        return profile

    if kind is ToolKind.MEMORY:
        command = str(inp.get("command", "view"))
        if command == "view":
            profile.read_only = True
        else:
            profile.add(Crossing.MEMORY_WRITE, f"memory {command}")
        return profile

    if kind is ToolKind.INTERNAL:
        profile.read_only = True
        return profile

    if kind is ToolKind.MESSAGE:
        profile.add(Crossing.MESSAGE_SEND, f"{call.name} sends a message")
        profile.add(Crossing.NETWORK_WRITE, f"{call.name} sends data outside the workspace")
        return profile

    if kind is ToolKind.MCP:
        ref = parse_mcp(call.name, spec, cfg.vetted_mcp_servers)
        profile.mcp = ref
        assert ref is not None
        profile.add(Crossing.EXTERNAL_TOOL, f"MCP server {ref.server!r}{'' if ref.vetted else ' (unvetted)'}")
        if ref.read_only:
            profile.read_only = True
        else:
            if ref.destructive:
                profile.add(Crossing.IRREVERSIBLE, f"MCP tool {ref.tool!r} may be destructive (destructiveHint default true{'; server unvetted' if not ref.vetted else ''})")
            if _MESSAGE_NAME.search(ref.tool):
                profile.add(Crossing.MESSAGE_SEND, f"MCP tool {ref.tool!r} appears to send or publish")
                profile.add(Crossing.NETWORK_WRITE, f"MCP tool {ref.tool!r} sends data")
        for u in extract_urls(" ".join(_strings(inp))):
            c.network_target(profile, host_from_url(u))
        for p in raw_paths:
            c.access(profile, p, c.resolve(p), not ref.read_only, call.name)
        return profile

    # UNKNOWN: trust the harness-defined ToolSpec flag (not an MCP hint).
    if spec is not None and spec.read_only:
        profile.read_only = True
        for p in raw_paths:
            c.access(profile, p, c.resolve(p), False, call.name)
    else:
        profile.add(Crossing.UNKNOWN_EFFECTS, f"tool {call.name!r} has no read-only declaration")
        for p in raw_paths:
            c.access(profile, p, c.resolve(p), True, call.name)
    return profile
