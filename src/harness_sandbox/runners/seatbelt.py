"""macOS Seatbelt runner: a generated SBPL profile run under ``/usr/bin/sandbox-exec``.

Research basis (§6.2: Claude Code, srt and Codex all use Seatbelt on macOS with per-invocation
generated profiles; srt's profile "allows only one localhost proxy port"; Codex keeps the resolved
gitdir read-only on macOS).

Profile shape (``deny default`` first, then narrow allows; ``require-not`` instead of relying on
rule ordering, as Codex's profiles do):

* process/signal/sysctl/mach basics needed by ordinary toolchains (a fixed mach-lookup list, not a
  blanket allow);
* ``file-read*`` everywhere *except* the read-deny rules; ``allow_read`` exceptions re-allow a
  subtree while still excluding any more specific deny inside it (srt precedence);
* ``file-write*`` only under writable roots, minus protected subpaths and deny rules, plus the
  usual ``/dev`` sinks;
* network: nothing for ``none``; exactly one ``localhost:<proxy port>`` outbound rule for
  ``allowlist``/``limited``; everything for ``open``.

The generator is pure text, so it is unit-tested on Linux CI; the runner itself needs macOS.
Resource limits come from the launcher (``setrlimit``); there is no seccomp on macOS.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from ..errors import SandboxUnavailableError
from ..policy import CompiledPolicy, NetworkMode, PathRule
from .base import BaseRunner

SANDBOX_EXEC = "/usr/bin/sandbox-exec"

_MACH_SERVICES = (
    "com.apple.system.opendirectoryd.libinfo", "com.apple.system.logger", "com.apple.system.notification_center",
    "com.apple.trustd.agent", "com.apple.trustd", "com.apple.cfprefsd.daemon", "com.apple.cfprefsd.agent",
    "com.apple.SecurityServer", "com.apple.coreservices.launchservicesd", "com.apple.diagnosticd", "com.apple.logd",
    "com.apple.system.DirectoryService.libinfo_v1", "com.apple.bsd.dirhelper", "com.apple.lsd.mapdb",
)

_BASE = """(version 1)
(deny default)
(allow process-exec)
(allow process-fork)
(allow signal (target same-sandbox))
(allow process-info* (target same-sandbox))
(allow sysctl-read)
(allow user-preference-read)
(allow ipc-posix-sem)
(allow ipc-posix-shm)
(allow pseudo-tty)
(allow file-ioctl (literal "/dev/tty") (regex #"^/dev/ttys[0-9]+$"))
(allow iokit-open (iokit-registry-entry-class "RootDomainUserClient"))
"""


def sbpl_string(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def glob_to_sbpl_regex(pattern: str) -> str:
    """POSIX-ERE regex for a path glob, matching the node and everything beneath it."""
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        elif pattern[i].isalnum() or pattern[i] in "/_-~":
            out.append(pattern[i])
            i += 1
        else:
            out.append(f"[{pattern[i]}]" if pattern[i] not in "]^\\" else f"\\{pattern[i]}")
            i += 1
    body = "".join(out)
    prefix = "^" if body.startswith(("/", "(.*/)?")) else "^(.*/)?"
    return f"{prefix}{body}(/.*)?$"


def _filter(rule_literal: str | None = None, rule: PathRule | None = None, compiled: CompiledPolicy | None = None) -> str:
    if rule is not None and rule.is_glob:
        src = rule.source
        if not src.startswith("**") and compiled is not None:
            src = compiled._literal(src)
        return f'(regex #"{glob_to_sbpl_regex(src)}")'
    return f"(subpath {sbpl_string(rule_literal or (rule.literal if rule else ''))})"


def build_seatbelt_profile(compiled: CompiledPolicy, *, proxy_port: int | None = None) -> str:
    """Return the SBPL profile text for ``compiled``."""
    lines = [_BASE.rstrip("\n")]
    lines.append("(allow mach-lookup " + " ".join(f"(global-name {sbpl_string(s)})" for s in _MACH_SERVICES) + ")")

    denies = [_filter(rule=r, compiled=compiled) for r in compiled.deny_read]
    denies = list(dict.fromkeys(denies))
    nots = "".join(f" (require-not {d})" for d in denies)
    lines.append(f'(allow file-read* (require-all (subpath "/"){nots}))')
    for allow in compiled.literal_allow_reads():
        inner = [_filter(rule=r, compiled=compiled) for r in compiled.deny_read
                 if r.is_glob or (r.literal and r.literal.startswith(allow.rstrip("/") + "/"))]
        extra = "".join(f" (require-not {d})" for d in dict.fromkeys(inner))
        lines.append(f"(allow file-read* (require-all (subpath {sbpl_string(allow)}){extra}))")
    lines.append('(allow file-read-metadata (literal "/"))')

    roots = list(dict.fromkeys([*compiled.writable_roots, *[os.path.realpath(r) for r in compiled.writable_roots]]))
    if roots:
        any_root = " ".join(f"(subpath {sbpl_string(r)})" for r in roots)
        wdenies = list(dict.fromkeys([*(_filter(rule=r, compiled=compiled) for r in compiled.deny_write), *denies]))
        wnots = "".join(f" (require-not {d})" for d in wdenies)
        lines.append(f"(allow file-write* (require-all (require-any {any_root}){wnots}))")
    lines.append('(allow file-write-data (literal "/dev/null") (literal "/dev/zero") (literal "/dev/tty") '
                 '(literal "/dev/stdout") (literal "/dev/stderr") (regex #"^/dev/ttys[0-9]+$") (regex #"^/dev/fd/[0-9]+$"))')

    net = compiled.policy.network.mode
    if net is NetworkMode.OPEN:
        lines.append("(allow network*)")
        lines.append("(allow system-socket)")
    elif net.uses_proxy:
        if proxy_port is None:
            raise ValueError("allowlist/limited network mode needs the proxy port")
        lines.append(f'(allow network-outbound (remote ip "localhost:{int(proxy_port)}"))')
    lines.append(f"; harness_sandbox policy {compiled.policy.fingerprint()} mode={compiled.policy.mode.value} network={net.value}")
    return "\n".join(lines) + "\n"


class SeatbeltRunner(BaseRunner):
    name = "seatbelt"
    isolation = "os"
    weak = False

    def __init__(self, *args: Any, sandbox_exec: str = SANDBOX_EXEC, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.sandbox_exec = sandbox_exec

    @classmethod
    def available(cls) -> bool:
        return sys.platform == "darwin" and os.path.exists(SANDBOX_EXEC)

    def build_argv(self, argv: list[str], *, command_id: str) -> tuple[list[str], str | None]:
        """Full argv (``sandbox-exec -p PROFILE python launcher ... -- argv``); usable on any OS."""
        spec = self.launcher_spec()
        spec["seccomp"] = False
        spec["no_new_privs"] = False
        proxy = self.proxy
        port = proxy.port if proxy is not None else None
        profile = build_seatbelt_profile(self.compiled, proxy_port=port)
        url = proxy.url(command_id) if proxy is not None else None
        return [self.sandbox_exec, "-p", profile, *self.launcher_argv(spec, argv)], url

    def _wrap(self, argv: list[str], *, cwd: str, command_id: str) -> tuple[list[str], str | None, list[str]]:
        if not self.available():
            raise SandboxUnavailableError("Seatbelt (sandbox-exec) is only available on macOS")
        full, url = self.build_argv(argv, command_id=command_id)
        return full, url, []
