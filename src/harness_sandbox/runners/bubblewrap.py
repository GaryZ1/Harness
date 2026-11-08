"""Linux bubblewrap runner: the default local OS sandbox (Codex default; srt Linux backend).

Research basis (§6.2 Codex linux-sandbox and srt; checklist §6.7 items 2-4; escape tests item 19):

* ``--ro-bind / /``: reads allowed by default (srt), writes denied everywhere...
* ...except writable roots re-bound read-write (workspace, session tmp, ``allow_write``);
* protected subpaths (``.git``, resolved gitdir, ``.harness``) re-bound read-only on top;
* read-denied paths masked: directories with an empty ``--tmpfs``, files and sockets with
  ``--ro-bind /dev/null``; ``allow_read`` exceptions re-bound read-only beneath a mask;
* glob denies (``**/.env``) expanded to concrete matches under the writable roots and masked;
* ``--unshare-net`` unless network mode is ``open``: the sandbox has no network at all, and in
  ``allowlist``/``limited`` mode the proxy's Unix-socket directory is bound in and the launcher bridges
  ``127.0.0.1:3128`` to it (srt: "network namespace removed entirely ... host proxies on
  bind-mounted Unix sockets");
* ``--unshare-user/pid/ipc/uts/cgroup``, ``--new-session`` (no TIOCSTI terminal injection),
  ``--die-with-parent`` (the tree dies with the harness), ``--cap-drop ALL``, ``--disable-userns``
  where supported, plus the launcher's seccomp filter and ``PR_SET_NO_NEW_PRIVS``;
* a fresh ``/tmp`` tmpfs, ``/dev`` and ``/proc`` for the new PID namespace.

Mounts are emitted shallowest-first so deeper rules overlay shallower ones, which is exactly the
precedence the policy defines (a deeper allow beats a shallower deny; a deeper deny beats both).
The environment is passed through ``Popen(env=...)``, never ``--setenv``, so values never appear in
``/proc/<pid>/cmdline``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from ..detect import detect_backends, find_trusted_binary
from ..errors import SandboxUnavailableError
from ..policy import CompiledPolicy, NetworkMode
from .base import LAUNCHER_PATH, BaseRunner


@dataclass(frozen=True)
class _Mount:
    dest: str
    priority: int        # same depth: masks after binds, exceptions after masks, sockets last
    args: tuple[str, ...]


def _depth(p: str) -> int:
    return len([x for x in p.split("/") if x])


def _under(p: str, root: str) -> bool:
    return p == root or p.startswith(root.rstrip("/") + "/")


def build_bwrap_argv(
    compiled: CompiledPolicy,
    *,
    cwd: str,
    bwrap: str = "bwrap",
    proxy_socket: str | None = None,
    disable_userns: bool = False,
    exists: Callable[[str], bool] = os.path.lexists,
    isdir: Callable[[str], bool] = os.path.isdir,
    realpath: Callable[[str], str] = os.path.realpath,
) -> list[str]:
    """Exact bwrap argv (without the trailing ``-- command``) for ``compiled``."""
    net = compiled.policy.network.mode
    argv = [bwrap, "--new-session", "--die-with-parent", "--unshare-user", "--unshare-pid", "--unshare-ipc",
            "--unshare-uts", "--unshare-cgroup-try"]
    if net is not NetworkMode.OPEN:
        argv.append("--unshare-net")
    if disable_userns and not compiled.policy.escape_hatches.allow_nested_namespaces:
        argv.append("--disable-userns")
    argv += ["--cap-drop", "ALL", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]

    mounts: list[_Mount] = []
    for root in compiled.writable_roots:
        r = realpath(root)
        if exists(r):
            mounts.append(_Mount(r, 1, ("--bind", r, r)))
    masked: list[str] = []
    for p in (*compiled.literal_deny_reads(), *compiled.find_glob_matches(compiled.glob_deny_reads())):
        r = realpath(p)
        if not exists(r) or r in masked or r == "/":
            continue
        masked.append(r)
        if isdir(r):
            mounts.append(_Mount(r, 2, ("--tmpfs", r)))
        else:
            mounts.append(_Mount(r, 2, ("--ro-bind", "/dev/null", r)))
    for p in compiled.literal_allow_reads():
        r = realpath(p)
        if exists(r) and any(_under(r, m) and r != m for m in masked):
            mounts.append(_Mount(r, 3, ("--ro-bind", r, r)))
    for p in (*compiled.literal_deny_writes(), *compiled.find_glob_matches(compiled.glob_deny_writes())):
        r = realpath(p)
        if exists(r) and r not in masked and any(_under(r, realpath(w)) for w in compiled.writable_roots):
            mounts.append(_Mount(r, 2, ("--ro-bind", r, r)))
    if proxy_socket and net.uses_proxy:
        d = os.path.dirname(proxy_socket)
        mounts.append(_Mount(d, 4, ("--bind", d, d)))

    mounts.sort(key=lambda m: (_depth(m.dest), m.priority, m.dest))
    seen: set[tuple[str, ...]] = set()
    for i, m in enumerate(mounts):
        if m.args in seen:
            continue
        seen.add(m.args)
        argv += list(m.args)
        if m.args[0] == "--tmpfs" and not any(_under(o.dest, m.dest) and o.dest != m.dest for o in mounts[i + 1:]):
            argv += ["--remount-ro", m.dest]
    argv += ["--chdir", cwd]
    return argv


class BubblewrapRunner(BaseRunner):
    name = "bubblewrap"
    isolation = "os"
    weak = False

    def __init__(self, *args: Any, bwrap: str | None = None, disable_userns: bool | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.bwrap = bwrap or find_trusted_binary("bwrap", workspace=self.workspace)
        if self.bwrap is None:
            raise SandboxUnavailableError("bubblewrap (bwrap) not found on PATH outside the workspace")
        if disable_userns is None:
            disable_userns = detect_backends(workspace=self.workspace).bwrap_disable_userns
        self.disable_userns = disable_userns

    @classmethod
    def available(cls) -> bool:
        r = detect_backends()
        return r.bwrap_usable and r.wsl != 1

    def _needs_unix_proxy(self) -> bool:
        return True

    def _wrap(self, argv: list[str], *, cwd: str, command_id: str) -> tuple[list[str], str | None, list[str]]:
        compiled = self.compiled
        if not compiled.check_read(self.python).allowed or not compiled.check_read(LAUNCHER_PATH).allowed:
            raise SandboxUnavailableError(f"python interpreter {self.python} is read-denied inside the sandbox")
        spec = self.launcher_spec()
        proxy = self.proxy
        proxy_url = None
        if proxy is not None:
            port = self.policy.network.proxy_port_in_sandbox
            spec["bridge"] = {"port": port, "socket": proxy.unix_socket_path}
            spec["loopback"] = True
            proxy_url = proxy.url(command_id, host="127.0.0.1", port=port)
        prefix = build_bwrap_argv(compiled, cwd=cwd, bwrap=self.bwrap, disable_userns=self.disable_userns,
                                  proxy_socket=proxy.unix_socket_path if proxy else None)
        return [*prefix, "--", *self.launcher_argv(spec, argv)], proxy_url, []
