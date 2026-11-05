"""Hardened local fallback runner. **Not a security boundary for the filesystem.**

Research basis: OpenAI's Agents SDK warns its Unix-local client "adds no OS-level confinement" and
smolagents says its local executor "is not a security sandbox" (§6.8 "mistaking a sandbox API for
isolation"). This runner is therefore marked ``weak = True`` and the backend chooser reports
``isolated=False`` for it, unless the harness itself is inside a detected container
(``external-sandbox`` mode, where the outer box is the boundary and this runner adds defence in depth).

What it does enforce, for every command and its whole process tree:

* environment allowlist and secret scrubbing (no tokens inside the box);
* ``RLIMIT_CPU``/``DATA``/``NPROC``/``FSIZE``/``NOFILE``/``CORE`` via the in-sandbox launcher;
* ``PR_SET_NO_NEW_PRIVS`` plus the seccomp filter (no nested namespaces, mount, ptrace, BPF...);
* new session + process-group kill on timeout and after exit; capped output;
* **network isolation when the host allows it**: with ``unshare --net`` (root or unprivileged user
  namespaces) the command gets an empty network namespace; in ``allowlist``/``limited`` mode the
  launcher bridges ``127.0.0.1:<port>`` to the egress proxy's Unix socket, exactly like the
  bubblewrap path. Without ``unshare`` the proxy variables are advisory only.

What it does **not** do: confine reads or writes. File tools must still call
``CompiledPolicy.check_read`` / ``check_write``.
"""

from __future__ import annotations

import os
from typing import Any

from ..detect import unshare_netns_argv
from ..policy import NetworkMode
from .base import BaseRunner


class LocalRunner(BaseRunner):
    name = "local"
    isolation = "process"
    weak = True

    def __init__(self, *args: Any, network_namespace: bool | str = "auto", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        wants_isolation = self.policy.network.mode is not NetworkMode.OPEN
        prefix = unshare_netns_argv() if (wants_isolation and network_namespace in ("auto", True)) else None
        if network_namespace is True and prefix is None and wants_isolation:
            from ..errors import SandboxUnavailableError

            raise SandboxUnavailableError("network_namespace=True but 'unshare --net' is not usable on this host")
        self.netns_prefix: tuple[str, ...] | None = prefix
        if prefix:
            self.isolation = "process+netns"

    def _needs_unix_proxy(self) -> bool:
        return self.netns_prefix is not None

    def _wrap(self, argv: list[str], *, cwd: str, command_id: str) -> tuple[list[str], str | None, list[str]]:
        spec = self.launcher_spec()
        notes: list[str] = []
        proxy_url = None
        mode = self.policy.network.mode
        proxy = self.proxy
        if self.netns_prefix:
            if proxy is not None:
                port = self.policy.network.proxy_port_in_sandbox
                spec["bridge"] = {"port": port, "socket": proxy.unix_socket_path}
                spec["loopback"] = True
                proxy_url = proxy.url(command_id, host="127.0.0.1", port=port)
            full = [*self.netns_prefix, *self.launcher_argv(spec, argv)]
        else:
            if proxy is not None:
                proxy_url = proxy.url(command_id)
                notes.append("no network namespace: proxy variables are advisory only")
            elif mode is NetworkMode.NONE:
                notes.append("no network namespace: network mode 'none' is NOT enforced")
            full = self.launcher_argv(spec, argv)
        return full, proxy_url, notes

    def describe(self) -> str:
        return f"local runner ({self.isolation}; filesystem NOT confined; pid {os.getpid()} harness outside the process group)"
