"""Detect which isolation backends this host offers and choose the strongest one.

Research basis (§6.2 Codex linux-sandbox, checklist §6.7 items 1, 2, 7; pitfalls §6.8):

* Prefer an OS sandbox: bubblewrap on Linux, Seatbelt (``sandbox-exec``) on macOS.
* Resolve helper binaries **only from PATH entries outside the workspace and CWD** (Codex), so a
  repository cannot ship its own ``bwrap`` ("picking up an attacker-supplied bwrap" pitfall).
* Detect user-namespace availability (Codex warns without it) and **refuse WSL1**.
* Inside a container, "use ``external-sandbox`` mode and keep the proxy rather than switching to full
  access" (pitfall: "disabling the sandbox because nesting fails"). Detection is from the environment
  (``/.dockerenv``, ``/run/.containerenv``, cgroup names), not from a flag the agent could set.
* A runner that is not a security boundary is reported as such (``isolated=False``) so the
  permission layer does not auto-allow in-boundary actions on top of it (the 84% prompt reduction
  is only safe when the boundary is real).
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from . import _launcher
from .errors import SandboxUnavailableError
from .policy import SandboxMode, SandboxPolicy

log = logging.getLogger("harness_sandbox")


def _inside(path: str, root: str | None) -> bool:
    if not root:
        return False
    p, r = os.path.realpath(path), os.path.realpath(root)
    return p == r or p.startswith(r.rstrip("/") + "/")


def find_trusted_binary(name: str, *, workspace: str | None = None, path: str | None = None, cwd: str | None = None) -> str | None:
    """``shutil.which`` that skips relative PATH entries and anything inside the workspace or CWD."""
    cwd = cwd or os.getcwd()
    if os.path.realpath(cwd) == "/":
        cwd = None  # every directory is "inside" /; the workspace check still applies
    for d in (path if path is not None else os.environ.get("PATH", os.defpath)).split(os.pathsep):
        if not d or not os.path.isabs(d) or _inside(d, workspace) or _inside(d, cwd):
            continue
        cand = os.path.join(d, name)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            real = os.path.realpath(cand)
            if _inside(real, workspace) or _inside(real, cwd):
                continue
            return cand
    return None


def wsl_version(proc_version: str | None = None) -> int | None:
    if proc_version is None:
        try:
            with open("/proc/version", encoding="utf-8") as fh:
                proc_version = fh.read()
        except OSError:
            return None
    low = proc_version.lower()
    if "microsoft" not in low:
        return None
    return 2 if ("microsoft-standard" in low or "wsl2" in low) else 1


def container_hint(root: str = "/") -> str:
    """Name of the container runtime we appear to be inside, or ``""``."""
    if os.path.exists(os.path.join(root, ".dockerenv")):
        return "docker"
    if os.path.exists(os.path.join(root, "run/.containerenv")):
        return "podman"
    if os.environ.get("container"):
        return os.environ["container"]
    try:
        with open(os.path.join(root, "proc/1/cgroup"), encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return ""
    for marker in ("kubepods", "docker", "containerd", "libpod", "lxc"):
        if marker in text:
            return marker
    return ""


def _probe(argv: list[str]) -> bool:
    try:
        return subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=10, check=False).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@lru_cache(maxsize=8)
def unshare_netns_argv(unshare: str | None = None) -> tuple[str, ...] | None:
    """Working ``unshare`` prefix that gives a private network namespace, or ``None``."""
    unshare = unshare or find_trusted_binary("unshare")
    if not unshare or not sys.platform.startswith("linux"):
        return None
    candidates = [(unshare, "--net")] if os.geteuid() == 0 else []
    candidates.append((unshare, "--net", "--map-root-user"))
    for c in candidates:
        if _probe([*c, "true"]):
            return c
    return None


@dataclass(frozen=True)
class BackendReport:
    platform: str
    arch: str
    bwrap: str | None = None
    bwrap_usable: bool = False
    bwrap_disable_userns: bool = False
    unshare_netns: tuple[str, ...] | None = None
    sandbox_exec: str | None = None
    docker: str | None = None
    podman: str | None = None
    container: str = ""
    wsl: int | None = None
    seccomp: bool = False
    notes: tuple[str, ...] = ()

    @property
    def inside_container(self) -> bool:
        return bool(self.container)

    def available(self) -> list[str]:
        """Usable backends, strongest first."""
        out = []
        if self.bwrap_usable and self.wsl != 1:
            out.append("bubblewrap")
        if self.sandbox_exec:
            out.append("seatbelt")
        if self.docker or self.podman:
            out.append("container")
        if self.inside_container:
            out.append("external")
        out.append("local+netns" if self.unshare_netns else "local")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__} | {"available": self.available()}


def detect_backends(*, workspace: str | None = None, probe: bool = True) -> BackendReport:
    plat = sys.platform
    arch = _launcher.machine()
    notes: list[str] = []
    wsl = wsl_version() if plat.startswith("linux") else None
    bwrap = find_trusted_binary("bwrap", workspace=workspace) if plat.startswith("linux") else None
    usable = disable_userns = False
    if bwrap and probe:
        usable = _probe([bwrap, "--ro-bind", "/", "/", "--unshare-user", "--unshare-net", "--die-with-parent", "true"])
        if not usable:
            notes.append("bwrap found but user namespaces are unavailable (Codex warns here); falling back")
        try:
            help_text = subprocess.run([bwrap, "--help"], capture_output=True, text=True, timeout=10).stdout
            disable_userns = "--disable-userns" in help_text
        except (OSError, subprocess.SubprocessError):
            pass
    if wsl == 1:
        notes.append("WSL1 detected: bubblewrap is refused (Codex rejects WSL1)")
        usable = False
    sbx = "/usr/bin/sandbox-exec" if plat == "darwin" and os.path.exists("/usr/bin/sandbox-exec") else None
    if plat.startswith("win"):
        notes.append("Windows sandbox (dedicated user + WFP) is not implemented; only the local fallback is available")
    return BackendReport(
        platform=plat, arch=arch, bwrap=bwrap, bwrap_usable=usable, bwrap_disable_userns=disable_userns,
        unshare_netns=unshare_netns_argv() if probe else None, sandbox_exec=sbx,
        docker=find_trusted_binary("docker", workspace=workspace), podman=find_trusted_binary("podman", workspace=workspace),
        container=container_hint(), wsl=wsl,
        seccomp=plat.startswith("linux") and arch in _launcher.SYSCALLS, notes=tuple(notes),
    )


@dataclass(frozen=True)
class BackendChoice:
    backend: str                 # bubblewrap | seatbelt | local
    mode: SandboxMode
    isolated: bool               # a real security boundary exists (OS sandbox or detected outer container)
    boundary: str                # os | external | none
    reason: str
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def auto_allow_in_boundary(self) -> bool:
        """Whether the permission layer may auto-allow in-boundary actions (Anthropic's 84%)."""
        return self.isolated and self.mode is not SandboxMode.FULL_ACCESS


def choose_backend(policy: SandboxPolicy, report: BackendReport | None = None, *, require_isolation: bool = False) -> BackendChoice:
    report = report or detect_backends()
    mode = policy.mode
    if mode is SandboxMode.FULL_ACCESS:
        return BackendChoice("local", mode, False, "none", "full-access mode: no sandbox", ("FULL ACCESS: no isolation",))
    if mode is SandboxMode.EXTERNAL_SANDBOX:
        return BackendChoice("local", mode, report.inside_container, "external" if report.inside_container else "none",
                             "external-sandbox mode: relying on the outer container/VM",
                             () if report.inside_container else ("external-sandbox requested but no container detected",))
    if report.platform.startswith("linux") and report.bwrap_usable and report.wsl != 1:
        return BackendChoice("bubblewrap", mode, True, "os", f"bubblewrap at {report.bwrap}")
    if report.platform == "darwin" and report.sandbox_exec:
        return BackendChoice("seatbelt", mode, True, "os", "macOS Seatbelt (sandbox-exec)")
    if report.inside_container:
        return BackendChoice("local", SandboxMode.EXTERNAL_SANDBOX, True, "external",
                             f"inside a {report.container} container without a nested OS sandbox: external-sandbox mode, proxy kept",
                             ("nested sandbox unavailable; relying on the outer container",))
    if require_isolation:
        raise SandboxUnavailableError(f"no OS sandbox available on {report.platform} ({'; '.join(report.notes) or 'install bubblewrap'})")
    warn = "NO OS SANDBOX: the local runner is not a security boundary (filesystem is not confined)"
    log.warning(warn)
    return BackendChoice("local", mode, False, "none", "fallback: hardened local runner", (warn,))


def create_runner(policy: SandboxPolicy, workspace: str, *, report: BackendReport | None = None,
                  require_isolation: bool = False, **kwargs: Any):
    """Instantiate the runner :func:`choose_backend` picks. ``kwargs`` go to the runner."""
    from .runners.bubblewrap import BubblewrapRunner
    from .runners.local import LocalRunner
    from .runners.seatbelt import SeatbeltRunner

    report = report or detect_backends(workspace=workspace)
    choice = choose_backend(policy, report, require_isolation=require_isolation)
    if choice.backend == "bubblewrap":
        runner = BubblewrapRunner(policy, workspace, bwrap=report.bwrap, disable_userns=report.bwrap_disable_userns, **kwargs)
    elif choice.backend == "seatbelt":
        runner = SeatbeltRunner(policy, workspace, **kwargs)
    else:
        runner = LocalRunner(policy if choice.mode is policy.mode else policy.with_mode(choice.mode), workspace, **kwargs)
    runner.choice = choice  # type: ignore[attr-defined]
    return runner


def host_summary() -> str:
    return f"{platform.system()} {platform.release()} ({_launcher.machine()})"


__all__ = ["BackendChoice", "BackendReport", "choose_backend", "create_runner", "detect_backends", "find_trusted_binary",
           "container_hint", "wsl_version", "unshare_netns_argv", "host_summary"]
