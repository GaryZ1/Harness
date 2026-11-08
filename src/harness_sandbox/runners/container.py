"""Container runner and command builder for Docker / Podman.

Research basis (§6.3 technique 9, checklist §6.7 item 7): plain Docker is what the C-compiler
swarm, Vals.ai SWE-bench runs, mini-swe-agent ``docker.py`` and the Agents SDK ``DockerSandboxClient``
use; the checklist reserves it for *trusted internal and eval workloads* (Firecracker/gVisor for
untrusted multi-tenant code, which this package reaches through
:class:`~harness_sandbox.environment.RemoteEnvironment`). Every invocation is a fresh
``run --rm`` container, which makes it stateless per action (mini-swe-agent) and disposable
(Managed Agents "cattle, not pets").

Hardening flags, all derived from the policy:

====================================  ===============================================
``--network none``                    unless network mode is ``open`` (then ``bridge``)
``--read-only`` + ``--tmpfs /tmp``    image filesystem immutable; scratch is ephemeral
``--cap-drop ALL``                    no capabilities
``--security-opt no-new-privileges``  setuid binaries cannot escalate
``--pids-limit`` / ``--memory``       ``limits.max_processes`` / ``limits.memory_bytes``
``--ulimit nofile/fsize/core/cpu``    the remaining rlimits
``--user uid:gid``                    files written to the workspace keep the host owner
``-v ws:/workspace[:ro]``             only the workspace is visible (``:ro`` in read-only mode)
``-v ws/.git:/workspace/.git:ro``     protected subpaths stay read-only
``-v /dev/null:<.env>:ro``            glob read-denies inside the workspace are masked
``--env-file``                        values never appear in ``ps`` output
====================================  ===============================================

In ``allowlist``/``limited`` mode the container still gets ``--network none``; the proxy's Unix
socket directory and this package's launcher are bind-mounted and the launcher bridges
``127.0.0.1:3128`` to the proxy (requires ``python3`` in the image; otherwise a ``PolicyError``).
Docker is never required at test time: :func:`build_container_argv` is pure.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Mapping

from ..detect import find_trusted_binary
from ..errors import PolicyError, SandboxUnavailableError
from ..policy import CompiledPolicy, NetworkMode, SandboxMode
from .base import LAUNCHER_PATH, BaseRunner, PreparedCommand

LAUNCHER_MOUNT = "/opt/harness-sandbox"


@dataclass(frozen=True)
class ContainerConfig:
    image: str
    engine: str = "docker"                 # docker | podman
    workdir: str = "/workspace"
    user: str | None = "auto"              # "auto" = host uid:gid; None = image default
    python_in_image: str | None = "python3"
    tmpfs_size: str = "1g"
    cpus: float | None = None
    init: bool = True                      # --init: reap zombies, forward signals
    extra_args: tuple[str, ...] = ()       # appended before the image (review: may weaken isolation)


def _map(path: str, workspace: str, workdir: str) -> str | None:
    if path == workspace:
        return workdir
    if path.startswith(workspace.rstrip("/") + "/"):
        return workdir.rstrip("/") + "/" + path[len(workspace.rstrip("/")) + 1:]
    return None


def build_container_argv(
    compiled: CompiledPolicy,
    config: ContainerConfig,
    argv: list[str],
    *,
    name: str | None = None,
    env_file: str | None = None,
    proxy_socket: str | None = None,
    launcher_spec_json: str | None = None,
    interactive: bool = False,
    cwd: str | None = None,
) -> list[str]:
    policy, lim = compiled.policy, compiled.policy.limits
    ws = compiled.workspace
    net = policy.network.mode
    out = [config.engine, "run", "--rm"]
    if interactive:
        out.append("-i")
    if config.init:
        out.append("--init")
    if name:
        out += ["--name", name]
    out += ["--network", "bridge" if net is NetworkMode.OPEN else "none", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges"]
    if lim.max_processes:
        out += ["--pids-limit", str(lim.max_processes)]
    if lim.memory_bytes:
        out += ["--memory", str(lim.memory_bytes), "--memory-swap", str(lim.memory_bytes)]
    if config.cpus:
        out += ["--cpus", f"{config.cpus:g}"]
    if lim.open_files:
        out += ["--ulimit", f"nofile={lim.open_files}:{lim.open_files}"]
    if lim.file_size_bytes:
        out += ["--ulimit", f"fsize={lim.file_size_bytes}:{lim.file_size_bytes}"]
    if lim.cpu_seconds:
        out += ["--ulimit", f"cpu={lim.cpu_seconds}:{lim.cpu_seconds + 5}"]
    if not lim.core_dumps:
        out += ["--ulimit", "core=0:0"]
    user = config.user
    if user == "auto":
        user = f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else None
    if user:
        out += ["--user", user]
    out += ["--tmpfs", f"/tmp:rw,nosuid,nodev,size={config.tmpfs_size}"]
    ws_writable = ws in compiled.writable_roots and policy.mode is not SandboxMode.READ_ONLY
    out += ["-v", f"{ws}:{config.workdir}:{'rw' if ws_writable else 'ro'}"]
    if ws_writable:
        for p in compiled.read_only_paths:
            inside = _map(p, ws, config.workdir)
            if inside and os.path.lexists(p):
                out += ["-v", f"{p}:{inside}:ro"]
    for p in compiled.find_glob_matches(compiled.glob_deny_reads()):
        inside = _map(p, ws, config.workdir)
        if inside:
            out += (["--tmpfs", f"{inside}:ro"] if os.path.isdir(p) else ["-v", f"/dev/null:{inside}:ro"])
    if env_file:
        out += ["--env-file", env_file]
    inner = list(argv)
    if net.uses_proxy:
        if not config.python_in_image:
            raise PolicyError("allowlist/limited network in a container needs python3 in the image for the proxy bridge")
        if not proxy_socket or launcher_spec_json is None:
            raise PolicyError("allowlist/limited network needs the proxy Unix socket")
        d = os.path.dirname(proxy_socket)
        out += ["-v", f"{d}:{d}", "-v", f"{os.path.dirname(LAUNCHER_PATH)}:{LAUNCHER_MOUNT}:ro"]
        inner = [config.python_in_image, "-I", "-S", f"{LAUNCHER_MOUNT}/{os.path.basename(LAUNCHER_PATH)}",
                 "--spec", launcher_spec_json, "--", *argv]
    wd = _map(cwd, ws, config.workdir) if cwd else None
    out += ["--workdir", wd or config.workdir]
    out += list(config.extra_args)
    out += [config.image, *inner]
    return out


class ContainerRunner(BaseRunner):
    name = "container"
    isolation = "container"
    weak = False

    def __init__(self, policy: Any, workspace: Any, *, config: ContainerConfig, engine_path: str | None = None, **kwargs: Any) -> None:
        super().__init__(policy, workspace, **kwargs)
        self.config = config
        self.engine_path = engine_path or find_trusted_binary(config.engine, workspace=self.workspace)

    @classmethod
    def available(cls, engine: str = "docker") -> bool:
        return find_trusted_binary(engine) is not None

    def _needs_unix_proxy(self) -> bool:
        return True

    def _tmpdir_inside(self) -> str | None:
        return "/tmp"

    def prepare(self, argv: list[str] | str, *, cwd: str | None = None, env: Mapping[str, str] | None = None,
                command_id: str | None = None) -> PreparedCommand:
        prepared = super().prepare(argv, cwd=cwd, env=env, command_id=command_id)
        fd, env_file = tempfile.mkstemp(prefix="harness-env-", suffix=".list")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for k, v in prepared.env.items():
                if k in ("PATH", "HOME") or "\n" in v:
                    continue  # the image owns PATH/HOME; env-file cannot carry newlines
                fh.write(f"{k}={v}\n")
        name = f"harness-{prepared.command_id}"
        prepared.argv[prepared.argv.index("--env-file") + 1] = env_file
        prepared.argv[prepared.argv.index("--name") + 1] = name
        prepared.env = dict(os.environ)  # the engine *client* runs outside the sandbox
        engine = self.engine_path or self.config.engine
        prepared.on_timeout.append(lambda: subprocess.run([engine, "kill", name], capture_output=True, timeout=30, check=False))
        prepared.cleanup.append(lambda: os.path.exists(env_file) and os.unlink(env_file))
        return prepared

    def _wrap(self, argv: list[str], *, cwd: str, command_id: str) -> tuple[list[str], str | None, list[str]]:
        if self.engine_path is None:
            raise SandboxUnavailableError(f"{self.config.engine} not found on PATH outside the workspace")
        proxy = self.proxy
        spec_json = None
        proxy_url = None
        if proxy is not None:
            import json

            port = self.policy.network.proxy_port_in_sandbox
            spec_json = json.dumps({"bridge": {"port": port, "socket": proxy.unix_socket_path}, "loopback": True,
                                    "no_new_privs": True, "seccomp": False}, separators=(",", ":"))
            proxy_url = proxy.url(command_id, host="127.0.0.1", port=port)
        cfg = self.config
        full = build_container_argv(self.compiled, cfg, argv, name="PLACEHOLDER", env_file="PLACEHOLDER",
                                    proxy_socket=proxy.unix_socket_path if proxy else None, launcher_spec_json=spec_json,
                                    interactive=True, cwd=cwd)
        full[0] = self.engine_path
        return full, proxy_url, []

    def describe(self) -> str:
        return f"{self.config.engine} container ({self.config.image})"
