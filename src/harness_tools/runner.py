"""Pluggable command execution for the ``bash`` tool.

Research basis (``reports/SOTA agentic harness reference.md`` §5.7 step 4, §5.3 D, §6):

* "Sandbox the shell, not each tool": every subprocess the model starts goes through one
  :class:`CommandRunner`, so a hardened runner (seatbelt/bubblewrap/container, egress proxy) can
  be swapped in without touching the tool. The sandbox package provides those; this module
  provides the protocol and a **local, unconfined** default suitable for trusted dev boxes,
  evals and tests (mini-swe-agent's stateless ``subprocess.run`` mode).
* Each call logs exit code, duration and output size; the whole **process group** is killed on
  timeout (so ``sleep 999 &``-style children do not outlive the call); captured output is
  capped in memory (the tool layer then applies the 25k-token result budget with offload).
* Environment hygiene (our choice): the child receives a minimal allowlisted environment, so API
  keys in the harness process are not inherited by model-run commands unless explicitly passed.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import IO, Iterable, Mapping, Protocol, runtime_checkable

from .errors import CommandRunnerError

DEFAULT_ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TMPDIR", "TZ", "USER", "LOGNAME", "SHELL")
DEFAULT_MAX_OUTPUT_BYTES = 1_000_000


@dataclass(frozen=True)
class CommandRequest:
    """One shell command to run.

    :param command: the shell command line (run with ``bash -c`` / ``sh -c``).
    :param cwd: absolute working directory.
    :param timeout_s: wall-clock limit; the process group is killed when it expires.
    :param env: extra environment variables layered over the runner's base environment.
    :param stdin: text fed to stdin (``None`` = closed / ``/dev/null``).
    :param max_output_bytes: per-stream capture cap; the remainder is counted but discarded.
    """

    command: str
    cwd: str
    timeout_s: float = 120.0
    env: Mapping[str, str] | None = None
    stdin: str | None = None
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES


@dataclass
class CommandResult:
    exit_code: int | None
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    output_truncated: bool = False
    sandbox_id: str | None = None
    violations: list[str] = field(default_factory=list)  # sandbox violations reported by hardened runners

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


@runtime_checkable
class CommandRunner(Protocol):
    """Runs a shell command and returns its captured result. Must never raise for a failing command
    (non-zero exit / timeout are results); raise :class:`CommandRunnerError` only when the runner
    itself is broken (cannot spawn, sandbox dead)."""

    def run(self, request: CommandRequest) -> CommandResult: ...


def _find_shell() -> list[str]:
    for sh in ("bash", "sh"):
        path = shutil.which(sh)
        if path:
            return [path, "-c"]
    if os.name == "nt":  # pragma: no cover - platform dependent
        return [os.environ.get("COMSPEC", "cmd.exe"), "/c"]
    raise CommandRunnerError("no shell found (bash or sh)")


def _drain(stream: IO[bytes], limit: int, sink: bytearray, counter: list[int]) -> None:
    try:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                break
            counter[0] += len(chunk)
            room = limit - len(sink)
            if room > 0:
                sink.extend(chunk[:room])
    except (OSError, ValueError):
        pass
    finally:
        try:
            stream.close()
        except OSError:
            pass


class LocalSubprocessRunner:
    """Unconfined local runner: stateless ``bash -c`` per call, process-group kill on timeout.

    .. warning:: This provides **no OS-level confinement**. Use a sandboxed runner for untrusted
       workloads (the research's "``LocalPythonExecutor`` is not a security sandbox" pitfall).

    :param inherit_env: pass the full parent environment (default ``False``: allowlist only).
    :param env_allowlist: variables copied from the parent when ``inherit_env`` is false.
    :param base_env: extra fixed variables for every command (e.g. ``{"CI": "1"}``).
    """

    def __init__(
        self,
        *,
        inherit_env: bool = False,
        env_allowlist: Iterable[str] = DEFAULT_ENV_ALLOWLIST,
        base_env: Mapping[str, str] | None = None,
        kill_grace_s: float = 2.0,
    ) -> None:
        self.inherit_env = inherit_env
        self.env_allowlist = tuple(env_allowlist)
        self.base_env = dict(base_env or {})
        self.kill_grace_s = kill_grace_s
        self._shell = _find_shell()

    def _env(self, request: CommandRequest) -> dict[str, str]:
        if self.inherit_env:
            env = dict(os.environ)
        else:
            env = {k: os.environ[k] for k in self.env_allowlist if k in os.environ}
        env.setdefault("PATH", os.defpath)
        env.update(self.base_env)
        env.update(dict(request.env or {}))
        env["PWD"] = request.cwd
        return env

    def run(self, request: CommandRequest) -> CommandResult:
        if not os.path.isdir(request.cwd):
            raise CommandRunnerError(f"working directory does not exist: {request.cwd}")
        posix = os.name == "posix"
        start = time.monotonic()
        try:
            proc = subprocess.Popen(
                [*self._shell, request.command],
                cwd=request.cwd,
                env=self._env(request),
                stdin=subprocess.PIPE if request.stdin is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=posix,
                close_fds=True,
            )
        except OSError as exc:
            raise CommandRunnerError(f"failed to start command: {exc}") from exc
        out, err = bytearray(), bytearray()
        out_n, err_n = [0], [0]
        assert proc.stdout is not None and proc.stderr is not None
        readers = [
            threading.Thread(target=_drain, args=(proc.stdout, request.max_output_bytes, out, out_n), daemon=True),
            threading.Thread(target=_drain, args=(proc.stderr, request.max_output_bytes, err, err_n), daemon=True),
        ]
        for t in readers:
            t.start()
        if request.stdin is not None and proc.stdin is not None:
            try:
                proc.stdin.write(request.stdin.encode("utf-8"))
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        timed_out = False
        try:
            proc.wait(timeout=request.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill(proc, posix)
        # Children that inherited the pipes may keep them open; the group kill above handles the
        # timeout case, and a bounded join keeps a stray background job from hanging the call.
        for t in readers:
            t.join(timeout=self.kill_grace_s if not timed_out else 1.0)
        if posix and not timed_out and any(t.is_alive() for t in readers):
            self._kill_group(proc.pid)
            for t in readers:
                t.join(timeout=1.0)
        duration = time.monotonic() - start
        return CommandResult(
            exit_code=None if timed_out else proc.returncode,
            stdout=out.decode("utf-8", errors="replace"),
            stderr=err.decode("utf-8", errors="replace"),
            duration_s=duration,
            timed_out=timed_out,
            stdout_bytes=out_n[0],
            stderr_bytes=err_n[0],
            output_truncated=out_n[0] > len(out) or err_n[0] > len(err),
        )

    @staticmethod
    def _kill_group(pid: int) -> None:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def _kill(self, proc: subprocess.Popen[bytes], posix: bool) -> None:
        if posix:
            self._kill_group(proc.pid)
        else:  # pragma: no cover - platform dependent
            proc.kill()
        try:
            proc.wait(timeout=self.kill_grace_s)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()


__all__ = [
    "CommandRequest",
    "CommandResult",
    "CommandRunner",
    "DEFAULT_ENV_ALLOWLIST",
    "DEFAULT_MAX_OUTPUT_BYTES",
    "LocalSubprocessRunner",
]
