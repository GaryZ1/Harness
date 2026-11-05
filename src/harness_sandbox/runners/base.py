"""The ``CommandRunner`` interface and the machinery every runner shares.

Interface (compatible with the sibling ``harness_tools.CommandRunner`` protocol, which this package
cannot import)::

    run(argv: list[str] | str, *, cwd: str, env: dict | None, timeout: float, stdin: str | None)
        -> CommandResult(exit_code, stdout, stderr, timed_out, truncated, duration_s, ...)

``CommandResult`` keeps those six fields first and in that order; the extra fields (``signal``,
``command_id``, ``sandbox``, ``violations``, byte counts, ``scrubbed_env``) all have defaults, so an
adapter can construct the sibling type with ``CommandResult(**{k: getattr(r, k) for k in SIX})``.
A string ``argv`` runs through ``/bin/bash -c`` (``/bin/sh`` when bash is absent).

Shared behaviour (research §6.3 techniques 1, 6, 14; checklist §6.7 items 5, 11, 17):

* **Whole process tree.** Every command starts in a new session/process group. On timeout the group
  gets SIGTERM, then SIGKILL after ``kill_grace_s``; after a normal exit the group is SIGKILLed too,
  so background children cannot outlive the call (OS sandboxes add a PID namespace on top).
* **Output caps.** Each stream keeps its head and tail up to ``max_output_bytes`` and reports
  ``truncated``; a runaway command cannot exhaust harness memory.
* **Stateless by default** (mini-swe-agent: "every action is completely independent"), which makes
  sandboxing and retries simple. Long-lived processes (stdio MCP servers, PTY shells) use
  :meth:`BaseRunner.prepare` to get the same sandboxed argv and environment for their own ``Popen``.
* **Violation feedback.** Proxy denials (attributed by command ID), timeouts, resource-limit signals
  and filesystem EROFS/EACCES errors become :class:`~harness_sandbox.violations.Violation` records.
* **Observability.** One ``sandbox.exec`` event per command (no output, no env values) tagged with
  the sandbox name, isolation level and command ID.
"""

from __future__ import annotations

import collections
import logging
import os
import shutil
import signal as _signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from harness_core.types import Event, EventSink, EventType, new_id
from harness_memory.security import SecretRedactor

from .. import _launcher
from ..environment_vars import build_env, proxy_env
from ..errors import PolicyError, SandboxError
from ..policy import CompiledPolicy, SandboxPolicy, canonicalize
from ..proxy import EgressProxy
from ..violations import Violation, detect_fs_violations, format_violations

log = logging.getLogger("harness_sandbox")

LAUNCHER_PATH = os.path.abspath(_launcher.__file__)
RESULT_FIELDS = ("exit_code", "stdout", "stderr", "timed_out", "truncated", "duration_s")
_REDACTOR = SecretRedactor()


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    truncated: bool = False
    duration_s: float = 0.0
    signal: int | None = None                  # fatal signal, if any (exit_code is then 128 + signal)
    command_id: str = ""
    sandbox: str = ""
    violations: tuple[Violation, ...] = ()
    stdout_bytes: int = 0                      # total bytes produced (before truncation)
    stderr_bytes: int = 0
    scrubbed_env: tuple[str, ...] = ()         # names removed from the environment

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def model_text(self) -> str:
        """stdout + stderr + ``<sandbox_violations>`` block, for a tool result."""
        parts = [self.stdout.rstrip("\n")] if self.stdout else []
        if self.stderr:
            parts.append(self.stderr.rstrip("\n"))
        if self.timed_out:
            parts.append(f"[timed out after {self.duration_s:.1f}s]")
        block = format_violations(self.violations)
        if block:
            parts.append(block)
        parts.append(f"[exit code {self.exit_code}]")
        return "\n".join(parts)


@runtime_checkable
class CommandRunner(Protocol):
    def run(self, argv: list[str] | str, *, cwd: str | None = None, env: dict[str, str] | None = None,
            timeout: float | None = None, stdin: str | None = None) -> CommandResult: ...


@dataclass
class PreparedCommand:
    """A fully sandboxed command line, ready for ``subprocess.Popen(argv, env=env, cwd=cwd,
    start_new_session=True)``. Used by :meth:`BaseRunner.run` and by long-lived processes."""

    argv: list[str]
    env: dict[str, str]
    cwd: str
    command_id: str
    scrubbed_env: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)
    on_timeout: list[Callable[[], None]] = field(default_factory=list)   # e.g. ``docker kill <name>``
    cleanup: list[Callable[[], None]] = field(default_factory=list)      # e.g. remove an env file

    def run_cleanup(self) -> None:
        for fn in self.cleanup:
            try:
                fn()
            except Exception:  # pragma: no cover - best effort
                log.exception("sandbox cleanup hook failed")

    def popen_kwargs(self) -> dict[str, Any]:
        return {"env": self.env, "cwd": self.cwd, "start_new_session": True, "close_fds": True}


class _CappedBuffer:
    """Keeps the first and last ``cap // 2`` bytes of a stream and counts the rest."""

    def __init__(self, cap: int) -> None:
        self.half = max(cap // 2, 1)
        self.head = bytearray()
        self.tail: collections.deque[bytes] = collections.deque()
        self.tail_len = 0
        self.total = 0

    def add(self, chunk: bytes) -> None:
        self.total += len(chunk)
        room = self.half - len(self.head)
        if room > 0:
            self.head += chunk[:room]
            chunk = chunk[room:]
        if chunk:
            self.tail.append(chunk)
            self.tail_len += len(chunk)
            while self.tail_len - len(self.tail[0]) >= self.half:
                self.tail_len -= len(self.tail.popleft())

    @property
    def truncated(self) -> bool:
        return self.total > len(self.head) + self.tail_len or self.tail_len > self.half

    def text(self) -> str:
        tail = b"".join(self.tail)
        if len(tail) > self.half:
            tail = tail[-self.half:]
        dropped = self.total - len(self.head) - len(tail)
        if dropped > 0:
            mid = f"\n[... {dropped} bytes truncated by sandbox output cap ...]\n".encode()
            return (bytes(self.head) + mid + tail).decode("utf-8", "replace")
        return (bytes(self.head) + tail).decode("utf-8", "replace")


def _reader(stream, buf: _CappedBuffer) -> None:
    try:
        for chunk in iter(lambda: stream.read1(65536) if hasattr(stream, "read1") else stream.read(65536), b""):
            buf.add(chunk)
    except (OSError, ValueError):
        pass


def _killpg(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def spawn_and_capture(prepared: PreparedCommand, *, stdin: str | None, timeout: float, max_output_bytes: int,
                      kill_grace_s: float = 2.0) -> tuple[int, _CappedBuffer, _CappedBuffer, bool, float]:
    """Run ``prepared`` with capped capture and process-group cleanup.

    Returns ``(returncode, stdout_buffer, stderr_buffer, timed_out, duration_s)``."""
    out, err = _CappedBuffer(max_output_bytes), _CappedBuffer(max_output_bytes)
    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(
            prepared.argv,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **prepared.popen_kwargs(),
        )
    except FileNotFoundError as exc:
        raise SandboxError(f"cannot start sandbox wrapper {prepared.argv[0]!r}: {exc}") from exc
    pgid = proc.pid
    threads = [threading.Thread(target=_reader, args=(proc.stdout, out), daemon=True),
               threading.Thread(target=_reader, args=(proc.stderr, err), daemon=True)]
    for t in threads:
        t.start()
    if stdin is not None:
        def feed() -> None:
            try:
                assert proc.stdin is not None
                proc.stdin.write(stdin.encode("utf-8"))
            except (BrokenPipeError, OSError):
                pass
            finally:
                try:
                    proc.stdin.close()  # type: ignore[union-attr]
                except OSError:
                    pass
        threading.Thread(target=feed, daemon=True).start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        for hook in prepared.on_timeout:
            try:
                hook()
            except Exception:  # pragma: no cover - best effort
                log.exception("sandbox timeout hook failed")
        _killpg(pgid, _signal.SIGTERM)
        try:
            proc.wait(timeout=kill_grace_s)
        except subprocess.TimeoutExpired:
            _killpg(pgid, _signal.SIGKILL)
            proc.wait()
    _killpg(pgid, _signal.SIGKILL)  # reap anything left in the process group
    for t in threads:
        t.join(timeout=5)
    for s in (proc.stdout, proc.stderr):
        try:
            s.close()  # type: ignore[union-attr]
        except OSError:
            pass
    return proc.returncode, out, err, timed_out, time.monotonic() - t0


class BaseRunner:
    """Common policy plumbing. Subclasses implement :meth:`_wrap` (and usually :meth:`available`)."""

    name = "base"
    isolation = "none"            # none | process | process+netns | os | container | remote
    weak = True                   # True: NOT a security boundary (filesystem not OS-confined)

    def __init__(
        self,
        policy: SandboxPolicy,
        workspace: str | os.PathLike[str],
        *,
        admin: bool = False,
        proxy: EgressProxy | None = None,
        session_tmp: str | os.PathLike[str] | None = None,
        home: str | None = None,
        host_env: Mapping[str, str] | None = None,
        event_sink: EventSink | None = None,
        session_id: str = "sandbox",
        python: str | None = None,
    ) -> None:
        policy.validate(admin=admin)
        self.policy = policy
        self.workspace = os.path.realpath(os.fspath(workspace))
        if not os.path.isdir(self.workspace):
            raise PolicyError(f"workspace does not exist: {self.workspace}")
        self._own_tmp = session_tmp is None and policy.filesystem.session_tmp
        self.session_tmp = os.fspath(session_tmp) if session_tmp else None
        self.home = home
        self.host_env = host_env
        self.event_sink = event_sink
        self.session_id = session_id
        self.python = python or sys.executable
        self._proxy = proxy
        self._own_proxy = False
        self._compiled: CompiledPolicy | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ lifecycle (lazy, per checklist 7/8)
    @classmethod
    def available(cls) -> bool:
        return True

    def _ensure_session_tmp(self) -> None:
        if self.session_tmp is None and self._own_tmp:
            self.session_tmp = tempfile.mkdtemp(prefix="harness-sbx-tmp-")
            os.chmod(self.session_tmp, 0o700)

    @property
    def compiled(self) -> CompiledPolicy:
        with self._lock:
            self._ensure_session_tmp()
            if self._compiled is None:
                self._compiled = self.policy.compile(self.workspace, home=self.home, session_tmp=self.session_tmp)
            return self._compiled

    def _needs_unix_proxy(self) -> bool:
        return False

    @property
    def proxy(self) -> EgressProxy | None:
        if not self.policy.network.mode.uses_proxy:
            return None
        with self._lock:
            if self._proxy is None:
                self._proxy = EgressProxy(self.policy, unix_socket=True if self._needs_unix_proxy() else None,
                                          tcp=not self._needs_unix_proxy(), event_sink=self.event_sink,
                                          session_id=self.session_id)
                self._own_proxy = True
            if not self._proxy.running:
                self._proxy.start()
            return self._proxy

    def close(self) -> None:
        if self._own_proxy and self._proxy is not None:
            self._proxy.stop()
            self._proxy = None
        if self._own_tmp and self.session_tmp and os.path.isdir(self.session_tmp):
            shutil.rmtree(self.session_tmp, ignore_errors=True)
            self.session_tmp = None
            self._compiled = None

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ building blocks
    def launcher_spec(self) -> dict[str, Any]:
        lim = self.policy.limits
        r: dict[str, list[int]] = {}
        if lim.cpu_seconds:
            r["cpu"] = [lim.cpu_seconds, lim.cpu_seconds + 5]   # SIGXCPU first, SIGKILL 5 s later
        if lim.memory_bytes:
            r[lim.memory_limit_kind] = [lim.memory_bytes, lim.memory_bytes]
        if lim.max_processes:
            r["nproc"] = [lim.max_processes, lim.max_processes]
        if lim.file_size_bytes:
            r["fsize"] = [lim.file_size_bytes, lim.file_size_bytes]
        if lim.open_files:
            r["nofile"] = [lim.open_files, lim.open_files]
        if not lim.core_dumps:
            r["core"] = [0, 0]
        hatches = self.policy.escape_hatches
        seccomp = sys.platform.startswith("linux") and _launcher.machine() in _launcher.SYSCALLS and not hatches.disable_seccomp
        return {"rlimits": r, "no_new_privs": sys.platform.startswith("linux"), "seccomp": seccomp,
                "block_namespaces": not hatches.allow_nested_namespaces, "umask": 0o022}

    def launcher_argv(self, spec: Mapping[str, Any], argv: Sequence[str]) -> list[str]:
        import json

        return [self.python, "-I", "-S", LAUNCHER_PATH, "--spec", json.dumps(spec, sort_keys=True, separators=(",", ":")), "--", *argv]

    def sandbox_vars(self, command_id: str, *, proxy_url: str | None) -> dict[str, str]:
        v: dict[str, str] = {"HARNESS_SANDBOX": self.name, "HARNESS_SANDBOX_COMMAND_ID": command_id}
        if proxy_url:
            v.update(proxy_env(proxy_url))
        return v

    @staticmethod
    def normalise_argv(argv: list[str] | str) -> list[str]:
        if isinstance(argv, str):
            shell = "/bin/bash" if os.path.exists("/bin/bash") else "/bin/sh"
            return [shell, "-c", argv]
        if not argv or not all(isinstance(a, str) for a in argv):
            raise PolicyError("argv must be a non-empty list of strings")
        return list(argv)

    def _wrap(self, argv: list[str], *, cwd: str, command_id: str) -> tuple[list[str], str | None, list[str]]:
        """Return ``(full_argv, proxy_url_inside_sandbox, notes)``."""
        raise NotImplementedError

    # ------------------------------------------------------------------ public API
    def prepare(self, argv: list[str] | str, *, cwd: str | None = None, env: Mapping[str, str] | None = None,
                command_id: str | None = None) -> PreparedCommand:
        command_id = command_id or new_id("cmd")
        args = self.normalise_argv(argv)
        cwd_abs = canonicalize(cwd or self.workspace, self.workspace)
        decision = self.compiled.check_read(cwd_abs)
        if not decision.allowed or not os.path.isdir(cwd_abs):
            raise _CwdRefused(Violation("filesystem", f"chdir {cwd_abs}", decision.rule if not decision.allowed else "cwd-missing", command_id))
        full, proxy_url, notes = self._wrap(args, cwd=cwd_abs, command_id=command_id)
        tmpdir = self._tmpdir_inside()
        svars = self.sandbox_vars(command_id, proxy_url=proxy_url)
        if tmpdir:
            svars["TMPDIR"] = tmpdir
        built = build_env(self.policy, host_env=self.host_env, extra=env, sandbox_vars=svars)
        return PreparedCommand(full, built.env, cwd_abs, command_id, built.scrubbed, notes)

    def _tmpdir_inside(self) -> str | None:
        return self.session_tmp

    def run(self, argv: list[str] | str, *, cwd: str | None = None, env: dict[str, str] | None = None,
            timeout: float | None = None, stdin: str | None = None, command_id: str | None = None) -> CommandResult:
        limits = self.policy.limits
        eff_timeout = limits.effective_timeout(timeout)
        try:
            prepared = self.prepare(argv, cwd=cwd, env=env, command_id=command_id)
        except _CwdRefused as refused:
            v = refused.violation
            result = CommandResult(126, "", f"harness-sandbox: working directory refused ({v.rule})\n", command_id=v.command_id,
                                   sandbox=self.name, violations=(v,))
            self._emit(result)
            return result
        try:
            rc, out, err, timed_out, duration = spawn_and_capture(
                prepared, stdin=stdin, timeout=eff_timeout, max_output_bytes=limits.max_output_bytes, kill_grace_s=limits.kill_grace_s)
        finally:
            prepared.run_cleanup()
        sig = -rc if rc < 0 else None
        exit_code = 128 + sig if sig else rc
        stdout, stderr = out.text(), err.text()
        violations = self._violations(prepared.command_id, exit_code, sig, stderr, timed_out, eff_timeout)
        if self.policy.redact_output:
            stdout, stderr = _REDACTOR.redact(stdout), _REDACTOR.redact(stderr)
        result = CommandResult(exit_code, stdout, stderr, timed_out, out.truncated or err.truncated, round(duration, 6), sig,
                               prepared.command_id, self.name, tuple(violations), out.total, err.total, prepared.scrubbed_env)
        self._emit(result)
        return result

    # ------------------------------------------------------------------ feedback & events
    def _violations(self, command_id: str, exit_code: int, sig: int | None, stderr: str, timed_out: bool, timeout: float) -> list[Violation]:
        out: list[Violation] = []
        if timed_out:
            out.append(Violation("timeout", f"wall clock > {timeout:g}s", "limits.max_timeout_s", command_id))
        elif sig == _signal.SIGXCPU:
            out.append(Violation("resource", "CPU time limit exceeded", "limits.cpu_seconds", command_id))
        elif sig == _signal.SIGXFSZ or "File too large" in stderr:
            out.append(Violation("resource", "file size limit exceeded", "limits.file_size_bytes", command_id))
        elif "MemoryError" in stderr or "Cannot allocate memory" in stderr:
            out.append(Violation("resource", "memory limit exceeded", "limits.memory_bytes", command_id))
        proxy = self._proxy
        if proxy is not None:
            for ev in proxy.events(command_id):
                if ev.decision == "deny":
                    out.append(Violation("network", f"{ev.method} {ev.host}:{ev.port}", ev.rule, command_id))
        if exit_code != 0:
            out.extend(detect_fs_violations(stderr, command_id))
        return out

    def _emit(self, r: CommandResult) -> None:
        if self.event_sink is None:
            return
        data = {"kind": "sandbox.exec", "sandbox": self.name, "isolation": self.isolation, "weak": self.weak,
                "command_id": r.command_id, "exit_code": r.exit_code, "timed_out": r.timed_out, "duration_s": r.duration_s,
                "truncated": r.truncated, "violations": [v.to_dict() for v in r.violations], "policy": self.policy.fingerprint()}
        try:
            self.event_sink.emit(Event(self.session_id, EventType.CUSTOM, data))
        except Exception:
            log.exception("sandbox event sink failed")


class _CwdRefused(Exception):
    def __init__(self, violation: Violation) -> None:
        super().__init__(violation.operation)
        self.violation = violation
