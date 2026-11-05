"""Execution environments: one interface over local, container and remote sandboxes.

Research basis (§6.2 Managed Agents; §6.3 techniques 11, 12, 17; checklist §6.7 items 7, 8):

* **Session, harness and sandbox are separate.** The environment exposes ``provision`` /
  ``execute`` / ``teardown`` (Managed Agents ``provision({resources})``, ``execute(name, input)``);
  the session log and the agent loop live outside it, so an agent that runs ``pkill -9 bash`` inside
  the box cannot kill its own harness.
* **Lazy provisioning.** Nothing is created until the first ``execute`` (p50 TTFT about -60%,
  p95 more than -90% in Managed Agents).
* **Sandbox death is a tool error.** A crashed or unreachable sandbox comes back as a
  :class:`~harness_sandbox.runners.base.CommandResult` with ``exit_code`` 125 and a
  ``sandbox_error`` violation the model can react to; the next call provisions a fresh box
  ("cattle, not pets").
* **Idempotent side effects.** :func:`idempotency_key` is ``hash(session_id, step_idx, tool_call_id)``
  so retried side-effecting tools are not applied twice.
* **Warm pools recycle only within a tenant and image** (:class:`EnvironmentPool`): recycling is a
  throughput optimisation, never a security feature.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import defaultdict
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from harness_core.types import new_id

from .errors import SandboxCrashedError, SandboxError
from .runners.base import BaseRunner, CommandResult
from .violations import Violation

log = logging.getLogger("harness_sandbox")

SANDBOX_ERROR_EXIT = 125


def idempotency_key(session_id: str, step_idx: int, tool_call_id: str) -> str:
    return hashlib.sha256(f"{session_id}\x1f{step_idx}\x1f{tool_call_id}".encode()).hexdigest()[:32]


def sandbox_error_result(env_id: str, exc: BaseException, *, command_id: str = "") -> CommandResult:
    v = Violation("policy", f"sandbox {env_id} failed: {type(exc).__name__}", "sandbox_error", command_id,
                  "The sandbox was lost or could not start. Retrying provisions a fresh sandbox; workspace changes not yet "
                  "persisted may be gone.")
    return CommandResult(SANDBOX_ERROR_EXIT, "", f"sandbox_error: {exc}\n", command_id=command_id, sandbox=env_id, violations=(v,))


@runtime_checkable
class ExecutionEnvironment(Protocol):
    id: str
    kind: str

    def provision(self) -> None: ...

    def execute(self, argv: list[str] | str, *, cwd: str | None = None, env: dict[str, str] | None = None,
                timeout: float | None = None, stdin: str | None = None) -> CommandResult: ...

    def teardown(self) -> None: ...

    @property
    def provisioned(self) -> bool: ...


class RunnerEnvironment:
    """Local, OS-sandboxed or container environment backed by a :class:`BaseRunner` factory."""

    def __init__(self, factory: Callable[[], BaseRunner], *, kind: str = "local", env_id: str | None = None) -> None:
        self.factory = factory
        self.kind = kind
        self.id = env_id or new_id("sbx")
        self._runner: BaseRunner | None = None
        self._lock = threading.Lock()
        self.provision_count = 0

    @property
    def provisioned(self) -> bool:
        return self._runner is not None

    @property
    def runner(self) -> BaseRunner | None:
        return self._runner

    def provision(self) -> None:
        with self._lock:
            if self._runner is None:
                self._runner = self.factory()
                self.provision_count += 1

    def execute(self, argv: list[str] | str, *, cwd: str | None = None, env: dict[str, str] | None = None,
                timeout: float | None = None, stdin: str | None = None) -> CommandResult:
        try:
            self.provision()
            assert self._runner is not None
            return self._runner.run(argv, cwd=cwd, env=env, timeout=timeout, stdin=stdin)
        except (SandboxError, OSError) as exc:
            log.warning("sandbox %s failed: %s", self.id, exc)
            self.teardown()
            return sandbox_error_result(self.id, exc)

    def teardown(self) -> None:
        with self._lock:
            if self._runner is not None:
                try:
                    self._runner.close()
                finally:
                    self._runner = None


@runtime_checkable
class RemoteBackend(Protocol):
    """Adapter for a hosted sandbox (Firecracker/E2B, gVisor/Modal, Daytona, k8s agent-sandbox...)."""

    def provision(self, resources: Mapping[str, Any]) -> str: ...

    def execute(self, sandbox_id: str, argv: list[str] | str, *, cwd: str | None, env: dict[str, str] | None,
                timeout: float | None, stdin: str | None) -> CommandResult: ...

    def destroy(self, sandbox_id: str) -> None: ...


class RemoteEnvironment:
    """Lazily provisioned remote sandbox; crashes become tool errors and the next call re-provisions."""

    kind = "remote"

    def __init__(self, backend: RemoteBackend, *, resources: Mapping[str, Any] | None = None, env_id: str | None = None) -> None:
        self.backend = backend
        self.resources = dict(resources or {})
        self.id = env_id or new_id("sbx")
        self.sandbox_id: str | None = None
        self._lock = threading.Lock()
        self.provision_count = 0

    @property
    def provisioned(self) -> bool:
        return self.sandbox_id is not None

    def provision(self) -> None:
        with self._lock:
            if self.sandbox_id is None:
                self.sandbox_id = self.backend.provision(self.resources)
                self.provision_count += 1

    def execute(self, argv: list[str] | str, *, cwd: str | None = None, env: dict[str, str] | None = None,
                timeout: float | None = None, stdin: str | None = None) -> CommandResult:
        try:
            self.provision()
            assert self.sandbox_id is not None
            return self.backend.execute(self.sandbox_id, argv, cwd=cwd, env=env, timeout=timeout, stdin=stdin)
        except (SandboxCrashedError, SandboxError, OSError, ConnectionError) as exc:
            dead, self.sandbox_id = self.sandbox_id, None
            if dead is not None:
                try:
                    self.backend.destroy(dead)
                except Exception:  # pragma: no cover - best effort on a dead box
                    log.debug("destroy of dead sandbox %s failed", dead)
            return sandbox_error_result(self.id, exc)

    def teardown(self) -> None:
        with self._lock:
            if self.sandbox_id is not None:
                sid, self.sandbox_id = self.sandbox_id, None
                self.backend.destroy(sid)


class EnvironmentPool:
    """Warm pool keyed by ``(tenant, image)``. Environments are only ever reused under the same key."""

    def __init__(self, factory: Callable[[str, str], ExecutionEnvironment], *, max_idle_per_key: int = 4) -> None:
        self.factory = factory
        self.max_idle_per_key = max_idle_per_key
        self._idle: dict[tuple[str, str], list[ExecutionEnvironment]] = defaultdict(list)
        self._owner: dict[str, tuple[str, str]] = {}
        self._lock = threading.Lock()

    def acquire(self, tenant: str, image: str) -> ExecutionEnvironment:
        key = (tenant, image)
        with self._lock:
            if self._idle[key]:
                env = self._idle[key].pop()
                self._owner[env.id] = key
                return env
        env = self.factory(tenant, image)
        with self._lock:
            self._owner[env.id] = key
        return env

    def release(self, env: ExecutionEnvironment, *, reset: Callable[[ExecutionEnvironment], bool] | None = None,
                healthy: bool = True) -> bool:
        """Return ``env`` to its own key's pool if healthy and ``reset`` succeeds; else tear it down."""
        with self._lock:
            key = self._owner.pop(env.id, None)
        ok = healthy and key is not None
        if ok and reset is not None:
            try:
                ok = bool(reset(env))
            except Exception:
                ok = False
        with self._lock:
            if ok and key is not None and len(self._idle[key]) < self.max_idle_per_key:
                self._idle[key].append(env)
                return True
        env.teardown()
        return False

    def idle(self, tenant: str, image: str) -> int:
        with self._lock:
            return len(self._idle[(tenant, image)])

    def close(self) -> None:
        with self._lock:
            envs = [e for lst in self._idle.values() for e in lst]
            self._idle.clear()
        for e in envs:
            e.teardown()
