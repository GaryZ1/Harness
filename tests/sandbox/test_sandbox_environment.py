"""Environments (lazy, crash-as-tool-error, pools), backend detection/choice and model-facing text."""

from __future__ import annotations

import signal

import pytest

from harness_sandbox import (
    BackendReport,
    CommandResult,
    EnvironmentPool,
    LocalRunner,
    NetworkMode,
    NetworkPolicy,
    RemoteEnvironment,
    RunnerEnvironment,
    SandboxCrashedError,
    SandboxMode,
    SandboxPolicy,
    SandboxUnavailableError,
    Violation,
    choose_backend,
    create_runner,
    describe_sandbox,
    format_violations,
    idempotency_key,
    sandbox_state_meta,
)
from harness_sandbox.detect import container_hint, wsl_version


@pytest.fixture(autouse=True)
def _deadline():
    signal.signal(signal.SIGALRM, lambda *_: pytest.fail("test exceeded 60s deadline"))
    signal.alarm(60)
    yield
    signal.alarm(0)


def report(**kw):
    base = dict(platform="linux", arch="x86_64")
    base.update(kw)
    return BackendReport(**base)


def test_runner_environment_is_lazy_and_reprovisions_after_failure(tmp_path):
    made = []

    def factory():
        made.append(1)
        return LocalRunner(SandboxPolicy(), tmp_path, network_namespace=False)

    env = RunnerEnvironment(factory)
    assert not env.provisioned and made == []  # nothing created before the first tool call
    assert env.execute(["echo", "x"], timeout=20).stdout == "x\n"
    assert env.provisioned and env.provision_count == 1
    env.execute(["true"], timeout=20)
    assert env.provision_count == 1
    env.runner.python = "/nonexistent/python"  # simulate a dead sandbox wrapper
    res = env.execute(["echo", "y"], timeout=20)
    assert res.exit_code == 125 and res.violations[0].rule == "sandbox_error" and not env.provisioned
    assert env.execute(["echo", "z"], timeout=20).stdout == "z\n" and env.provision_count == 2
    env.teardown()


class FakeRemote:
    def __init__(self):
        self.provisioned, self.destroyed, self.crash_next = [], [], False

    def provision(self, resources):
        sid = f"vm{len(self.provisioned)}"
        self.provisioned.append((sid, dict(resources)))
        return sid

    def execute(self, sandbox_id, argv, *, cwd, env, timeout, stdin):
        if self.crash_next:
            self.crash_next = False
            raise SandboxCrashedError(f"{sandbox_id} OOM-killed")
        return CommandResult(0, f"{sandbox_id}:{argv}", "", sandbox=sandbox_id)

    def destroy(self, sandbox_id):
        self.destroyed.append(sandbox_id)


def test_remote_environment_crash_is_a_tool_error_then_fresh_box():
    be = FakeRemote()
    env = RemoteEnvironment(be, resources={"image": "py312", "cpu": 2})
    assert be.provisioned == []
    assert env.execute("ls").stdout == "vm0:ls"
    be.crash_next = True
    err = env.execute("make")
    assert err.exit_code == 125 and "OOM-killed" in err.stderr and be.destroyed == ["vm0"]
    assert "sandbox_error" in format_violations(err.violations)
    assert env.execute("ls").stdout == "vm1:ls" and be.provisioned[1][1] == {"image": "py312", "cpu": 2}
    env.teardown()
    assert be.destroyed == ["vm0", "vm1"]


def test_pool_never_recycles_across_tenants():
    created = []

    class Env:
        def __init__(self, tenant, image):
            self.id, self.kind, self.torn = f"{tenant}-{image}-{len(created)}", "fake", False
            created.append(self)

        def teardown(self):
            self.torn = True

    pool = EnvironmentPool(lambda t, i: Env(t, i), max_idle_per_key=1)
    a = pool.acquire("tenantA", "img")
    assert pool.release(a, reset=lambda e: True)
    assert pool.acquire("tenantB", "img") is not a          # other tenant: fresh env
    again = pool.acquire("tenantA", "img")
    assert again is a                                        # same tenant + image: recycled
    assert not pool.release(again, reset=lambda e: False) and again.torn  # failed reset: destroyed
    pool.close()


def test_idempotency_key_is_stable_and_distinct():
    k = idempotency_key("s1", 3, "toolu_1")
    assert k == idempotency_key("s1", 3, "toolu_1") and len(k) == 32
    assert k != idempotency_key("s1", 4, "toolu_1") and k != idempotency_key("s13", 3, "toolu_1"[1:])


def test_choose_backend_matrix():
    pol = SandboxPolicy()
    c = choose_backend(pol, report(bwrap="/usr/bin/bwrap", bwrap_usable=True))
    assert (c.backend, c.isolated, c.auto_allow_in_boundary) == ("bubblewrap", True, True)
    assert choose_backend(pol, report(bwrap="/usr/bin/bwrap", bwrap_usable=True, wsl=1)).backend == "local"  # WSL1 refused
    mac = choose_backend(pol, report(platform="darwin", sandbox_exec="/usr/bin/sandbox-exec"))
    assert mac.backend == "seatbelt" and mac.isolated
    inside = choose_backend(pol, report(container="docker"))
    assert inside.mode is SandboxMode.EXTERNAL_SANDBOX and inside.isolated and inside.boundary == "external"
    weak = choose_backend(pol, report())
    assert weak.backend == "local" and not weak.isolated and not weak.auto_allow_in_boundary and weak.warnings
    with pytest.raises(SandboxUnavailableError):
        choose_backend(pol, report(), require_isolation=True)
    full = choose_backend(SandboxPolicy(mode=SandboxMode.FULL_ACCESS), report(bwrap_usable=True))
    assert full.boundary == "none" and not full.auto_allow_in_boundary


def test_create_runner_follows_choice(tmp_path):
    r = create_runner(SandboxPolicy(), str(tmp_path), report=report(container="podman"), network_namespace=False)
    try:
        assert isinstance(r, LocalRunner) and r.policy.mode is SandboxMode.EXTERNAL_SANDBOX and r.choice.isolated
    finally:
        r.close()


def test_platform_detection_helpers(tmp_path):
    assert wsl_version("Linux version 4.4.0-19041-Microsoft") == 1
    assert wsl_version("Linux version 5.15.90.1-microsoft-standard-WSL2") == 2
    assert wsl_version("Linux version 6.8.0-generic") is None
    (tmp_path / ".dockerenv").write_text("")
    assert container_hint(str(tmp_path)) == "docker"


def test_model_facing_description_is_stable_and_honest(tmp_path):
    pol = SandboxPolicy(network=NetworkPolicy(mode=NetworkMode.ALLOWLIST, allowed_domains=("pypi.org", "docs.python.org")))
    a = describe_sandbox(pol, workspace="/w")
    assert a == describe_sandbox(pol, workspace="/w")  # cache-stable
    assert "docs.python.org, pypi.org" in a and "<sandbox_violations>" in a and "/w" in a
    weak = describe_sandbox(pol, workspace="/w", choice=choose_backend(pol, report()))
    assert "advisory" in weak
    meta = sandbox_state_meta(pol)["harness/sandbox-state"]
    assert meta["mode"] == "workspace-write" and meta["network"] == "allowlist" and meta["policy_fingerprint"] == pol.fingerprint()


def test_violation_block_escapes_markup():
    block = format_violations([Violation("network", "GET <evil>&", 'rule"x', "cmd_1")])
    assert block.startswith("<sandbox_violations>") and "&lt;evil&gt;&amp;" in block and "rule&quot;x" in block
    assert format_violations([]) == ""
