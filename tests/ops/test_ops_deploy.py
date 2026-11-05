"""Rainbow deploys (version pinning, deterministic split, drain/retire/rollback) and per-model flags."""

from __future__ import annotations

import threading

import pytest

from harness_ops import ConfigError, DeployError, DeploymentRegistry, FeatureFlags, SessionAbortedError, VersionState, prompts_tools_sha


@pytest.fixture
def reg(tmp_path, clock):
    r = DeploymentRegistry(tmp_path / "deploy.db", clock=clock)
    r.register("v1", prompts_tools_sha("system v1", [{"name": "read"}]), weight=100)
    r.register("v2", prompts_tools_sha("system v2", [{"name": "read"}, {"name": "edit"}]))
    return r


def test_sessions_finish_on_the_version_they_started_on(reg):
    early = [reg.pin(f"s{i}").version_id for i in range(50)]
    assert set(early) == {"v1"}
    reg.shift("v2", 100)  # full cut-over for NEW sessions
    assert all(reg.pin(f"s{i}").version_id == "v1" for i in range(50))  # in-flight stay pinned
    assert reg.pin("new-session").version_id == "v2"
    assert reg.version("v1").state is VersionState.DRAINING and reg.inflight("v1") == 50
    with pytest.raises(DeployError):
        reg.retire("v1")  # refused while sessions are in flight
    for i in range(50):
        reg.finish(f"s{i}")
    assert reg.is_drained("v1")
    reg.retire("v1")
    with pytest.raises(DeployError):
        reg.set_weights({"v1": 10, "v2": 90})  # retired versions take no traffic
    v2 = reg.version("v2")
    assert v2.span_attributes() == {"harness.version": "v2", "harness.prompts_tools_sha": v2.prompts_tools_sha}


def test_split_is_deterministic_proportional_and_ramps_monotonically(reg, tmp_path, clock):
    reg.shift("v2", 10)
    keys = [f"user-{i}" for i in range(4000)]
    at10 = {k: reg.route(k) for k in keys}
    share = sum(v == "v2" for v in at10.values()) / len(keys)
    assert 0.08 < share < 0.12
    other = DeploymentRegistry(tmp_path / "deploy.db", clock=clock)  # another process, same state
    assert all(other.route(k) == at10[k] for k in keys[:200])
    reg.shift("v2", 50)
    at50 = {k: reg.route(k) for k in keys}
    assert all(at50[k] == "v2" for k in keys if at10[k] == "v2")  # ramping only ADDS sessions to the new version
    assert 0.46 < sum(v == "v2" for v in at50.values()) / len(keys) < 0.54


def test_rollback_and_abort_inflight(reg):
    reg.shift("v2", 100)
    bad = [reg.pin(f"b{i}").version_id for i in range(3)]
    assert bad == ["v2"] * 3
    affected = reg.rollback("v2", "v1")
    assert affected == ["b0", "b1", "b2"] and reg.version("v2").state is VersionState.ROLLED_BACK
    assert reg.pin("b0").version_id == "v2"  # default: in-flight sessions keep their version
    assert reg.pin("fresh").version_id == "v1"
    reg.register("v3", "sha3")
    reg.shift("v3", 100)
    reg.pin("c0")
    reg.rollback("v3", "v1", abort_inflight=True)
    with pytest.raises(SessionAbortedError):
        reg.pin("c0")
    with pytest.raises(DeployError):
        reg.register("v1", "different-sha")  # versions are immutable
    with pytest.raises(DeployError):
        reg.drain("v1")  # the only version taking traffic


def test_concurrent_pins_for_one_session_agree(reg):
    reg.shift("v2", 50)
    got = []
    barrier = threading.Barrier(8)

    def pin():
        barrier.wait()
        got.append(reg.pin("shared-session").version_id)

    ts = [threading.Thread(target=pin) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(set(got)) == 1


def test_feature_flags_per_model_frozen_per_session(reg):
    flags = FeatureFlags.from_dict(
        {
            "flags": {
                "todo_tool": {"default": True, "models": {"test-model-new*": False, "test-model-new-exact": True}},
                "parallel_tools": {"default": True, "rollout_percent": 30},
                "apply_patch": {"default": True, "versions": ["v2"]},
            }
        }
    )
    snap = flags.snapshot("test-model-new-2", routing_key="s1", version="v1")
    assert snap.enabled("todo_tool") is False and snap.enabled("apply_patch") is False
    assert flags.value("todo_tool", "test-model-new-exact") is True  # exact id beats pattern
    assert flags.value("apply_patch", "test-model", version="v2") is True
    on = sum(flags.value("parallel_tools", "test-model", routing_key=f"s{i}") for i in range(2000)) / 2000
    assert 0.26 < on < 0.34
    pin = reg.pin("s1", snapshot=snap.to_dict())
    assert pin.snapshot["values"]["todo_tool"] is False  # frozen with the session
    with pytest.raises(ConfigError):
        FeatureFlags.from_dict({"flags": {"x": {"rollout_percent": 150}}})
    with pytest.raises(KeyError):
        flags.value("nope", "test-model")
