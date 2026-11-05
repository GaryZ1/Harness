from __future__ import annotations

import pytest

from harness import HarnessConfig, HarnessConfigError

SAMPLE = """
model: ${HARNESS_MODEL}
effort: high
workspace: ./ws
base_dir: ./base
user_id: alice
context_window_tokens: 200000
permissions:
  mode: accept-edits
  interactive: false
  policy_files: [policy.yaml]
sandbox:
  mode: workspace-write
budgets:
  max_turns: 40
features:
  memory: true
  subagents: true
  tool_search: false
  mcp_servers:
    - {name: tracker, command: tracker-mcp, args: [--stdio]}
model_roles:
  worker: ${HARNESS_WORKER_MODEL:-test-model-worker}
"""


def test_model_is_required_and_never_defaulted(tmp_path):
    with pytest.raises(HarnessConfigError, match="model is required"):
        HarnessConfig.from_dict({"workspace": str(tmp_path)})
    with pytest.raises(HarnessConfigError, match="model is required"):
        HarnessConfig.from_yaml_text("model: ${HARNESS_MODEL}\n", env={})
    with pytest.raises(HarnessConfigError):
        HarnessConfig(model="")


def test_yaml_loading_env_expansion_and_relative_paths(tmp_path):
    path = tmp_path / "harness.yaml"
    path.write_text(SAMPLE)
    cfg = HarnessConfig.from_yaml(path, env={"HARNESS_MODEL": "test-model"})
    assert cfg.model == "test-model" and cfg.effort == "high" and cfg.user_id == "alice"
    assert cfg.workspace == str((tmp_path / "ws").resolve())
    assert cfg.base_dir == str((tmp_path / "base").resolve())
    assert cfg.permissions.policy_files == [str((tmp_path / "policy.yaml").resolve())]
    assert cfg.permissions.mode == "accept-edits" and cfg.permissions.interactive is False
    assert cfg.budgets.max_turns == 40 and cfg.features.subagents is True
    assert cfg.features.mcp_servers[0].name == "tracker" and cfg.features.mcp_servers[0].args == ["--stdio"]
    assert cfg.model_for_role("worker") == "test-model-worker"
    assert cfg.state_path == cfg.base_path / "harness" and cfg.telemetry_path == cfg.state_path / "telemetry"


def test_model_roles_fall_back_to_the_main_model():
    cfg = HarnessConfig.from_yaml_text("model: test-model\nmodel_roles: {reviewer: test-model-reviewer}\n")
    assert cfg.model_for_role("reviewer") == "test-model-reviewer"
    assert cfg.model_for_role("worker") == "test-model"


def test_unknown_keys_and_bad_values_are_rejected():
    with pytest.raises(HarnessConfigError, match="unknown config keys: permisions"):
        HarnessConfig.from_dict({"model": "test-model", "permisions": {}})
    with pytest.raises(HarnessConfigError, match="unknown config keys: sandbox.policy"):
        HarnessConfig.from_dict({"model": "test-model", "sandbox": {"policy": "x"}})
    with pytest.raises(HarnessConfigError, match="effort"):
        HarnessConfig.from_dict({"model": "test-model", "effort": "extreme"})
    with pytest.raises(HarnessConfigError, match="permissions.mode"):
        HarnessConfig.from_dict({"model": "test-model", "permissions": {"mode": "yolo"}})
