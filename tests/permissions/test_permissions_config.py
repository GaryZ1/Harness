"""harness-policy.toml loader/validator and layer precedence."""

import builtins
import json
import re
from pathlib import Path

import pytest
import yaml

from harness_core.types import PermissionBehavior, RunContext, ToolCall
from harness_permissions import (
    SCHEMA,
    Layer,
    PolicyEngine,
    PolicyLoadError,
    PolicyValidationError,
    Workspace,
    default_policy,
    discover_policy_files,
    load_layers,
    load_policy_file,
    parse_policy,
    resolve,
)
from harness_permissions import config as config_mod

# Verbatim from reports/SOTA agentic harness reference.md §7.7 ("our schema").
RESEARCH_EXAMPLE = '''
# harness-policy.toml (our schema)
[defaults]
interactive_mode = "auto"          # plan | accept-edits | auto | dont-ask | bypass
headless_mode    = "dont-ask"
sandbox          = "workspace-write"
deny_wins        = true

[denials]
escalate_consecutive = 3
escalate_total       = 20
headless_on_limit    = "terminate"

[rules]
allow = ["Read(*)", "Grep(*)", "Glob(*)", "Bash(git status*)", "Bash(npm test*)"]
deny  = ["Bash(git push --force*)", "Edit(.git/hooks/**)", "Read(~/.ssh/**)", "Read(**/.env)"]

[classifier]
stages          = ["single_token_block_biased", "cot_on_flag"]
inputs          = ["user_messages", "tool_calls"]   # never tool_results / assistant text
environment     = ["current_git_repo"]
block_groups    = ["destroy_or_exfiltrate", "degrade_security_posture",
                   "cross_trust_boundary", "bypass_review_or_affect_others"]

[taint]
sources        = ["web_fetch", "browser", "mcp:openWorld", "email", "out_of_repo_files", "subagent_report"]
strict_actions = ["network_write", "message_send", "git_push_non_session_branch",
                  "new_egress_domain", "memory_write"]

[mcp]
pin_versions             = true
hash_tool_descriptions   = true
on_description_change    = "block_and_ask"
trust_annotations_from   = ["vetted"]

[hooks]
on_failure = "block"
timeouts_s = { command = 600, http = 600, mcp_tool = 600, prompt = 30, agent = 60 }
'''


def test_research_example_loads_and_matches_defaults(tmp_path):
    f = tmp_path / "harness-policy.toml"
    f.write_text(RESEARCH_EXAMPLE)
    pol = resolve([load_policy_file(f)])
    assert pol.interactive_mode.value == "auto" and pol.headless_mode.value == "dont-ask"
    assert pol["denials.escalate_consecutive"] == 3 and pol["denials.escalate_total"] == 20
    assert {r.source for r in pol.rules} >= {"Bash(git status*)", "Read(**/.env)"}
    assert pol["hooks.timeouts_s"]["prompt"] == 30
    defaults = default_policy()
    for key in ("defaults.sandbox", "denials.headless_on_limit", "mcp.on_description_change", "hooks.on_failure", "classifier.stages"):
        assert pol[key] == defaults[key], key


def test_yaml_and_json_load_the_same_schema(tmp_path):
    toml_pol = resolve([load_policy_file(_write(tmp_path / "p.toml", RESEARCH_EXAMPLE))])
    data = config_mod._load_toml(RESEARCH_EXAMPLE, Path("x.toml"))
    y = resolve([load_policy_file(_write(tmp_path / "p.yaml", yaml.safe_dump(data)))])
    j = resolve([load_policy_file(_write(tmp_path / "p.json", json.dumps(data)))])
    for key in SCHEMA:
        assert toml_pol[key] == y[key] == j[key], key
    with pytest.raises(PolicyLoadError):
        load_policy_file(_write(tmp_path / "p.ini", "x=1"))
    with pytest.raises(PolicyLoadError):
        load_policy_file(_write(tmp_path / "bad.toml", "[defaults\n"))


def _write(path, text):
    path.write_text(text)
    return path


def test_missing_toml_parser_gives_a_clear_error(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name in {"tomllib", "tomli"}:
            raise ModuleNotFoundError(name)
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(PolicyLoadError, match="tomli.*yaml or .json"):
        load_policy_file(_write(tmp_path / "p.toml", RESEARCH_EXAMPLE))


def test_validation_collects_every_problem():
    with pytest.raises(PolicyValidationError) as exc:
        parse_policy({
            "defaults": {"deny_wins": False, "interactive_mode": "turbo", "colour": "red"},
            "rules": {"allow": ["Bash(*)", "Bash(python *)", "Bash(npm run *)"]},
            "classifier": {"inputs": ["user_messages", "tool_results"]},
            "denials": {"escalate_total": 0},
            "exec_policy": {"prefix_rules": [
                {"pattern": ["git", "status"], "decision": "allow", "justification": "read-only", "match": ["git push"]},
                {"pattern": ["rm"], "decision": "forbidden", "justification": ""},
            ]},
            "nonsense": {},
        })
    problems = "\n".join(exc.value.problems)
    for needle in ["deny_wins", "turbo", "colour", "Bash(*)", "Bash(python *)", "Bash(npm run *)", "tool_results", "escalate_total",
                   "should match", "justification is required", "nonsense"]:
        assert needle in problems, needle
    assert len(exc.value.problems) >= 11


def test_layer_precedence_and_narrow_only_merging():
    managed = parse_policy({"defaults": {"interactive_mode": "default"}, "denials": {"escalate_consecutive": 5},
                            "rules": {"deny": ["Bash(curl *)"]}}, Layer.MANAGED)
    project = parse_policy({"defaults": {"interactive_mode": "auto"}, "denials": {"escalate_consecutive": 2},
                            "network": {"block_private": False}, "rules": {"allow": ["Bash(curl https://api.example/*)"]}}, Layer.PROJECT)
    user = parse_policy({"defaults": {"interactive_mode": "plan"}}, Layer.USER)
    pol = resolve([user, project, managed])
    assert pol.interactive_mode.value == "plan"              # least permissive wins
    assert pol["denials.escalate_consecutive"] == 2          # min
    assert pol["network.block_private"] is True              # only managed may relax a default-on protection
    assert [r.layer for r in pol.rules][:1] == ["managed"]   # sorted by authority
    relaxed = resolve([parse_policy({"network": {"block_private": False}}, Layer.MANAGED)])
    assert relaxed["network.block_private"] is False


def test_plugins_and_managed_only_allows():
    with pytest.raises(PolicyValidationError, match="plugin policy may only add restrictions"):
        parse_policy({"rules": {"allow": ["Bash(make test)"]}}, Layer.PLUGIN)
    with pytest.raises(PolicyValidationError, match="cannot add allow exec rules"):
        parse_policy({"exec_policy": {"prefix_rules": [{"pattern": ["make"], "decision": "allow", "justification": "x"}]}}, Layer.PLUGIN)
    plugin = parse_policy({"rules": {"deny": ["Bash(make deploy*)"]}}, Layer.PLUGIN)
    managed = parse_policy({"defaults": {"allow_managed_rules_only": True}, "rules": {"allow": ["Bash(make lint)"]}}, Layer.MANAGED)
    project = parse_policy({"rules": {"allow": ["Bash(make test)"]}}, Layer.PROJECT)
    pol = resolve([plugin, managed, project])
    assert {r.source for r in pol.rules} == {"Bash(make deploy*)", "Bash(make lint)"}


def test_loaded_policy_files_are_protected(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    f = _write(repo / "custom-policy.yaml", "rules:\n  allow: ['Edit(**)']\n")
    pol = load_layers(project=[f])
    eng = PolicyEngine(pol, workspace=Workspace(roots=[str(repo)], temp_roots=[]))
    ctx = RunContext("s", cwd=str(repo), metadata={"permission_mode": "accept-edits"})
    assert eng.check(ToolCall("1", "Edit", {"file_path": "custom-policy.yaml"}), None, ctx).behavior is PermissionBehavior.DENY
    assert eng.check(ToolCall("2", "Bash", {"command": "echo '{}' > custom-policy.yaml"}), None, ctx).behavior is PermissionBehavior.DENY
    assert eng.check(ToolCall("3", "Edit", {"file_path": "src/x.py"}), None, ctx).behavior is PermissionBehavior.ALLOW


def test_relative_workspace_roots_resolve_against_policy_file(tmp_path, monkeypatch):
    monkeypatch.delenv("HARNESS_MANAGED_POLICY", raising=False)
    (tmp_path / "proj" / ".harness").mkdir(parents=True)
    f = _write(tmp_path / "proj" / ".harness" / "harness-policy.toml", '[workspace]\nroots = [".."]\n')
    assert load_policy_file(f).values["workspace.roots"] == [str((tmp_path / "proj").resolve())]
    found = discover_policy_files(tmp_path / "proj")
    assert found[Layer.PROJECT] == [f]


def test_docs_reference_every_schema_key():
    doc = (Path(__file__).resolve().parents[2] / "docs" / "permissions" / "policy-file.md").read_text()
    documented = set(re.findall(r"`([a-z_]+\.[a-z_]+)`", doc)) | ({"version"} if "`version`" in doc else set())
    missing = [k for k in SCHEMA if k not in documented]
    assert not missing, missing
