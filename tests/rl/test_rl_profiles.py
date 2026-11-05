from __future__ import annotations

import pytest
import yaml

from harness_rl import (
    HarnessProfile,
    ProfileAmbiguityError,
    ProfileError,
    ProfileMismatchMonitor,
    ProfileNotFoundError,
    ProfileRegistry,
    parse_profile,
)

CONFIG = {
    "fallback": "generic",
    "profiles": [
        {"name": "patch-family", "family": "family-p", "match": ["family-p-*"], "edit_format": "apply_patch_v4a", "edit_tool": "apply_patch",
         "tools": [{"name": "apply_patch", "description": "V4A patch"}, "exec_command", "update_plan"], "reasoning_replay": "openai_encrypted_reasoning",
         "tool_result_role": "tool", "compaction": {"mode": "client", "trigger_ratio": 0.9}, "scaffolds": {"todo": False, "planner": True}},
        {"name": "edit-family", "family": "family-e", "match": ["family-e-*"], "edit_format": "str_replace", "edit_tool": "edit",
         "tools": ["read", "edit", "write", "bash", "glob", "grep"], "reasoning_replay": "anthropic_thinking",
         "tool_aliases": {"apply_patch": "edit"}, "scaffolds": {"todo": True, "think": True, "evaluator_gate": True}},
        {"name": "edit-family-large", "match": ["family-e-large-*"], "edit_tool": "edit", "tools": ["edit", "bash"]},
        {"name": "generic", "match": ["__never__"], "edit_tool": "write", "tools": ["write", "bash"]},
    ],
}


@pytest.fixture
def reg(tmp_path):
    p = tmp_path / "profiles.yaml"
    p.write_text(yaml.safe_dump(CONFIG))
    return ProfileRegistry.from_file(p)


def test_select_by_glob_most_specific_wins(reg):
    assert reg.select("family-p-2-mini").profile.name == "patch-family"
    sel = reg.select("family-e-large-7")
    assert sel.profile.name == "edit-family-large" and sel.matched_pattern == "family-e-large-*" and not sel.fallback


def test_no_silent_generic_fallback(reg):
    with pytest.raises(ProfileNotFoundError):
        reg.select("unknown-model")
    sel = reg.select("unknown-model", allow_fallback=True)
    assert sel.fallback and sel.profile.name == "generic"


def test_ambiguous_profiles_rejected():
    r = ProfileRegistry.from_mapping({"profiles": [
        {"name": "a", "match": ["m-*"], "edit_tool": "e", "tools": ["e"]}, {"name": "b", "match": ["m-*"], "edit_tool": "e", "tools": ["e"]}]})
    with pytest.raises(ProfileAmbiguityError):
        r.select("m-1")


@pytest.mark.parametrize("patch", [
    {"tool_result_role": "user_text"},           # tool results as user messages break native training
    {"edit_tool": "apply_patch", "tools": ["bash"]},
    {"match": []},
    {"reasoning_replay": "made_up"},
    {"scaffolds": {"todo": "yes"}},
    {"compaction": {"trigger_ratio": 2}},
])
def test_profile_invariants(patch):
    base = {"name": "x", "match": ["x-*"], "edit_tool": "edit", "tools": ["edit"]}
    with pytest.raises(ProfileError):
        parse_profile(base | patch)


def test_mismatch_monitor_gives_actionable_error(reg):
    mon = ProfileMismatchMonitor(reg.get("edit-family"))
    assert mon.check("edit") is None
    out = mon.check("apply_patch")
    assert out.is_error and "Use 'edit' instead" in out.content and out.metadata["profile_mismatch"]
    mon.check("apply_patch")
    assert mon.snapshot() == {"apply_patch": 2} and mon.total == 2


def test_ablations_and_hash(reg):
    p: HarnessProfile = reg.get("edit-family")
    abl = dict(p.ablations())
    assert set(abl) == {"todo", "think", "evaluator_gate"}
    assert abl["todo"].scaffold("todo") is False and abl["todo"].scaffold("think") is True
    assert abl["todo"].content_hash() != p.content_hash() == reg.get("edit-family").content_hash()
    assert [t.name for t in reg.get("patch-family").tools][0] == "apply_patch"
