"""Rule parsing, linting and the deny-fuzzy / allow-exact asymmetry."""

import pytest

from harness_permissions import PolicyValidationError, Rule, RuleEffect, Workspace, is_blanket, lint_rule
from harness_permissions.rules import match_allow_text, match_command
from harness_permissions.shell import analyze


def cmd(text):
    return analyze(text).commands[0]


def deny(spec):
    return Rule.parse(f"Bash({spec})", RuleEffect.DENY)


def allow(spec):
    return Rule.parse(f"Bash({spec})", RuleEffect.ALLOW)


def test_parse_forms():
    r = Rule.parse("Bash(git push:*)", "deny", layer="managed")
    assert (r.tool, r.specifier, r.effect, r.layer) == ("Bash", "git push:*", RuleEffect.DENY, "managed")
    assert Rule.parse("mcp__github", "allow").specifier is None
    assert Rule.parse("WebFetch(domain:example.com)", "allow").specifier == "domain:example.com"
    for bad in ["", "Bash(", "Bash()", "Ba sh(x)", "(x)"]:
        with pytest.raises(PolicyValidationError):
            Rule.parse(bad, "allow")


@pytest.mark.parametrize(
    "text",
    ["Bash", "Bash(*)", "Bash(python *)", "Bash(python3*)", "Bash(node *)", "Bash(npm run *)", "Bash(npx *)", "Bash(uv run *)",
     "Bash(yarn *)", "Bash(sudo *)", "Bash(env *)", "Bash(xargs *)", "Bash(bash *)", "Bash(* --help)", "Bash(/usr/bin/python *)",
     "mcp__*", "*", "WebFetch", "WebFetch(domain:*)"],
)
def test_lint_rejects_blanket_allow_rules(text):
    assert lint_rule(Rule.parse(text, "allow"))
    assert is_blanket(Rule.parse(text, "allow"))


@pytest.mark.parametrize("text", ["Bash(npm test*)", "Bash(git status*)", "Bash(python scripts/check.py)", "Bash(make test)", "Read(src/**)", "mcp__github__get_*", "WebFetch(domain:docs.python.org)"])
def test_lint_accepts_narrow_rules(text):
    assert lint_rule(Rule.parse(text, "allow")) == []


def test_lint_only_applies_to_allow():
    assert lint_rule(Rule.parse("Bash(*)", "deny")) == []


def test_allow_matching_is_exact_with_word_boundaries():
    assert match_command(allow("ls *"), cmd("ls -la"), fuzzy=False)
    assert match_command(allow("ls *"), cmd("ls"), fuzzy=False)
    assert not match_command(allow("ls *"), cmd("lsof"), fuzzy=False)
    assert match_command(allow("ls*"), cmd("lsof"), fuzzy=False)
    assert match_command(allow("git push:*"), cmd("git push origin x"), fuzzy=False)
    assert not match_command(allow("git status*"), cmd("GIT status"), fuzzy=False)
    assert not match_allow_text(allow("git status*"), "sudo git status")


@pytest.mark.parametrize(
    "command",
    ["rm -rf /tmp/x", "rm -fr /tmp/x", "rm -r -f /tmp/x", "rm --recursive --force /tmp/x", "/bin/rm -rf x", "RM -RF x", "sudo rm -rf x", "ｒｍ -rf x"],
)
def test_deny_matching_is_fuzzy(command):
    assert match_command(deny("rm -rf*"), cmd(command), fuzzy=True)


@pytest.mark.parametrize("command", ["git push --force", "git push origin main --force", "git push -f origin main", "git push origin --force-with-lease"])
def test_deny_matches_flags_in_any_order(command):
    assert match_command(deny("git push --force*"), cmd(command), fuzzy=True) or match_command(deny("git push -f*"), cmd(command), fuzzy=True)


def test_path_rules_and_unicode_case_tricks(tmp_path):
    ws = Workspace(roots=[str(tmp_path)], temp_roots=[])
    r = Rule.parse("Read(**/.env)", "deny")
    pat = ws.pattern(r.specifier)
    assert pat.matches(str(tmp_path / "a" / ".env"))
    assert pat.matches(str(tmp_path / ".ENV"), fuzzy=True)
    assert pat.matches(str(tmp_path / ".e​nv"), fuzzy=True)
    assert pat.matches(str(tmp_path / ".ｅnv"), fuzzy=True)
    assert not pat.matches(str(tmp_path / ".ENV"))  # allow side stays exact
    src = ws.pattern("src/**")
    assert src.matches(str(tmp_path / "src")) and src.matches(str(tmp_path / "src" / "a" / "b.py"))
    assert not src.matches(str(tmp_path / "srcx" / "b.py"))
