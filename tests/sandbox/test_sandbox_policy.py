"""Policy evaluation: modes, srt precedence, protected subpaths, loading, escape hatches, env scrubbing."""

from __future__ import annotations

import json
import os
import signal

import pytest

from harness_sandbox import (
    EnvPolicy,
    EscapeHatchError,
    EscapeHatches,
    FilesystemPolicy,
    NetworkMode,
    NetworkPolicy,
    PolicyError,
    ResourceLimits,
    SandboxMode,
    SandboxPolicy,
    build_env,
    domain_matches,
    load_policy,
)


@pytest.fixture(autouse=True)
def _deadline():
    signal.signal(signal.SIGALRM, lambda *_: pytest.fail("test exceeded 60s deadline"))
    signal.alarm(60)
    yield
    signal.alarm(0)


@pytest.fixture
def ws(tmp_path):
    w = tmp_path / "ws"
    (w / ".git" / "hooks").mkdir(parents=True)
    (w / "src").mkdir()
    (w / "src" / "app.py").write_text("print(1)\n")
    (w / ".env").write_text("API_KEY=x\n")
    return w


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "home"
    (h / ".ssh").mkdir(parents=True)
    (h / ".ssh" / "id_ed25519").write_text("KEY")
    (h / ".aws").mkdir()
    (h / ".aws" / "config").write_text("[default]\n")
    return h


def compiled(ws, home, policy=None, **kw):
    return (policy or SandboxPolicy()).compile(ws, home=str(home), session_tmp=kw.get("tmp"))


def test_defaults_are_workspace_write_with_network_denied():
    p = SandboxPolicy().validate()
    assert p.mode is SandboxMode.WORKSPACE_WRITE
    assert p.network.mode is NetworkMode.NONE
    assert "~/.ssh" in p.filesystem.deny_read and "**/.env" in p.filesystem.deny_read
    assert p.filesystem.read_only_subpaths == (".git", ".harness")
    assert p.weakenings() == []


def test_secrets_are_read_denied_by_default(ws, home):
    c = compiled(ws, home)
    d = c.check_read(home / ".ssh" / "id_ed25519")
    assert not d.allowed and d.rule == "deny_read:~/.ssh"
    assert not c.check_read(ws / ".env").allowed
    assert not c.check_read(ws / "sub" / ".env.production").allowed
    assert c.check_read(ws / "src" / "app.py").allowed
    assert c.check_read("/usr/lib").allowed  # reads allowed by default (srt)


def test_more_specific_allow_beats_deny_but_glob_deny_stays(ws, home):
    p = SandboxPolicy(filesystem=FilesystemPolicy(allow_read=("~/.aws/config", "~/.ssh")))
    c = compiled(ws, home, p)
    assert c.check_read(home / ".aws" / "config").allowed            # deeper allow wins
    assert not c.check_read(home / ".aws" / "credentials").allowed   # rest of ~/.aws still denied
    assert not c.check_read(home / ".ssh" / "id_ed25519").allowed     # equal specificity: deny wins
    p2 = SandboxPolicy(filesystem=FilesystemPolicy(allow_read=(str(ws),)))
    assert not compiled(ws, home, p2).check_read(ws / ".env").allowed  # **/.env stays denied


def test_writes_are_allow_only_and_protected_subpaths_read_only(ws, home, tmp_path):
    tmp = tmp_path / "sess"
    tmp.mkdir()
    c = compiled(ws, home, tmp=tmp)
    assert c.check_write(ws / "src" / "new.py").allowed
    assert c.check_write(tmp / "scratch").allowed
    hook = c.check_write(ws / ".git" / "hooks" / "pre-commit")
    assert not hook.allowed and hook.rule.startswith("read_only_subpaths")
    assert not c.check_write(ws / ".harness" / "settings.json").allowed
    assert not c.check_write("/etc/passwd").allowed
    assert not c.check_write(home / "notes.txt").allowed
    assert not c.check_write(ws / ".env").allowed  # secret paths are not writable either


def test_symlink_escape_and_traversal_are_rejected(ws, home):
    (ws / "link_out").symlink_to(home)
    (ws / "ssh_link").symlink_to(home / ".ssh")
    c = compiled(ws, home)
    assert not c.check_write(ws / "link_out" / "evil.sh").allowed       # resolves outside the workspace
    assert not c.check_read(ws / "ssh_link" / "id_ed25519").allowed     # resolves into ~/.ssh
    assert not c.check_write(f"{ws}/src/../../outside.txt").allowed
    with pytest.raises(PolicyError):
        c.check_read("src/%2e%2e%2f%2e%2e%2fetc/passwd")
    with pytest.raises(PolicyError):
        c.check_write("src/a\x00b")


def test_worktree_resolved_gitdir_is_protected(tmp_path, home):
    main_git = tmp_path / "main" / ".git"
    (main_git / "worktrees" / "wt" ).mkdir(parents=True)
    (main_git / "hooks").mkdir()
    (main_git / "worktrees" / "wt" / "commondir").write_text("../..\n")
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {main_git / 'worktrees' / 'wt'}\n")
    p = SandboxPolicy(filesystem=FilesystemPolicy(allow_write=(str(main_git.parent),)))
    c = p.compile(wt, home=str(home))
    assert str(main_git) in c.read_only_paths
    assert not c.check_write(main_git / "hooks" / "post-checkout").allowed


def test_read_only_mode_has_no_writable_workspace(ws, home):
    c = compiled(ws, home, SandboxPolicy(mode=SandboxMode.READ_ONLY))
    assert c.writable_roots == ()
    assert not c.check_write(ws / "src" / "app.py").allowed
    assert c.check_read(ws / "src" / "app.py").allowed


def test_escape_hatches_and_full_access_need_admin():
    with pytest.raises(EscapeHatchError):
        SandboxPolicy(mode=SandboxMode.FULL_ACCESS).validate()
    with pytest.raises(EscapeHatchError):
        SandboxPolicy(escape_hatches=EscapeHatches(allow_non_loopback_proxy=True)).validate()
    with pytest.raises(EscapeHatchError):
        SandboxPolicy.from_dict({"escape_hatches": {"allowed_private_cidrs": ["10.0.0.0/8"]}})
    p = SandboxPolicy.from_dict({"escape_hatches": {"allowed_private_cidrs": ["10.0.0.0/8"]}}, admin=True)
    assert p.weakenings() == ["escape_hatches.allowed_private_cidrs"]


def test_load_yaml_and_json_extend_default_denies(tmp_path):
    y = tmp_path / "p.yaml"
    y.write_text(
        "mode: read-only\n"
        "filesystem:\n  deny_read: ['~/.config/secret-tool']\n"
        "network:\n  mode: allowlist\n  allowed_domains: ['pypi.org', '*.example.com']\n  allowed_ports: [443]\n"
        "limits:\n  cpu_seconds: 30\n  default_timeout_s: 10\n"
        "env:\n  set: {CI: '1'}\n  deny: ['MY_*']\n"
    )
    p = load_policy(y)
    assert p.mode is SandboxMode.READ_ONLY and p.network.mode is NetworkMode.ALLOWLIST
    assert "~/.ssh" in p.filesystem.deny_read and "~/.config/secret-tool" in p.filesystem.deny_read
    assert p.network.allowed_domains == ("pypi.org", "*.example.com") and p.network.allowed_ports == (443,)
    assert p.limits.cpu_seconds == 30 and p.limits.default_timeout_s == 10.0
    assert "MY_*" in p.env.deny and "*TOKEN*" in p.env.deny
    j = tmp_path / "p.json"
    j.write_text(json.dumps(p.to_dict()))
    assert load_policy(j).fingerprint() == p.fingerprint()


@pytest.mark.parametrize("bad", [
    {"mode": "yolo"},
    {"unknown": 1},
    {"filesystem": {"deny_reed": []}},
    {"network": {"mode": "allowlist", "allowed_domains": ["https://pypi.org"]}},
    {"network": {"allowed_domains": ["py*.org"]}},
    {"limits": {"cpu_seconds": -1}},
    {"limits": {"default_timeout_s": 900}},
    {"env": {"set": {"BAD NAME": "x"}}},
    {"redact_output": "yes"},
])
def test_invalid_policies_are_rejected(bad):
    with pytest.raises(PolicyError):
        SandboxPolicy.from_dict(bad)


def test_domain_matching_exact_and_wildcard():
    assert domain_matches("pypi.org", "PyPI.org.")
    assert not domain_matches("pypi.org", "evil-pypi.org")
    assert domain_matches("*.example.com", "a.b.example.com")
    assert not domain_matches("*.example.com", "example.com")
    net = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allowed_domains=("*.example.com",), denied_domains=("bad.example.com",))
    assert net.domain_decision("bad.example.com") == (False, "denied_domains:bad.example.com")
    assert net.domain_decision("ok.example.com")[0] is True
    assert net.domain_decision("other.org") == (None, "")


def test_timeout_capped_by_policy():
    lim = ResourceLimits(default_timeout_s=5, max_timeout_s=20)
    assert lim.effective_timeout(None) == 5 and lim.effective_timeout(100) == 20
    with pytest.raises(PolicyError):
        lim.effective_timeout(0)


def test_env_allowlist_scrubs_secrets_and_injects_proxy():
    host = {"PATH": "/usr/bin", "HOME": "/home/u", "LANG": "C.UTF-8", "LC_ALL": "C", "AWS_SECRET_ACCESS_KEY": "abc",
            "GITHUB_TOKEN": "ghp_" + "a" * 36, "RANDOM_VAR": "1", "HTTPS_PROXY": "http://user:pw@corp:8080"}
    p = SandboxPolicy(env=EnvPolicy(set={"CI": "1"}))
    built = build_env(p, host_env=host, extra={"MY_API_KEY": "zzz", "SAFE": "ok", "LEAK": "sk-ant-" + "b" * 30},
                      sandbox_vars={"HTTPS_PROXY": "http://cmd:sandbox@127.0.0.1:3128"})
    env = built.env
    assert env["PATH"] == "/usr/bin" and env["LC_ALL"] == "C" and env["CI"] == "1" and env["SAFE"] == "ok"
    assert "RANDOM_VAR" not in env  # not allowlisted
    assert "AWS_SECRET_ACCESS_KEY" not in env and "GITHUB_TOKEN" not in env  # never allowlisted
    for name in ("MY_API_KEY", "LEAK"):  # passed explicitly, still scrubbed (by name / by value)
        assert name not in env and name in built.scrubbed
    assert env["HTTPS_PROXY"] == "http://cmd:sandbox@127.0.0.1:3128"  # ours, not the host's


def test_env_secret_passthrough_is_an_admin_escape_hatch():
    p = SandboxPolicy(escape_hatches=EscapeHatches(env_secret_passthrough=("DEPLOY_TOKEN",)))
    with pytest.raises(EscapeHatchError):
        p.validate()
    built = build_env(p.validate(admin=True), host_env={}, extra={"DEPLOY_TOKEN": "t"})
    assert built.env["DEPLOY_TOKEN"] == "t"


def test_glob_matches_expand_for_os_masks(ws, home):
    (ws / "pkg").mkdir()
    (ws / "pkg" / ".env.local").write_text("X=1")
    c = compiled(ws, home)
    found = c.find_glob_matches(c.glob_deny_reads())
    assert str(ws / ".env") in found and str(ws / "pkg" / ".env.local") in found
    assert all(os.path.exists(f) for f in found)
