import pytest

from harness_memory import PathSandbox, PathSecurityError, SecretRedactor, TaintTracker
from harness_memory.security import REDACTED


@pytest.mark.parametrize(
    "bad",
    ["../etc/passwd", "a/../../b", "..\\windows", "%2e%2e%2fsecret", "%252e%252e%252f", "/abs/path", "C:\\x", "a\x00b"],
)
def test_sandbox_rejects_traversal(tmp_path, bad):
    with pytest.raises(PathSecurityError):
        PathSandbox(tmp_path).resolve(bad)


def test_sandbox_normalises(tmp_path):
    sb = PathSandbox(tmp_path)
    assert sb.normalise("./a//b/c.md") == "a/b/c.md"
    assert sb.is_root("")
    assert sb.resolve("x/y.md") == tmp_path.resolve() / "x" / "y.md"


def test_sandbox_rejects_symlink_escape(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(outside)
    with pytest.raises(PathSecurityError):
        PathSandbox(root).resolve("link/file.md")


@pytest.mark.parametrize(
    "secret",
    [
        "sk-ant-api03-abcdefghijklmnopqrstuvwxyz012345",
        "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        "AKIAABCDEFGHIJKLMNOP",
        "xoxb-1234567890-abcdefghij",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----",
    ],
)
def test_redactor_scrubs_tokens(secret):
    out = SecretRedactor().redact(f"value: {secret} end")
    assert secret not in out and REDACTED in out


def test_redactor_assignment_and_url_credentials():
    r = SecretRedactor()
    assert r.redact('API_KEY="abcd1234efgh"') == f'API_KEY="{REDACTED}"'
    assert REDACTED in r.redact("postgres://admin:hunter2pass@db:5432/x")
    assert "admin" in r.redact("postgres://admin:hunter2pass@db:5432/x")
    assert r.redact("token_count: 123456") == "token_count: 123456"
    assert r.redact("Authorization: Bearer abcdefghijklmnopqrstuvwxyz").startswith("Authorization: Bearer ")


def test_taint_tracker():
    t = TaintTracker()
    assert not t.record_tool_output("read")
    assert t.memory_mode == "clean"
    assert t.record_tool_output("web_fetch")
    assert t.memory_mode == "polluted" and t.reasons == ["web_fetch"]
    t2 = TaintTracker()
    t2.record_tool_output("read", untrusted=True)
    assert t2.polluted
