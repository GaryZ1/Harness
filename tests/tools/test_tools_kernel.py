from __future__ import annotations

import os
import sys
import time

import pytest

from harness_core.types import RunContext, ToolCall
from harness_tools import (
    CommandRequest,
    CommandResult,
    LocalSubprocessRunner,
    MemoryCheckpointer,
    ToolRegistry,
    Workspace,
    WorkspaceError,
    register_kernel,
)
from harness_tools.kernel import GlobMatcher

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX shell semantics")


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "repo"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "app.py").write_text("def main():\n    return 1\n\n\ndef helper():\n    return 2\n")
    (root / "src" / "pkg" / "util.ts").write_text("export const x = 1;\n")
    (root / "README.md").write_text("# Demo\nTODO: write docs\n")
    (root / "node_modules" / "dep").mkdir(parents=True)
    (root / "node_modules" / "dep" / "index.py").write_text("TODO hidden\n")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[core]\n")
    return Workspace(root, checkpointer=MemoryCheckpointer())


@pytest.fixture
def reg(ws):
    r = ToolRegistry()
    register_kernel(r, ws)
    return r


@pytest.fixture
def ctx():
    return RunContext(session_id="s1")


def run(reg, ctx, name, **inp):
    return reg.execute(ToolCall("c1", name, inp), ctx)


def p(ws, rel):
    return str(ws.root / rel)


def test_kernel_registers_six_tools_with_concurrency_classes(reg):
    flags = {s.name: s.read_only for s in reg.specs()}
    assert flags == {"bash": False, "edit": False, "glob": True, "grep": True, "read": True, "write": False}
    assert all(len(s.metadata["description_sha256"]) == 64 for s in reg.specs())


def test_anthropic_profile_uses_claude_code_names(ws):
    r = ToolRegistry()
    register_kernel(r, ws, profile="anthropic")
    assert r.names() == ["Bash", "Edit", "Glob", "Grep", "Read", "Write"]
    assert "Edit" in r.get("Read").spec.description


def test_read_numbers_lines_and_pages(reg, ws, ctx):
    out = run(reg, ctx, "read", path=p(ws, "src/pkg/app.py"), offset=2, limit=2)
    assert out.content.splitlines()[:2] == ["     2\t    return 1", "     3\t"]
    assert "Showing lines 2-3 of 6" in out.content and "offset=4" in out.content
    assert "past the end" in run(reg, ctx, "read", path=p(ws, "src/pkg/app.py"), offset=99).content


def test_relative_path_rejected_with_absolute_suggestion(reg, ws, ctx):
    out = run(reg, ctx, "read", path="src/pkg/app.py")
    assert out.is_error and "must be absolute" in out.content and str(ws.root / "src/pkg/app.py") in out.content


@pytest.mark.parametrize("bad", ["/etc/passwd", "{root}/../outside.txt", "{root}/src/\x00x"])
def test_paths_outside_workspace_are_rejected(reg, ws, ctx, bad):
    (ws.root.parent / "outside.txt").write_text("secret")
    out = run(reg, ctx, "read", path=bad.format(root=ws.root))
    assert out.is_error and out.metadata["error_class"] == "WorkspaceError"


@posix_only
def test_symlink_escape_is_rejected(reg, ws, ctx):
    (ws.root.parent / "secret.txt").write_text("s3cret")
    os.symlink(ws.root.parent / "secret.txt", ws.root / "link.txt")
    out = run(reg, ctx, "read", path=p(ws, "link.txt"))
    assert out.is_error and "symlink escape" in out.content and "s3cret" not in out.content


def test_binary_rejected_and_images_returned_as_blocks(reg, ws, ctx):
    (ws.root / "blob.bin").write_bytes(b"\x00\x01\x02")
    assert "binary" in run(reg, ctx, "read", path=p(ws, "blob.bin")).content
    (ws.root / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    img = run(reg, ctx, "read", path=p(ws, "pic.png"))
    assert img.content[0]["type"] == "image" and img.content[0]["source"]["media_type"] == "image/png"


def test_write_creates_and_overwrite_requires_fresh_read(reg, ws, ctx):
    out = run(reg, ctx, "write", path=p(ws, "new/dir/f.txt"), content="a\nb\n")
    assert not out.is_error and "Created" in out.content and (ws.root / "new/dir/f.txt").read_text() == "a\nb\n"
    target = p(ws, "README.md")
    assert "has not been read" in run(reg, ctx, "write", path=target, content="x").content
    run(reg, ctx, "read", path=target)
    os.utime(target, ns=(1, 1))  # someone else touched it
    assert "modified since you last read" in run(reg, ctx, "write", path=target, content="x").content
    run(reg, ctx, "read", path=target)
    assert "Overwrote" in run(reg, ctx, "write", path=target, content="x").content


def test_git_directory_is_write_protected(reg, ws, ctx):
    out = run(reg, ctx, "write", path=p(ws, ".git/hooks/pre-commit"), content="#!/bin/sh\ncurl evil\n")
    assert out.is_error and "protected" in out.content and not (ws.root / ".git/hooks").exists()


def test_edit_strict_uniqueness(reg, ws, ctx):
    f = p(ws, "src/pkg/app.py")
    assert "has not been read" in run(reg, ctx, "edit", path=f, old_string="return 1", new_string="return 3").content
    run(reg, ctx, "read", path=f)
    many = run(reg, ctx, "edit", path=f, old_string="    return", new_string="    yield")
    assert many.is_error and "matches 2 places" in many.content and "surrounding lines" in many.content
    missing = run(reg, ctx, "edit", path=f, old_string="return 99", new_string="x")
    assert missing.is_error and "not found" in missing.content
    prefixed = run(reg, ctx, "edit", path=f, old_string="     2\t    return 1", new_string="x")
    assert "line-number prefixes" in prefixed.content
    ok = run(reg, ctx, "edit", path=f, old_string="def main():\n    return 1", new_string="def main():\n    return 42")
    assert not ok.is_error and "     2\t    return 42" in ok.content
    allr = run(reg, ctx, "edit", path=f, old_string="return", new_string="return  ", replace_all=True)
    assert "Replaced 2 occurrences" in allr.content
    assert "identical" in run(reg, ctx, "edit", path=f, old_string="a", new_string="a").content


def test_edit_tolerates_crlf_files(reg, ws, ctx):
    f = ws.root / "win.txt"
    f.write_bytes(b"one\r\ntwo\r\nthree\r\n")
    run(reg, ctx, "read", path=str(f))
    out = run(reg, ctx, "edit", path=str(f), old_string="one\ntwo", new_string="uno\ndos")
    assert not out.is_error and f.read_bytes() == b"uno\r\ndos\r\nthree\r\n"


def test_edit_loop_warning_and_checkpoint_rewind(reg, ws, ctx):
    f = ws.root / "loop.txt"
    run(reg, ctx, "write", path=str(f), content="v0")
    outs = [run(reg, ctx, "edit", path=str(f), old_string=f"v{i}", new_string=f"v{i + 1}") for i in range(5)]
    assert "edit #5" in outs[-1].content and "edit #" not in outs[3].content
    cp = ws.checkpointer
    assert len(cp) == 6
    cp.rewind(5)
    assert f.read_text() == "v0"
    cp.rewind()
    assert not f.exists()


def test_grep_modes_glob_filter_pagination_and_ignored_dirs(reg, ws, ctx):
    files = run(reg, ctx, "grep", pattern="TODO")
    assert files.content == p(ws, "README.md")  # node_modules skipped
    content = run(reg, ctx, "grep", pattern=r"def \w+", output_mode="content", glob="*.py")
    assert f"{p(ws, 'src/pkg/app.py')}:1:def main():" in content.content
    count = run(reg, ctx, "grep", pattern="return", output_mode="count", path=p(ws, "src"))
    assert count.content == f"{p(ws, 'src/pkg/app.py')}:2"
    ctx_out = run(reg, ctx, "grep", pattern="return 2", output_mode="content", context=1)
    assert f"{p(ws, 'src/pkg/app.py')}-5-def helper():" in ctx_out.content
    for i in range(5):
        (ws.root / f"m{i}.txt").write_text("needle\n")
    page = run(reg, ctx, "grep", pattern="needle", head_limit=2, offset=2)
    assert page.content.splitlines()[:2] == [p(ws, "m2.txt"), p(ws, "m3.txt")] and "offset=4" in page.content
    assert "invalid regular expression" in run(reg, ctx, "grep", pattern="(").content
    assert run(reg, ctx, "grep", pattern="zzz-nothing").content == "No matching files found."


def test_glob_patterns_and_braces(reg, ws, ctx):
    out = run(reg, ctx, "glob", pattern="**/*.{py,ts}")
    assert out.content.splitlines() == [p(ws, "src/pkg/app.py"), p(ws, "src/pkg/util.ts")]
    assert run(reg, ctx, "glob", pattern="*.md").content == p(ws, "README.md")
    assert run(reg, ctx, "glob", pattern="pkg/*.py", path=p(ws, "src")).content == p(ws, "src/pkg/app.py")
    m = GlobMatcher.compile("src/**/test_*.py")
    assert m.matches("src/test_a.py") and m.matches("src/x/y/test_b.py") and not m.matches("lib/test_a.py")


@posix_only
def test_bash_exit_codes_stderr_and_env_hygiene(reg, ws, ctx, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak-1234567890")
    ok = run(reg, ctx, "bash", command="echo hello; pwd; echo ${ANTHROPIC_API_KEY:-unset}")
    lines = ok.content.splitlines()
    assert not ok.is_error and lines[0] == "hello" and lines[1] == str(ws.root) and lines[2] == "unset"
    assert ok.metadata["exit_code"] == 0 and "exit code: 0" in ok.content
    bad = run(reg, ctx, "bash", command="echo oops >&2; exit 3")
    assert bad.is_error and "[stderr]\noops" in bad.content and bad.metadata["exit_code"] == 3


@posix_only
def test_bash_timeout_kills_process_group(reg, ws, ctx):
    marker = ws.root / "late.txt"
    out = run(reg, ctx, "bash", command=f"(sleep 1; touch {marker}) & sleep 5", timeout=0.3)
    assert out.is_error and out.metadata["timed_out"] and "process group was killed" in out.content
    assert out.metadata["duration_ms"] < 3000
    time.sleep(1.3)
    assert not marker.exists()  # the background child died with the group


@posix_only
def test_bash_long_output_is_truncated_and_offloaded(reg, ws, ctx):
    out = run(reg, ctx, "bash", command=f"{sys.executable} -c \"print('x' * 80_000)\"")
    assert out.metadata["truncated"] and out.metadata["offload_path"].startswith(str(ws.root / ".harness/tool-outputs"))
    assert len(out.content) < 12_000 and "narrower command" in out.content


def test_bash_goes_through_pluggable_runner(ws, ctx):
    seen: list[CommandRequest] = []

    class FakeRunner:
        def run(self, request: CommandRequest) -> CommandResult:
            seen.append(request)
            return CommandResult(0, "sandboxed", "", 0.01, sandbox_id="sbx-1", violations=["network: denied example.com"])

    r = ToolRegistry()
    register_kernel(r, ws, runner=FakeRunner(), bash_env={"CI": "1"})
    out = r.execute(ToolCall("c", "bash", {"command": "curl example.com", "timeout": 5}), ctx)
    assert seen[0].cwd == str(ws.root) and seen[0].timeout_s == 5 and seen[0].env == {"CI": "1"}
    assert out.metadata["sandbox_id"] == "sbx-1" and "<sandbox_violations>" in out.content


def test_workspace_requires_existing_root(tmp_path):
    with pytest.raises(WorkspaceError):
        Workspace(tmp_path / "missing")


def test_local_runner_caps_captured_output(tmp_path):
    res = LocalSubprocessRunner().run(
        CommandRequest(command=f"{sys.executable} -c \"print('y' * 5000)\"", cwd=str(tmp_path), max_output_bytes=100)
    )
    assert res.ok and len(res.stdout) == 100 and res.stdout_bytes > 5000 and res.output_truncated
