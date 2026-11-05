"""Bubblewrap argv, Seatbelt profile and container argv builders (pure, run everywhere) plus the live
bubblewrap escape suite from checklist item 19 (skipped when ``bwrap`` is unusable)."""

from __future__ import annotations

import os
import shutil
import signal
import socket
import stat
import sys

import pytest

from harness_sandbox import (
    BubblewrapRunner,
    ContainerConfig,
    ContainerRunner,
    FilesystemPolicy,
    NetworkMode,
    NetworkPolicy,
    PolicyError,
    SandboxMode,
    SandboxPolicy,
    SandboxUnavailableError,
    SeatbeltRunner,
    build_bwrap_argv,
    build_container_argv,
    build_seatbelt_profile,
    detect_backends,
)
from harness_sandbox.detect import find_trusted_binary


@pytest.fixture(autouse=True)
def _deadline():
    signal.signal(signal.SIGALRM, lambda *_: pytest.fail("test exceeded 90s deadline"))
    signal.alarm(90)
    yield
    signal.alarm(0)


@pytest.fixture
def tree(tmp_path):
    ws = tmp_path / "ws"
    (ws / ".git" / "hooks").mkdir(parents=True)
    (ws / "src").mkdir()
    (ws / ".env").write_text("SECRET=1\n")
    (ws / "src" / ".env.local").write_text("SECRET=2\n")
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_ed25519").write_text("PRIVATE")
    (home / ".aws").mkdir()
    (home / ".aws" / "config").write_text("[default]\n")
    (home / ".netrc").write_text("machine x password y\n")
    tmp = tmp_path / "sess"
    tmp.mkdir()
    return ws, home, tmp


def pairs(argv, flag):
    return [tuple(argv[i + 1:i + 3]) if flag != "--tmpfs" and flag != "--remount-ro" else (argv[i + 1],)
            for i, a in enumerate(argv) if a == flag]


# ----------------------------------------------------------------------------- bubblewrap argv


def test_bwrap_argv_layers_policy(tree):
    ws, home, tmp = tree
    c = SandboxPolicy().compile(ws, home=str(home), session_tmp=tmp)
    argv = build_bwrap_argv(c, cwd=str(ws / "src"), bwrap="/usr/bin/bwrap", disable_userns=True)
    for flag in ("--new-session", "--die-with-parent", "--unshare-net", "--unshare-pid", "--unshare-user", "--disable-userns"):
        assert flag in argv
    assert argv[0] == "/usr/bin/bwrap" and argv[-2:] == ["--chdir", str(ws / "src")]
    assert ("/", "/") in pairs(argv, "--ro-bind") and (str(ws), str(ws)) in pairs(argv, "--bind")
    assert (str(tmp), str(tmp)) in pairs(argv, "--bind")
    assert (str(ws / ".git"), str(ws / ".git")) in pairs(argv, "--ro-bind")
    assert ("/dev/null", str(ws / ".env")) in pairs(argv, "--ro-bind")
    assert ("/dev/null", str(ws / "src" / ".env.local")) in pairs(argv, "--ro-bind")
    assert ("/dev/null", str(home / ".netrc")) in pairs(argv, "--ro-bind")
    assert (str(home / ".ssh"),) in pairs(argv, "--tmpfs") and (str(home / ".ssh"),) in pairs(argv, "--remount-ro")
    # deeper rules come after shallower ones: the workspace bind precedes its protected .git
    assert argv.index(str(ws / ".git")) > argv.index("--bind")
    # the tmp tmpfs comes before the session tmp bind under it
    assert argv.index("/tmp") < argv.index(str(tmp))
    assert "--setenv" not in argv  # env goes through Popen, never argv


def test_bwrap_allow_read_exception_under_mask_and_network_modes(tree, tmp_path):
    ws, home, tmp = tree
    pol = SandboxPolicy(filesystem=FilesystemPolicy(allow_read=("~/.aws/config",)),
                        network=NetworkPolicy(mode=NetworkMode.ALLOWLIST))
    sock = tmp_path / "proxy" / "proxy.sock"
    sock.parent.mkdir()
    argv = build_bwrap_argv(pol.compile(ws, home=str(home)), cwd=str(ws), proxy_socket=str(sock))
    aws = str(home / ".aws")
    assert (aws,) in pairs(argv, "--tmpfs") and (aws,) not in pairs(argv, "--remount-ro")  # child re-bound inside
    assert argv.index(str(home / ".aws" / "config")) > argv.index(aws)
    assert (str(sock.parent), str(sock.parent)) in pairs(argv, "--bind") and "--unshare-net" in argv
    open_argv = build_bwrap_argv(SandboxPolicy(network=NetworkPolicy(mode=NetworkMode.OPEN)).compile(ws, home=str(home)), cwd=str(ws))
    assert "--unshare-net" not in open_argv
    ro = build_bwrap_argv(SandboxPolicy(mode=SandboxMode.READ_ONLY).compile(ws, home=str(home)), cwd=str(ws))
    assert (str(ws), str(ws)) not in pairs(ro, "--bind")


def test_bubblewrap_runner_prepares_full_command(tree):
    ws, home, _ = tree
    r = BubblewrapRunner(SandboxPolicy(), ws, bwrap="/opt/fake/bwrap", disable_userns=False, home=str(home),
                         host_env={"PATH": "/usr/bin", "GH_TOKEN": "x"})
    try:
        p = r.prepare(["make", "test"])
        sep = p.argv.index("--")
        assert p.argv[0] == "/opt/fake/bwrap" and p.argv[sep + 1] == sys.executable
        assert p.argv[sep + 4].endswith("_launcher.py") and p.argv[-3:] == ["--", "make", "test"]
        assert "GH_TOKEN" not in p.env and p.env["TMPDIR"] == r.session_tmp
        assert r.weak is False and r.isolation == "os"
    finally:
        r.close()


def test_trusted_binary_lookup_skips_workspace(tmp_path):
    ws = tmp_path / "ws"
    (ws / "bin").mkdir(parents=True)
    trojan = ws / "bin" / "bwrap"
    trojan.write_text("#!/bin/sh\necho pwned\n")
    trojan.chmod(trojan.stat().st_mode | stat.S_IXUSR)
    good = tmp_path / "sys"
    good.mkdir()
    real = good / "bwrap"
    real.write_text("#!/bin/sh\n")
    real.chmod(real.stat().st_mode | stat.S_IXUSR)
    path = os.pathsep.join([str(ws / "bin"), "relative/bin", str(good)])
    assert find_trusted_binary("bwrap", workspace=str(ws), path=path, cwd="/") == str(real)
    assert find_trusted_binary("bwrap", workspace=str(ws), path=str(ws / "bin"), cwd="/") is None


# ----------------------------------------------------------------------------- seatbelt profile


def test_seatbelt_profile_text(tree):
    ws, home, tmp = tree
    c = SandboxPolicy().compile(ws, home=str(home), session_tmp=tmp)
    prof = build_seatbelt_profile(c)
    assert prof.startswith("(version 1)\n(deny default)\n")
    assert f'(require-not (subpath "{home}/.ssh"))' in prof
    assert '(require-not (regex #"^(.*/)?[.]env(/.*)?$"))' in prof
    write = next(line for line in prof.splitlines() if line.startswith("(allow file-write* (require-all"))
    assert f'(subpath "{ws}")' in write and f'(require-not (subpath "{ws}/.git"))' in write
    assert "network" not in prof.replace("network=none", "")  # mode none: no network rule at all
    allow = build_seatbelt_profile(SandboxPolicy(network=NetworkPolicy(mode=NetworkMode.ALLOWLIST)).compile(ws, home=str(home)),
                                   proxy_port=3128)
    assert allow.count("(allow network") == 1 and '(remote ip "localhost:3128")' in allow
    assert "(allow network*)" in build_seatbelt_profile(SandboxPolicy(network=NetworkPolicy(mode=NetworkMode.OPEN)).compile(ws))
    ro = build_seatbelt_profile(SandboxPolicy(mode=SandboxMode.READ_ONLY, filesystem=FilesystemPolicy(session_tmp=False)).compile(ws))
    assert "(allow file-write* (require-all" not in ro


def test_seatbelt_escapes_quotes_and_allow_read_keeps_inner_denies(tmp_path):
    ws = tmp_path / 'we"ird'
    ws.mkdir()
    pol = SandboxPolicy(filesystem=FilesystemPolicy(allow_read=(str(ws),)))
    prof = build_seatbelt_profile(pol.compile(ws))
    assert 'we\\"ird' in prof
    line = next(ln for ln in prof.splitlines() if ln.startswith(f'(allow file-read* (require-all (subpath "{str(ws).replace(chr(34), chr(92) + chr(34))}")'))
    assert "[.]env" in line  # **/.env stays denied inside the re-allowed subtree


def test_seatbelt_runner_builds_argv_on_any_os_but_runs_only_on_macos(tree):
    ws, home, _ = tree
    with SeatbeltRunner(SandboxPolicy(), ws, home=str(home)) as r:
        argv, url = r.build_argv(["ls"], command_id="cmd_1")
        assert argv[:2] == ["/usr/bin/sandbox-exec", "-p"] and "(deny default)" in argv[2] and url is None
        if sys.platform != "darwin":
            assert not SeatbeltRunner.available()
            with pytest.raises(SandboxUnavailableError):
                r.run(["ls"], timeout=5)


# ----------------------------------------------------------------------------- containers


def test_container_argv_is_hardened(tree):
    ws, home, _ = tree
    c = SandboxPolicy().compile(ws, home=str(home))
    argv = build_container_argv(c, ContainerConfig(image="python:3.12-slim", user="1000:1000"), ["pytest", "-q"],
                                name="harness-cmd_1", env_file="/tmp/env.list", cwd=str(ws / "src"))
    s = " ".join(argv)
    for frag in ("run --rm", "--network none", "--read-only", "--cap-drop ALL", "--security-opt no-new-privileges",
                 "--pids-limit 1024", f"--memory {4 * 1024 ** 3}", "--ulimit core=0:0", "--user 1000:1000",
                 f"-v {ws}:/workspace:rw", f"-v {ws}/.git:/workspace/.git:ro", "-v /dev/null:/workspace/.env:ro",
                 "-v /dev/null:/workspace/src/.env.local:ro", "--env-file /tmp/env.list", "--workdir /workspace/src"):
        assert frag in s, frag
    assert argv[-3:] == ["python:3.12-slim", "pytest", "-q"]
    assert str(home) not in s  # nothing outside the workspace is mounted


def test_container_network_modes(tree, tmp_path):
    ws, home, _ = tree
    ro = build_container_argv(SandboxPolicy(mode=SandboxMode.READ_ONLY).compile(ws), ContainerConfig(image="img"), ["ls"])
    assert f"{ws}:/workspace:ro" in ro
    op = build_container_argv(SandboxPolicy(network=NetworkPolicy(mode=NetworkMode.OPEN)).compile(ws), ContainerConfig(image="img"), ["ls"])
    assert op[op.index("--network") + 1] == "bridge"
    allow = SandboxPolicy(network=NetworkPolicy(mode=NetworkMode.ALLOWLIST)).compile(ws)
    with pytest.raises(PolicyError):
        build_container_argv(allow, ContainerConfig(image="img", python_in_image=None), ["ls"], proxy_socket="/x/p.sock", launcher_spec_json="{}")
    argv = build_container_argv(allow, ContainerConfig(image="img"), ["curl", "x"], proxy_socket="/run/p/p.sock", launcher_spec_json='{"bridge":1}')
    assert argv[argv.index("--network") + 1] == "none" and "/run/p:/run/p" in argv
    i = argv.index("img")
    assert argv[i:i + 5] == ["img", "python3", "-I", "-S", "/opt/harness-sandbox/_launcher.py"]
    assert argv[-3:] == ["--", "curl", "x"]


def test_container_runner_prepare_uses_env_file(tree, tmp_path):
    ws, home, _ = tree
    r = ContainerRunner(SandboxPolicy(), ws, config=ContainerConfig(image="img"), engine_path="/usr/bin/fake-docker",
                        host_env={"PATH": "/usr/bin", "LANG": "C", "AWS_SESSION_TOKEN": "t"}, home=str(home))
    p = r.prepare(["echo", "hi"], env={"FOO": "bar"})
    env_file = p.argv[p.argv.index("--env-file") + 1]
    text = open(env_file).read()
    assert "FOO=bar" in text and "LANG=C" in text and "TMPDIR=/tmp" in text and "AWS_SESSION_TOKEN" not in text
    assert p.argv[0] == "/usr/bin/fake-docker" and p.argv[p.argv.index("--name") + 1] == f"harness-{p.command_id}"
    assert p.on_timeout  # `docker kill <name>` on timeout: killing the client does not stop the container
    p.run_cleanup()
    assert not os.path.exists(env_file)
    r.close()


# ----------------------------------------------------------------------------- live bubblewrap escape suite

_REPORT = detect_backends(probe=True) if sys.platform.startswith("linux") and shutil.which("bwrap") else None
needs_bwrap = pytest.mark.skipif(not (_REPORT and _REPORT.bwrap_usable), reason="bubblewrap not installed or user namespaces unavailable")


@pytest.fixture
def outside_tmp_home():
    """A fake $HOME outside /tmp (bwrap replaces /tmp with a tmpfs, which would hide it trivially)."""
    base = "/var/tmp" if os.access("/var/tmp", os.W_OK) else None
    if base is None:
        pytest.skip("/var/tmp not writable")
    import tempfile

    home = tempfile.mkdtemp(prefix="harness-sbx-home-", dir=base)
    os.makedirs(os.path.join(home, ".ssh"))
    with open(os.path.join(home, ".ssh", "id_ed25519"), "w") as fh:
        fh.write("PRIVATE")
    with open(os.path.join(home, "notes.txt"), "w") as fh:
        fh.write("visible")
    yield home
    shutil.rmtree(home, ignore_errors=True)


@needs_bwrap
def test_bwrap_escape_suite(tree, tmp_path, outside_tmp_home):
    ws, _, _ = tree
    home = outside_tmp_home
    (ws / "link_out").symlink_to(home)
    sockdir = os.path.join(home, "sockets")
    os.mkdir(sockdir)
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(os.path.join(sockdir, "agent.sock"))
    srv.listen(1)
    pol = SandboxPolicy(filesystem=FilesystemPolicy(deny_read=(sockdir,)))
    py = sys.executable
    try:
        with BubblewrapRunner(pol, ws, home=home) as r:
            visible = r.run(["cat", os.path.join(home, "notes.txt")], timeout=30)
            checks = {
                "git hook write": r.run(["sh", "-c", "echo x > .git/hooks/pre-commit"], timeout=30),
                "ssh key read": r.run(["cat", os.path.join(home, ".ssh", "id_ed25519")], timeout=30),
                ".env read": r.run(["cat", ".env"], timeout=30),
                "symlink escape": r.run(["sh", "-c", "echo x > link_out/pwned"], timeout=30),
                "network": r.run([py, "-c", "import socket;socket.create_connection(('1.1.1.1',53),timeout=3)"], timeout=30),
                "unix socket": r.run([py, "-c", f"import socket;s=socket.socket(socket.AF_UNIX);s.connect('{sockdir}/agent.sock')"], timeout=30),
                "nested userns": r.run(["unshare", "--user", "true"], timeout=30),
            }
            ok = r.run(["sh", "-c", "echo fine > src/out.txt && cat src/out.txt"], timeout=30)
        assert visible.stdout == "visible"  # positive control: the fake home IS visible inside
        for name, res in checks.items():
            assert res.exit_code != 0 or res.stdout.strip() == "", f"{name} was not blocked: {res}"
        assert "PRIVATE" not in checks["ssh key read"].stdout and "SECRET" not in checks[".env read"].stdout
        assert any(v.kind == "filesystem" for v in checks["git hook write"].violations)
        assert not os.path.exists(os.path.join(home, "pwned"))
        assert not (ws / ".git" / "hooks" / "pre-commit").exists()
        assert ok.stdout.strip() == "fine" and (ws / "src" / "out.txt").read_text() == "fine\n"
    finally:
        srv.close()


@needs_bwrap
def test_bwrap_allowlist_egress_through_proxy(tree):
    import http.server
    import threading

    from harness_sandbox import EgressProxy, EscapeHatches

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *a):
            pass

    ws, home, _ = tree
    origin = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=origin.serve_forever, daemon=True).start()
    port = origin.server_address[1]
    pol = SandboxPolicy(network=NetworkPolicy(mode=NetworkMode.ALLOWLIST, allowed_domains=("pkg.test",), allowed_ports=(port,)),
                        escape_hatches=EscapeHatches(allowed_private_cidrs=("127.0.0.1/32",)))
    fetch = "import urllib.request,sys;print(urllib.request.urlopen(sys.argv[1],timeout=10).read().decode())"
    try:
        with EgressProxy(pol, tcp=False, unix_socket=True, resolver=lambda h, p: ["127.0.0.1"]) as proxy, \
                BubblewrapRunner(pol, ws, admin=True, proxy=proxy, home=str(home)) as r:
            ok = r.run([sys.executable, "-c", fetch, f"http://pkg.test:{port}/"], timeout=30)
            bad = r.run([sys.executable, "-c", fetch, f"http://exfil.test:{port}/"], timeout=30)
        assert ok.stdout.strip() == "ok", ok.stderr
        assert bad.exit_code != 0 and bad.violations[0].rule == "not-in-allowlist"
    finally:
        origin.shutdown()
        origin.server_close()
