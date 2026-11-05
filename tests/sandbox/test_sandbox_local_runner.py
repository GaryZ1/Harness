"""Hardened local runner, tested for real: limits, process-tree kill, env scrubbing, seccomp, netns + proxy."""

from __future__ import annotations

import http.server
import os
import signal
import socket
import subprocess
import sys
import threading
import time

import pytest

from harness_sandbox import (
    EgressProxy,
    EnvPolicy,
    EscapeHatches,
    LocalRunner,
    NetworkMode,
    NetworkPolicy,
    ResourceLimits,
    SandboxPolicy,
    prepare_mcp_server,
)
from harness_sandbox.detect import unshare_netns_argv
from harness_sandbox.runners.base import RESULT_FIELDS
from harness_sandbox.seccomp import SYSCALLS, machine

LINUX = sys.platform.startswith("linux")
needs_seccomp = pytest.mark.skipif(not (LINUX and machine() in SYSCALLS), reason="seccomp filter needs Linux x86_64/aarch64")
needs_netns = pytest.mark.skipif(not (LINUX and unshare_netns_argv()), reason="'unshare --net' unavailable (no userns/root)")
PY = sys.executable


@pytest.fixture(autouse=True)
def _deadline():
    signal.signal(signal.SIGALRM, lambda *_: pytest.fail("test exceeded 90s deadline"))
    signal.alarm(90)
    yield
    signal.alarm(0)


@pytest.fixture
def ws(tmp_path):
    w = tmp_path / "ws"
    w.mkdir()
    return w


def alive(pid: int) -> bool:
    """True if ``pid`` is running (zombies awaiting a non-reaping PID 1 count as dead)."""
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False
    except OSError:  # pragma: no cover - non-Linux
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False


def runner(ws, policy=None, **kw):
    kw.setdefault("network_namespace", False)
    return LocalRunner(policy or SandboxPolicy(), ws, **kw)


def test_basic_run_and_result_shape(ws):
    with runner(ws) as r:
        res = r.run(["echo", "hi"], timeout=20)
    assert (res.exit_code, res.stdout, res.timed_out, res.truncated) == (0, "hi\n", False, False)
    assert tuple(f for f in res.__dataclass_fields__)[:6] == RESULT_FIELDS
    assert res.command_id.startswith("cmd_") and res.sandbox == "local" and res.ok
    assert r.weak is True and "exit code 0" in res.model_text()


def test_shell_string_stdin_cwd_and_tmpdir(ws):
    (ws / "sub").mkdir()
    with runner(ws) as r:
        res = r.run("cat; pwd; echo $HARNESS_SANDBOX; echo $TMPDIR", cwd="sub", stdin="from-stdin\n", timeout=20)
        lines = res.stdout.splitlines()
        assert lines[0] == "from-stdin" and lines[1] == str(ws / "sub") and lines[2] == "local"
        assert lines[3] == r.session_tmp and os.path.isdir(lines[3])
        tmp = r.session_tmp
    assert not os.path.exists(tmp)  # session tmp expires with the runner


def test_environment_is_allowlisted_and_scrubbed(ws):
    host = {"PATH": os.environ["PATH"], "HOME": str(ws), "SECRET_TOKEN": "abc123", "UNLISTED": "1"}
    pol = SandboxPolicy(env=EnvPolicy(set={"FROM_POLICY": "yes"}))
    with runner(ws, pol, host_env=host) as r:
        res = r.run([PY, "-c", "import os,json;print(json.dumps(dict(os.environ)))"], env={"OPENAI_API_KEY": "sk-zzz", "OK_VAR": "1"},
                    timeout=20)
    import json

    env = json.loads(res.stdout)
    assert "SECRET_TOKEN" not in env and "UNLISTED" not in env and "OPENAI_API_KEY" not in env
    assert env["OK_VAR"] == "1" and env["FROM_POLICY"] == "yes" and env["HARNESS_SANDBOX_COMMAND_ID"] == res.command_id
    assert "OPENAI_API_KEY" in res.scrubbed_env


def test_timeout_kills_the_whole_process_tree(ws):
    pidfile = ws / "bg.pid"
    with runner(ws, SandboxPolicy(limits=ResourceLimits(kill_grace_s=0.5))) as r:
        t0 = time.monotonic()
        res = r.run(f"sleep 60 & echo $! > {pidfile}; sleep 60", timeout=1)
        assert time.monotonic() - t0 < 15
    assert res.timed_out and not res.ok and res.violations[0].kind == "timeout"
    time.sleep(0.2)
    assert not alive(int(pidfile.read_text()))


def test_background_children_do_not_outlive_a_normal_exit(ws):
    pidfile = ws / "bg.pid"
    with runner(ws) as r:
        res = r.run(f"sleep 60 >/dev/null 2>&1 & echo $! > {pidfile}", timeout=20)
    assert res.exit_code == 0
    time.sleep(0.2)
    assert not alive(int(pidfile.read_text()))


def test_output_cap_keeps_head_and_tail(ws):
    pol = SandboxPolicy(limits=ResourceLimits(max_output_bytes=4096))
    with runner(ws, pol) as r:
        res = r.run([PY, "-c", "print('A'*50000 + 'END')"], timeout=20)
    assert res.truncated and res.stdout_bytes == 50004
    assert res.stdout.startswith("A" * 100) and res.stdout.rstrip().endswith("END") and "bytes truncated" in res.stdout
    assert len(res.stdout) < 5000


@pytest.mark.skipif(not LINUX, reason="RLIMIT semantics checked on Linux")
def test_cpu_limit_sends_sigxcpu(ws):
    pol = SandboxPolicy(limits=ResourceLimits(cpu_seconds=1))
    with runner(ws, pol) as r:
        res = r.run([PY, "-c", "while True: pass"], timeout=30)
    assert res.signal == signal.SIGXCPU and res.exit_code == 128 + signal.SIGXCPU and not res.timed_out
    assert res.violations[0].rule == "limits.cpu_seconds"


@pytest.mark.skipif(not LINUX, reason="RLIMIT semantics checked on Linux")
def test_file_size_and_memory_limits(ws):
    pol = SandboxPolicy(limits=ResourceLimits(file_size_bytes=1024 * 1024, memory_bytes=256 * 1024 * 1024))
    with runner(ws, pol) as r:
        big = r.run([PY, "-c", "open('big.bin','wb').write(b'x'*(3*1024*1024))"], timeout=20)
        mem = r.run([PY, "-c", "b = bytearray(1024*1024*1024)"], timeout=20)
    assert big.exit_code != 0 and (ws / "big.bin").stat().st_size <= 1024 * 1024
    assert any(v.rule == "limits.file_size_bytes" for v in big.violations)
    assert mem.exit_code != 0 and any(v.rule == "limits.memory_bytes" for v in mem.violations)


def test_rlimits_and_no_new_privs_are_applied(ws):
    pol = SandboxPolicy(limits=ResourceLimits(open_files=256, max_processes=777))
    code = ("import resource as r;print(r.getrlimit(r.RLIMIT_NOFILE)[0], r.getrlimit(r.RLIMIT_NPROC)[0], r.getrlimit(r.RLIMIT_CORE)[1]);"
            "print(open('/proc/self/status').read().split('NoNewPrivs:')[1].split()[0] if __import__('sys').platform.startswith('linux') else 1)")
    with runner(ws, pol) as r:
        res = r.run([PY, "-c", code], timeout=20)
    nofile, nproc, core = res.stdout.split()[:3]
    assert (nofile, nproc, core) == ("256", "777", "0")
    assert res.stdout.split()[3] == "1"


@needs_seccomp
def test_seccomp_blocks_nested_namespaces_and_ptrace(ws):
    code = ("import ctypes,os;libc=ctypes.CDLL(None,use_errno=True);"
            "print(libc.unshare(0x10000000), ctypes.get_errno());"   # CLONE_NEWUSER
            "print(libc.ptrace(0,0,0,0), ctypes.get_errno());"
            "pid=os.fork();os._exit(0) if pid==0 else os.waitpid(pid,0);print('fork-ok')")
    with runner(ws) as r:
        res = r.run([PY, "-c", code], timeout=20)
    lines = res.stdout.split("\n")
    assert lines[0] == "-1 1" and lines[1] == "-1 1" and lines[2] == "fork-ok", res.stderr
    if subprocess.run(["sh", "-c", "command -v unshare"], capture_output=True).returncode == 0:
        with runner(ws) as r:
            assert r.run(["unshare", "--user", "true"], timeout=20).exit_code != 0
    hatch = SandboxPolicy(escape_hatches=EscapeHatches(allow_nested_namespaces=True))
    with runner(ws, hatch, admin=True) as r:
        out = r.run([PY, "-c", "import ctypes;l=ctypes.CDLL(None,use_errno=True);print(l.ptrace(0,0,0,0), ctypes.get_errno())"], timeout=20)
    assert out.stdout.strip() == "-1 1"  # ptrace stays blocked even with the namespace hatch


def test_missing_command_and_refused_cwd(ws, tmp_path):
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    with runner(ws, home=str(home)) as r:
        assert r.run(["definitely-not-a-command-xyz"], timeout=20).exit_code == 127
        res = r.run(["ls"], cwd=str(home / ".ssh"), timeout=20)
        assert res.exit_code == 126 and res.violations[0].rule == "deny_read:~/.ssh"
        assert "<sandbox_violations>" in res.model_text()
        assert r.run(["ls"], cwd=str(ws / "nope"), timeout=20).violations[0].rule == "cwd-missing"


def test_output_redaction_and_event_sink(ws):
    events = []

    class Sink:
        def emit(self, e):
            events.append(e)
            return e

    with runner(ws, event_sink=Sink(), session_id="sess1") as r:
        res = r.run(["echo", "key=sk-ant-" + "a" * 30], timeout=20)
    assert "sk-ant-" not in res.stdout and "[REDACTED_SECRET]" in res.stdout
    (ev,) = events
    assert ev.session_id == "sess1" and ev.data["kind"] == "sandbox.exec" and ev.data["weak"] is True
    assert ev.data["command_id"] == res.command_id and "stdout" not in ev.data


def test_prepared_command_for_long_lived_processes(ws):
    with runner(ws) as r:
        p = prepare_mcp_server(r, PY, ["-c", "import sys;print(sys.stdin.readline().upper(),end='')"])
        proc = subprocess.Popen(p.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, **p.popen_kwargs())
        out, _ = proc.communicate("hello mcp\n", timeout=20)
    assert out == "HELLO MCP\n" and p.env["HARNESS_SANDBOX_COMMAND_ID"] == p.command_id


@needs_netns
def test_network_none_removes_the_network(ws):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        with LocalRunner(SandboxPolicy(), ws, network_namespace=True) as r:
            assert r.isolation == "process+netns"
            res = r.run([PY, "-c", f"import socket;socket.create_connection(('127.0.0.1',{port}),timeout=3)"], timeout=20)
        assert res.exit_code != 0 and ("ConnectionRefused" in res.stderr or "unreachable" in res.stderr)
    finally:
        srv.close()


@needs_netns
def test_allowlist_egress_through_bridge_and_proxy(ws):
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *a):
            pass

    origin = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=origin.serve_forever, daemon=True).start()
    port = origin.server_address[1]
    pol = SandboxPolicy(network=NetworkPolicy(mode=NetworkMode.ALLOWLIST, allowed_domains=("pkg.test",), allowed_ports=(port,)),
                        escape_hatches=EscapeHatches(allowed_private_cidrs=("127.0.0.1/32",)))
    proxy = EgressProxy(pol, tcp=False, unix_socket=True, resolver=lambda h, p: ["127.0.0.1"])
    fetch = "import urllib.request,sys;print(urllib.request.urlopen(sys.argv[1],timeout=10).read().decode())"
    try:
        with proxy, LocalRunner(pol, ws, admin=True, proxy=proxy, network_namespace=True) as r:
            ok = r.run([PY, "-c", fetch, f"http://pkg.test:{port}/x"], timeout=30)
            bad = r.run([PY, "-c", fetch, f"http://exfil.test:{port}/x"], timeout=30)
            direct = r.run([PY, "-c", f"import socket;socket.create_connection(('127.0.0.1',{port}),timeout=3)"], timeout=30)
            assert ok.stdout.strip() == "ok", ok.stderr
            assert bad.exit_code != 0 and bad.violations[0].kind == "network" and bad.violations[0].rule == "not-in-allowlist"
            assert direct.exit_code != 0  # no route around the proxy
            assert {e.command_id for e in proxy.events()} == {ok.command_id, bad.command_id}
    finally:
        origin.shutdown()
        origin.server_close()
