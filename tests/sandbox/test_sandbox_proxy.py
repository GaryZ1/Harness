"""Egress proxy against local servers on 127.0.0.1 (offline; a fake resolver maps test names)."""

from __future__ import annotations

import http.server
import signal
import socket
import socketserver
import struct
import threading
import urllib.error
import urllib.request

import pytest

from harness_core.types import Event
from harness_sandbox import (
    EgressProxy,
    EscapeHatches,
    NetworkMode,
    NetworkPolicy,
    ProxyError,
    SandboxPolicy,
    ip_block_reason,
)


@pytest.fixture(autouse=True)
def _deadline():
    signal.signal(signal.SIGALRM, lambda *_: pytest.fail("test exceeded 60s deadline"))
    signal.alarm(60)
    yield
    signal.alarm(0)


class _Origin(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = f"origin saw {self.path} host={self.headers.get('Host')} pa={self.headers.get('Proxy-Authorization')}".encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(n)
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


class _Echo(socketserver.BaseRequestHandler):
    def handle(self):
        data = self.request.recv(1024)
        self.request.sendall(b"echo:" + data)


@pytest.fixture
def origin():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Origin)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def echo():
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _Echo)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


DNS = {"allowed.test": ["127.0.0.1"], "api.allowed.test": ["127.0.0.1"], "evil.test": ["10.0.0.5"],
       "metadata.test": ["169.254.169.254"], "mixed.test": ["127.0.0.1", "192.168.1.1"]}


def resolver(host, port):
    if host not in DNS:
        raise socket.gaierror("no such host")
    return DNS[host]


def make_proxy(ports, *, mode=NetworkMode.ALLOWLIST, domains=("allowed.test", "*.allowed.test", "evil.test", "metadata.test", "mixed.test"),
               cidrs=("127.0.0.1/32",), **kw):
    pol = SandboxPolicy(network=NetworkPolicy(mode=mode, allowed_domains=domains, allowed_ports=tuple(ports), **kw.pop("net", {})),
                        escape_hatches=EscapeHatches(allowed_private_cidrs=cidrs))
    return EgressProxy(pol, resolver=resolver, **kw)


def opener(proxy, cmd="cmd_test1"):
    url = proxy.url(cmd)
    return urllib.request.build_opener(urllib.request.ProxyHandler({"http": url, "https": url}))


def wait_events(proxy, n=1, cmd=None):
    for _ in range(200):
        evs = proxy.events(cmd)
        if len(evs) >= n:
            return evs
        threading.Event().wait(0.02)
    return proxy.events(cmd)


def raw(proxy, data: bytes) -> bytes:
    with socket.create_connection(("127.0.0.1", proxy.port), timeout=10) as s:
        s.sendall(data)
        chunks = []
        while True:
            c = s.recv(65536)
            if not c:
                break
            chunks.append(c)
    return b"".join(chunks)


def test_allowlisted_http_request_is_forwarded_and_audited(origin):
    with make_proxy([origin]) as proxy:
        body = opener(proxy).open(f"http://allowed.test:{origin}/hello?x=1", timeout=10).read().decode()
        assert "origin saw /hello?x=1" in body and f"host=allowed.test:{origin}" in body
        assert "pa=None" in body  # Proxy-Authorization is stripped before the origin
        ev = wait_events(proxy, 1, "cmd_test1")
        assert len(ev) == 1 and ev[0].decision == "allow" and ev[0].rule == "allowed_domains:allowed.test"
        assert ev[0].resolved_ip == "127.0.0.1" and ev[0].bytes_down > len(body) and ev[0].bytes_up > 0
        st = proxy.stats()  # recorded together with the event
        assert st.allowed == 1 and st.denied == 0 and st.per_host["allowed.test"]["requests"] == 1


def test_post_body_streams_through(origin):
    with make_proxy([origin]) as proxy:
        req = urllib.request.Request(f"http://api.allowed.test:{origin}/up", data=b"x" * 5000, method="POST")
        assert opener(proxy).open(req, timeout=10).read() == b"x" * 5000
        assert wait_events(proxy)[0].bytes_up >= 5000


def test_non_allowlisted_domain_is_denied_with_feedback(origin):
    with make_proxy([origin]) as proxy:
        with pytest.raises(urllib.error.HTTPError) as ei:
            opener(proxy, "cmd_x").open(f"http://unknown.test:{origin}/", timeout=10)
        assert ei.value.code == 403
        assert ei.value.headers["X-Sandbox-Violation"] == "not-in-allowlist"
        assert b"Blocked by sandbox egress policy" in ei.value.read()
        (ev,) = proxy.events("cmd_x")
        assert ev.decision == "deny" and ev.host == "unknown.test"


@pytest.mark.parametrize("host,rule", [("evil.test", "private-ip:10.0.0.5"), ("metadata.test", "metadata-ip:169.254.169.254"),
                                       ("mixed.test", "private-ip:192.168.1.1")])
def test_allowlisted_name_resolving_to_private_ip_is_blocked(origin, host, rule):
    with make_proxy([origin]) as proxy:
        resp = raw(proxy, f"GET http://{host}:{origin}/ HTTP/1.1\r\nHost: {host}\r\n\r\n".encode())
        assert resp.startswith(b"HTTP/1.1 403") and rule.encode() in resp
        assert wait_events(proxy)[0].rule == rule


def test_loopback_blocked_without_admin_cidr(origin):
    with make_proxy([origin], cidrs=()) as proxy:
        resp = raw(proxy, f"GET http://allowed.test:{origin}/ HTTP/1.1\r\n\r\n".encode())
        assert resp.startswith(b"HTTP/1.1 403") and b"loopback-ip:127.0.0.1" in resp
        resp = raw(proxy, f"GET http://169.254.169.254:{origin}/latest/meta-data HTTP/1.1\r\n\r\n".encode())
        assert resp.startswith(b"HTTP/1.1 403")  # IP literal: not in the allowlist either


def test_connect_tunnel_relays_bytes_and_counts(echo):
    with make_proxy([echo]) as proxy:
        with socket.create_connection(("127.0.0.1", proxy.port), timeout=10) as s:
            s.sendall(f"CONNECT allowed.test:{echo} HTTP/1.1\r\nHost: allowed.test:{echo}\r\n\r\n".encode())
            head = s.recv(1024)
            assert head.startswith(b"HTTP/1.1 200")
            s.sendall(b"ping")
            assert s.recv(1024) == b"echo:ping"
        ev = wait_events(proxy)[0]
        assert ev.protocol == "connect" and ev.decision == "allow" and ev.bytes_up == 4 and ev.bytes_down == 9


def test_port_not_allowed_and_denied_domain_wins(echo):
    with make_proxy([443], net={"denied_domains": ("api.allowed.test",)}) as proxy:
        assert b"port-not-allowed" in raw(proxy, f"CONNECT allowed.test:{echo} HTTP/1.1\r\n\r\n".encode())
        assert b"denied_domains:api.allowed.test" in raw(proxy, b"CONNECT api.allowed.test:443 HTTP/1.1\r\n\r\n")


def test_socks5_connect_with_command_attribution(echo):
    with make_proxy([echo]) as proxy, socket.create_connection(("127.0.0.1", proxy.port), timeout=10) as s:
        s.sendall(b"\x05\x01\x02")
        assert s.recv(2) == b"\x05\x02"
        user = b"cmd_socks"
        s.sendall(b"\x01" + bytes([len(user)]) + user + b"\x01x")
        assert s.recv(2) == b"\x01\x00"
        host = b"allowed.test"
        s.sendall(b"\x05\x01\x00\x03" + bytes([len(host)]) + host + struct.pack("!H", echo))
        assert s.recv(10)[1] == 0
        s.sendall(b"hi")
        assert s.recv(100) == b"echo:hi"
    with make_proxy([echo]) as proxy, socket.create_connection(("127.0.0.1", proxy.port), timeout=10) as s:
        s.sendall(b"\x05\x01\x00")
        assert s.recv(2) == b"\x05\x00"
        s.sendall(b"\x05\x01\x00\x03\x09evil.test" + struct.pack("!H", echo))
        assert s.recv(10)[1] == 0x02  # connection not allowed by ruleset
        assert wait_events(proxy)[0].rule == "private-ip:10.0.0.5"


def test_limited_mode_is_read_only(origin):
    with make_proxy([origin], mode=NetworkMode.LIMITED) as proxy:
        assert opener(proxy).open(f"http://allowed.test:{origin}/r", timeout=10).status == 200
        req = urllib.request.Request(f"http://allowed.test:{origin}/w", data=b"exfil", method="POST")
        with pytest.raises(urllib.error.HTTPError) as ei:
            opener(proxy).open(req, timeout=10)
        assert ei.value.headers["X-Sandbox-Violation"] == "limited-mode:method-POST"
        assert b"limited-mode:https-and-socks-require-mitm" in raw(proxy, f"CONNECT allowed.test:{origin} HTTP/1.1\r\n\r\n".encode())


def test_interactive_decider_is_consulted_once_per_host(origin):
    calls = []

    def decider(host, port, cmd):
        calls.append(host)
        return host == "allowed.test"

    with make_proxy([origin], domains=(), decider=decider, net={"ask_on_unknown": True}) as proxy:
        for _ in range(2):
            assert opener(proxy).open(f"http://allowed.test:{origin}/", timeout=10).status == 200
        assert calls == ["allowed.test"]
        assert wait_events(proxy)[0].rule == "user-approved"
    with make_proxy([origin], domains=(), decider=decider) as proxy:  # headless: never asks
        with pytest.raises(urllib.error.HTTPError):
            opener(proxy).open(f"http://allowed.test:{origin}/", timeout=10)
        assert calls == ["allowed.test"]


def test_events_go_to_event_sink_and_unix_socket_listener(origin, tmp_path):
    seen: list[Event] = []

    class Sink:
        def emit(self, event):
            seen.append(event)
            return event

    with make_proxy([origin], tcp=False, unix_socket=True, event_sink=Sink(), session_id="s1") as proxy:
        assert proxy.unix_socket_path and proxy.port == 0
        with socket.socket(socket.AF_UNIX) as s:
            s.settimeout(10)
            s.connect(proxy.unix_socket_path)
            s.sendall(f"GET http://allowed.test:{origin}/u HTTP/1.1\r\n\r\n".encode())
            assert s.recv(4096).startswith(b"HTTP/1.0 200")
        for _ in range(100):
            if seen:
                break
            threading.Event().wait(0.05)
        path = proxy.unix_socket_path
    assert seen[0].session_id == "s1" and seen[0].data["kind"] == "sandbox.egress" and seen[0].data["decision"] == "allow"
    import os
    assert not os.path.exists(path)  # socket removed on stop


def test_bind_must_be_loopback():
    with pytest.raises(ProxyError):
        EgressProxy(NetworkPolicy(mode=NetworkMode.ALLOWLIST), host="0.0.0.0")
    p = EgressProxy(SandboxPolicy(escape_hatches=EscapeHatches(allow_non_loopback_proxy=True)), host="0.0.0.0")
    assert p.allow_non_loopback


def test_ip_block_reasons():
    assert ip_block_reason("8.8.8.8") is None
    assert ip_block_reason("169.254.169.254") == "metadata-ip:169.254.169.254"
    assert ip_block_reason("::ffff:10.1.2.3") == "private-ip:10.1.2.3"
    assert ip_block_reason("100.64.0.1").startswith("private-ip")
    assert ip_block_reason("::1") == "loopback-ip:::1"
    assert ip_block_reason("fd00:ec2::254") == "metadata-ip:fd00:ec2::254"
    assert ip_block_reason("10.0.0.1", ["10.0.0.0/8"]) is None


def test_network_none_denies_everything(origin):
    with make_proxy([origin], mode=NetworkMode.NONE) as proxy:
        assert b"network-mode:none" in raw(proxy, f"GET http://allowed.test:{origin}/ HTTP/1.1\r\n\r\n".encode())
