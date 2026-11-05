"""Egress proxy: the only route out of the sandbox (HTTP forward, HTTPS ``CONNECT`` and SOCKS5).

Research basis (§6.2, §6.3 technique 4, checklist §6.7 item 4, pitfalls §6.8):

* "Both layers are required": a filesystem sandbox alone still lets secrets leave over the network.
  srt and ``codex-network-proxy`` both run HTTP and SOCKS5 proxies *outside* the sandbox and remove
  the sandbox's own network. This proxy serves both protocols on one listener (the first byte
  ``0x05`` selects SOCKS5), on loopback TCP and/or a Unix socket that runners bind-mount into the
  sandbox (srt's Linux design).
* **Domain allowlist, deny by default**: exact names and ``*.example.com`` subdomain wildcards;
  ``denied_domains`` always wins. Unknown domains are denied headless, or passed to an interactive
  ``decider`` (Claude Code "asks the user about new domains").
* **Private-IP blocking after DNS resolution** (Codex): an allowlisted name that resolves to a
  private, loopback, link-local (incl. the ``169.254.169.254`` metadata endpoint), CGNAT or otherwise
  non-global address is blocked. The proxy connects to the exact address it vetted, so a second DNS
  answer cannot rebind the connection. Admins can exempt CIDRs (escape hatch).
* **Loopback-only binds** unless ``allow_non_loopback_proxy`` (Codex
  ``dangerously_allow_non_loopback_proxy``).
* **Audit**: one :class:`ProxyEvent` per request with decision, rule, resolved IP and byte counts,
  kept in a bounded ring and emitted to an :class:`~harness_core.types.EventSink`. Requests are
  attributed to commands through the proxy URL's username (``http://<command_id>:sandbox@...``),
  mirroring srt's ``commandId`` violation attribution.
* ``limited`` mode is Codex's read-only network: plain-HTTP ``GET``/``HEAD``/``OPTIONS`` only.
  Codex enforces it inside HTTPS with a managed MITM CA; that needs a TLS stack beyond the stdlib,
  so here ``CONNECT`` and SOCKS fail closed in ``limited`` mode (documented deviation).
"""

from __future__ import annotations

import base64
import collections
import ipaddress
import logging
import os
import selectors
import shutil
import socket
import socketserver
import struct
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Sequence

from harness_core.types import Clock, Event, EventSink, EventType, utcnow

from .errors import ProxyError
from .policy import NetworkMode, NetworkPolicy, SandboxPolicy, normalise_host

log = logging.getLogger("harness_sandbox.proxy")

Resolver = Callable[[str, int], Sequence[str]]
Decider = Callable[[str, int, str], bool]

MAX_HEADER_BYTES = 64 * 1024
_HOP_BY_HOP = {"proxy-authorization", "proxy-connection", "connection", "keep-alive", "te", "trailer", "upgrade", "proxy-authenticate"}
LIMITED_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
METADATA_IPS = frozenset({"169.254.169.254", "fd00:ec2::254", "169.254.170.2"})


def system_resolver(host: str, port: int) -> list[str]:
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(str(i[4][0]) for i in infos))


def ip_block_reason(ip: str, allowed_cidrs: Iterable[str] = ()) -> str | None:
    """Why ``ip`` must not be reached, or ``None`` if it is a public address."""
    addr = ipaddress.ip_address(ip.split("%", 1)[0])
    for cidr in allowed_cidrs:
        if addr in ipaddress.ip_network(cidr, strict=False):
            return None
    embedded = None
    if isinstance(addr, ipaddress.IPv6Address):
        embedded = addr.ipv4_mapped or addr.sixtofour or (addr.teredo[1] if addr.teredo else None)
    if embedded is not None:
        return ip_block_reason(str(embedded), allowed_cidrs)
    if str(addr) in METADATA_IPS:
        return f"metadata-ip:{addr}"
    if addr.is_loopback:
        return f"loopback-ip:{addr}"
    if addr.is_link_local:
        return f"link-local-ip:{addr}"
    if addr.is_multicast or addr.is_unspecified or addr.is_reserved:
        return f"reserved-ip:{addr}"
    if addr.is_private or not addr.is_global:
        return f"private-ip:{addr}"
    return None


@dataclass(frozen=True)
class Decision:
    allowed: bool
    rule: str
    ip: str = ""
    status: int = 200


@dataclass(frozen=True)
class ProxyEvent:
    ts: str
    command_id: str
    protocol: str            # http | connect | socks5
    method: str
    host: str
    port: int
    decision: str            # allow | deny
    rule: str
    resolved_ip: str = ""
    status: int = 0
    bytes_up: int = 0
    bytes_down: int = 0
    duration_s: float = 0.0


@dataclass
class ProxyStats:
    requests: int = 0
    allowed: int = 0
    denied: int = 0
    bytes_up: int = 0
    bytes_down: int = 0
    per_host: dict[str, dict[str, int]] = field(default_factory=dict)


class EgressProxy:
    """Threaded policy proxy. Use as a context manager or call :meth:`start` / :meth:`stop`."""

    def __init__(
        self,
        policy: SandboxPolicy | NetworkPolicy,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        tcp: bool = True,
        unix_socket: str | bool | None = None,
        resolver: Resolver | None = None,
        decider: Decider | None = None,
        event_sink: EventSink | None = None,
        session_id: str = "sandbox",
        connect_timeout: float = 10.0,
        idle_timeout: float = 120.0,
        max_events: int = 10_000,
        clock: Clock = utcnow,
    ) -> None:
        if isinstance(policy, SandboxPolicy):
            self.network, hatches = policy.network, policy.escape_hatches
        else:
            self.network, hatches = policy, None
        self.allow_non_loopback = bool(hatches and hatches.allow_non_loopback_proxy)
        self.allowed_private_cidrs = tuple(hatches.allowed_private_cidrs) if hatches else ()
        if tcp and not self.allow_non_loopback and not _is_loopback(host):
            raise ProxyError(f"refusing to bind the egress proxy to non-loopback address {host!r}")
        self.host, self._port, self.tcp = host, port, tcp
        self._unix_request = unix_socket
        self.unix_socket_path: str | None = None
        self._private_dir: str | None = None
        self.resolver = resolver or system_resolver
        self.decider = decider
        self.event_sink = event_sink
        self.session_id = session_id
        self.connect_timeout = connect_timeout
        self.idle_timeout = idle_timeout
        self.clock = clock
        self._events: collections.deque[ProxyEvent] = collections.deque(maxlen=max_events)
        self._stats = ProxyStats()
        self._lock = threading.Lock()
        self._asked: dict[str, bool] = {}
        self._servers: list[socketserver.BaseServer] = []
        self._threads: list[threading.Thread] = []

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "EgressProxy":
        if self._servers:
            return self
        handler = _make_handler(self)
        try:
            if self.tcp:
                cls = _ThreadingTCPServer6 if ":" in self.host else _ThreadingTCPServer
                srv = cls((self.host, self._port), handler)
                self._port = srv.server_address[1]
                self._servers.append(srv)
            if self._unix_request:
                if self._unix_request is True:
                    self._private_dir = tempfile.mkdtemp(prefix="harness-proxy-")
                    os.chmod(self._private_dir, 0o700)
                    path = os.path.join(self._private_dir, "proxy.sock")
                else:
                    path = str(self._unix_request)
                if os.path.exists(path):
                    os.unlink(path)
                usrv = _ThreadingUnixServer(path, handler)
                os.chmod(path, 0o600)
                self.unix_socket_path = path
                self._servers.append(usrv)
        except OSError as exc:
            self.stop()
            raise ProxyError(f"egress proxy failed to start: {exc}") from exc
        for srv in self._servers:
            t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.2}, name="harness-egress-proxy", daemon=True)
            t.start()
            self._threads.append(t)
        return self

    def stop(self) -> None:
        for srv in self._servers:
            try:
                srv.shutdown()
            except Exception:  # pragma: no cover - shutdown before serve_forever
                pass
            srv.server_close()
        for t in self._threads:
            t.join(timeout=5)
        self._servers.clear()
        self._threads.clear()
        if self.unix_socket_path and os.path.exists(self.unix_socket_path):
            os.unlink(self.unix_socket_path)
        if self._private_dir:
            shutil.rmtree(self._private_dir, ignore_errors=True)
            self._private_dir = None

    def __enter__(self) -> "EgressProxy":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @property
    def running(self) -> bool:
        return bool(self._servers)

    @property
    def port(self) -> int:
        return self._port

    def url(self, command_id: str = "", *, host: str | None = None, port: int | None = None) -> str:
        """Proxy URL for ``HTTP(S)_PROXY``; the username tags requests with ``command_id``."""
        h = host or (f"[{self.host}]" if ":" in self.host else self.host)
        auth = f"{command_id}:sandbox@" if command_id else ""
        return f"http://{auth}{h}:{port or self._port}"

    # ------------------------------------------------------------------ audit
    def events(self, command_id: str | None = None) -> list[ProxyEvent]:
        with self._lock:
            evs = list(self._events)
        return evs if command_id is None else [e for e in evs if e.command_id == command_id]

    def stats(self) -> ProxyStats:
        with self._lock:
            s = self._stats
            return ProxyStats(s.requests, s.allowed, s.denied, s.bytes_up, s.bytes_down, {k: dict(v) for k, v in s.per_host.items()})

    def _record(self, ev: ProxyEvent) -> None:
        with self._lock:
            self._events.append(ev)
            s = self._stats
            s.requests += 1
            if ev.decision == "allow":
                s.allowed += 1
            else:
                s.denied += 1
            s.bytes_up += ev.bytes_up
            s.bytes_down += ev.bytes_down
            ph = s.per_host.setdefault(ev.host, {"requests": 0, "bytes_up": 0, "bytes_down": 0})
            ph["requests"] += 1
            ph["bytes_up"] += ev.bytes_up
            ph["bytes_down"] += ev.bytes_down
        level = logging.INFO if ev.decision == "allow" else logging.WARNING
        log.log(level, "egress %s %s %s:%d -> %s (%s)", ev.protocol, ev.method, ev.host, ev.port, ev.decision, ev.rule)
        if self.event_sink is not None:
            try:
                self.event_sink.emit(Event(self.session_id, EventType.CUSTOM, {"kind": "sandbox.egress", **asdict(ev)}))
            except Exception:  # never let a broken sink take down egress enforcement
                log.exception("egress audit sink failed")

    # ------------------------------------------------------------------ policy
    def decide(self, host: str, port: int, *, method: str = "CONNECT", protocol: str = "connect", command_id: str = "") -> Decision:
        net = self.network
        h = normalise_host(host)
        if net.mode is NetworkMode.NONE:
            return Decision(False, "network-mode:none", status=403)
        if port not in net.allowed_ports:
            return Decision(False, f"port-not-allowed:{port}", status=403)
        if net.mode is NetworkMode.LIMITED:
            if protocol != "http":
                return Decision(False, "limited-mode:https-and-socks-require-mitm", status=403)
            if method.upper() not in LIMITED_METHODS:
                return Decision(False, f"limited-mode:method-{method.upper()}", status=403)
        if net.mode is not NetworkMode.OPEN:
            allowed, rule = net.domain_decision(h)
            if allowed is False:
                return Decision(False, rule, status=403)
            if allowed is None:
                if not (net.ask_on_unknown and self.decider):
                    return Decision(False, "not-in-allowlist", status=403)
                with self._lock:
                    cached = self._asked.get(h)
                if cached is None:
                    try:
                        cached = bool(self.decider(h, port, command_id))
                    except Exception:
                        log.exception("egress decider failed; denying")
                        cached = False
                    with self._lock:
                        self._asked[h] = cached
                if not cached:
                    return Decision(False, "user-denied", status=403)
                rule = "user-approved"
        else:
            rule = "network-mode:open"
        try:
            ips = [h] if _is_ip(h) else list(self.resolver(h, port))
        except (OSError, UnicodeError) as exc:
            return Decision(False, f"dns-failure:{type(exc).__name__}", status=502)
        if not ips:
            return Decision(False, "dns-failure:no-addresses", status=502)
        for ip in ips:  # any non-global answer blocks the name entirely (rebinding defence)
            reason = ip_block_reason(ip, self.allowed_private_cidrs)
            if reason:
                return Decision(False, reason, ip=ip, status=403)
        return Decision(True, rule, ip=ips[0])

    def _connect(self, ip: str, port: int) -> socket.socket:
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        s = socket.socket(family, socket.SOCK_STREAM)
        s.settimeout(self.connect_timeout)
        try:
            s.connect((ip, port))
        except OSError:
            s.close()
            raise
        s.settimeout(None)
        return s

    def _relay(self, a: socket.socket, b: socket.socket, initial_up: bytes = b"", *, end_on_upstream_eof: bool = False) -> tuple[int, int]:
        """Pump bytes both ways until both sides close (half-close aware) or the idle timeout passes.

        ``end_on_upstream_eof`` ends the relay as soon as the origin closes (plain HTTP with
        ``Connection: close``), so a client that keeps its socket open cannot pin a thread."""
        up = down = 0
        if initial_up:
            b.sendall(initial_up)
            up += len(initial_up)
        sel = selectors.DefaultSelector()
        sel.register(a, selectors.EVENT_READ, "up")
        sel.register(b, selectors.EVENT_READ, "down")
        open_dirs = {"up", "down"}
        try:
            while open_dirs:
                ready = sel.select(timeout=self.idle_timeout)
                if not ready:
                    break
                for key, _ in ready:
                    src = key.fileobj
                    dst = b if key.data == "up" else a
                    try:
                        data = src.recv(65536)  # type: ignore[union-attr]
                    except OSError:
                        data = b""
                    if not data:
                        if key.data == "down" and end_on_upstream_eof:
                            return up, down
                        sel.unregister(src)
                        open_dirs.discard(key.data)
                        try:
                            dst.shutdown(socket.SHUT_WR)
                        except OSError:
                            pass
                        continue
                    try:
                        dst.sendall(data)
                    except OSError:
                        return up, down
                    if key.data == "up":
                        up += len(data)
                    else:
                        down += len(data)
        finally:
            sel.close()
        return up, down


# ----------------------------------------------------------------------------- servers & handler


class _ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128


class _ThreadingTCPServer6(_ThreadingTCPServer):
    address_family = socket.AF_INET6


class _ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    request_queue_size = 128


def _is_ip(h: str) -> bool:
    try:
        ipaddress.ip_address(h)
        return True
    except ValueError:
        return False


def _is_loopback(host: str) -> bool:
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _read_head(sock: socket.socket) -> tuple[bytes, bytes]:
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(8192)
        if not chunk:
            break
        buf += chunk
        if len(buf) > MAX_HEADER_BYTES:
            raise ValueError("header too large")
    head, sep, rest = buf.partition(b"\r\n\r\n")
    if not sep:
        raise ValueError("incomplete request head")
    return head, rest


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    out = b""
    while len(out) < n:
        chunk = sock.recv(n - len(out))
        if not chunk:
            raise ConnectionError("short read")
        out += chunk
    return out


def _command_from_auth(value: str) -> str:
    try:
        scheme, _, token = value.partition(" ")
        if scheme.lower() != "basic":
            return ""
        user = base64.b64decode(token.strip()).decode("utf-8", "replace").split(":", 1)[0]
        return user[:64] if all(c.isalnum() or c in "_-." for c in user) else ""
    except (ValueError, UnicodeError):
        return ""


def _split_hostport(value: str, default_port: int) -> tuple[str, int]:
    v = value.strip()
    if v.startswith("["):
        host, _, rest = v[1:].partition("]")
        port = int(rest[1:]) if rest.startswith(":") else default_port
        return host, port
    if v.count(":") == 1:
        host, _, p = v.partition(":")
        return host, int(p)
    return v, default_port


def _make_handler(proxy: EgressProxy) -> type[socketserver.BaseRequestHandler]:
    class Handler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            conn: socket.socket = self.request
            conn.settimeout(proxy.idle_timeout)
            try:
                first = conn.recv(1, socket.MSG_PEEK)
                if not first:
                    return
                if first == b"\x05":
                    self._socks5(conn)
                else:
                    self._http(conn)
            except (OSError, ValueError, ConnectionError) as exc:
                log.debug("proxy connection error: %s", exc)

        # -------------------------------------------------------------- HTTP / CONNECT
        def _deny_http(self, conn: socket.socket, d: Decision, host: str) -> None:
            body = (f"Blocked by sandbox egress policy.\nhost: {host}\nrule: {d.rule}\n"
                    "hint: use an allowlisted domain, work offline, or ask the user for a scoped exception.\n").encode()
            reason = "Forbidden" if d.status == 403 else "Bad Gateway"
            conn.sendall(
                f"HTTP/1.1 {d.status} {reason}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {len(body)}\r\n"
                f"X-Sandbox-Violation: {d.rule}\r\nConnection: close\r\n\r\n".encode() + body
            )

        def _http(self, conn: socket.socket) -> None:
            t0 = time.monotonic()
            try:
                head, rest = _read_head(conn)
            except ValueError:
                conn.sendall(b"HTTP/1.1 431 Request Header Fields Too Large\r\nConnection: close\r\nContent-Length: 0\r\n\r\n")
                return
            lines = head.decode("iso-8859-1").split("\r\n")
            parts = lines[0].split(" ")
            if len(parts) != 3:
                conn.sendall(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\nContent-Length: 0\r\n\r\n")
                return
            method, target, version = parts
            headers = [(k.strip(), v.strip()) for k, _, v in (ln.partition(":") for ln in lines[1:] if ln)]
            cmd = next((_command_from_auth(v) for k, v in headers if k.lower() == "proxy-authorization"), "")
            if method.upper() == "CONNECT":
                host, port = _split_hostport(target, 443)
                protocol, path = "connect", ""
            else:
                if not target.lower().startswith("http://"):
                    conn.sendall(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\nContent-Length: 0\r\n\r\n")
                    return
                hostport, slash, path = target[7:].partition("/")
                path = slash + path if slash else "/"
                host, port = _split_hostport(hostport.rpartition("@")[2], 80)
                protocol = "http"
            d = proxy.decide(host, port, method=method, protocol=protocol, command_id=cmd)
            if not d.allowed:
                self._deny_http(conn, d, host)
                proxy._record(ProxyEvent(proxy.clock().isoformat(), cmd, protocol, method, normalise_host(host), port, "deny",
                                         d.rule, d.ip, d.status, 0, 0, round(time.monotonic() - t0, 6)))
                return
            try:
                upstream = proxy._connect(d.ip, port)
            except OSError as exc:
                bad = Decision(False, f"upstream-connect-failed:{type(exc).__name__}", d.ip, 502)
                self._deny_http(conn, bad, host)
                proxy._record(ProxyEvent(proxy.clock().isoformat(), cmd, protocol, method, normalise_host(host), port, "deny",
                                         bad.rule, d.ip, 502, 0, 0, round(time.monotonic() - t0, 6)))
                return
            conn.settimeout(None)
            with upstream:
                if protocol == "connect":
                    conn.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                    up, down = proxy._relay(conn, upstream, rest)
                    status = 200
                else:
                    kept = [(k, v) for k, v in headers if k.lower() not in _HOP_BY_HOP]
                    if not any(k.lower() == "host" for k, _ in kept):
                        kept.insert(0, ("Host", host if port == 80 else f"{host}:{port}"))
                    req = f"{method} {path} {version}\r\n" + "".join(f"{k}: {v}\r\n" for k, v in kept) + "Connection: close\r\n\r\n"
                    up, down = proxy._relay(conn, upstream, req.encode("iso-8859-1") + rest, end_on_upstream_eof=True)
                    status = 0
            proxy._record(ProxyEvent(proxy.clock().isoformat(), cmd, protocol, method, normalise_host(host), port, "allow",
                                     d.rule, d.ip, status, up, down, round(time.monotonic() - t0, 6)))

        # -------------------------------------------------------------- SOCKS5 (RFC 1928 / 1929)
        def _socks5(self, conn: socket.socket) -> None:
            t0 = time.monotonic()
            ver, n = _recv_exact(conn, 2)
            methods = _recv_exact(conn, n)
            cmd = ""
            if 0x02 in methods:
                conn.sendall(b"\x05\x02")
                _, ulen = _recv_exact(conn, 2)
                user = _recv_exact(conn, ulen).decode("utf-8", "replace")
                (plen,) = _recv_exact(conn, 1)
                _recv_exact(conn, plen)
                cmd = user[:64] if all(c.isalnum() or c in "_-." for c in user) else ""
                conn.sendall(b"\x01\x00")
            elif 0x00 in methods:
                conn.sendall(b"\x05\x00")
            else:
                conn.sendall(b"\x05\xff")
                return
            _, command, _, atyp = _recv_exact(conn, 4)
            if atyp == 1:
                host = socket.inet_ntop(socket.AF_INET, _recv_exact(conn, 4))
            elif atyp == 4:
                host = socket.inet_ntop(socket.AF_INET6, _recv_exact(conn, 16))
            elif atyp == 3:
                (ln,) = _recv_exact(conn, 1)
                host = _recv_exact(conn, ln).decode("idna")
            else:
                conn.sendall(b"\x05\x08\x00\x01\x00\x00\x00\x00\x00\x00")
                return
            (port,) = struct.unpack("!H", _recv_exact(conn, 2))
            if command != 1:  # only CONNECT; BIND/UDP ASSOCIATE are refused
                conn.sendall(b"\x05\x07\x00\x01\x00\x00\x00\x00\x00\x00")
                return
            d = proxy.decide(host, port, method="CONNECT", protocol="socks5", command_id=cmd)
            if not d.allowed:
                conn.sendall(b"\x05\x02\x00\x01\x00\x00\x00\x00\x00\x00")
                proxy._record(ProxyEvent(proxy.clock().isoformat(), cmd, "socks5", "CONNECT", normalise_host(host), port, "deny",
                                         d.rule, d.ip, d.status, 0, 0, round(time.monotonic() - t0, 6)))
                return
            try:
                upstream = proxy._connect(d.ip, port)
            except OSError:
                conn.sendall(b"\x05\x05\x00\x01\x00\x00\x00\x00\x00\x00")
                proxy._record(ProxyEvent(proxy.clock().isoformat(), cmd, "socks5", "CONNECT", normalise_host(host), port, "deny",
                                         "upstream-connect-failed", d.ip, 502, 0, 0, round(time.monotonic() - t0, 6)))
                return
            conn.settimeout(None)
            with upstream:
                conn.sendall(b"\x05\x00\x00\x01\x00\x00\x00\x00\x00\x00")
                up, down = proxy._relay(conn, upstream)
            proxy._record(ProxyEvent(proxy.clock().isoformat(), cmd, "socks5", "CONNECT", normalise_host(host), port, "allow",
                                     d.rule, d.ip, 0, up, down, round(time.monotonic() - t0, 6)))

    return Handler
