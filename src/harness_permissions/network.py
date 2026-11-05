"""Network-target normalisation: URL -> host, IDNA, private-address detection, domain allowlists.

Egress policy belongs to the sandbox proxy (research §6, §7.3 technique 26); this module is the
*policy-side* view used to classify a call before it runs:

* hosts are extracted with WHATWG-like leniency (backslashes become slashes, userinfo is
  stripped) so ``https://github.com@evil.com/`` and ``https://evil.com\\@github.com`` resolve to
  the host the client would really contact;
* hosts are lower-cased, trailing dots removed and IDNA-encoded, so a Cyrillic look-alike of an
  allowlisted domain does **not** match it;
* "Private/LAN destinations are always blocked": loopback, RFC 1918, link-local (incl. the cloud
  metadata endpoint ``169.254.169.254``), CGNAT, unique-local IPv6 and ``*.localhost`` /
  ``*.internal`` names are reported as private. Numeric IPv4 forms (``2130706433``, ``0x7f.1``)
  are decoded with ``inet_aton`` semantics. DNS rebinding (a public name resolving to a private
  IP) can only be stopped by the egress proxy, which resolves names.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import urllib.parse
from typing import Iterable

_URL = re.compile(r"(?i)\b(?:https?|wss?|ftp|sftp|ssh|git|file)://[^\s'\"<>]+")
_HOSTISH = re.compile(r"^(?:[A-Za-z0-9-_]+\.)+[A-Za-z][A-Za-z0-9-]{1,62}(?::\d+)?(?:/.*)?$")
_PRIVATE_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".home.arpa", ".intranet")
_METADATA_HOSTS = {"metadata.google.internal", "metadata", "instance-data", "metadata.azure.internal"}


def normalize_host(host: str) -> str:
    """Lower-case, strip brackets/trailing dot, IDNA-encode. Returns '' for garbage."""
    h = host.strip().strip("[]").rstrip(".").lower()
    if not h:
        return ""
    try:
        ipaddress.ip_address(h)
        return h
    except ValueError:
        pass
    try:
        return h.encode("idna").decode("ascii")
    except UnicodeError:
        # Mixed-script or otherwise un-encodable: keep a punycode-ish marker so it never matches.
        return "xn--invalid-" + h.encode("unicode_escape").decode("ascii")


def host_from_url(url: str) -> str:
    """Host a client would contact for ``url`` ('' if none)."""
    u = url.strip().replace("\\", "/")
    if "://" not in u:
        u = "http://" + u
    try:
        parts = urllib.parse.urlsplit(u)
        host = parts.hostname or ""
    except ValueError:
        return ""
    return normalize_host(host)


def extract_urls(text: str) -> list[str]:
    return [m.group(0).rstrip(").,;") for m in _URL.finditer(text)]


def looks_like_host(token: str) -> bool:
    return bool(_HOSTISH.match(token))


def _numeric_ipv4(host: str) -> ipaddress.IPv4Address | None:
    if not re.fullmatch(r"[0-9a-fA-Fx.]+", host):
        return None
    try:
        return ipaddress.IPv4Address(socket.inet_aton(host))
    except OSError:
        return None


def is_private_host(host: str) -> bool:
    """True for loopback, private, link-local, CGNAT, reserved and local-only names."""
    h = normalize_host(host)
    if not h:
        return False
    if h in {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"} or h in _METADATA_HOSTS:
        return True
    if h.endswith(_PRIVATE_SUFFIXES):
        return True
    addr: ipaddress.IPv4Address | ipaddress.IPv6Address | None
    try:
        addr = ipaddress.ip_address(h)
    except ValueError:
        addr = _numeric_ipv4(h)
    if addr is None:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    cgnat = isinstance(addr, ipaddress.IPv4Address) and addr in ipaddress.IPv4Network("100.64.0.0/10")
    return bool(addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved or addr.is_unspecified or addr.is_multicast or cgnat)


def domain_matches(host: str, pattern: str) -> bool:
    """``example.com`` matches exactly; ``*.example.com`` matches strict subdomains only."""
    h = normalize_host(host)
    p = pattern.strip().lower()
    if p.startswith("domain:"):
        p = p[len("domain:"):]
    if p == "*":
        return bool(h)
    if p.startswith("*."):
        base = normalize_host(p[2:])
        return bool(base) and h.endswith("." + base)
    return bool(h) and h == normalize_host(p)


def any_domain_matches(host: str, patterns: Iterable[str]) -> bool:
    return any(domain_matches(host, p) for p in patterns)
