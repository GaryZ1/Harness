# Egress proxy

Module: `harness_sandbox.proxy`.

## Purpose

`EgressProxy` is the only route out of a sandbox whose own network has been removed. It is a threaded forward proxy. One listener speaks HTTP (absolute-form requests), HTTPS `CONNECT` and SOCKS5 `CONNECT`; a first byte of `0x05` selects SOCKS. It listens on loopback TCP and/or a private Unix socket. For every request it applies the domain allowlist, resolves DNS itself, blocks non-public addresses, connects to the exact address it vetted, relays bytes, and records an audit event with byte counts.

## Research basis

* "Both layers are required" (Anthropic). `srt` and `codex-network-proxy` run HTTP and SOCKS5 proxies outside the sandbox, and `srt` on Linux exposes them through bind-mounted Unix sockets (§6.2, §6.3 technique 4).
* Codex: "Hostnames that resolve to private IPs stay blocked even if allowlisted" (closes the "allowlisted name that points at the LAN" pitfall). The checklist adds link-local and metadata ranges as our addition (§6.7 item 4).
* Non-loopback binds need `dangerously_allow_non_loopback_proxy` (Codex), so here `allow_non_loopback_proxy`.
* Interactive sessions ask about new domains; headless sessions deny (Claude Code; checklist item 4).
* Codex `mode = "limited"` is read-only network (§6.3 technique 5).
* Violations are attributed by command (`srt` `commandId`), and proxy decisions are recorded as events (checklist items 5 and 17).

## API

```python
from harness_sandbox import EgressProxy, SandboxPolicy, NetworkPolicy, NetworkMode

policy = SandboxPolicy(network=NetworkPolicy(mode=NetworkMode.ALLOWLIST,
                                             allowed_domains=("pypi.org", "*.pythonhosted.org"),
                                             denied_domains=("upload.pypi.org",)))
with EgressProxy(policy, unix_socket=True, tcp=False, event_sink=sink, session_id="s1",
                 decider=None) as proxy:              # decider(host, port, command_id) -> bool, if ask_on_unknown
    proxy.unix_socket_path                            # bind-mount this directory into the sandbox
    proxy.url("cmd_123", host="127.0.0.1", port=3128) # http://cmd_123:sandbox@127.0.0.1:3128 (for HTTP(S)_PROXY)
    proxy.decide("files.pythonhosted.org", 443)       # Decision(allowed, rule, ip, status)
    proxy.events("cmd_123")                           # [ProxyEvent(...)]
    proxy.stats()                                     # ProxyStats(requests, allowed, denied, bytes_up, bytes_down, per_host)
```

Runners create and own a proxy automatically when the policy's network mode is `allowlist` or `limited`. Pass `proxy=` to share one proxy across runners, or to inject a resolver in tests. Inside the sandbox, `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY` (`socks5h://`), their lowercase forms, `PIP_PROXY` and the npm proxy variables point at it. `NO_PROXY=localhost,127.0.0.1,::1` refers to the sandbox's *own* loopback, because the sandbox has its own network namespace.

### Decision order

1. Network mode `none`: deny (`network-mode:none`).
2. Port not in `allowed_ports`: deny (`port-not-allowed:<port>`).
3. `limited`: only plain-HTTP `GET`/`HEAD`/`OPTIONS`. `CONNECT` and SOCKS are denied (`limited-mode:...`).
4. `denied_domains` match: deny. `allowed_domains` match (exact, or `*.example.com` for subdomains only): continue. Unknown: deny (`not-in-allowlist`), or ask `decider` once per host when `ask_on_unknown` is set.
5. Resolve (IP literals are used as is). Any answer that is metadata, loopback, link-local, multicast, reserved, unspecified, private, CGNAT or otherwise non-global denies the name entirely. IPv4-mapped, 6to4 and Teredo IPv6 addresses are unwrapped first. Admin `allowed_private_cidrs` are exempt. DNS failure gives 502.
6. Connect to the first vetted address (10 s timeout) and relay (idle timeout 120 s).

Denials return `403` with an `X-Sandbox-Violation: <rule>` header and a plain-text body with a hint (SOCKS reply `0x02`). Runners turn the matching audit events into `network` violations.

## Defaults

| Setting | Default | Source |
|---|---|---|
| Bind | `127.0.0.1`, ephemeral port; or Unix socket in a fresh `0700` dir, socket `0600` | Codex binds loopback (`127.0.0.1:3128` HTTP, `:8081` SOCKS5); ephemeral host port and single listener are our choice |
| Port inside the sandbox | 3128 (via the launcher bridge) | Codex HTTP default |
| Protocols | HTTP forward, `CONNECT`, SOCKS5 `CONNECT` (no-auth or user/pass for command tagging) | `srt`, Codex (HTTP + SOCKS5) |
| Private-IP blocking | on, including `169.254.169.254`, `169.254.170.2`, `fd00:ec2::254`, `100.64/10` | Codex (private/LAN); metadata, link-local and CGNAT are our addition |
| Unknown domains | denied (headless) | `srt`, checklist |
| `limited` mode | plain-HTTP read methods only; HTTPS fails closed | Codex uses an HTTPS MITM CA; **deviation**, see below |
| Header limit | 64 KiB | our choice |
| Connect / idle timeout | 10 s / 120 s | our choice |
| Audit ring | 10,000 events in memory, plus `EventSink` | our choice |

## Invariants

* The proxy never binds a non-loopback address without the admin hatch (`ProxyError` otherwise).
* It connects only to an address that passed the IP check. There is no second resolution, so DNS rebinding cannot swap the target between check and connect.
* `Proxy-Authorization`, `Proxy-Connection` and other hop-by-hop headers are stripped before the origin; requests are forced to `Connection: close`.
* A broken event sink is logged and never disables enforcement.
* Stopping the proxy removes the socket file and its private directory.

## Failure modes

| Situation | Behaviour |
|---|---|
| Client ignores `HTTP_PROXY` | in bwrap, netns-local and containers there is no other route (connection refused or unreachable). In the plain local runner the proxy is advisory |
| Allowlisted host resolves to a mixed public/private set | denied (rebinding defence) |
| Certificate-pinned clients | unaffected (no MITM) |
| HTTP keep-alive / pipelining | one request per connection (`Connection: close`) |
| Non-HTTP TCP (git over ssh, databases) | SOCKS5 `CONNECT` only, subject to the same rules; UDP and BIND are refused |

## Deviations from the research

* **No MITM CA for `limited` mode.** Codex enforces read-only semantics inside HTTPS with a managed CA. Doing that needs a TLS stack beyond the stdlib (certificate generation), so `limited` here fails closed for HTTPS. Use `allowlist` for HTTPS hosts.
* **One listener for HTTP and SOCKS5** instead of two ports. The protocol is sniffed from the first byte.

## Platform support

| Platform | Proxy | Unix-socket listener |
|---|---|---|
| Linux | yes | yes (used by bwrap, netns-local and containers) |
| macOS | yes | yes (Seatbelt uses the TCP loopback port) |
| Windows | TCP only, not tested | no |
