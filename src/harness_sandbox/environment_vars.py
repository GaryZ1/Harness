"""Build the environment a sandboxed process sees: allowlist, secret scrubbing, proxy injection.

Research basis: "Secrets inside the box" is a named pitfall (§6.8: "tokens in environment
variables"); checklist §6.7 item 9 says nothing long-lived goes inside the sandbox and model API
keys are injected harness-side, never through sandbox env vars. The egress proxy is wired in with
``HTTP(S)_PROXY`` / ``ALL_PROXY`` so ordinary tools (curl, pip, npm, git-over-https) use it.

Order of precedence (later wins): host vars matching ``env.allow`` -> caller ``env`` -> ``env.set``
-> sandbox-controlled vars (proxy, ``TMPDIR``). Every name matching ``env.deny`` and every value the
:class:`~harness_memory.security.SecretRedactor` flags is dropped at the end, whatever its source,
unless an admin listed the name in ``escape_hatches.env_secret_passthrough``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

from harness_memory.security import SecretRedactor

from .policy import SandboxPolicy, env_name_matches

_REDACTOR = SecretRedactor()


@dataclass(frozen=True)
class EnvBuild:
    env: dict[str, str]
    scrubbed: tuple[str, ...] = field(default_factory=tuple)   # names removed (never values)


def proxy_env(url: str, *, no_proxy: str = "localhost,127.0.0.1,::1") -> dict[str, str]:
    """Proxy variables in both cases, plus ``ALL_PROXY`` (the proxy also speaks SOCKS5)."""
    socks = url.replace("http://", "socks5h://", 1)
    return {
        "HTTP_PROXY": url, "HTTPS_PROXY": url, "http_proxy": url, "https_proxy": url,
        "ALL_PROXY": socks, "all_proxy": socks, "NO_PROXY": no_proxy, "no_proxy": no_proxy,
        "PIP_PROXY": url, "npm_config_proxy": url, "npm_config_https_proxy": url,
    }


def build_env(
    policy: SandboxPolicy,
    *,
    host_env: Mapping[str, str] | None = None,
    extra: Mapping[str, str] | None = None,
    sandbox_vars: Mapping[str, str] | None = None,
) -> EnvBuild:
    host = dict(os.environ if host_env is None else host_env)
    ep = policy.env
    passthrough = set(policy.escape_hatches.env_secret_passthrough)
    env: dict[str, str] = {k: v for k, v in host.items() if env_name_matches(k, ep.allow)}
    env.update({str(k): str(v) for k, v in (extra or {}).items()})
    env.update(ep.set)
    controlled = dict(sandbox_vars or {})
    scrubbed: list[str] = []
    for name in list(env):
        if name in passthrough or name in controlled:
            continue
        value = env[name]
        if env_name_matches(name, ep.deny) or (ep.scrub_secret_values and _REDACTOR.contains_secret(f"{value}")):
            scrubbed.append(name)
            del env[name]
    env.update(controlled)
    env.setdefault("PATH", os.defpath)
    return EnvBuild(env, tuple(sorted(scrubbed)))
