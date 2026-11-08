"""Model-facing sandbox description and the MCP ``_meta`` sandbox state.

Research basis (§6.3 techniques 8 and 19, checklist §6.7 items 6 and 13):

* Codex injects a per-mode description (``permissions/sandbox_mode/*.md``) so the model plans inside
  the boundary instead of probing it. :func:`describe_sandbox` renders a **stable** text (sorted, no
  timestamps or per-command values) so it does not break the prompt cache. It is context, never
  enforcement ("CLAUDE.md content is context, not enforced configuration").
* Codex lets MCP servers opt into ``codex/sandbox-state`` and receive the policy in ``_meta``;
  :func:`sandbox_state_meta` builds the equivalent payload under ``harness/sandbox-state``.
* :func:`prepare_mcp_server` launches a stdio MCP server inside the same sandbox as the shell
  (srt's ``"command": "srt"`` wrapping), returning a :class:`PreparedCommand` for ``Popen``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .detect import BackendChoice
from .policy import NetworkMode, SandboxMode, SandboxPolicy
from .runners.base import BaseRunner, PreparedCommand

_MODE_TEXT = {
    SandboxMode.READ_ONLY: "Commands run in a read-only sandbox: you can read files but cannot modify the workspace.",
    SandboxMode.WORKSPACE_WRITE: "Commands run in a sandbox that can write only inside the workspace and the session tmp directory.",
    SandboxMode.EXTERNAL_SANDBOX: "Commands run inside an externally provided container/VM; the workspace is writable.",
    SandboxMode.FULL_ACCESS: "Commands run WITHOUT a sandbox (full access). Be careful with destructive operations.",
}
_NET_TEXT = {
    NetworkMode.NONE: "Network access is disabled.",
    NetworkMode.ALLOWLIST: "Network access goes through an egress proxy that only allows these domains: {domains}.",
    NetworkMode.LIMITED: "Network access is read-only (plain HTTP GET/HEAD/OPTIONS) to these domains: {domains}.",
    NetworkMode.OPEN: "Network access is unrestricted.",
}


def describe_sandbox(policy: SandboxPolicy, *, workspace: str, choice: BackendChoice | None = None) -> str:
    net = policy.network
    domains = ", ".join(sorted(net.allowed_domains)) or "(none)"
    lines = [
        "# Sandbox",
        _MODE_TEXT[policy.mode],
        f"Workspace: {workspace}",
        _NET_TEXT[net.mode].format(domains=domains),
        "Protected (read-only even inside the workspace): " + ", ".join(sorted(policy.filesystem.read_only_subpaths)) + ".",
        "Credential locations are unreadable (for example ~/.ssh, ~/.aws, **/.env).",
        f"Commands time out after {policy.limits.default_timeout_s:g}s by default (max {policy.limits.max_timeout_s:g}s).",
        "If a command is blocked you will see a <sandbox_violations> block: choose an in-boundary alternative or ask the "
        "user for a scoped exception. Do not try to work around the sandbox.",
    ]
    if choice is not None and not choice.isolated:
        lines.append("Note: no OS sandbox is active on this host; the restrictions above are partly advisory.")
    return "\n".join(lines) + "\n"


def sandbox_state_meta(policy: SandboxPolicy, *, choice: BackendChoice | None = None) -> dict[str, Any]:
    return {"harness/sandbox-state": {
        "mode": policy.mode.value,
        "network": policy.network.mode.value,
        "allowed_domains": sorted(policy.network.allowed_domains) if policy.network.mode.uses_proxy else [],
        "backend": choice.backend if choice else None,
        "isolated": choice.isolated if choice else None,
        "policy_fingerprint": policy.fingerprint(),
    }}


def prepare_mcp_server(runner: BaseRunner, command: str, args: Sequence[str] = (), *, env: Mapping[str, str] | None = None,
                       cwd: str | None = None) -> PreparedCommand:
    """Sandboxed argv/env for a stdio MCP server; spawn with ``Popen(p.argv, stdin=PIPE, stdout=PIPE,
    **p.popen_kwargs())``. Restart it whenever ``policy.fingerprint()`` changes."""
    return runner.prepare([command, *args], cwd=cwd, env=env)
