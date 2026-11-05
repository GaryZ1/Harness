"""harness-sandbox: execution environment and sandboxing for agentic harnesses.

"Sandbox the shell, not each tool": one OS sandbox around every command and its whole process
tree, plus an egress proxy for every outbound byte, because "both layers are required"
(``reports/SOTA agentic harness reference.md`` §6).

=====================================  =========================================================
Component                              Module
=====================================  =========================================================
Declarative policy (YAML/JSON)         :mod:`harness_sandbox.policy`
Environment scrubbing                  :mod:`harness_sandbox.environment_vars`
Runners (``CommandRunner``)            :mod:`harness_sandbox.runners` (bubblewrap, seatbelt,
                                       container, local fallback)
In-sandbox launcher / seccomp          :mod:`harness_sandbox._launcher`, :mod:`~.seccomp`
Egress proxy (HTTP, CONNECT, SOCKS5)   :mod:`harness_sandbox.proxy`
Backend detection and choice           :mod:`harness_sandbox.detect`
Execution environments, pools          :mod:`harness_sandbox.environment`
Workspace snapshots                    :mod:`harness_sandbox.snapshot`
Violation feedback                     :mod:`harness_sandbox.violations`
Model-facing description, MCP meta     :mod:`harness_sandbox.describe`
=====================================  =========================================================
"""

from .describe import describe_sandbox, prepare_mcp_server, sandbox_state_meta
from .detect import BackendChoice, BackendReport, choose_backend, create_runner, detect_backends, find_trusted_binary
from .environment import (
    EnvironmentPool,
    ExecutionEnvironment,
    RemoteBackend,
    RemoteEnvironment,
    RunnerEnvironment,
    idempotency_key,
)
from .environment_vars import build_env, proxy_env
from .errors import (
    EscapeHatchError,
    PolicyError,
    ProxyError,
    SandboxCrashedError,
    SandboxError,
    SandboxUnavailableError,
    SnapshotError,
)
from .policy import (
    DEFAULT_ALLOWED_DOMAINS,
    DEFAULT_DENY_READ,
    DEFAULT_READ_ONLY_SUBPATHS,
    CompiledPolicy,
    EnvPolicy,
    EscapeHatches,
    FilesystemPolicy,
    NetworkMode,
    NetworkPolicy,
    PathDecision,
    ResourceLimits,
    SandboxMode,
    SandboxPolicy,
    domain_matches,
    load_policy,
)
from .proxy import EgressProxy, ProxyEvent, ProxyStats, ip_block_reason
from .runners import (
    BaseRunner,
    BubblewrapRunner,
    CommandResult,
    CommandRunner,
    ContainerConfig,
    ContainerRunner,
    LocalRunner,
    PreparedCommand,
    SeatbeltRunner,
    build_bwrap_argv,
    build_container_argv,
    build_seatbelt_profile,
)
from .snapshot import Snapshot, SnapshotDiff, WorkspaceSnapshotter
from .violations import Violation, format_violations

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_ALLOWED_DOMAINS", "DEFAULT_DENY_READ", "DEFAULT_READ_ONLY_SUBPATHS",
    "BackendChoice", "BackendReport", "BaseRunner", "BubblewrapRunner", "CommandResult", "CommandRunner", "CompiledPolicy",
    "ContainerConfig", "ContainerRunner", "EgressProxy", "EnvPolicy", "EnvironmentPool", "EscapeHatchError", "EscapeHatches",
    "ExecutionEnvironment", "FilesystemPolicy", "LocalRunner", "NetworkMode", "NetworkPolicy", "PathDecision", "PolicyError",
    "PreparedCommand", "ProxyError", "ProxyEvent", "ProxyStats", "RemoteBackend", "RemoteEnvironment", "ResourceLimits",
    "RunnerEnvironment", "SandboxCrashedError", "SandboxError", "SandboxMode", "SandboxPolicy", "SandboxUnavailableError",
    "SeatbeltRunner", "Snapshot", "SnapshotDiff", "SnapshotError", "Violation", "WorkspaceSnapshotter", "build_bwrap_argv",
    "build_container_argv", "build_env", "build_seatbelt_profile", "choose_backend", "create_runner", "describe_sandbox",
    "detect_backends", "domain_matches", "find_trusted_binary", "format_violations", "idempotency_key", "ip_block_reason",
    "load_policy", "prepare_mcp_server", "proxy_env", "sandbox_state_meta",
]
