"""harness-tools: the tool layer of the harness (registry, kernel tools, catalog, MCP client).

==========================  ===============================================  ==========================
Component                   Module                                           Implements
==========================  ===============================================  ==========================
Registry / executor         :mod:`harness_tools.registry`                    ``ToolExecutor`` protocol
JSON Schema                 :mod:`harness_tools.schema`                      validator + compiler
Output budgets              :mod:`harness_tools.output`                      25k-token cap + offload
Workspace confinement       :mod:`harness_tools.workspace`                   absolute paths, sandbox
Command runner              :mod:`harness_tools.runner`                      ``CommandRunner`` protocol
Kernel tools                :mod:`harness_tools.kernel`                      read/write/edit/bash/grep/glob
Tool catalog                :mod:`harness_tools.catalog`                     deferred loading + search
Design lint / pins          :mod:`harness_tools.design`                      budgets, lint, rug-pull pins
MCP client                  :mod:`harness_tools.mcp`                         stdio JSON-RPC 2.0
==========================  ===============================================  ==========================

Typical wiring::

    registry = ToolRegistry()
    ws = Workspace("/abs/path/to/repo", checkpointer=MemoryCheckpointer())
    register_kernel(registry, ws)
    enable_tool_search(registry)
    connect_mcp_servers(registry, [MCPServerConfig("github", "github-mcp-server", ("stdio",))])
    registry.freeze()
    request = ModelRequest(system=..., messages=..., tools=registry.specs())
    reply = registry.run_tool_uses(response, ctx)   # ONE user message answering every tool_use
"""

from .catalog import (
    ToolCatalog,
    apply_tiering,
    definition_tokens,
    enable_tool_search,
    server_tool_search_definition,
    server_tool_search_spec,
    should_enable_tool_search,
)
from .design import DescriptionPins, LintFinding, fingerprint, lint_tool, lint_tools
from .errors import (
    CommandRunnerError,
    FileStateError,
    MCPConnectionError,
    MCPError,
    MCPProtocolError,
    MCPTimeoutError,
    SchemaError,
    ToolDefinitionError,
    ToolInputError,
    ToolNotFoundError,
    ToolsError,
    ToolTimeoutError,
    WorkspaceError,
)
from .kernel import Kernel, register_kernel
from .mcp import (
    MCPClient,
    MCPInputRequired,
    MCPServerConfig,
    MCPToolset,
    RemoteMCPServer,
    connect_mcp_servers,
    mcp_connector_params,
    mcp_tool_name,
)
from .output import OutputPolicy, apply_budget, estimate_tokens
from .registry import FailureTracker, Tool, ToolCallRecord, ToolInvocation, ToolRegistry, function_tool, tool_results_message
from .runner import CommandRequest, CommandResult, CommandRunner, LocalSubprocessRunner
from .schema import compile_schema, lint_schema, schema_from_function, validate
from .workspace import Checkpointer, MemoryCheckpointer, Workspace

__version__ = "0.1.0"

__all__ = [
    "Checkpointer",
    "CommandRequest",
    "CommandResult",
    "CommandRunner",
    "CommandRunnerError",
    "DescriptionPins",
    "FailureTracker",
    "FileStateError",
    "Kernel",
    "LintFinding",
    "LocalSubprocessRunner",
    "MCPClient",
    "MCPConnectionError",
    "MCPError",
    "MCPInputRequired",
    "MCPProtocolError",
    "MCPServerConfig",
    "MCPTimeoutError",
    "MCPToolset",
    "MemoryCheckpointer",
    "OutputPolicy",
    "RemoteMCPServer",
    "SchemaError",
    "Tool",
    "ToolCallRecord",
    "ToolCatalog",
    "ToolDefinitionError",
    "ToolInputError",
    "ToolInvocation",
    "ToolNotFoundError",
    "ToolRegistry",
    "ToolTimeoutError",
    "ToolsError",
    "Workspace",
    "WorkspaceError",
    "apply_budget",
    "apply_tiering",
    "compile_schema",
    "connect_mcp_servers",
    "definition_tokens",
    "enable_tool_search",
    "estimate_tokens",
    "fingerprint",
    "function_tool",
    "lint_schema",
    "lint_tool",
    "lint_tools",
    "mcp_connector_params",
    "mcp_tool_name",
    "register_kernel",
    "schema_from_function",
    "server_tool_search_definition",
    "server_tool_search_spec",
    "should_enable_tool_search",
    "tool_results_message",
    "validate",
]
