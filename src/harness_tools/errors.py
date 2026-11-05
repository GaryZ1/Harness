"""Exception hierarchy for :mod:`harness_tools`.

Errors raised *inside* a tool call never escape :meth:`ToolRegistry.execute`: they are converted
into ``is_error: true`` tool results so the model can self-correct (MCP SEP-1303; OpenAI Agents
SDK ``"return_error_to_model"``). These classes exist so handlers can signal *which* kind of
failure happened and so configuration mistakes (bad schemas, duplicate names) fail loudly at
registration time instead of mid-session.
"""

from __future__ import annotations


class ToolsError(Exception):
    """Base class for every error raised by this package."""


class ToolDefinitionError(ToolsError):
    """A tool definition is invalid (bad name, duplicate, malformed schema, deferred search tool...)."""


class SchemaError(ToolDefinitionError):
    """A JSON Schema is malformed or uses a construct the validator cannot honour."""


class ToolInputError(ToolsError):
    """The model supplied input the tool cannot accept. The message is shown to the model.

    ``example`` (optional) is a corrected input the model can copy.
    """

    def __init__(self, message: str, *, example: dict | None = None) -> None:
        super().__init__(message)
        self.example = example


class ToolNotFoundError(ToolsError):
    """The requested tool is not registered."""


class ToolTimeoutError(ToolsError):
    """A tool exceeded its per-call timeout."""


class WorkspaceError(ToolInputError):
    """A path escaped the workspace, was relative where absolute is required, or is protected."""


class FileStateError(ToolInputError):
    """An edit/overwrite precondition failed (file not read first, or modified since it was read)."""


class CommandRunnerError(ToolsError):
    """The command runner could not start or supervise a process."""


class MCPError(ToolsError):
    """Base class for MCP client failures."""


class MCPConnectionError(MCPError):
    """The MCP server process is not running, exited, or closed its pipes."""


class MCPTimeoutError(MCPError):
    """An MCP request did not receive a response within its timeout."""


class MCPProtocolError(MCPError):
    """The server returned a JSON-RPC error or a malformed message.

    ``code`` is the JSON-RPC error code when one was returned.
    """

    def __init__(self, message: str, *, code: int | None = None, data: object = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


__all__ = [
    "CommandRunnerError",
    "FileStateError",
    "MCPConnectionError",
    "MCPError",
    "MCPProtocolError",
    "MCPTimeoutError",
    "SchemaError",
    "ToolDefinitionError",
    "ToolInputError",
    "ToolNotFoundError",
    "ToolTimeoutError",
    "ToolsError",
    "WorkspaceError",
]
