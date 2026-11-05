"""Client-side MCP over stdio (JSON-RPC 2.0), adapted into the :class:`ToolRegistry`.

Research basis (``reports/SOTA agentic harness reference.md`` §5.3 E, §5.5 "MCP", §5.7 steps 10,
11, 14, 15 and §5.8 "Protocols and packaging"):

* **Target 2026-07-28, fall back for older servers.** The client first calls ``server/discover``
  (stateless core: every request then carries ``_meta`` with
  ``io.modelcontextprotocol/protocolVersion`` / ``clientCapabilities`` / ``clientInfo``); if the
  server rejects it, it falls back to the ``initialize`` + ``notifications/initialized``
  handshake (protocol ``2025-11-25``).
* **Multi round-trip requests.** A result with ``resultType: "input_required"`` is surfaced to an
  ``input_handler`` (the user) and the request is retried with ``inputResponses`` +
  ``requestState``; without a handler the tool returns an error result.
* **Cache-stable lists.** ``tools/list`` is paginated, sorted by name and cached by ``ttlMs``
  (or until ``notifications/tools/list_changed``).
* **Timeouts everywhere**; a timed-out request is cancelled with ``notifications/cancelled``.
* **Namespacing** ``mcp__<server>__<tool>`` (Claude Code convention), <= 64 chars.
* **Untrusted by default.** MCP tools are ``open_world`` (their output taints the session) and
  ``defer_loading`` (Claude Code defers MCP tools by default). ``ToolAnnotations``
  (``readOnlyHint`` false, ``destructiveHint`` true, ``idempotentHint`` false, ``openWorldHint``
  true by default) are honoured **only** for servers configured ``trusted=True``; otherwise they
  are recorded but ignored, because annotations from unvetted servers are hints an attacker
  controls.
* **Rug-pull defence.** With :class:`~harness_tools.design.DescriptionPins`, tools whose
  definition changed since approval are refused (or flagged) instead of silently registered.
* Deliberately not implemented (deprecated or out of scope): Sampling, Roots, Logging, HTTP+SSE;
  server-initiated requests are answered with ``-32601``. The tasks extension and Streamable HTTP
  are left to a later phase.

:func:`mcp_connector_params` emits the *server-side* MCP connector request parameters instead,
for deployments that let the Anthropic API call remote MCP servers directly.
"""

from __future__ import annotations

import collections
import hashlib
import itertools
import json
import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

from harness_core.types import ToolOutcome, ToolSpec

from .design import DescriptionPins, fingerprint
from .errors import MCPConnectionError, MCPError, MCPProtocolError, MCPTimeoutError
from .registry import Tool, ToolInvocation, ToolRegistry
from .runner import DEFAULT_ENV_ALLOWLIST

log = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION = "2026-07-28"
MCP_LEGACY_PROTOCOL_VERSION = "2025-11-25"
MCP_CONNECTOR_BETA = "mcp-client-2025-11-20"
JSONRPC_METHOD_NOT_FOUND = -32601
META_PREFIX = "io.modelcontextprotocol/"
MAX_INPUT_ROUNDS = 4
MAX_DESCRIPTION_CHARS = 2_048
CLIENT_INFO = {"name": "harness-tools", "version": "0.1.0"}

InputHandler = Callable[[str, str, Any], Any]
"""``(server_name, method, input_requests) -> input_responses``; raise to refuse."""


class MCPInputRequired(MCPError):
    """The server needs user input (``resultType: input_required``) and no handler was configured."""

    def __init__(self, message: str, input_requests: Any) -> None:
        super().__init__(message)
        self.input_requests = input_requests


# ----------------------------------------------------------------------------- configuration & data


@dataclass(frozen=True)
class MCPServerConfig:
    """A local MCP server launched as a subprocess speaking newline-delimited JSON-RPC on stdio.

    Secrets for the server belong in ``env`` (read them from the environment or a secret store at
    call time); they are never logged. ``inherit_env=False`` passes only an allowlisted base
    environment plus ``env``.
    """

    name: str
    command: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    cwd: str | None = None
    inherit_env: bool = False
    startup_timeout_s: float = 30.0
    request_timeout_s: float = 30.0
    call_timeout_s: float = 120.0
    trusted: bool = False
    defer_loading: bool = True


@dataclass(frozen=True)
class MCPTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    annotations: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MCPCallResult:
    content: list[dict[str, Any]]
    structured: Any = None
    is_error: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------- transport


class StdioTransport:
    """Owns the server subprocess and its pipes. One reader thread dispatches incoming messages."""

    def __init__(self, config: MCPServerConfig, on_message: Callable[[dict[str, Any]], None], on_close: Callable[[], None]) -> None:
        self.config = config
        self._on_message = on_message
        self._on_close = on_close
        self._proc: subprocess.Popen[bytes] | None = None
        self._write_lock = threading.Lock()
        self._stderr: collections.deque[str] = collections.deque(maxlen=50)

    def _env(self) -> dict[str, str]:
        env = dict(os.environ) if self.config.inherit_env else {k: os.environ[k] for k in DEFAULT_ENV_ALLOWLIST if k in os.environ}
        env.update(dict(self.config.env or {}))
        return env

    def start(self) -> None:
        try:
            self._proc = subprocess.Popen(
                [self.config.command, *self.config.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.config.cwd,
                env=self._env(),
                close_fds=True,
                start_new_session=os.name == "posix",
            )
        except OSError as exc:
            raise MCPConnectionError(f"failed to start MCP server {self.config.name!r}: {exc}") from exc
        threading.Thread(target=self._read_stdout, name=f"mcp-{self.config.name}-out", daemon=True).start()
        threading.Thread(target=self._read_stderr, name=f"mcp-{self.config.name}-err", daemon=True).start()

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stderr_tail(self, n: int = 5) -> list[str]:
        return list(self._stderr)[-n:]

    def send(self, message: dict[str, Any]) -> None:
        if not self.alive or self._proc is None or self._proc.stdin is None:
            raise MCPConnectionError(f"MCP server {self.config.name!r} is not running")
        line = json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._write_lock:
            try:
                self._proc.stdin.write(line.encode("utf-8"))
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as exc:
                raise MCPConnectionError(f"MCP server {self.config.name!r} closed its input: {exc}") from exc

    def _read_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            for raw in self._proc.stdout:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    log.debug("mcp %s: ignoring non-JSON stdout line", self.config.name)
                    continue
                if isinstance(msg, dict):
                    self._on_message(msg)
        except (OSError, ValueError):
            pass
        finally:
            self._on_close()

    def _read_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        try:
            for raw in self._proc.stderr:
                self._stderr.append(raw.decode("utf-8", errors="replace").rstrip())
        except (OSError, ValueError):
            pass

    def close(self, timeout: float = 2.0) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=timeout)
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream:
                    stream.close()
            except OSError:
                pass


# ----------------------------------------------------------------------------- client


class _Pending:
    __slots__ = ("event", "response")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.response: dict[str, Any] | None = None


class MCPClient:
    """Synchronous, thread-safe MCP client for one stdio server.

    >>> with MCPClient(MCPServerConfig("github", "github-mcp")) as client:   # doctest: +SKIP
    ...     tools = client.list_tools()
    ...     result = client.call_tool("search_issues", {"query": "bug"})
    """

    def __init__(
        self,
        config: MCPServerConfig,
        *,
        input_handler: InputHandler | None = None,
        clock: Callable[[], float] = time.monotonic,
        client_info: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        self.input_handler = input_handler
        self.clock = clock
        self.client_info = dict(client_info or CLIENT_INFO)
        self.mode: Literal["stateless", "legacy"] | None = None
        self.protocol_version: str | None = None
        self.server_info: dict[str, Any] = {}
        self.capabilities: dict[str, Any] = {}
        self.instructions: str | None = None
        self._ids = itertools.count(1)
        self._pending: dict[int, _Pending] = {}
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._dead: MCPError | None = None  # set once the transport is gone; guarded by _lock
        self._tools_cache: tuple[float, list[MCPTool]] | None = None
        self._transport = StdioTransport(config, self._dispatch, self._on_transport_close)

    # ------------------------------------------------------------------ lifecycle
    def __enter__(self) -> "MCPClient":
        self.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def connected(self) -> bool:
        return self.mode is not None and self._transport.alive and not self._closed.is_set() and self._dead is None

    def connect(self) -> "MCPClient":
        """Start the server and negotiate: ``server/discover`` first, ``initialize`` as fallback."""
        if self.mode is not None:
            return self
        self._transport.start()
        timeout = self.config.startup_timeout_s
        try:
            try:
                result = self._rpc("server/discover", {"_meta": self._meta()}, timeout)
                versions = result.get("supportedVersions") or result.get("protocolVersions") or []
                version = result.get("protocolVersion") or (
                    MCP_PROTOCOL_VERSION if not versions or MCP_PROTOCOL_VERSION in versions else versions[-1]
                )
                mode: Literal["stateless", "legacy"] = "stateless"
            except MCPProtocolError:  # pre-2026-07-28 server: classic handshake
                result = self._rpc(
                    "initialize",
                    {"protocolVersion": MCP_LEGACY_PROTOCOL_VERSION, "capabilities": {}, "clientInfo": self.client_info},
                    timeout,
                )
                version = result.get("protocolVersion") or MCP_LEGACY_PROTOCOL_VERSION
                mode = "legacy"
                self._notify("notifications/initialized", {})
        except BaseException:
            self.close()
            raise
        self.mode = mode
        self.protocol_version = str(version)
        self.server_info = dict(result.get("serverInfo") or {})
        self.capabilities = dict(result.get("capabilities") or {})
        self.instructions = result.get("instructions")
        return self

    def close(self) -> None:
        self._closed.set()
        self._transport.close()
        self._fail_pending(MCPConnectionError(f"MCP client for {self.config.name!r} was closed"))

    def stderr_tail(self, n: int = 5) -> list[str]:
        return self._transport.stderr_tail(n)

    # ------------------------------------------------------------------ JSON-RPC plumbing
    def _meta(self) -> dict[str, Any]:
        return {
            META_PREFIX + "protocolVersion": MCP_PROTOCOL_VERSION,
            META_PREFIX + "clientCapabilities": {},
            META_PREFIX + "clientInfo": self.client_info,
        }

    def _dispatch(self, msg: dict[str, Any]) -> None:
        if "method" in msg:
            method = str(msg.get("method"))
            if "id" in msg:  # server -> client request: we support none of them (Sampling/Roots are deprecated)
                if method == "ping":
                    reply: dict[str, Any] = {"jsonrpc": "2.0", "id": msg["id"], "result": {}}
                else:
                    reply = {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": JSONRPC_METHOD_NOT_FOUND, "message": f"client does not support {method}"}}
                try:
                    self._transport.send(reply)
                except MCPConnectionError:
                    pass
            elif method == "notifications/tools/list_changed":
                with self._lock:
                    self._tools_cache = None
            return
        rid = msg.get("id")
        if not isinstance(rid, int):
            return
        with self._lock:
            pending = self._pending.pop(rid, None)
        if pending is not None:
            pending.response = msg
            pending.event.set()

    def _on_transport_close(self) -> None:
        self._fail_pending(MCPConnectionError(f"MCP server {self.config.name!r} exited or closed its output"))

    def _fail_pending(self, exc: MCPError) -> None:
        # Marking the client dead under the same lock that registers requests closes the race
        # where the server's stdout hit EOF but the process has not been reaped yet.
        with self._lock:
            if self._dead is None:
                self._dead = exc
            pending, self._pending = self._pending, {}
        for p in pending.values():
            p.response = {"__error__": exc}
            p.event.set()

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._transport.send({"jsonrpc": "2.0", "method": method, "params": params})

    def _rpc(self, method: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        rid = next(self._ids)
        pending = _Pending()
        with self._lock:
            if self._dead is not None:
                raise MCPConnectionError(str(self._dead))
            self._pending[rid] = pending
        try:
            self._transport.send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        except MCPConnectionError:
            with self._lock:
                self._pending.pop(rid, None)
            raise
        if not pending.event.wait(timeout):
            with self._lock:
                self._pending.pop(rid, None)
            try:
                self._notify("notifications/cancelled", {"requestId": rid, "reason": "timeout"})
            except MCPConnectionError:
                pass
            raise MCPTimeoutError(f"MCP server {self.config.name!r} did not answer {method} within {timeout:g}s")
        resp = pending.response or {}
        if "__error__" in resp:
            raise resp["__error__"]
        if "error" in resp:
            err = resp["error"] or {}
            raise MCPProtocolError(
                f"MCP server {self.config.name!r} returned error for {method}: {err.get('message', 'unknown error')}",
                code=err.get("code"),
                data=err.get("data"),
            )
        result = resp.get("result")
        if not isinstance(result, dict):
            raise MCPProtocolError(f"MCP server {self.config.name!r} returned a malformed result for {method}")
        return result

    def request(self, method: str, params: Mapping[str, Any] | None = None, *, timeout: float | None = None) -> dict[str, Any]:
        """Send one request (with ``_meta`` in stateless mode) and resolve ``input_required`` rounds."""
        if self.mode is None:
            self.connect()
        if self._closed.is_set():
            raise MCPConnectionError(f"MCP client for {self.config.name!r} is closed")
        base = dict(params or {})
        if self.mode == "stateless":
            base["_meta"] = {**self._meta(), **dict(base.get("_meta") or {})}
        t = self.config.request_timeout_s if timeout is None else timeout
        current = base
        for _ in range(MAX_INPUT_ROUNDS):
            result = self._rpc(method, current, t)
            if result.get("resultType") != "input_required":
                return result
            requests = result.get("inputRequests")
            if self.input_handler is None:
                raise MCPInputRequired(f"MCP server {self.config.name!r} requires user input to complete {method}", requests)
            responses = self.input_handler(self.config.name, method, requests)
            current = {**base, "inputResponses": responses}
            if "requestState" in result:
                current["requestState"] = result["requestState"]
        raise MCPProtocolError(f"MCP server {self.config.name!r} kept requesting input for {method} after {MAX_INPUT_ROUNDS} rounds")

    # ------------------------------------------------------------------ tools
    def list_tools(self, *, refresh: bool = False) -> list[MCPTool]:
        """All tools (paginated), sorted by name, cached by ``ttlMs`` / until ``list_changed``."""
        with self._lock:
            cached = self._tools_cache
        if cached is not None and not refresh and self.clock() < cached[0]:
            return list(cached[1])
        tools: list[MCPTool] = []
        cursor: str | None = None
        ttl_ms: float | None = None
        for _ in range(1000):  # hard stop against cursor loops
            params: dict[str, Any] = {"cursor": cursor} if cursor else {}
            result = self.request("tools/list", params)
            for raw in result.get("tools") or []:
                if isinstance(raw, dict) and isinstance(raw.get("name"), str):
                    schema = raw.get("inputSchema") if isinstance(raw.get("inputSchema"), dict) else {"type": "object"}
                    tools.append(
                        MCPTool(
                            name=raw["name"],
                            description=str(raw.get("description") or ""),
                            input_schema=dict(schema),
                            annotations=dict(raw.get("annotations") or {}),
                            output_schema=raw.get("outputSchema") if isinstance(raw.get("outputSchema"), dict) else None,
                            raw=dict(raw),
                        )
                    )
            if isinstance(result.get("ttlMs"), (int, float)):
                ttl_ms = float(result["ttlMs"]) if ttl_ms is None else min(ttl_ms, float(result["ttlMs"]))
            cursor = result.get("nextCursor")
            if not cursor:
                break
        tools.sort(key=lambda t: t.name)
        expiry = float("inf") if ttl_ms is None else self.clock() + ttl_ms / 1000.0
        with self._lock:
            self._tools_cache = (expiry, tools)
        return list(tools)

    def call_tool(self, name: str, arguments: Mapping[str, Any], *, timeout: float | None = None) -> MCPCallResult:
        result = self.request(
            "tools/call", {"name": name, "arguments": dict(arguments)}, timeout=self.config.call_timeout_s if timeout is None else timeout
        )
        content = result.get("content")
        return MCPCallResult(
            content=[c for c in content if isinstance(c, dict)] if isinstance(content, list) else [],
            structured=result.get("structuredContent"),
            is_error=bool(result.get("isError", False)),
            raw=result,
        )


# ----------------------------------------------------------------------------- registry adapter


_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


def mcp_tool_name(server: str, tool: str, *, max_len: int = 64) -> str:
    """``mcp__<server>__<tool>``, sanitised to ``[A-Za-z0-9_-]`` and capped (hash suffix when too long)."""
    name = f"mcp__{_UNSAFE.sub('_', server)}__{_UNSAFE.sub('_', tool)}"
    if len(name) <= max_len:
        return name
    digest = hashlib.sha256(name.encode()).hexdigest()[:8]
    return f"{name[: max_len - 9]}_{digest}"


def _block_to_api(block: dict[str, Any]) -> dict[str, Any]:
    t = block.get("type")
    if t == "text":
        return {"type": "text", "text": str(block.get("text", ""))}
    if t == "image" and block.get("data"):
        return {"type": "image", "source": {"type": "base64", "media_type": block.get("mimeType", "image/png"), "data": block["data"]}}
    if t == "resource":
        res = block.get("resource") or {}
        if "text" in res:
            return {"type": "text", "text": f"[resource {res.get('uri', '')}]\n{res['text']}"}
        return {"type": "text", "text": f"[binary resource {res.get('uri', '')} ({res.get('mimeType', 'unknown type')}) omitted]"}
    if t == "resource_link":
        return {"type": "text", "text": f"[resource link] {block.get('name') or ''} {block.get('uri', '')}".strip()}
    if t == "audio":
        return {"type": "text", "text": f"[audio content ({block.get('mimeType', 'audio')}) omitted]"}
    return {"type": "text", "text": json.dumps(block, ensure_ascii=False, default=str)}


def outcome_from_result(result: MCPCallResult, *, untrusted: bool, source: str, wrap_untrusted: bool = False) -> ToolOutcome:
    """Map an MCP ``CallToolResult`` to a :class:`ToolOutcome` (text-only results become a string)."""
    blocks = [_block_to_api(b) for b in result.content]
    if not blocks and result.structured is not None:
        blocks = [{"type": "text", "text": json.dumps(result.structured, ensure_ascii=False, indent=2, default=str)}]
    if wrap_untrusted and untrusted:
        for b in blocks:
            if b["type"] == "text":
                b["text"] = f'<untrusted_content source="{source}">\n{b["text"]}\n</untrusted_content>'
    meta: dict[str, Any] = {"untrusted": untrusted, "mcp_source": source}
    if result.structured is not None:
        meta["structured_content"] = result.structured
    if not blocks:
        return ToolOutcome("(no output)", result.is_error, meta)
    if all(b["type"] == "text" for b in blocks):
        return ToolOutcome("\n".join(b["text"] for b in blocks), result.is_error, meta)
    return ToolOutcome(blocks, result.is_error, meta)


PinPolicy = Literal["reject", "warn", "approve"]


class MCPToolset:
    """Adapts one :class:`MCPClient`'s tools into registry :class:`Tool` objects.

    :param pins: approved fingerprints; with ``on_unpinned="reject"`` (default) new or changed
        tools are not registered and are listed in :attr:`rejected`; ``"warn"`` registers them
        and logs; ``"approve"`` registers and pins them (trust-on-first-use; changes still rejected).
    :param include / exclude: original tool names to keep / drop.
    :param wrap_untrusted: wrap untrusted text output in ``<untrusted_content>`` tags.
    """

    def __init__(
        self,
        client: MCPClient,
        *,
        pins: DescriptionPins | None = None,
        on_unpinned: PinPolicy = "reject",
        include: Iterable[str] | None = None,
        exclude: Iterable[str] = (),
        wrap_untrusted: bool = False,
    ) -> None:
        self.client = client
        self.pins = pins
        self.on_unpinned = on_unpinned
        self.include = set(include) if include is not None else None
        self.exclude = set(exclude)
        self.wrap_untrusted = wrap_untrusted
        self.rejected: list[tuple[str, str]] = []
        self.names: dict[str, str] = {}  # registry name -> original MCP name

    def _allowed(self, t: MCPTool, fp: str) -> bool:
        if self.pins is None:
            return True
        key = f"{self.client.config.name}/{t.name}"
        status = self.pins.status(key, fp)
        if status == "approved":
            return True
        if status == "changed" or self.on_unpinned == "reject":
            self.rejected.append((t.name, status))
            log.warning("MCP tool %s is %s since approval; not registered", key, status)
            return False
        if self.on_unpinned == "approve":
            self.pins.approve(key, fp)
        else:
            log.warning("MCP tool %s is not pinned (registered anyway)", key)
        return True

    def tools(self) -> list[Tool]:
        cfg = self.client.config
        out: list[Tool] = []
        self.rejected = []
        for t in self.client.list_tools():
            if (self.include is not None and t.name not in self.include) or t.name in self.exclude:
                continue
            fp = fingerprint(t.raw or {"name": t.name, "description": t.description, "inputSchema": t.input_schema})
            if not self._allowed(t, fp):
                continue
            out.append(self._adapt(t, fp, cfg))
        return out

    def _adapt(self, t: MCPTool, fp: str, cfg: MCPServerConfig) -> Tool:
        ann = t.annotations
        trusted = cfg.trusted
        read_only = bool(ann.get("readOnlyHint", False)) if trusted else False
        open_world = bool(ann.get("openWorldHint", True)) if trusted else True
        destructive = (not read_only) and (bool(ann.get("destructiveHint", True)) if trusted else True)
        idempotent = bool(ann.get("idempotentHint", False)) if trusted else False
        name = mcp_tool_name(cfg.name, t.name)
        self.names[name] = t.name
        desc = t.description.strip() or f"MCP tool {t.name} from server {cfg.name}."
        if len(desc) > MAX_DESCRIPTION_CHARS:
            desc = desc[: MAX_DESCRIPTION_CHARS - 3] + "..."
        schema = dict(t.input_schema)
        schema.setdefault("type", "object")
        source = f"mcp:{cfg.name}"
        original = t.name
        client = self.client
        wrap = self.wrap_untrusted

        def handler(inp: dict[str, Any], inv: ToolInvocation) -> ToolOutcome:
            try:
                result = client.call_tool(original, inp)
            except MCPInputRequired:
                return ToolOutcome(
                    f"{name} needs interactive user input, which is not available here. Ask the user, or use another approach.",
                    True,
                    {"error_class": "MCPInputRequired"},
                )
            except MCPTimeoutError as exc:
                return ToolOutcome(f"{exc}. Retry with a smaller request, or try later.", True, {"error_class": "MCPTimeoutError"})
            except MCPConnectionError as exc:
                return ToolOutcome(f"{exc}. The tool is unavailable for now.", True, {"error_class": "MCPConnectionError"})
            except MCPProtocolError as exc:
                # The message text comes from the server: treat it as untrusted content.
                return ToolOutcome(str(exc), True, {"error_class": "MCPProtocolError", "untrusted": True, "mcp_source": source})
            return outcome_from_result(result, untrusted=open_world or not trusted, source=f"{source}/{original}", wrap_untrusted=wrap)

        spec = ToolSpec(
            name=name,
            description=desc,
            input_schema=schema,
            read_only=read_only,
            open_world=open_world,
            defer_loading=cfg.defer_loading,
            metadata={
                "mcp_server": cfg.name,
                "mcp_tool": t.name,
                "mcp_annotations": dict(ann),
                "mcp_trusted": trusted,
                "fingerprint": fp,
            },
        )
        return Tool(
            spec=spec,
            handler=handler,
            timeout_s=cfg.call_timeout_s + 5.0,
            idempotent=idempotent,
            destructive=destructive,
            source=source,
        )

    def register(self, registry: ToolRegistry) -> list[str]:
        """Register the adapted tools; returns their registry names (sorted)."""
        names = []
        for tool in self.tools():
            registry.register(tool)
            names.append(tool.name)
        return sorted(names)


def connect_mcp_servers(
    registry: ToolRegistry, configs: Sequence[MCPServerConfig], *, input_handler: InputHandler | None = None, **toolset_kwargs: Any
) -> list[MCPClient]:
    """Connect each server and register its tools. A server that fails to start is logged and skipped."""
    clients = []
    for cfg in configs:
        client = MCPClient(cfg, input_handler=input_handler)
        try:
            client.connect()
            MCPToolset(client, **toolset_kwargs).register(registry)
        except MCPError as exc:
            log.warning("MCP server %s unavailable: %s", cfg.name, exc)
            client.close()
            continue
        clients.append(client)
    return clients


# ----------------------------------------------------------------------------- server-side connector


@dataclass(frozen=True)
class RemoteMCPServer:
    """A remote MCP server for the Anthropic server-side connector.

    ``authorization_token_env`` names an environment variable holding the OAuth token; the token
    is read at call time and never stored in code or config.
    """

    name: str
    url: str
    authorization_token_env: str | None = None
    defer_loading: bool = True
    allowed_tools: tuple[str, ...] | None = None


def mcp_connector_params(servers: Sequence[RemoteMCPServer], *, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Request parameters for the Anthropic MCP connector (beta ``mcp-client-2025-11-20``).

    Returns ``{"mcp_servers": [...], "tools": [{"type": "mcp_toolset", ...}], "betas": [...]}``;
    merge ``tools`` into the request's tool array and ``betas`` into its beta headers.
    """
    env = os.environ if environ is None else environ
    mcp_servers, tools = [], []
    for s in servers:
        entry: dict[str, Any] = {"type": "url", "url": s.url, "name": s.name}
        if s.authorization_token_env:
            token = env.get(s.authorization_token_env)
            if not token:
                raise MCPConnectionError(f"environment variable {s.authorization_token_env} (token for MCP server {s.name!r}) is not set")
            entry["authorization_token"] = token
        mcp_servers.append(entry)
        toolset: dict[str, Any] = {"type": "mcp_toolset", "mcp_server_name": s.name}
        if s.defer_loading:
            toolset["default_config"] = {"defer_loading": True}
        if s.allowed_tools is not None:
            toolset["default_config"] = {**toolset.get("default_config", {}), "enabled": False}
            toolset["configs"] = {t: {"enabled": True} for t in sorted(s.allowed_tools)}
        tools.append(toolset)
    return {"mcp_servers": mcp_servers, "tools": tools, "betas": [MCP_CONNECTOR_BETA]}


__all__ = [
    "CLIENT_INFO",
    "InputHandler",
    "MCPCallResult",
    "MCPClient",
    "MCPInputRequired",
    "MCPServerConfig",
    "MCPTool",
    "MCPToolset",
    "MCP_CONNECTOR_BETA",
    "MCP_LEGACY_PROTOCOL_VERSION",
    "MCP_PROTOCOL_VERSION",
    "RemoteMCPServer",
    "StdioTransport",
    "connect_mcp_servers",
    "mcp_connector_params",
    "mcp_tool_name",
    "outcome_from_result",
]
