# MCP client: `harness_tools.mcp`

**Purpose:** connect local MCP servers over stdio (JSON-RPC 2.0), list and call their tools, and adapt them into the registry as namespaced, deferred, untrusted tools. A helper also emits the Anthropic *server-side* MCP connector parameters.

## Research basis

* **Target MCP 2026-07-28** (§5.7 step 14). The core is stateless and the `initialize` handshake is gone. Servers MUST implement `server/discover`, and every request carries `_meta` with `io.modelcontextprotocol/protocolVersion` and `clientCapabilities` (and SHOULD carry `clientInfo`). Older servers need a fallback to `initialize`.
* **Multi Round-Trip Requests** (SEP-2322): a result with `resultType: "input_required"` carries `inputRequests`, and the client retries with `inputResponses` plus `requestState`.
* **Cacheable lists** (SEP-2549): `ttlMs`/`cacheScope`, with a deterministic `tools/list` order "to improve LLM prompt cache hit rates".
* **Deprecated features not to build on:** Sampling, Roots, Logging, HTTP+SSE, protocol sessions.
* **Annotations are hints** (`readOnlyHint` false, `destructiveHint` true, `idempotentHint` false, `openWorldHint` true by default) and are untrusted unless the server is vetted. Tool poisoning and rug pulls are real, so pin and hash approved descriptions (§5.7 step 15, §7).
* **Defer all MCP tools** and namespace them `server__tool` (§5.7 step 10; Claude Code defers MCP tools by default).
* Validation errors are tool errors (`isError`), not protocol errors (SEP-1303).

## API

```python
from harness_tools import MCPClient, MCPServerConfig, MCPToolset, DescriptionPins, connect_mcp_servers

cfg = MCPServerConfig(
    name="tracker", command="tracker-mcp", args=("--stdio",),
    env={"TRACKER_TOKEN": os.environ["TRACKER_TOKEN"]},   # secrets from the environment, never in code
    request_timeout_s=30, call_timeout_s=120, trusted=False, defer_loading=True,
)
with MCPClient(cfg, input_handler=ask_user) as client:     # ask_user(server, method, input_requests) -> responses
    client.mode, client.protocol_version                    # "stateless", "2026-07-28"  (or "legacy", "2025-11-25")
    tools = client.list_tools()                             # paginated, sorted, cached by ttlMs / list_changed
    result = client.call_tool("search_issues", {"query": "login bug"})

    pins = DescriptionPins(path="~/.harness/mcp-pins.json")
    toolset = MCPToolset(client, pins=pins, on_unpinned="reject")   # or "warn" / "approve" (trust on first use)
    toolset.register(registry)                              # -> ["mcp__tracker__search_issues", ...]
    toolset.rejected                                        # [("create_issue", "changed"), ...]

clients = connect_mcp_servers(registry, [cfg1, cfg2])       # skips servers that fail to start; close() them at session end
```

### How a call is mapped

| MCP `CallToolResult` | `ToolOutcome` |
|---|---|
| only `text` blocks | joined string |
| `image` | Anthropic image block (`source.type = base64`) |
| `resource` (embedded) | text `[resource uri]\n<text>`, or a binary placeholder |
| `resource_link` / `audio` | descriptive text |
| no content, `structuredContent` | pretty JSON text; `metadata["structured_content"]` keeps the value |
| `isError: true` | `is_error=True` |
| JSON-RPC error | `is_error=True`, `untrusted=True` (the message text comes from the server) |
| timeout / server gone / `input_required` without a handler | `is_error=True` with a recovery hint |

Registry inputs are validated against the server's `inputSchema` before the server is called.

### Server-side connector

```python
from harness_tools import RemoteMCPServer, mcp_connector_params
params = mcp_connector_params([RemoteMCPServer("docs", "https://mcp.example.com/mcp",
                                               authorization_token_env="DOCS_MCP_TOKEN",
                                               allowed_tools=("search",))])
# {"mcp_servers": [{"type": "url", "url": ..., "name": "docs", "authorization_token": <from env>}],
#  "tools": [{"type": "mcp_toolset", "mcp_server_name": "docs",
#             "default_config": {"defer_loading": True, "enabled": False}, "configs": {"search": {"enabled": True}}}],
#  "betas": ["mcp-client-2025-11-20"]}
```

## Defaults

| Setting | Default | Anchor |
|---|---|---|
| Protocol | `2026-07-28` via `server/discover`; fallback `initialize` with `2025-11-25` | MCP 2026-07-28 changelog |
| Tool naming | `mcp__<server>__<tool>`, `[A-Za-z0-9_-]`, ≤ 64 chars (hash suffix when longer) | Claude Code convention; Anthropic name rule |
| `defer_loading` | `True` | Claude Code |
| Trust | `trusted=False`: `open_world=True`, `read_only=False`, `destructive=True`, `idempotent=False` | MCP annotation defaults treated as untrusted |
| Startup / request / call timeout | 30 s / 30 s / 120 s | our choice |
| `input_required` rounds | 4 | our choice |
| Description length | truncated at 2,048 chars | our choice (budget) |
| Child environment | allowlist + `env` | our choice (no secret leakage) |
| Server→client requests | `ping` answered; everything else `-32601` | Sampling/Roots deprecated (SEP-2577) |

## Invariants (enforced in code)

* Results from untrusted servers are always `metadata["untrusted"] = True`, so the registry taints the session. For trusted servers this holds unless `openWorldHint` is false.
* Annotations from untrusted servers are recorded in `spec.metadata["mcp_annotations"]` but never change `read_only` or `destructive`, so they cannot unlock parallel execution or auto-approval.
* With pins, a tool whose fingerprint (SHA-256 over name, description, `inputSchema` and annotations) changed since approval is never registered, whatever `on_unpinned` says.
* Each request has a timeout. A timed-out request is removed and `notifications/cancelled` is sent.
* When the server exits, every pending and future request fails fast with `MCPConnectionError`. The client is marked dead under the same lock that registers requests, so no request can hang on a server that is already gone.
* `tools/list` output is sorted by name, and pagination stops after 1,000 pages to guard against cursor loops.

## Failure modes

| Failure | Behaviour |
|---|---|
| Server binary missing | `MCPConnectionError` from `connect()`; `connect_mcp_servers` logs and skips it |
| Non-JSON lines on stdout | ignored (logged at debug) |
| Server crash mid-call | pending call fails with `MCPConnectionError`; the registry returns an error result |
| Slow server | `MCPTimeoutError` / error result "Retry with a smaller request, or try later" |
| Description changed after approval (rug pull) | tool not registered; listed in `toolset.rejected`; warning logged |
| Not implemented yet | Streamable HTTP transport, the tasks extension (`tasks/get`), OAuth/CIMD, resources/prompts. Use the server-side connector for remote servers. |
