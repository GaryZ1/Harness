from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from harness_core.types import RunContext, ToolCall
from harness_tools import (
    DescriptionPins,
    MCPClient,
    MCPConnectionError,
    MCPInputRequired,
    MCPServerConfig,
    MCPTimeoutError,
    MCPToolset,
    RemoteMCPServer,
    ToolRegistry,
    connect_mcp_servers,
    enable_tool_search,
    mcp_connector_params,
    mcp_tool_name,
)

SERVER = str(Path(__file__).with_name("tools_fake_mcp_server.py"))


def config(*flags: str, name: str = "fake", **kw) -> MCPServerConfig:
    kw.setdefault("request_timeout_s", 5.0)
    kw.setdefault("call_timeout_s", 5.0)
    return MCPServerConfig(name=name, command=sys.executable, args=(SERVER, *flags), **kw)


@pytest.fixture
def client():
    c = MCPClient(config()).connect()
    yield c
    c.close()


@pytest.fixture
def ctx():
    return RunContext(session_id="s")


def test_stateless_discover_and_meta_on_every_request(client):
    assert client.mode == "stateless" and client.protocol_version == "2026-07-28" and client.server_info["name"] == "fake"
    meta = json.loads(client.call_tool("whoami", {}).content[0]["text"])["meta"]
    assert meta["io.modelcontextprotocol/protocolVersion"] == "2026-07-28"
    assert meta["io.modelcontextprotocol/clientInfo"]["name"] == "harness-tools"


def test_legacy_server_falls_back_to_initialize():
    with MCPClient(config("--legacy")) as c:
        assert c.mode == "legacy" and c.protocol_version == "2025-11-25" and c.server_info["name"] == "fake-legacy"
        out = json.loads(c.call_tool("whoami", {}).content[0]["text"])
        assert out["legacy"] and out["meta"] is None  # no stateless _meta sent to old servers


def test_tools_list_is_paginated_sorted_and_cached_by_ttl(client):
    names = [t.name for t in client.list_tools()]
    assert names == sorted(names) and len(names) == 10  # 4 pages of 3, served in reverse order
    now = [0.0]
    client.clock = lambda: now[0]
    first = client.list_tools(refresh=True)
    assert client.list_tools() == first
    now[0] = 61.0  # past ttlMs=60000
    assert [t.name for t in client.list_tools()] == names


def test_call_results_text_structured_error_and_image(client):
    assert client.call_tool("echo", {"text": "hi"}).content == [{"type": "text", "text": "hi"}]
    assert client.call_tool("add", {"a": 2, "b": 3}).structured == {"sum": 5}
    assert client.call_tool("fail", {}).is_error


def test_input_required_round_trip():
    seen = []

    def handler(server, method, requests):
        seen.append((server, method, requests))
        return {"confirm": {"action": "accept"}}

    with MCPClient(config(), input_handler=handler) as c:
        text = c.call_tool("ask", {}).content[0]["text"]
    assert seen == [("fake", "tools/call", {"confirm": {"message": "Proceed?"}})]
    assert text == 'got: {"confirm": {"action": "accept"}} state=s1'
    with MCPClient(config()) as c, pytest.raises(MCPInputRequired):
        c.call_tool("ask", {})


def test_request_timeout_and_server_requests(client):
    with pytest.raises(MCPTimeoutError):
        client.call_tool("slow", {"seconds": 1.5}, timeout=0.2)
    sampled = client.call_tool("sample", {}, timeout=5).content[0]["text"]
    assert sampled == "client replied with error code -32601"  # Sampling is deprecated; client refuses


def test_server_crash_fails_pending_and_later_calls(client):
    with pytest.raises(MCPConnectionError):
        client.call_tool("crash", {})
    with pytest.raises(MCPConnectionError):
        client.call_tool("echo", {"text": "x"})


def test_namespacing_is_sanitised_and_bounded():
    assert mcp_tool_name("git hub", "list.issues") == "mcp__git_hub__list_issues"
    long = mcp_tool_name("s" * 40, "t" * 40)
    assert len(long) == 64 and long.startswith("mcp__sss") and long != mcp_tool_name("s" * 40, "t" * 41)


def test_toolset_registers_deferred_untrusted_tools(client, ctx):
    reg = ToolRegistry()
    enable_tool_search(reg)
    names = MCPToolset(client).register(reg)
    assert "mcp__fake__echo" in names
    spec = reg.get("mcp__fake__echo").spec
    # annotations from an untrusted server are recorded but ignored
    assert spec.defer_loading and spec.open_world and not spec.read_only and spec.metadata["mcp_annotations"]["readOnlyHint"]
    out = reg.execute(ToolCall("c", "mcp__fake__echo", {"text": "hello"}), ctx)
    assert out.content == "hello" and out.metadata["untrusted"] and ctx.tainted
    invalid = reg.execute(ToolCall("c", "mcp__fake__add", {"a": "two"}), ctx)  # validated client-side first
    assert invalid.is_error and "/b: is a required property" in invalid.content
    assert json.loads(reg.execute(ToolCall("c", "mcp__fake__add", {"a": 1, "b": 2}), ctx).content) == {"sum": 3}
    err = reg.execute(ToolCall("c", "mcp__fake__rpc_error", {}), ctx)
    assert err.is_error and "missing widget" in err.content and err.metadata["untrusted"]
    img = reg.execute(ToolCall("c", "mcp__fake__image", {}), ctx)
    assert img.content[1]["type"] == "image" and img.content[1]["source"]["media_type"] == "image/png"
    assert reg.execute(ToolCall("c", "mcp__fake__fail", {}), ctx).is_error
    assert "interactive user input" in reg.execute(ToolCall("c", "mcp__fake__ask", {}), ctx).content


def test_trusted_server_annotations_are_honoured(ctx):
    with MCPClient(config(trusted=True, defer_loading=False)) as c:
        reg = ToolRegistry()
        MCPToolset(c, include=["echo", "add"]).register(reg)
        echo, add = reg.get("mcp__fake__echo"), reg.get("mcp__fake__add")
        assert echo.spec.read_only and not echo.spec.open_world and not echo.destructive and echo.parallel_safe
        assert add.spec.open_world and add.destructive and not add.spec.defer_loading
        assert not reg.execute(ToolCall("c", "mcp__fake__echo", {"text": "t"}), ctx).metadata["untrusted"]


def test_pins_reject_changed_descriptions(tmp_path):
    pins = DescriptionPins(path=tmp_path / "pins.json")
    with MCPClient(config()) as c:
        reg = ToolRegistry()
        MCPToolset(c, pins=pins, on_unpinned="approve", include=["echo"]).register(reg)
        assert "mcp__fake__echo" in reg
    with MCPClient(config("--desc-version", "2")) as c:  # the server swapped in a poisoned description
        reg = ToolRegistry()
        ts = MCPToolset(c, pins=DescriptionPins(path=tmp_path / "pins.json"), on_unpinned="approve", include=["echo"])
        assert ts.register(reg) == [] and ts.rejected == [("echo", "changed")]


def test_connect_mcp_servers_skips_broken_servers(tmp_path):
    reg = ToolRegistry()
    enable_tool_search(reg)
    broken = MCPServerConfig(name="broken", command=str(tmp_path / "does-not-exist"))
    clients = connect_mcp_servers(reg, [broken, config(name="ok")])
    try:
        assert [c.config.name for c in clients] == ["ok"] and "mcp__ok__echo" in reg
        reg.freeze()  # deferred MCP tools + a search tool form a valid array
    finally:
        for c in clients:
            c.close()


def test_server_side_connector_params_read_token_from_env():
    params = mcp_connector_params(
        [RemoteMCPServer("docs", "https://mcp.example.com/sse", authorization_token_env="DOCS_TOKEN", allowed_tools=("search",))],
        environ={"DOCS_TOKEN": "tok-123"},
    )
    assert params["betas"] == ["mcp-client-2025-11-20"]
    assert params["mcp_servers"] == [{"type": "url", "url": "https://mcp.example.com/sse", "name": "docs", "authorization_token": "tok-123"}]
    assert params["tools"][0]["mcp_server_name"] == "docs" and params["tools"][0]["default_config"]["defer_loading"] is True
    assert params["tools"][0]["configs"] == {"search": {"enabled": True}}
    with pytest.raises(MCPConnectionError, match="DOCS_TOKEN"):
        mcp_connector_params([RemoteMCPServer("docs", "https://x", authorization_token_env="DOCS_TOKEN")], environ={})
