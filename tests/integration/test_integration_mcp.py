"""features.mcp_servers: stdio MCP servers are connected into the registry, deferred behind tool_search."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from integration_support import build, text_response

from harness.config import FeaturesConfig, McpServerConfig

logging.getLogger("harness_sandbox").setLevel(logging.ERROR)

SERVER = str(Path(__file__).resolve().parents[1] / "tools" / "tools_fake_mcp_server.py")


def test_mcp_tools_are_registered_deferred_with_tool_search(tmp_path):
    features = FeaturesConfig(memory=False, mcp_servers=[McpServerConfig("fake", sys.executable, [SERVER])])
    h, client = build(tmp_path, [text_response("ok")], features=features)
    try:
        specs = {s.name: s for s in h.registry.specs()}
        mcp = [n for n in specs if n.startswith("mcp__fake__")]
        assert "mcp__fake__echo" in mcp and all(specs[n].defer_loading for n in mcp)
        assert "tool_search" in specs and not specs["tool_search"].defer_loading
        assert h.registry.frozen and len(h.mcp_clients) == 1
        h.run("hi")
        sent = {t.name for t in client.requests[0].tools}
        assert "tool_search" in sent and "read" in sent
    finally:
        h.end_session()
    assert not h.mcp_clients[0].connected  # end_session closed the server
