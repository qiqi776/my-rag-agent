from __future__ import annotations

import pytest

from src.interfaces.mcp.models import MCPTextContent, MCPTool, MCPToolResult
from src.interfaces.mcp.protocol_handler import MCPProtocolHandler


class _StubServer:
    server_name = "minimal-modular-rag-mcp"
    server_version = "0.1.0"

    def list_tools(self) -> list[MCPTool]:
        return [
            MCPTool(
                name="query_knowledge",
                description="Search",
                input_schema={"type": "object"},
            )
        ]

    def call_tool(self, name: str, arguments: dict[str, object]) -> MCPToolResult:
        return MCPToolResult(
            content=[MCPTextContent(f"called {name}")],
            structured_content={"kind": "tool_result", "name": name, "arguments": arguments},
        )


@pytest.mark.unit
def test_protocol_handler_supports_initialize_tools_list_and_tools_call() -> None:
    handler = MCPProtocolHandler(_StubServer())

    initialize = handler.handle_payload(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    )
    tools_list = handler.handle_payload(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    )
    tool_call = handler.handle_payload(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "query_knowledge", "arguments": {"query": "paging"}},
        }
    )

    assert initialize["result"]["serverInfo"]["name"] == "minimal-modular-rag-mcp"
    assert tools_list["result"]["tools"][0]["name"] == "query_knowledge"
    assert tool_call["result"]["structuredContent"]["name"] == "query_knowledge"


@pytest.mark.unit
def test_protocol_handler_returns_stable_errors() -> None:
    handler = MCPProtocolHandler(_StubServer())

    unknown_method = handler.handle_payload(
        {"jsonrpc": "2.0", "id": 1, "method": "missing", "params": {}}
    )
    invalid_params = handler.handle_payload(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "", "arguments": {}},
        }
    )

    assert unknown_method["error"]["code"] == -32601
    assert invalid_params["error"]["code"] == -32602
