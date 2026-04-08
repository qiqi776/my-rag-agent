"""MCP tool for listing collections."""

from __future__ import annotations

from typing import Any

from src.interfaces.mcp.dependencies import MCPDependencies
from src.interfaces.mcp.mappers import map_collection_list
from src.interfaces.mcp.models import MCPTool, MCPToolResult

LIST_COLLECTIONS_TOOL = MCPTool(
    name="list_collections",
    description="List known collections in the vector store.",
    input_schema={
        "type": "object",
        "properties": {},
    },
)


class ListCollectionsTool:
    """Adapter from MCP tool arguments to DocumentService.list_collections."""

    def __init__(self, dependencies: MCPDependencies) -> None:
        self.dependencies = dependencies

    def handle(self, arguments: dict[str, Any]) -> MCPToolResult:
        del arguments
        collections = self.dependencies.document_service.list_collections()
        return map_collection_list(collections)


def register_list_collections_tool(server: Any, dependencies: MCPDependencies) -> None:
    """Register the list-collections tool with an MCP server."""

    server.register_tool(LIST_COLLECTIONS_TOOL, ListCollectionsTool(dependencies).handle)
