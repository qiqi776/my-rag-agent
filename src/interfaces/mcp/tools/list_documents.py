"""MCP tool for document listing."""

from __future__ import annotations

from typing import Any

from src.interfaces.mcp.dependencies import MCPDependencies
from src.interfaces.mcp.mappers import map_document_list
from src.interfaces.mcp.models import MCPTool, MCPToolResult

LIST_DOCUMENTS_TOOL = MCPTool(
    name="list_documents",
    description="List ingested documents, optionally filtered by collection.",
    input_schema={
        "type": "object",
        "properties": {
            "collection": {
                "type": "string",
                "description": "Optional collection name.",
            }
        },
    },
)


def _optional_collection(arguments: dict[str, Any]) -> str | None:
    value = arguments.get("collection")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("'collection' must be a non-empty string when provided")
    return value


class ListDocumentsTool:
    """Adapter from MCP tool arguments to DocumentService.list_documents."""

    def __init__(self, dependencies: MCPDependencies) -> None:
        self.dependencies = dependencies

    def handle(self, arguments: dict[str, Any]) -> MCPToolResult:
        collection = _optional_collection(arguments)
        documents = self.dependencies.document_service.list_documents(collection=collection)
        return map_document_list(documents, collection=collection)


def register_list_documents_tool(server: Any, dependencies: MCPDependencies) -> None:
    """Register the list-documents tool with an MCP server."""

    server.register_tool(LIST_DOCUMENTS_TOOL, ListDocumentsTool(dependencies).handle)
