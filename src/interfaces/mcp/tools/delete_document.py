"""MCP tool for document deletion."""

from __future__ import annotations

from typing import Any

from src.interfaces.mcp.dependencies import MCPDependencies
from src.interfaces.mcp.mappers import map_delete_result
from src.interfaces.mcp.models import MCPTool, MCPToolResult

DELETE_DOCUMENT_TOOL = MCPTool(
    name="delete_document",
    description="Delete a document from a collection.",
    input_schema={
        "type": "object",
        "properties": {
            "doc_id": {"type": "string", "description": "Target document id."},
            "collection": {
                "type": "string",
                "description": "Optional collection name.",
            },
        },
        "required": ["doc_id"],
    },
)


def _required_doc_id(arguments: dict[str, Any]) -> str:
    value = arguments.get("doc_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("'doc_id' must be a non-empty string")
    return value


def _optional_collection(arguments: dict[str, Any]) -> str | None:
    value = arguments.get("collection")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("'collection' must be a non-empty string when provided")
    return value


class DeleteDocumentTool:
    """Adapter from MCP tool arguments to DocumentService.delete_document."""

    def __init__(self, dependencies: MCPDependencies) -> None:
        self.dependencies = dependencies

    def handle(self, arguments: dict[str, Any]) -> MCPToolResult:
        doc_id = _required_doc_id(arguments)
        collection = _optional_collection(arguments)
        result = self.dependencies.document_service.delete_document(
            doc_id=doc_id,
            collection=collection,
        )
        return map_delete_result(result)


def register_delete_document_tool(server: Any, dependencies: MCPDependencies) -> None:
    """Register the delete-document tool with an MCP server."""

    server.register_tool(DELETE_DOCUMENT_TOOL, DeleteDocumentTool(dependencies).handle)
