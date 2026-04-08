"""MCP tool for document summary/detail lookup."""

from __future__ import annotations

from typing import Any

from src.interfaces.mcp.dependencies import MCPDependencies
from src.interfaces.mcp.mappers import map_document_detail
from src.interfaces.mcp.models import MCPTool, MCPToolResult

GET_DOCUMENT_SUMMARY_TOOL = MCPTool(
    name="get_document_summary",
    description="Return a document summary with preview and metadata.",
    input_schema={
        "type": "object",
        "properties": {
            "doc_id": {"type": "string", "description": "Target document id."},
            "collection": {
                "type": "string",
                "description": "Optional collection override.",
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


class GetDocumentSummaryTool:
    """Adapter from MCP tool arguments to DocumentService.get_document_summary."""

    def __init__(self, dependencies: MCPDependencies) -> None:
        self.dependencies = dependencies

    def handle(self, arguments: dict[str, Any]) -> MCPToolResult:
        detail = self.dependencies.document_service.get_document_summary(
            doc_id=_required_doc_id(arguments),
            collection=_optional_collection(arguments),
        )
        return map_document_detail(detail)


def register_get_document_summary_tool(server: Any, dependencies: MCPDependencies) -> None:
    """Register the get-document-summary tool with an MCP server."""

    server.register_tool(
        GET_DOCUMENT_SUMMARY_TOOL,
        GetDocumentSummaryTool(dependencies).handle,
    )
