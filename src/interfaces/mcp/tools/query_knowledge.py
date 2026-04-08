"""MCP tool for query operations."""

from __future__ import annotations

from typing import Any

from src.interfaces.mcp.dependencies import MCPDependencies
from src.interfaces.mcp.mappers import map_search_output
from src.interfaces.mcp.models import MCPTool, MCPToolResult

QUERY_KNOWLEDGE_TOOL = MCPTool(
    name="query_knowledge",
    description="Search the knowledge base and return structured results with citations.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query text."},
            "collection": {
                "type": "string",
                "description": "Optional collection name.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of results to return.",
                "minimum": 1,
            },
            "mode": {
                "type": "string",
                "description": "Retrieval mode override.",
                "enum": ["dense", "hybrid"],
            },
            "doc_type": {
                "type": "string",
                "description": "Optional document type filter, for example 'pdf' or 'text'.",
            },
        },
        "required": ["query"],
    },
)


def _require_string(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' must be a non-empty string")
    return value


def _optional_string(arguments: dict[str, Any], key: str) -> str | None:
    value = arguments.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' must be a non-empty string when provided")
    return value


def _optional_top_k(arguments: dict[str, Any]) -> int | None:
    value = arguments.get("top_k")
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ValueError("'top_k' must be a positive integer")
    return value


class QueryKnowledgeTool:
    """Adapter from MCP tool arguments to SearchService."""

    def __init__(self, dependencies: MCPDependencies) -> None:
        self.dependencies = dependencies

    def handle(self, arguments: dict[str, Any]) -> MCPToolResult:
        query = _require_string(arguments, "query")
        collection = _optional_string(arguments, "collection")
        mode = _optional_string(arguments, "mode")
        if mode is not None and mode not in {"dense", "hybrid"}:
            raise ValueError("'mode' must be one of: dense, hybrid")

        output = self.dependencies.search_service.search(
            query=query,
            collection=collection,
            top_k=_optional_top_k(arguments),
            mode=mode,
            filters=self._filters(arguments),
        )
        return map_search_output(output)

    def _filters(self, arguments: dict[str, Any]) -> dict[str, Any] | None:
        doc_type = _optional_string(arguments, "doc_type")
        if doc_type is None:
            return None
        return {"doc_type": doc_type}


def register_query_knowledge_tool(server: Any, dependencies: MCPDependencies) -> None:
    """Register the query tool with an MCP server."""

    server.register_tool(QUERY_KNOWLEDGE_TOOL, QueryKnowledgeTool(dependencies).handle)
