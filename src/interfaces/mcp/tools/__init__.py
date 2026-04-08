"""MCP tool registration helpers."""

from src.interfaces.mcp.tools.delete_document import register_delete_document_tool
from src.interfaces.mcp.tools.get_document_summary import register_get_document_summary_tool
from src.interfaces.mcp.tools.list_collections import register_list_collections_tool
from src.interfaces.mcp.tools.list_documents import register_list_documents_tool
from src.interfaces.mcp.tools.query_knowledge import register_query_knowledge_tool

__all__ = [
    "register_delete_document_tool",
    "register_get_document_summary_tool",
    "register_list_collections_tool",
    "register_list_documents_tool",
    "register_query_knowledge_tool",
]
