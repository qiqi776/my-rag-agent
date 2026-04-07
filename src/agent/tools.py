"""Built-in agent-ready tools that reuse existing application services."""

from __future__ import annotations

from typing import Any

from src.agent.contracts import (
    AgentTool,
    AnswerServiceLike,
    DocumentServiceLike,
    SearchServiceLike,
    ToolRequest,
    ToolResult,
    ToolSpec,
)
from src.agent.dependencies import AgentDependencies
from src.agent.registry import ToolRegistry


def _context_metadata(request: ToolRequest) -> dict[str, Any]:
    return request.context.to_dict()


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


def _optional_positive_int(arguments: dict[str, Any], key: str) -> int | None:
    value = arguments.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"'{key}' must be a positive integer when provided")
    return value


class SearchKnowledgeTool(AgentTool):
    """Agent-ready wrapper around SearchService.search."""

    _spec = ToolSpec(
        name="search_knowledge",
        description="Search the knowledge base and return structured results with citations.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text."},
                "collection": {"type": "string", "description": "Optional collection name."},
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
            },
            "required": ["query"],
        },
    )

    def __init__(self, search_service: SearchServiceLike) -> None:
        self.search_service = search_service

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def execute(self, request: ToolRequest) -> ToolResult:
        mode = _optional_string(request.arguments, "mode")
        if mode is not None and mode not in {"dense", "hybrid"}:
            raise ValueError("'mode' must be one of: dense, hybrid")

        output = self.search_service.search(
            query=_require_string(request.arguments, "query"),
            collection=_optional_string(request.arguments, "collection"),
            top_k=_optional_positive_int(request.arguments, "top_k"),
            mode=mode,
        )
        return ToolResult(
            name=self.spec.name,
            ok=True,
            content=f"Found {output.result_count} search result(s).",
            structured_content={
                "kind": "search_output",
                **output.to_dict(),
            },
            metadata=_context_metadata(request),
        )


class AnswerQuestionTool(AgentTool):
    """Agent-ready wrapper around AnswerService.answer."""

    _spec = ToolSpec(
        name="answer_question",
        description="Generate an answer from retrieved context with citations.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Question text."},
                "collection": {"type": "string", "description": "Optional collection name."},
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of supporting results.",
                    "minimum": 1,
                },
                "mode": {
                    "type": "string",
                    "description": "Retrieval mode override.",
                    "enum": ["dense", "hybrid"],
                },
            },
            "required": ["query"],
        },
    )

    def __init__(self, answer_service: AnswerServiceLike) -> None:
        self.answer_service = answer_service

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def execute(self, request: ToolRequest) -> ToolResult:
        mode = _optional_string(request.arguments, "mode")
        if mode is not None and mode not in {"dense", "hybrid"}:
            raise ValueError("'mode' must be one of: dense, hybrid")

        output = self.answer_service.answer(
            query=_require_string(request.arguments, "query"),
            collection=_optional_string(request.arguments, "collection"),
            top_k=_optional_positive_int(request.arguments, "top_k"),
            mode=mode,
        )
        return ToolResult(
            name=self.spec.name,
            ok=True,
            content=output.answer,
            structured_content={
                "kind": "answer_output",
                **output.to_dict(),
            },
            metadata=_context_metadata(request),
        )


class ListDocumentsAgentTool(AgentTool):
    """Agent-ready wrapper around DocumentService.list_documents."""

    _spec = ToolSpec(
        name="list_documents",
        description="List ingested documents, optionally filtered by collection.",
        input_schema={
            "type": "object",
            "properties": {
                "collection": {"type": "string", "description": "Optional collection name."},
            },
        },
    )

    def __init__(self, document_service: DocumentServiceLike) -> None:
        self.document_service = document_service

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def execute(self, request: ToolRequest) -> ToolResult:
        collection = _optional_string(request.arguments, "collection")
        documents = self.document_service.list_documents(collection=collection)
        return ToolResult(
            name=self.spec.name,
            ok=True,
            content=f"Found {len(documents)} document(s).",
            structured_content={
                "kind": "document_list",
                "collection": collection,
                "count": len(documents),
                "documents": [document.to_dict() for document in documents],
            },
            metadata=_context_metadata(request),
        )


class DeleteDocumentAgentTool(AgentTool):
    """Agent-ready wrapper around DocumentService.delete_document."""

    _spec = ToolSpec(
        name="delete_document",
        description="Delete a document from a collection.",
        input_schema={
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "Target document id."},
                "collection": {"type": "string", "description": "Optional collection name."},
            },
            "required": ["doc_id"],
        },
    )

    def __init__(self, document_service: DocumentServiceLike) -> None:
        self.document_service = document_service

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def execute(self, request: ToolRequest) -> ToolResult:
        result = self.document_service.delete_document(
            doc_id=_require_string(request.arguments, "doc_id"),
            collection=_optional_string(request.arguments, "collection"),
        )
        return ToolResult(
            name=self.spec.name,
            ok=True,
            content=(
                f"Deleted {result.deleted_chunks} chunk(s)."
                if result.deleted
                else "Document not found."
            ),
            structured_content={
                "kind": "delete_document_result",
                **result.to_dict(),
            },
            metadata=_context_metadata(request),
        )


def create_tool_registry(dependencies: AgentDependencies) -> ToolRegistry:
    """Build the default agent-ready tool registry."""

    return ToolRegistry(
        [
            SearchKnowledgeTool(dependencies.search_service),
            AnswerQuestionTool(dependencies.answer_service),
            ListDocumentsAgentTool(dependencies.document_service),
            DeleteDocumentAgentTool(dependencies.document_service),
        ]
    )
