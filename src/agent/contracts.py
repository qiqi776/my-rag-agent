"""Core contracts for agent-ready tools and workflow orchestration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Protocol

from src.application.document_service import DeleteDocumentResult, DocumentSummary
from src.application.ingest_service import IngestedDocument
from src.core.types import Metadata
from src.response.answer_builder import AnswerOutput
from src.response.response_builder import SearchOutput


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Shared request-scoped metadata passed through tool calls."""

    workflow_id: str | None = None
    trace_id: str | None = None
    metadata: Metadata = field(default_factory=dict)

    def to_dict(self) -> Metadata:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ToolRequest:
    """Normalized tool invocation payload."""

    name: str
    arguments: Metadata = field(default_factory=dict)
    context: ToolContext = field(default_factory=ToolContext)

    def to_dict(self) -> Metadata:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Static description of an agent-ready tool."""

    name: str
    description: str
    input_schema: Metadata

    def to_dict(self) -> Metadata:
        return asdict(self)


@dataclass(slots=True)
class ToolResult:
    """Stable outcome returned by an agent-ready tool."""

    name: str
    ok: bool
    content: str
    structured_content: Metadata = field(default_factory=dict)
    error: str | None = None
    error_code: str | None = None
    metadata: Metadata = field(default_factory=dict)

    def to_dict(self) -> Metadata:
        return asdict(self)

    @classmethod
    def error_result(
        cls,
        name: str,
        message: str,
        *,
        code: str,
        metadata: Metadata | None = None,
    ) -> ToolResult:
        return cls(
            name=name,
            ok=False,
            content="",
            structured_content={
                "kind": "tool_error",
                "error": {
                    "code": code,
                    "message": message,
                },
            },
            error=message,
            error_code=code,
            metadata=metadata or {},
        )


class AgentTool(ABC):
    """Abstract base class for agent-ready tools."""

    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        """Return the static tool definition."""

    @abstractmethod
    def execute(self, request: ToolRequest) -> ToolResult:
        """Execute the tool."""


class SearchServiceLike(Protocol):
    """Search-service surface needed by agent tools."""

    def search(
        self,
        query: str,
        collection: str | None = None,
        top_k: int | None = None,
        mode: str | None = None,
    ) -> SearchOutput: ...


class AnswerServiceLike(Protocol):
    """Answer-service surface needed by agent tools."""

    def answer(
        self,
        query: str,
        collection: str | None = None,
        top_k: int | None = None,
        mode: str | None = None,
    ) -> AnswerOutput: ...


class DocumentServiceLike(Protocol):
    """Document-service surface needed by agent tools."""

    def list_documents(self, collection: str | None = None) -> list[DocumentSummary]: ...

    def delete_document(
        self,
        doc_id: str,
        collection: str | None = None,
    ) -> DeleteDocumentResult: ...


class IngestServiceLike(Protocol):
    """Ingest-service surface reserved for future agent tools."""

    def ingest_path(
        self,
        path: str,
        collection: str | None = None,
    ) -> list[IngestedDocument]: ...
