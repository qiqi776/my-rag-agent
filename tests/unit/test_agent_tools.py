from __future__ import annotations

import pytest

from src.agent.contracts import ToolContext, ToolRequest
from src.agent.tools import (
    AnswerQuestionTool,
    DeleteDocumentAgentTool,
    ListDocumentsAgentTool,
    SearchKnowledgeTool,
)
from src.application.document_service import DeleteDocumentResult, DocumentSummary
from src.response.answer_builder import AnswerCitation, AnswerOutput
from src.response.response_builder import Citation, SearchOutput, SearchResultItem


class StubSearchService:
    def search(
        self,
        query: str,
        collection: str | None = None,
        top_k: int | None = None,
        mode: str | None = None,
    ) -> SearchOutput:
        return SearchOutput(
            query=query,
            normalized_query=query,
            collection=collection or "knowledge",
            retrieval_mode=mode or "dense",
            result_count=1,
            results=[
                SearchResultItem(
                    rank=1,
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    score=0.9,
                    text="semantic embeddings",
                    source_path="/tmp/python.txt",
                    collection=collection or "knowledge",
                    chunk_index=0,
                    metadata={},
                )
            ],
            citations=[
                Citation(
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    source_path="/tmp/python.txt",
                    collection=collection or "knowledge",
                    chunk_index=0,
                    score=0.9,
                )
            ],
        )


class StubAnswerService:
    def answer(
        self,
        query: str,
        collection: str | None = None,
        top_k: int | None = None,
        mode: str | None = None,
    ) -> AnswerOutput:
        return AnswerOutput(
            query=query,
            normalized_query=query,
            collection=collection or "knowledge",
            retrieval_mode=mode or "dense",
            answer=f"answer for {query}",
            citations=[
                AnswerCitation(
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    source_path="/tmp/python.txt",
                    collection=collection or "knowledge",
                    chunk_index=0,
                    score=0.9,
                )
            ],
            supporting_results=[],
        )


class StubDocumentService:
    def list_documents(self, collection: str | None = None) -> list[DocumentSummary]:
        return [
            DocumentSummary(
                doc_id="doc-1",
                source_path="/tmp/python.txt",
                collection=collection or "knowledge",
                chunk_count=2,
            )
        ]

    def delete_document(
        self,
        doc_id: str,
        collection: str | None = None,
    ) -> DeleteDocumentResult:
        return DeleteDocumentResult(
            doc_id=doc_id,
            collection=collection or "knowledge",
            deleted_chunks=2,
        )


def _context() -> ToolContext:
    return ToolContext(workflow_id="wf-1")


@pytest.mark.unit
def test_search_knowledge_tool_returns_structured_search_output() -> None:
    tool = SearchKnowledgeTool(StubSearchService())

    result = tool.execute(
        ToolRequest(
            name="search_knowledge",
            arguments={"query": "semantic embeddings", "collection": "knowledge"},
            context=_context(),
        )
    )

    assert result.ok is True
    assert result.content == "Found 1 search result(s)."
    assert result.structured_content["kind"] == "search_output"
    assert result.metadata["workflow_id"] == "wf-1"


@pytest.mark.unit
def test_answer_question_tool_returns_structured_answer_output() -> None:
    tool = AnswerQuestionTool(StubAnswerService())

    result = tool.execute(
        ToolRequest(
            name="answer_question",
            arguments={"query": "semantic embeddings", "collection": "knowledge"},
            context=_context(),
        )
    )

    assert result.ok is True
    assert result.structured_content["kind"] == "answer_output"
    assert result.structured_content["answer"] == "answer for semantic embeddings"


@pytest.mark.unit
def test_list_documents_tool_returns_structured_document_list() -> None:
    tool = ListDocumentsAgentTool(StubDocumentService())

    result = tool.execute(
        ToolRequest(
            name="list_documents",
            arguments={"collection": "knowledge"},
            context=_context(),
        )
    )

    assert result.ok is True
    assert result.structured_content["kind"] == "document_list"
    assert result.structured_content["count"] == 1


@pytest.mark.unit
def test_delete_document_tool_returns_delete_result() -> None:
    tool = DeleteDocumentAgentTool(StubDocumentService())

    result = tool.execute(
        ToolRequest(
            name="delete_document",
            arguments={"doc_id": "doc-1", "collection": "knowledge"},
            context=_context(),
        )
    )

    assert result.ok is True
    assert result.structured_content["kind"] == "delete_document_result"
    assert result.structured_content["deleted"] is True
