from __future__ import annotations

from src.application.document_service import DeleteDocumentResult, DocumentSummary
from src.interfaces.mcp.mappers import (
    map_delete_result,
    map_document_list,
    map_error,
    map_search_output,
)
from src.response.response_builder import Citation, SearchOutput, SearchResultItem


def _search_output() -> SearchOutput:
    return SearchOutput(
        query="semantic embeddings",
        normalized_query="semantic embeddings",
        collection="knowledge",
        retrieval_mode="hybrid",
        result_count=1,
        results=[
            SearchResultItem(
                rank=1,
                chunk_id="chunk-1",
                doc_id="doc-1",
                score=0.98,
                text="semantic embeddings support retrieval",
                source_path="/tmp/python.txt",
                collection="knowledge",
                chunk_index=0,
                metadata={"rrf_sources": ["dense", "sparse"]},
            )
        ],
        citations=[
            Citation(
                chunk_id="chunk-1",
                doc_id="doc-1",
                source_path="/tmp/python.txt",
                collection="knowledge",
                chunk_index=0,
                score=0.98,
            )
        ],
    )


def test_map_search_output_preserves_structured_payload() -> None:
    result = map_search_output(_search_output())

    assert not result.is_error
    assert result.structured_content["kind"] == "search_output"
    assert result.structured_content["citations"][0]["source_path"] == "/tmp/python.txt"
    assert "Search Results" in result.content[0].text


def test_map_document_list_returns_machine_and_human_views() -> None:
    result = map_document_list(
        [
            DocumentSummary(
                doc_id="doc-1",
                source_path="/tmp/python.txt",
                collection="knowledge",
                chunk_count=2,
            )
        ],
        collection="knowledge",
    )

    assert not result.is_error
    assert result.structured_content["documents"][0]["doc_id"] == "doc-1"
    assert "Documents" in result.content[0].text


def test_map_delete_result_preserves_deleted_flag() -> None:
    result = map_delete_result(
        DeleteDocumentResult(
            doc_id="doc-1",
            collection="knowledge",
            deleted_chunks=2,
        )
    )

    assert not result.is_error
    assert result.structured_content["deleted"] is True
    assert "Document Deleted" in result.content[0].text


def test_map_error_marks_result_as_error() -> None:
    result = map_error("bad request", code="invalid_params")

    assert result.is_error
    assert result.structured_content["error"]["code"] == "invalid_params"
    assert result.content[0].text == "Error: bad request"
