from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.interfaces.mcp.tools.query_knowledge import QueryKnowledgeTool
from src.response.response_builder import Citation, SearchOutput, SearchResultItem


class StubSearchService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def search(
        self,
        query: str,
        collection: str | None = None,
        top_k: int | None = None,
        mode: str | None = None,
    ) -> SearchOutput:
        self.calls.append(
            {
                "query": query,
                "collection": collection,
                "top_k": top_k,
                "mode": mode,
            }
        )
        return SearchOutput(
            query=query,
            normalized_query=query.strip(),
            collection=collection or "default",
            retrieval_mode=mode or "dense",
            result_count=1,
            results=[
                SearchResultItem(
                    rank=1,
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    score=1.0,
                    text="result text",
                    source_path="/tmp/doc.txt",
                    collection=collection or "default",
                    chunk_index=0,
                    metadata={},
                )
            ],
            citations=[
                Citation(
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    source_path="/tmp/doc.txt",
                    collection=collection or "default",
                    chunk_index=0,
                    score=1.0,
                )
            ],
        )


@pytest.mark.unit
def test_query_tool_calls_search_service_and_returns_structured_payload() -> None:
    search_service = StubSearchService()
    tool = QueryKnowledgeTool(SimpleNamespace(search_service=search_service))

    result = tool.handle(
        {
            "query": "semantic embeddings",
            "collection": "knowledge",
            "top_k": 2,
            "mode": "hybrid",
        }
    )

    assert search_service.calls == [
        {
            "query": "semantic embeddings",
            "collection": "knowledge",
            "top_k": 2,
            "mode": "hybrid",
        }
    ]
    assert result.structured_content["kind"] == "search_output"
    assert result.structured_content["citations"][0]["collection"] == "knowledge"


@pytest.mark.unit
def test_query_tool_rejects_invalid_mode() -> None:
    tool = QueryKnowledgeTool(SimpleNamespace(search_service=StubSearchService()))

    with pytest.raises(ValueError, match="mode"):
        tool.handle({"query": "semantic embeddings", "mode": "other"})
