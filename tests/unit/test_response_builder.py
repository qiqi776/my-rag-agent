from __future__ import annotations

import pytest

from src.core.types import ProcessedQuery, RetrievalResult
from src.response.response_builder import ResponseBuilder


def _query() -> ProcessedQuery:
    return ProcessedQuery(
        original_query=" semantic embeddings ",
        normalized_query="semantic embeddings",
        collection="knowledge",
        top_k=2,
    )


def _result(
    chunk_id: str,
    score: float,
    *,
    source_path: str = "/tmp/python.txt",
    collection: str = "knowledge",
    chunk_index: int = 0,
    metadata: dict[str, object] | None = None,
) -> RetrievalResult:
    payload = {
        "source_path": source_path,
        "collection": collection,
        "chunk_index": chunk_index,
    }
    if metadata:
        payload.update(metadata)
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        score=score,
        text=f"text for {chunk_id}",
        metadata=payload,
    )


@pytest.mark.unit
def test_response_builder_builds_dense_output_with_citations() -> None:
    builder = ResponseBuilder()

    output = builder.build(
        _query(),
        [
            _result("chunk-a", 0.91, chunk_index=3),
            _result("chunk-b", 0.82, chunk_index=4),
        ],
        retrieval_mode="dense",
    )

    assert output.query == " semantic embeddings "
    assert output.normalized_query == "semantic embeddings"
    assert output.collection == "knowledge"
    assert output.retrieval_mode == "dense"
    assert output.result_count == 2
    assert [item.rank for item in output.results] == [1, 2]
    assert output.results[0].source_path == "/tmp/python.txt"
    assert output.citations[0].chunk_id == "chunk-a"
    assert output.citations[0].chunk_index == 3
    assert output.citations[0].collection == "knowledge"


@pytest.mark.unit
def test_response_builder_preserves_hybrid_metadata() -> None:
    builder = ResponseBuilder()

    output = builder.build(
        _query(),
        [
            _result(
                "chunk-a",
                0.021,
                metadata={
                    "rrf_sources": ["dense", "sparse"],
                    "rrf_source_ranks": {"dense": 1, "sparse": 2},
                },
            )
        ],
        retrieval_mode="hybrid",
    )

    assert output.retrieval_mode == "hybrid"
    assert output.results[0].metadata["rrf_sources"] == ["dense", "sparse"]
    assert output.results[0].metadata["rrf_source_ranks"] == {"dense": 1, "sparse": 2}


@pytest.mark.unit
def test_response_builder_falls_back_to_query_collection_for_missing_metadata() -> None:
    builder = ResponseBuilder()

    output = builder.build(
        _query(),
        [
            RetrievalResult(
                chunk_id="chunk-a",
                doc_id="doc-a",
                score=0.5,
                text="fallback result",
                metadata={"source_path": "/tmp/fallback.txt"},
            )
        ],
        retrieval_mode="dense",
    )

    assert output.results[0].collection == "knowledge"
    assert output.results[0].chunk_index is None
    assert output.citations[0].collection == "knowledge"


@pytest.mark.unit
def test_response_builder_returns_stable_empty_output() -> None:
    builder = ResponseBuilder()

    output = builder.build(_query(), [], retrieval_mode="dense")

    assert output.result_count == 0
    assert output.results == []
    assert output.citations == []
    assert output.to_dict()["results"] == []
