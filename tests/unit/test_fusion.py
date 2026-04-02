from __future__ import annotations

import pytest

from src.core.types import RetrievalResult
from src.retrieval.fusion import rrf_fuse


def _result(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        score=score,
        text=f"text for {chunk_id}",
        metadata={
            "source_path": f"/tmp/{chunk_id}.txt",
            "collection": "default",
        },
    )


@pytest.mark.unit
def test_rrf_fuse_boosts_chunks_seen_in_multiple_rankings() -> None:
    dense_results = [
        _result("chunk-a", 0.9),
        _result("chunk-b", 0.8),
        _result("chunk-c", 0.7),
    ]
    sparse_results = [
        _result("chunk-b", 4.0),
        _result("chunk-d", 3.0),
        _result("chunk-a", 2.0),
    ]

    fused = rrf_fuse(
        [
            ("dense", dense_results),
            ("sparse", sparse_results),
        ],
        k=60,
        top_k=3,
    )

    assert [item.chunk_id for item in fused] == ["chunk-b", "chunk-a", "chunk-d"]
    assert fused[0].metadata["rrf_sources"] == ["dense", "sparse"]
    assert fused[0].metadata["rrf_source_ranks"] == {"dense": 2, "sparse": 1}


@pytest.mark.unit
def test_rrf_fuse_preserves_order_for_single_source() -> None:
    dense_results = [
        _result("chunk-1", 1.0),
        _result("chunk-2", 0.5),
        _result("chunk-3", 0.1),
    ]

    fused = rrf_fuse([("dense", dense_results)], k=60, top_k=2)

    assert [item.chunk_id for item in fused] == ["chunk-1", "chunk-2"]
    assert fused[0].metadata["rrf_sources"] == ["dense"]
