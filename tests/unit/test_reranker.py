from __future__ import annotations

import pytest

from src.adapters.reranker.fake_reranker import FakeReranker
from src.response.response_builder import SearchResultItem


def _item(rank: int, text: str, score: float, chunk_id: str) -> SearchResultItem:
    return SearchResultItem(
        rank=rank,
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        score=score,
        text=text,
        source_path=f"/tmp/{chunk_id}.txt",
        collection="knowledge",
        chunk_index=rank - 1,
        metadata={},
    )


@pytest.mark.unit
def test_fake_reranker_promotes_higher_overlap_results() -> None:
    reranker = FakeReranker()

    reranked = reranker.rerank(
        "semantic embeddings",
        [
            _item(1, "cooking recipes and heat", 0.99, "chunk-a"),
            _item(2, "semantic embeddings improve retrieval", 0.50, "chunk-b"),
        ],
    )

    assert [item.chunk_id for item in reranked] == ["chunk-b", "chunk-a"]
    assert reranked[0].metadata["original_rank"] == 2
    assert reranked[0].metadata["rerank_overlap"] >= 1


@pytest.mark.unit
def test_fake_reranker_rejects_invalid_top_k() -> None:
    reranker = FakeReranker()

    with pytest.raises(ValueError, match="top_k"):
        reranker.rerank("semantic embeddings", [], top_k=0)
