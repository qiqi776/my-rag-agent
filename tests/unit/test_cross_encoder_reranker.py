from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.adapters.reranker.cross_encoder_reranker import CrossEncoderReranker
from src.response.response_builder import SearchResultItem


def _item(rank: int, chunk_id: str, text: str, score: float) -> SearchResultItem:
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


class _FakeCrossEncoder:
    next_scores: list[float] = []

    def __init__(self, model: str) -> None:
        self.model = model
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(pairs)
        return type(self).next_scores


class _ArrayLikeScores:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return list(self._values)


@pytest.mark.unit
def test_cross_encoder_reranker_orders_by_predicted_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.adapters.reranker.cross_encoder_reranker.importlib.util.find_spec",
        lambda name: object(),
    )
    monkeypatch.setattr(
        "src.adapters.reranker.cross_encoder_reranker.importlib.import_module",
        lambda name: SimpleNamespace(CrossEncoder=_FakeCrossEncoder),
    )
    _FakeCrossEncoder.next_scores = _ArrayLikeScores([0.15, 0.91])  # type: ignore[assignment]

    reranker = CrossEncoderReranker(model="cross-encoder-mini")
    reranked = reranker.rerank(
        "virtual memory",
        [
            _item(1, "chunk-a", "disk scheduling", 0.9),
            _item(2, "chunk-b", "virtual memory gives each process an address space", 0.8),
        ],
    )

    assert [item.chunk_id for item in reranked] == ["chunk-b", "chunk-a"]
    assert reranked[0].metadata["rerank_score"] == pytest.approx(0.91)


@pytest.mark.unit
def test_cross_encoder_reranker_rejects_unexpected_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.adapters.reranker.cross_encoder_reranker.importlib.util.find_spec",
        lambda name: object(),
    )
    monkeypatch.setattr(
        "src.adapters.reranker.cross_encoder_reranker.importlib.import_module",
        lambda name: SimpleNamespace(CrossEncoder=_FakeCrossEncoder),
    )
    _FakeCrossEncoder.next_scores = [0.2]

    reranker = CrossEncoderReranker(model="cross-encoder-mini")

    with pytest.raises(ValueError, match="unexpected score payload"):
        reranker.rerank(
            "virtual memory",
            [
                _item(1, "chunk-a", "disk scheduling", 0.9),
                _item(2, "chunk-b", "virtual memory", 0.8),
            ],
        )
