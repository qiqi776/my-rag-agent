from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.adapters.reranker.llm_reranker import LLMReranker
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


class _FakeChatCompletionsAPI:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))]
        )


class _FakeOpenAIClient:
    init_kwargs: list[dict[str, object]] = []
    next_content = '{"ranking": []}'

    def __init__(self, **kwargs: object) -> None:
        type(self).init_kwargs.append(kwargs)
        self.chat = SimpleNamespace(completions=_FakeChatCompletionsAPI(type(self).next_content))


class _FakeAzureOpenAIClient(_FakeOpenAIClient):
    pass


@pytest.mark.unit
def test_llm_reranker_orders_results_from_llm_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.adapters.reranker.llm_reranker.importlib.util.find_spec",
        lambda name: object(),
    )
    monkeypatch.setattr(
        "src.adapters.reranker.llm_reranker.importlib.import_module",
        lambda name: SimpleNamespace(OpenAI=_FakeOpenAIClient, AzureOpenAI=_FakeAzureOpenAIClient),
    )
    _FakeOpenAIClient.init_kwargs.clear()
    _FakeOpenAIClient.next_content = (
        '{"ranking": [{"chunk_id": "chunk-b", "score": 0.98}, '
        '{"chunk_id": "chunk-a", "score": 0.87}]}'
    )

    reranker = LLMReranker(model="rerank-1", api_key="test-key")
    reranked = reranker.rerank(
        "semantic embeddings",
        [
            _item(1, "chunk-a", "semantic embeddings help retrieval", 0.8),
            _item(2, "chunk-b", "semantic embeddings improve ranking", 0.7),
        ],
    )

    assert [item.chunk_id for item in reranked] == ["chunk-b", "chunk-a"]
    assert reranked[0].metadata["original_rank"] == 2
    assert reranked[0].metadata["rerank_score"] == pytest.approx(0.98)


@pytest.mark.unit
def test_llm_reranker_rejects_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.adapters.reranker.llm_reranker.importlib.util.find_spec",
        lambda name: object(),
    )
    monkeypatch.setattr(
        "src.adapters.reranker.llm_reranker.importlib.import_module",
        lambda name: SimpleNamespace(OpenAI=_FakeOpenAIClient, AzureOpenAI=_FakeAzureOpenAIClient),
    )
    _FakeOpenAIClient.next_content = "not-json"

    reranker = LLMReranker(model="rerank-1", api_key="test-key")

    with pytest.raises(ValueError, match="invalid JSON"):
        reranker.rerank("semantic embeddings", [_item(1, "chunk-a", "text", 0.8)])
