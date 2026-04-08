from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.adapters.embedding.openai_embedding import OpenAIEmbedding
from src.core.errors import ConfigError


class _FakeEmbeddingsAPI:
    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors
        self.calls: list[dict[str, object]] = []

    def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
        self.calls.append({"model": model, "input": input})
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=vector) for vector in self._vectors]
        )


class _FakeOpenAIClient:
    init_kwargs: list[dict[str, object]] = []
    next_vectors: list[list[float]] = []

    def __init__(self, **kwargs: object) -> None:
        type(self).init_kwargs.append(kwargs)
        self.embeddings = _FakeEmbeddingsAPI(type(self).next_vectors)


class _FakeAzureOpenAIClient(_FakeOpenAIClient):
    pass


@pytest.mark.unit
def test_openai_embedding_batches_texts_and_normalizes_blank_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.adapters.embedding.openai_embedding.importlib.util.find_spec",
        lambda name: object(),
    )
    monkeypatch.setattr(
        "src.adapters.embedding.openai_embedding.importlib.import_module",
        lambda name: SimpleNamespace(OpenAI=_FakeOpenAIClient, AzureOpenAI=_FakeAzureOpenAIClient),
    )
    _FakeOpenAIClient.init_kwargs.clear()
    _FakeOpenAIClient.next_vectors = [[0.1, 0.2], [0.3, 0.4]]

    embedding = OpenAIEmbedding(
        provider="openai",
        model="text-embedding-3-small",
        dimensions=2,
        api_key="test-key",
        base_url="https://api.openai.com/v1",
    )

    vectors = embedding.embed_texts(["hello world", "   "])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert _FakeOpenAIClient.init_kwargs == [
        {
            "api_key": "test-key",
            "base_url": "https://api.openai.com/v1",
        }
    ]
    assert embedding._client.embeddings.calls == [  # pyright: ignore[reportPrivateUsage]
        {
            "model": "text-embedding-3-small",
            "input": ["hello world", " "],
        }
    ]


@pytest.mark.unit
def test_openai_embedding_uses_azure_client_and_deployment_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.adapters.embedding.openai_embedding.importlib.util.find_spec",
        lambda name: object(),
    )
    monkeypatch.setattr(
        "src.adapters.embedding.openai_embedding.importlib.import_module",
        lambda name: SimpleNamespace(OpenAI=_FakeOpenAIClient, AzureOpenAI=_FakeAzureOpenAIClient),
    )
    _FakeAzureOpenAIClient.init_kwargs.clear()
    _FakeAzureOpenAIClient.next_vectors = [[1.0, 2.0, 3.0]]

    embedding = OpenAIEmbedding(
        provider="azure",
        model="text-embedding-3-large",
        dimensions=3,
        api_key="azure-key",
        azure_endpoint="https://example.openai.azure.com/",
        deployment_name="embed-deployment",
        api_version="2024-02-15-preview",
    )

    vector = embedding.embed_text("azure embedding")

    assert vector == [1.0, 2.0, 3.0]
    assert _FakeAzureOpenAIClient.init_kwargs == [
        {
            "api_key": "azure-key",
            "azure_endpoint": "https://example.openai.azure.com/",
            "api_version": "2024-02-15-preview",
        }
    ]
    assert embedding._client.embeddings.calls == [  # pyright: ignore[reportPrivateUsage]
        {
            "model": "embed-deployment",
            "input": ["azure embedding"],
        }
    ]


@pytest.mark.unit
def test_openai_embedding_requires_openai_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.adapters.embedding.openai_embedding.importlib.util.find_spec",
        lambda name: None,
    )

    with pytest.raises(ConfigError, match="openai"):
        OpenAIEmbedding(
            provider="openai",
            model="text-embedding-3-small",
            dimensions=2,
            api_key="missing-package",
        )


@pytest.mark.unit
def test_openai_embedding_rejects_response_size_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.adapters.embedding.openai_embedding.importlib.util.find_spec",
        lambda name: object(),
    )
    monkeypatch.setattr(
        "src.adapters.embedding.openai_embedding.importlib.import_module",
        lambda name: SimpleNamespace(OpenAI=_FakeOpenAIClient, AzureOpenAI=_FakeAzureOpenAIClient),
    )
    _FakeOpenAIClient.next_vectors = [[0.1, 0.2]]

    embedding = OpenAIEmbedding(
        provider="openai",
        model="text-embedding-3-small",
        dimensions=2,
        api_key="test-key",
    )

    with pytest.raises(ValueError, match="response size mismatch"):
        embedding.embed_texts(["first", "second"])
