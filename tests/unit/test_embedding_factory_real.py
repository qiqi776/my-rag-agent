from __future__ import annotations

from pathlib import Path

import pytest

from src.adapters.embedding.base_embedding import BaseEmbedding
from src.adapters.embedding.factory import create_embedding
from src.core.errors import ConfigError
from src.core.settings import load_settings


def _write_settings(path: Path, provider_block: str) -> None:
    path.write_text(
        f"""
project:
  name: "minimal-modular-rag"
  environment: "test"
ingestion:
  default_collection: "default"
  chunk_size: 80
  chunk_overlap: 10
  supported_extensions:
    - ".txt"
retrieval:
  dense_top_k: 3
adapters:
  loader:
    provider: "text"
  embedding:
{provider_block}
  vector_store:
    provider: "memory"
    storage_path: "./data/db/vector_store.json"
  llm:
    provider: "fake"
  reranker:
    provider: "fake"
observability:
  trace_enabled: false
  trace_file: "./data/traces/test.jsonl"
  log_level: "INFO"
""".strip(),
        encoding="utf-8",
    )


class _StubEmbedding(BaseEmbedding):
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    @property
    def dimensions(self) -> int:
        return 3

    def embed_text(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]


@pytest.mark.unit
def test_embedding_factory_creates_openai_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    _write_settings(
        config_path,
        """
    provider: "openai"
    model: "text-embedding-3-small"
    api_key: "embedding-key"
    base_url: "https://api.openai.com/v1"
    dimensions: 1536
""".rstrip(),
    )
    settings = load_settings(config_path)
    captured: dict[str, object] = {}

    def _build_stub(**kwargs: object) -> BaseEmbedding:
        captured.update(kwargs)
        return _StubEmbedding(**kwargs)

    monkeypatch.setattr("src.adapters.embedding.factory.OpenAIEmbedding", _build_stub)

    embedding = create_embedding(settings)

    assert isinstance(embedding, _StubEmbedding)
    assert captured == {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "dimensions": 1536,
        "api_key": "embedding-key",
        "base_url": "https://api.openai.com/v1",
        "azure_endpoint": None,
        "deployment_name": None,
        "api_version": None,
    }


@pytest.mark.unit
def test_embedding_factory_rejects_real_provider_without_required_fields(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "settings.yaml"
    _write_settings(
        config_path,
        """
    provider: "openai"
    model: "text-embedding-3-small"
    dimensions: 1536
""".rstrip(),
    )

    with pytest.raises(ConfigError, match="api_key"):
        load_settings(config_path)
