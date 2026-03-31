from __future__ import annotations

from pathlib import Path

import pytest

from src.adapters.embedding.base_embedding import BaseEmbedding
from src.adapters.embedding.factory import create_embedding
from src.adapters.loader.base_loader import BaseLoader
from src.adapters.loader.factory import create_loader
from src.adapters.vector_store.base_vector_store import BaseVectorStore
from src.adapters.vector_store.factory import create_vector_store
from src.adapters.vector_store.in_memory_store import InMemoryVectorStore
from src.adapters.vector_store.local_json_store import LocalJsonVectorStore
from src.core.errors import ConfigError
from src.core.settings import load_settings


def _write_settings(
    path: Path,
    storage_path: Path,
    vector_provider: str = "local_json",
    loader_provider: str = "text",
    embedding_provider: str = "fake",
) -> None:
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
    - ".md"
retrieval:
  dense_top_k: 3
adapters:
  loader:
    provider: "{loader_provider}"
  embedding:
    provider: "{embedding_provider}"
    dimensions: 16
  vector_store:
    provider: "{vector_provider}"
    storage_path: "{storage_path}"
observability:
  trace_enabled: false
  trace_file: "./data/traces/test.jsonl"
  log_level: "INFO"
""".strip(),
        encoding="utf-8",
    )


@pytest.mark.unit
def test_factories_create_supported_adapters(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    _write_settings(config_path, tmp_path / "store.json", vector_provider="local_json")
    settings = load_settings(config_path)

    loader = create_loader(settings)
    embedding = create_embedding(settings)
    vector_store = create_vector_store(settings)

    assert isinstance(loader, BaseLoader)
    assert isinstance(embedding, BaseEmbedding)
    assert isinstance(vector_store, BaseVectorStore)
    assert isinstance(vector_store, LocalJsonVectorStore)


@pytest.mark.unit
def test_vector_store_factory_supports_memory_provider(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    _write_settings(config_path, tmp_path / "unused.json", vector_provider="memory")
    settings = load_settings(config_path)

    vector_store = create_vector_store(settings)

    assert isinstance(vector_store, InMemoryVectorStore)


@pytest.mark.unit
def test_factory_rejects_unknown_provider(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    _write_settings(config_path, tmp_path / "store.json", vector_provider="unsupported")
    settings = load_settings(config_path)

    with pytest.raises(ConfigError, match="Unsupported vector_store provider"):
        create_vector_store(settings)
