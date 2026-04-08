from __future__ import annotations

from pathlib import Path

import pytest

from src.core.errors import ConfigError
from src.core.settings import load_settings


def _write_config(path: Path, adapters_yaml: str) -> None:
    path.write_text(
        f"""
project:
  name: "minimal-modular-rag"
  environment: "test"
ingestion:
  default_collection: "default"
  chunk_size: 500
  chunk_overlap: 50
  supported_extensions:
    - ".txt"
retrieval:
  mode: "dense"
  dense_top_k: 5
generation:
  max_context_results: 3
  max_answer_chars: 400
adapters:
  loader:
    provider: "text"
{adapters_yaml}
  vector_store:
    provider: "memory"
    storage_path: "./data/db/vector_store.json"
observability:
  trace_enabled: true
  trace_file: "./data/traces/app.jsonl"
  log_level: "INFO"
""".strip(),
        encoding="utf-8",
    )


@pytest.mark.unit
def test_load_settings_supports_real_provider_configuration_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    _write_config(
        config_path,
        """
  embedding:
    provider: "openai"
    model: "text-embedding-3-small"
    api_key: "embedding-key"
    base_url: "https://api.openai.com/v1"
    dimensions: 1536
  llm:
    provider: "azure"
    model: "gpt-4o-mini"
    api_key: "llm-key"
    azure_endpoint: "https://example.openai.azure.com/"
    deployment_name: "gpt-4o-mini"
    api_version: "2024-02-15-preview"
    temperature: 0.2
  reranker:
    provider: "llm"
    model: "rerank-1"
    api_key: "reranker-key"
    base_url: "https://api.openai.com/v1"
""".rstrip(),
    )

    settings = load_settings(config_path)

    assert settings.adapters.embedding.provider == "openai"
    assert settings.adapters.embedding.model == "text-embedding-3-small"
    assert settings.adapters.embedding.api_key == "embedding-key"
    assert settings.adapters.embedding.base_url == "https://api.openai.com/v1"
    assert settings.adapters.embedding.dimensions == 1536
    assert settings.adapters.llm.provider == "azure"
    assert settings.adapters.llm.model == "gpt-4o-mini"
    assert settings.adapters.llm.azure_endpoint == "https://example.openai.azure.com/"
    assert settings.adapters.llm.deployment_name == "gpt-4o-mini"
    assert settings.adapters.llm.api_version == "2024-02-15-preview"
    assert settings.adapters.llm.temperature == pytest.approx(0.2)
    assert settings.adapters.reranker.provider == "llm"
    assert settings.adapters.reranker.model == "rerank-1"
    assert settings.adapters.reranker.api_key == "reranker-key"
    assert settings.adapters.reranker.base_url == "https://api.openai.com/v1"


@pytest.mark.unit
def test_load_settings_rejects_missing_openai_api_key(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    _write_config(
        config_path,
        """
  embedding:
    provider: "openai"
    model: "text-embedding-3-small"
  llm:
    provider: "fake"
  reranker:
    provider: "fake"
""".rstrip(),
    )

    with pytest.raises(ConfigError, match="adapters.embedding.api_key"):
        load_settings(config_path)


@pytest.mark.unit
def test_load_settings_rejects_missing_azure_endpoint(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    _write_config(
        config_path,
        """
  embedding:
    provider: "fake"
  llm:
    provider: "azure"
    model: "gpt-4o-mini"
    api_key: "llm-key"
    deployment_name: "gpt-4o-mini"
    api_version: "2024-02-15-preview"
  reranker:
    provider: "fake"
""".rstrip(),
    )

    with pytest.raises(ConfigError, match="adapters.llm.azure_endpoint"):
        load_settings(config_path)


@pytest.mark.unit
def test_load_settings_keeps_fake_provider_defaults_compatible(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    _write_config(
        config_path,
        """
  embedding:
    provider: "fake"
    dimensions: 16
  llm:
    provider: "fake"
  reranker:
    provider: "fake"
""".rstrip(),
    )

    settings = load_settings(config_path)

    assert settings.adapters.embedding.model is None
    assert settings.adapters.embedding.api_key is None
    assert settings.adapters.llm.model is None
    assert settings.adapters.llm.temperature == pytest.approx(0.0)
    assert settings.adapters.reranker.model is None
