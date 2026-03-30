from __future__ import annotations

from pathlib import Path

import pytest

from src.core.errors import ConfigError
from src.core.settings import load_settings


def _write_settings(path: Path, chunk_overlap: int = 50) -> None:
    path.write_text(
        f"""
project:
  name: "minimal-modular-rag"
  environment: "test"
ingestion:
  default_collection: "default"
  chunk_size: 500
  chunk_overlap: {chunk_overlap}
  supported_extensions:
    - ".txt"
    - ".md"
retrieval:
  dense_top_k: 5
adapters:
  loader:
    provider: "text"
  embedding:
    provider: "fake"
    dimensions: 16
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
def test_load_settings_success(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    _write_settings(config_path)

    settings = load_settings(config_path)

    assert settings.project.name == "minimal-modular-rag"
    assert settings.ingestion.chunk_size == 500
    assert settings.adapters.embedding.dimensions == 16


@pytest.mark.unit
def test_load_settings_rejects_invalid_overlap(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    _write_settings(config_path, chunk_overlap=500)

    with pytest.raises(ConfigError, match="chunk_overlap"):
        load_settings(config_path)

