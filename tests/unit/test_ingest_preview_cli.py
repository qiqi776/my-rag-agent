from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.interfaces.cli.ingest_preview import main as ingest_preview_main


def _write_text_settings(path: Path) -> None:
    path.write_text(
        """
project:
  name: "minimal-modular-rag"
  environment: "test"
ingestion:
  default_collection: "knowledge"
  chunk_size: 120
  chunk_overlap: 20
  supported_extensions:
    - ".txt"
    - ".md"
retrieval:
  dense_top_k: 3
adapters:
  loader:
    provider: "text"
  embedding:
    provider: "fake"
    dimensions: 16
  vector_store:
    provider: "local_json"
    storage_path: "./data/db/vector_store.json"
observability:
  trace_enabled: false
  trace_file: "./data/traces/test.jsonl"
  log_level: "INFO"
""".strip(),
        encoding="utf-8",
    )


@pytest.mark.unit
def test_ingest_preview_cli_shows_excerpt_without_writing_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "settings.yaml"
    _write_text_settings(config_path)
    text_file = tmp_path / "notes.txt"
    text_file.write_text(
        "Virtual memory lets processes use isolated address spaces for protection.",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mrag-ingest-preview",
            str(text_file),
            "--config",
            str(config_path),
            "--collection",
            "knowledge",
        ],
    )

    assert ingest_preview_main() == 0

    output = capsys.readouterr().out
    assert "Previewed 1 document" in output
    assert "quality_status=n/a" in output
    assert "will_ingest=true" in output
    assert "Virtual memory lets processes use isolated address spaces" in output
    assert not (tmp_path / "store.json").exists()


@pytest.mark.unit
def test_ingest_preview_cli_reports_no_supported_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "settings.yaml"
    _write_text_settings(config_path)
    empty_dir = tmp_path / "docs"
    empty_dir.mkdir()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mrag-ingest-preview",
            str(empty_dir),
            "--config",
            str(config_path),
        ],
    )

    assert ingest_preview_main() == 0
    assert "No supported files found" in capsys.readouterr().out
