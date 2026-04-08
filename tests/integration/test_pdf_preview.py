from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.interfaces.cli.ingest_preview import main as ingest_preview_main

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "ingestion"


def _write_pdf_settings(path: Path, storage_path: Path) -> None:
    path.write_text(
        f"""
project:
  name: "minimal-modular-rag"
  environment: "test"
ingestion:
  default_collection: "knowledge"
  chunk_size: 200
  chunk_overlap: 20
  supported_extensions:
    - ".pdf"
  transforms:
    enabled: true
    order:
      - "metadata_enrichment"
      - "chunk_refinement"
    metadata_enrichment:
      enabled: true
      section_title_max_length: 80
    chunk_refinement:
      enabled: true
      collapse_whitespace: true
retrieval:
  dense_top_k: 3
adapters:
  loader:
    provider: "pdf"
  embedding:
    provider: "fake"
    dimensions: 16
  vector_store:
    provider: "local_json"
    storage_path: "{storage_path}"
observability:
  trace_enabled: false
  trace_file: "./data/traces/test.jsonl"
  log_level: "INFO"
""".strip(),
        encoding="utf-8",
    )


@pytest.mark.integration
def test_pdf_preview_reports_quality_and_does_not_write_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "settings.yaml"
    storage_path = tmp_path / "store.json"
    _write_pdf_settings(config_path, storage_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mrag-ingest-preview",
            str(FIXTURE_DIR / "simple.pdf"),
            "--config",
            str(config_path),
            "--collection",
            "knowledge",
        ],
    )

    assert ingest_preview_main() == 0

    output = capsys.readouterr().out
    assert "Previewed 1 document" in output
    assert "page_count=1" in output
    assert "quality_status=good" in output
    assert "suggested_loader=pdf" in output
    assert "Minimal Modular RAG PDF" in output
    assert not storage_path.exists()
