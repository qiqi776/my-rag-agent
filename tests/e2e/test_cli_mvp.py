from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.adapters.vector_store.factory import create_vector_store
from src.core.settings import load_settings
from src.interfaces.cli.documents import main as docs_main
from src.interfaces.cli.ingest import main as ingest_main
from src.interfaces.cli.query import main as query_main


def _write_settings(path: Path, storage_path: Path, trace_path: Path) -> None:
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
    provider: "text"
  embedding:
    provider: "fake"
    dimensions: 16
  vector_store:
    provider: "local_json"
    storage_path: "{storage_path}"
observability:
  trace_enabled: true
  trace_file: "{trace_path}"
  log_level: "INFO"
""".strip(),
        encoding="utf-8",
    )


@pytest.mark.e2e
def test_cli_mvp_ingest_query_and_document_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "settings.yaml"
    storage_path = tmp_path / "store.json"
    trace_path = tmp_path / "trace.jsonl"
    _write_settings(config_path, storage_path, trace_path)

    text_file = tmp_path / "python.txt"
    text_file.write_text(
        "Python retrieval systems use semantic embeddings to answer questions.",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mrag-ingest",
            str(text_file),
            "--collection",
            "knowledge",
            "--config",
            str(config_path),
        ],
    )
    assert ingest_main() == 0
    assert "Ingested 1 document" in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mrag-query",
            "semantic embeddings",
            "--collection",
            "knowledge",
            "--config",
            str(config_path),
        ],
    )
    assert query_main() == 0
    query_output = capsys.readouterr().out
    assert "Found 1 result(s)" in query_output
    assert "python.txt" in query_output

    settings = load_settings(config_path)
    records = create_vector_store(settings).list_records("knowledge")
    assert records
    doc_id = records[0].doc_id

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mrag-docs",
            "list",
            "--collection",
            "knowledge",
            "--config",
            str(config_path),
        ],
    )
    assert docs_main() == 0
    list_output = capsys.readouterr().out
    assert doc_id[:12] in list_output
    assert "python.txt" in list_output

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mrag-docs",
            "delete",
            doc_id,
            "--collection",
            "knowledge",
            "--config",
            str(config_path),
        ],
    )
    assert docs_main() == 0
    assert "Deleted document" in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mrag-query",
            "semantic embeddings",
            "--collection",
            "knowledge",
            "--config",
            str(config_path),
        ],
    )
    assert query_main() == 0
    assert "No results found" in capsys.readouterr().out
