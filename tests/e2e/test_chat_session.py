from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.interfaces.cli.chat import main as chat_main
from src.interfaces.cli.ingest import main as ingest_main


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
  mode: "hybrid"
  dense_top_k: 3
  sparse_top_k: 3
  rrf_k: 60
generation:
  max_context_results: 2
  max_answer_chars: 240
adapters:
  loader:
    provider: "text"
  embedding:
    provider: "fake"
    dimensions: 16
  vector_store:
    provider: "local_json"
    storage_path: "{storage_path}"
  llm:
    provider: "fake"
  reranker:
    provider: "fake"
observability:
  trace_enabled: true
  trace_file: "{trace_path}"
  log_level: "INFO"
""".strip(),
        encoding="utf-8",
    )


@pytest.mark.e2e
def test_chat_session_answers_question_from_ingested_corpus(
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
        "Virtual memory gives each process an isolated address space and simplifies memory management.",
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
    capsys.readouterr()

    inputs = iter(["virtual memory", "/exit"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mrag-chat",
            "--collection",
            "knowledge",
            "--config",
            str(config_path),
        ],
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    assert chat_main() == 0
    output = capsys.readouterr().out
    assert "Interactive RAG chat started" in output
    assert "virtual memory" in output.lower()
    assert "python.txt" in output
