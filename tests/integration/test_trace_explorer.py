from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.adapters.embedding.factory import create_embedding
from src.adapters.loader.factory import create_loader
from src.adapters.vector_store.factory import create_vector_store
from src.application.ingest_service import IngestService
from src.application.search_service import SearchService
from src.core.settings import load_settings
from src.interfaces.cli.answer import main as answer_main
from src.interfaces.cli.traces import main as traces_main
from src.observability.trace_store import TraceStore
from src.retrieval.sparse_retriever import SparseRetriever


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


@pytest.mark.integration
def test_trace_cli_lists_shows_and_summarizes_traces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "settings.yaml"
    storage_path = tmp_path / "store.json"
    trace_path = tmp_path / "trace.jsonl"
    _write_settings(config_path, storage_path, trace_path)

    settings = load_settings(config_path)
    text_file = tmp_path / "python.txt"
    text_file.write_text(
        "Python retrieval systems use semantic embeddings to answer questions.",
        encoding="utf-8",
    )
    IngestService(
        settings=settings,
        loader=create_loader(settings),
        embedding=create_embedding(settings),
        vector_store=create_vector_store(settings),
        trace_store=TraceStore(trace_path),
    ).ingest_path(text_file, collection="knowledge")
    vector_store = create_vector_store(settings)
    SearchService(
        settings=settings,
        embedding=create_embedding(settings),
        vector_store=vector_store,
        sparse_retriever=SparseRetriever(vector_store),
        trace_store=TraceStore(trace_path),
    ).search("semantic embeddings", collection="knowledge", top_k=1)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mrag-answer",
            "semantic embeddings",
            "--collection",
            "knowledge",
            "--config",
            str(config_path),
        ],
    )
    assert answer_main() == 0
    capsys.readouterr()

    monkeypatch.setattr(
        sys,
        "argv",
        ["mrag-traces", "--config", str(config_path), "list", "--limit", "5"],
    )
    assert traces_main() == 0
    list_payload = json.loads(capsys.readouterr().out)
    assert len(list_payload["traces"]) == 4
    first_trace_id = list_payload["traces"][0]["trace_id"]

    monkeypatch.setattr(
        sys,
        "argv",
        ["mrag-traces", "--config", str(config_path), "show", first_trace_id],
    )
    assert traces_main() == 0
    show_payload = json.loads(capsys.readouterr().out)
    assert show_payload["trace_id"] == first_trace_id
    assert show_payload["trace_type"] in {"query", "answer", "ingestion"}

    monkeypatch.setattr(
        sys,
        "argv",
        ["mrag-traces", "--config", str(config_path), "stats"],
    )
    assert traces_main() == 0
    stats_payload = json.loads(capsys.readouterr().out)
    assert stats_payload["total_traces"] == 4
    assert stats_payload["trace_counts"]["answer"] == 1
    assert stats_payload["trace_counts"]["query"] == 2
