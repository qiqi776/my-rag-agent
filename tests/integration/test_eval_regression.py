from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.adapters.embedding.factory import create_embedding
from src.adapters.loader.factory import create_loader
from src.adapters.vector_store.factory import create_vector_store
from src.application.ingest_service import IngestService
from src.core.settings import load_settings
from src.interfaces.cli.eval import main as eval_main
from src.observability.trace_store import TraceStore


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
def test_eval_cli_runs_retrieval_and_answer_regressions(
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
    ingested = IngestService(
        settings=settings,
        loader=create_loader(settings),
        embedding=create_embedding(settings),
        vector_store=create_vector_store(settings),
        trace_store=TraceStore(trace_path),
    ).ingest_path(text_file, collection="knowledge")

    retrieval_fixture = tmp_path / "retrieval_cases.json"
    retrieval_fixture.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "name": "semantic-python",
                        "query": "semantic embeddings",
                        "collection": "knowledge",
                        "top_k": 1,
                        "mode": "dense",
                        "expected_doc_ids": [ingested[0].doc_id],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    answer_fixture = tmp_path / "answer_cases.json"
    answer_fixture.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "name": "semantic-answer",
                        "query": "semantic embeddings",
                        "collection": "knowledge",
                        "top_k": 1,
                        "mode": "hybrid",
                        "expected_keywords": ["semantic", "embeddings"],
                        "expected_source_paths": ["python.txt"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mrag-eval",
            "--config",
            str(config_path),
            "all",
            "--retrieval-fixtures",
            str(retrieval_fixture),
            "--answer-fixtures",
            str(answer_fixture),
        ],
    )
    assert eval_main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["retrieval"]["passed_cases"] == 1
    assert payload["answer"]["passed_cases"] == 1
