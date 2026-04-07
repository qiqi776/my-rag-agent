from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.interfaces.cli.agent import main as agent_main
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
def test_agent_cli_lists_tools_and_runs_workflow(
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
    capsys.readouterr()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mrag-agent",
            "--config",
            str(config_path),
            "list-tools",
        ],
    )
    assert agent_main() == 0
    list_payload = json.loads(capsys.readouterr().out)
    assert [tool["name"] for tool in list_payload["tools"]] == [
        "answer_question",
        "delete_document",
        "list_documents",
        "search_knowledge",
    ]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mrag-agent",
            "--config",
            str(config_path),
            "run-workflow",
            "research_and_answer",
            "semantic embeddings",
            "--collection",
            "knowledge",
            "--mode",
            "hybrid",
        ],
    )
    assert agent_main() == 0
    workflow_payload = json.loads(capsys.readouterr().out)
    assert workflow_payload["status"] == "completed"
    assert workflow_payload["final_output"]["tools_used"] == [
        "search_knowledge",
        "answer_question",
    ]
    assert workflow_payload["final_output"]["citations"][0]["source_path"].endswith("python.txt")
