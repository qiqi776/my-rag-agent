from __future__ import annotations

from pathlib import Path

import pytest

from src.adapters.embedding.factory import create_embedding
from src.adapters.loader.factory import create_loader
from src.adapters.vector_store.factory import create_vector_store
from src.agent.dependencies import build_agent_dependencies
from src.agent.tools import create_tool_registry
from src.agent.workflows import create_workflow_runner
from src.application.ingest_service import IngestService
from src.core.settings import load_settings
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
def test_agent_workflow_runs_search_then_answer_against_real_services(tmp_path: Path) -> None:
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

    dependencies = build_agent_dependencies(config_path)
    registry = create_tool_registry(dependencies)
    runner = create_workflow_runner(registry)

    state = runner.run(
        "research_and_answer",
        {
            "query": "semantic embeddings",
            "collection": "knowledge",
            "mode": "hybrid",
            "search_top_k": 2,
            "answer_top_k": 1,
        },
    )

    assert state.status == "completed"
    assert state.final_output["answer"]
    assert state.final_output["tools_used"] == ["search_knowledge", "answer_question"]
    assert state.final_output["citations"][0]["source_path"].endswith("python.txt")
