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
from src.interfaces.mcp.server import main as mcp_main
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
  mode: "dense"
  dense_top_k: 3
  sparse_top_k: 3
  rrf_k: 60
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
def test_mcp_smoke_list_tools_and_call_query(
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

    monkeypatch.setattr(
        sys,
        "argv",
        ["mrag-mcp", "list-tools", "--config", str(config_path)],
    )
    assert mcp_main() == 0
    list_output = json.loads(capsys.readouterr().out)
    assert [tool["name"] for tool in list_output["tools"]] == [
        "delete_document",
        "list_documents",
        "query_knowledge",
    ]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mrag-mcp",
            "call-tool",
            "query_knowledge",
            "--arguments-json",
            json.dumps(
                {
                    "query": "semantic embeddings",
                    "collection": "knowledge",
                }
            ),
            "--config",
            str(config_path),
        ],
    )
    assert mcp_main() == 0
    call_output = json.loads(capsys.readouterr().out)
    assert call_output["isError"] is False
    assert call_output["structuredContent"]["kind"] == "search_output"
    assert call_output["structuredContent"]["citations"][0]["source_path"].endswith(
        "python.txt"
    )
