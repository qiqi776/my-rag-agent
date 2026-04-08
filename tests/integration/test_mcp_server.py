from __future__ import annotations

from pathlib import Path

import pytest

from src.adapters.embedding.factory import create_embedding
from src.adapters.loader.factory import create_loader
from src.adapters.vector_store.factory import create_vector_store
from src.application.ingest_service import IngestService
from src.core.settings import load_settings
from src.interfaces.mcp.protocol_handler import MCPProtocolHandler
from src.interfaces.mcp.server import create_mcp_server
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


@pytest.mark.integration
def test_mcp_server_reuses_application_services_for_query_and_documents(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    storage_path = tmp_path / "store.json"
    trace_path = tmp_path / "trace.jsonl"
    _write_settings(config_path, storage_path, trace_path)

    settings = load_settings(config_path)
    text_file = tmp_path / "python.txt"
    text_file.write_text(
        "Semantic embeddings help Python retrieval systems answer questions.",
        encoding="utf-8",
    )

    ingest_service = IngestService(
        settings=settings,
        loader=create_loader(settings),
        embedding=create_embedding(settings),
        vector_store=create_vector_store(settings),
        trace_store=TraceStore(trace_path),
    )
    ingest_result = ingest_service.ingest_path(text_file, collection="knowledge")[0]

    server = create_mcp_server(config_path)

    tool_names = [tool.name for tool in server.list_tools()]
    assert tool_names == [
        "delete_document",
        "get_document_summary",
        "list_collections",
        "list_documents",
        "query_knowledge",
    ]

    query_result = server.call_tool(
        "query_knowledge",
        {
            "query": "semantic embeddings",
            "collection": "knowledge",
            "mode": "hybrid",
        },
    )
    assert not query_result.is_error
    assert query_result.structured_content["kind"] == "search_output"
    assert query_result.structured_content["citations"][0]["source_path"].endswith("python.txt")

    list_result = server.call_tool("list_documents", {"collection": "knowledge"})
    assert not list_result.is_error
    assert list_result.structured_content["documents"][0]["doc_id"] == ingest_result.doc_id

    collections_result = server.call_tool("list_collections", {})
    assert not collections_result.is_error
    assert collections_result.structured_content["collections"] == ["knowledge"]

    detail_result = server.call_tool(
        "get_document_summary",
        {"doc_id": ingest_result.doc_id, "collection": "knowledge"},
    )
    assert not detail_result.is_error
    assert detail_result.structured_content["found"] is True
    assert detail_result.structured_content["doc_id"] == ingest_result.doc_id

    delete_result = server.call_tool(
        "delete_document",
        {"doc_id": ingest_result.doc_id, "collection": "knowledge"},
    )
    assert not delete_result.is_error
    assert delete_result.structured_content["deleted"] is True

    not_found_result = server.call_tool("missing_tool", {})
    assert not_found_result.is_error
    assert not_found_result.structured_content["error"]["code"] == "tool_not_found"

    protocol_handler = MCPProtocolHandler(server)
    payload = protocol_handler.handle_payload(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        }
    )
    assert payload["result"]["tools"][0]["name"] == "delete_document"
