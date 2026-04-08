from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.adapters.embedding.factory import create_embedding
from src.adapters.loader.factory import create_loader
from src.adapters.vector_store.factory import create_vector_store
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
retrieval:
  mode: "dense"
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


def _read_framed_message(stream: object) -> dict[str, object]:
    assert hasattr(stream, "readline")
    assert hasattr(stream, "read")
    headers: dict[str, str] = {}
    while True:
        line = stream.readline()
        assert isinstance(line, bytes)
        assert line
        if line in {b"\r\n", b"\n"}:
            break
        key, value = line.decode("ascii").strip().split(":", 1)
        headers[key.strip().lower()] = value.strip()
    content_length = int(headers["content-length"])
    body = stream.read(content_length)
    assert isinstance(body, bytes)
    return json.loads(body.decode("utf-8"))


def _send_request(process: subprocess.Popen[bytes], payload: dict[str, object]) -> dict[str, object]:
    assert process.stdin is not None
    assert process.stdout is not None
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    header = (
        f"Content-Length: {len(body)}\r\n"
        "Content-Type: application/json\r\n"
        "\r\n"
    ).encode("ascii")
    process.stdin.write(header + body)
    process.stdin.flush()
    return _read_framed_message(process.stdout)


@pytest.mark.e2e
def test_mcp_stdio_client_roundtrip(tmp_path: Path) -> None:
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

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "src.interfaces.mcp.server",
            "serve-stdio",
            "--config",
            str(config_path),
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )
    try:
        initialize = _send_request(
            process,
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        )
        tools_list = _send_request(
            process,
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
        tool_call = _send_request(
            process,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "query_knowledge",
                    "arguments": {
                        "query": "semantic embeddings",
                        "collection": "knowledge",
                    },
                },
            },
        )
        invalid = _send_request(
            process,
            {"jsonrpc": "2.0", "id": 4, "method": "missing", "params": {}},
        )
    finally:
        if process.stdin is not None:
            process.stdin.close()
        process.terminate()
        process.wait(timeout=5)

    assert initialize["result"]["serverInfo"]["name"] == "minimal-modular-rag-mcp"
    assert any(tool["name"] == "query_knowledge" for tool in tools_list["result"]["tools"])
    assert tool_call["result"]["structuredContent"]["kind"] == "search_output"
    assert invalid["error"]["code"] == -32601
