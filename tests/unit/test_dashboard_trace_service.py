from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.observability.dashboard.services.trace_service import TraceService


@pytest.fixture()
def trace_file(tmp_path: Path) -> Path:
    path = tmp_path / "trace.jsonl"
    payloads = [
        {
            "trace_id": "ing-1",
            "trace_type": "ingestion",
            "started_at": "2026-01-01T00:00:00",
            "finished_at": "2026-01-01T00:00:01",
            "total_elapsed_ms": 120.0,
            "metadata": {"source_path": "/tmp/doc.pdf"},
            "stages": [
                {"stage": "load", "elapsed_ms": 10.0, "data": {"page_count": 1}},
                {"stage": "embed", "elapsed_ms": 20.0, "data": {"chunk_count": 2}},
            ],
        },
        {
            "trace_id": "q-1",
            "trace_type": "query",
            "started_at": "2026-01-02T00:00:00",
            "finished_at": "2026-01-02T00:00:00",
            "total_elapsed_ms": 40.0,
            "metadata": {"query": "semantic embeddings"},
            "stages": [
                {"stage": "dense_retrieve", "elapsed_ms": 15.0, "data": {"candidate_result_count": 3}},
            ],
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload) + "\n")
    return path


@pytest.mark.unit
def test_trace_service_reads_trace_lists_and_stage_rows(trace_file: Path) -> None:
    service = TraceService(trace_file=trace_file)

    traces = service.list_traces("ingestion")
    detail = service.get_trace("ing-1")
    stage_rows = service.get_stage_rows("ing-1")
    summary = service.summarize("query")

    assert len(traces) == 1
    assert detail is not None
    assert detail["trace_id"] == "ing-1"
    assert stage_rows[0]["stage_name"] == "load"
    assert stage_rows[0]["data"]["page_count"] == 1
    assert summary["total_traces"] == 1
