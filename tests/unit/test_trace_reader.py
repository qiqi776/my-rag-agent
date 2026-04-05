from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.observability.trace_reader import TraceReader


def _write_trace_file(path: Path) -> None:
    traces = [
        {
            "trace_id": "trace-answer",
            "trace_type": "answer",
            "started_at": "2026-04-04T08:02:00+00:00",
            "finished_at": "2026-04-04T08:02:01+00:00",
            "total_elapsed_ms": 14.5,
            "metadata": {"query": "semantic embeddings"},
            "stages": [
                {"stage": "retrieve", "data": {"result_count": 2}},
                {"stage": "rerank", "data": {"result_count": 1}},
                {"stage": "generate_answer", "data": {"answer_chars": 120}},
            ],
        },
        {
            "trace_id": "trace-query",
            "trace_type": "query",
            "started_at": "2026-04-04T08:01:00+00:00",
            "finished_at": "2026-04-04T08:01:01+00:00",
            "total_elapsed_ms": 10.0,
            "metadata": {"query": "semantic embeddings"},
            "stages": [
                {"stage": "dense_retrieve", "data": {"result_count": 2}},
            ],
        },
        {
            "trace_id": "trace-ingestion",
            "trace_type": "ingestion",
            "started_at": "2026-04-04T08:00:00+00:00",
            "finished_at": "2026-04-04T08:00:01+00:00",
            "total_elapsed_ms": 20.0,
            "metadata": {"source_path": "/tmp/python.txt"},
            "stages": [
                {"stage": "split", "data": {"chunk_count": 3}},
                {"stage": "store", "data": {"upserted_records": 3}},
            ],
        },
    ]
    path.write_text(
        "\n".join(json.dumps(trace) for trace in traces) + "\n",
        encoding="utf-8",
    )


@pytest.mark.unit
def test_trace_reader_lists_latest_first_and_filters(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    _write_trace_file(trace_path)

    reader = TraceReader(trace_path)

    traces = reader.list_traces(limit=2)
    assert [trace.trace_id for trace in traces] == ["trace-answer", "trace-query"]

    query_only = reader.list_traces(trace_type="query")
    assert [trace.trace_id for trace in query_only] == ["trace-query"]


@pytest.mark.unit
def test_trace_reader_gets_single_trace_and_summarizes(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    _write_trace_file(trace_path)

    reader = TraceReader(trace_path)
    trace = reader.get_trace("trace-answer")

    assert trace is not None
    assert trace.stage("generate_answer") is not None

    summary = reader.summarize()
    assert summary.total_traces == 3
    assert summary.trace_counts == {"answer": 1, "ingestion": 1, "query": 1}
    assert summary.stage_counts["generate_answer"] == 1
    assert summary.average_answer_chars == 120.0
    assert summary.average_ingested_chunks == 3.0
