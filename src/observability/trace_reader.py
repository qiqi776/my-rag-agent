"""Structured trace reading and summarization utilities."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.core.settings import resolve_path
from src.core.trace import TraceType


def _float_or_zero(value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _query_result_count(trace: TraceRecord) -> int | None:
    for stage_name in ("rrf_fuse", "dense_retrieve"):
        stage = trace.stage(stage_name)
        if stage is None:
            continue
        final_result_count = stage["data"].get("final_result_count")
        if isinstance(final_result_count, int):
            return final_result_count

    for stage_name in ("rrf_fuse", "dense_retrieve"):
        stage = trace.stage(stage_name)
        if stage is None:
            continue
        result_count = stage["data"].get("result_count")
        if isinstance(result_count, int):
            return result_count
    return None


def _answer_chars(trace: TraceRecord) -> int | None:
    stage = trace.stage("generate_answer")
    if stage is None:
        return None
    answer_chars = stage["data"].get("answer_chars")
    return answer_chars if isinstance(answer_chars, int) else None


def _ingested_chunk_count(trace: TraceRecord) -> int | None:
    for stage_name, field_name in (("store", "upserted_records"), ("split", "chunk_count")):
        stage = trace.stage(stage_name)
        if stage is None:
            continue
        value = stage["data"].get(field_name)
        if isinstance(value, int):
            return value
    return None


@dataclass(slots=True)
class TraceRecord:
    """A single structured trace record loaded from JSONL."""

    trace_id: str
    trace_type: TraceType
    started_at: str
    finished_at: str | None
    total_elapsed_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)
    stages: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceRecord:
        return cls(
            trace_id=str(data["trace_id"]),
            trace_type=data["trace_type"],
            started_at=str(data["started_at"]),
            finished_at=data.get("finished_at"),
            total_elapsed_ms=_float_or_zero(data.get("total_elapsed_ms")),
            metadata=dict(data.get("metadata", {})),
            stages=list(data.get("stages", [])),
        )

    def stage(self, name: str) -> dict[str, Any] | None:
        for stage in self.stages:
            if stage.get("stage") == name:
                return stage
        return None

    def summary_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_elapsed_ms": self.total_elapsed_ms,
            "stage_count": len(self.stages),
            "metadata": self.metadata,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_elapsed_ms": self.total_elapsed_ms,
            "metadata": self.metadata,
            "stages": self.stages,
        }

    def stage_rows(self) -> list[dict[str, Any]]:
        """Return UI-friendly stage timing rows."""

        rows: list[dict[str, Any]] = []
        for stage in self.stages:
            payload = stage.get("data", {})
            rows.append(
                {
                    "stage_name": str(stage.get("stage", "")),
                    "elapsed_ms": _float_or_zero(stage.get("elapsed_ms")),
                    "data": payload if isinstance(payload, dict) else {},
                }
            )
        return rows


@dataclass(slots=True)
class TraceSummary:
    """Aggregate statistics for a set of traces."""

    trace_type: str
    total_traces: int
    average_total_elapsed_ms: float
    trace_counts: dict[str, int] = field(default_factory=dict)
    stage_counts: dict[str, int] = field(default_factory=dict)
    average_query_result_count: float | None = None
    empty_query_traces: int = 0
    average_answer_chars: float | None = None
    empty_answer_traces: int = 0
    average_ingested_chunks: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TraceReader:
    """Read and summarize append-only JSONL trace files."""

    def __init__(self, trace_file: str | Path) -> None:
        self.path = resolve_path(trace_file)

    def read_all(self) -> list[TraceRecord]:
        if not self.path.exists():
            raise FileNotFoundError(f"Trace file not found: {self.path}")

        traces: list[TraceRecord] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                traces.append(TraceRecord.from_dict(json.loads(raw)))
        traces.sort(key=lambda item: item.started_at, reverse=True)
        return traces

    def list_traces(
        self,
        trace_type: TraceType | None = None,
        limit: int | None = None,
    ) -> list[TraceRecord]:
        traces = self.read_all()
        if trace_type is not None:
            traces = [trace for trace in traces if trace.trace_type == trace_type]
        if limit is not None:
            return traces[:limit]
        return traces

    def get_trace(self, trace_id: str) -> TraceRecord | None:
        for trace in self.read_all():
            if trace.trace_id == trace_id:
                return trace
        return None

    def summarize(self, trace_type: TraceType | None = None) -> TraceSummary:
        traces = self.list_traces(trace_type=trace_type)
        trace_counts = Counter(trace.trace_type for trace in traces)
        stage_counts = Counter(
            stage["stage"]
            for trace in traces
            for stage in trace.stages
            if isinstance(stage.get("stage"), str)
        )

        average_total_elapsed_ms = 0.0
        if traces:
            average_total_elapsed_ms = round(
                sum(trace.total_elapsed_ms for trace in traces) / len(traces),
                2,
            )

        query_counts = [count for trace in traces if (count := _query_result_count(trace)) is not None]
        answer_chars = [count for trace in traces if (count := _answer_chars(trace)) is not None]
        ingested_chunks = [
            count for trace in traces if (count := _ingested_chunk_count(trace)) is not None
        ]

        return TraceSummary(
            trace_type=trace_type or "all",
            total_traces=len(traces),
            average_total_elapsed_ms=average_total_elapsed_ms,
            trace_counts=dict(sorted(trace_counts.items())),
            stage_counts=dict(sorted(stage_counts.items())),
            average_query_result_count=(
                round(sum(query_counts) / len(query_counts), 2) if query_counts else None
            ),
            empty_query_traces=sum(1 for count in query_counts if count == 0),
            average_answer_chars=(
                round(sum(answer_chars) / len(answer_chars), 2) if answer_chars else None
            ),
            empty_answer_traces=sum(1 for count in answer_chars if count == 0),
            average_ingested_chunks=(
                round(sum(ingested_chunks) / len(ingested_chunks), 2)
                if ingested_chunks
                else None
            ),
        )
