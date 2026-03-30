"""Trace context types for ingestion and query flows."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

TraceType = Literal["ingestion", "query"]


@dataclass(slots=True)
class TraceContext:
    """Request-scoped trace context."""

    trace_type: TraceType
    metadata: dict[str, Any] = field(default_factory=dict)
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    finished_at: str | None = None
    stages: list[dict[str, Any]] = field(default_factory=list)
    _started_mono: float = field(default_factory=time.monotonic, repr=False)

    def record_stage(self, stage: str, data: dict[str, Any], elapsed_ms: float | None = None) -> None:
        entry: dict[str, Any] = {
            "stage": stage,
            "timestamp": datetime.now(UTC).isoformat(),
            "data": data,
        }
        if elapsed_ms is not None:
            entry["elapsed_ms"] = round(elapsed_ms, 2)
        self.stages.append(entry)

    def finish(self) -> None:
        self.finished_at = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        end = time.monotonic()
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_elapsed_ms": round((end - self._started_mono) * 1000.0, 2),
            "metadata": self.metadata,
            "stages": self.stages,
        }

