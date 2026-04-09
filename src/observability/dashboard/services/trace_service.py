"""Dashboard trace service built on top of TraceReader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.core.settings import load_settings
from src.observability.trace_reader import TraceReader, TraceRecord

_DASHBOARD_CONFIG_ENV = "MRAG_DASHBOARD_CONFIG"


def _resolve_settings_path(path: str | Path | None) -> str | Path | None:
    return path or os.environ.get(_DASHBOARD_CONFIG_ENV)


class TraceService:
    """Read and format traces for dashboard consumers."""

    def __init__(
        self,
        trace_file: str | Path | None = None,
        *,
        trace_reader: TraceReader | None = None,
        settings_path: str | Path | None = None,
    ) -> None:
        self._settings_path = _resolve_settings_path(settings_path)
        if trace_reader is not None:
            self._reader = trace_reader
        else:
            resolved_trace_file = trace_file
            if resolved_trace_file is None:
                settings = load_settings(self._settings_path)
                resolved_trace_file = settings.observability.trace_file
            self._reader = TraceReader(resolved_trace_file)

    def list_traces(
        self,
        trace_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return trace summary payloads."""

        return [
            trace.summary_dict()
            for trace in self._reader.list_traces(trace_type=trace_type, limit=limit)
        ]

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """Return a full trace payload."""

        trace = self._reader.get_trace(trace_id)
        return trace.to_dict() if trace is not None else None

    def get_stage_rows(
        self,
        trace: str | TraceRecord | dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Return stage timing rows for a trace id or payload."""

        if isinstance(trace, str):
            record = self._reader.get_trace(trace)
            return [] if record is None else record.stage_rows()
        if isinstance(trace, TraceRecord):
            return trace.stage_rows()
        return TraceRecord.from_dict(trace).stage_rows()

    def summarize(self, trace_type: str | None = None) -> dict[str, Any]:
        """Return aggregate trace statistics."""

        return self._reader.summarize(trace_type=trace_type).to_dict()

