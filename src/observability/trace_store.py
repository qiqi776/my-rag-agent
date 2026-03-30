"""JSONL trace persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.core.settings import resolve_path
from src.core.trace import TraceContext


class TraceStore:
    """Append-only JSONL trace store."""

    def __init__(self, trace_file: str | Path) -> None:
        self.path = resolve_path(trace_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, trace: TraceContext | dict[str, Any]) -> None:
        payload = trace.to_dict() if isinstance(trace, TraceContext) else trace
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

