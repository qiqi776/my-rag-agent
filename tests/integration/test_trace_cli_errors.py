from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.interfaces.cli.traces import main as traces_main


@pytest.mark.integration
def test_traces_cli_reports_missing_trace_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_trace_path = tmp_path / "missing.jsonl"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mrag-traces",
            "--trace-file",
            str(missing_trace_path),
            "stats",
        ],
    )

    assert traces_main() == 1
    payload = json.loads(capsys.readouterr().out)
    assert "Trace file not found" in payload["error"]
