"""Dashboard evaluation service with local history persistence."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core.settings import resolve_path
from src.evaluation.runtime import (
    default_fixture_paths,
    run_all_evaluations,
    run_answer_evaluation,
    run_retrieval_evaluation,
)

_DASHBOARD_CONFIG_ENV = "MRAG_DASHBOARD_CONFIG"
_DASHBOARD_EVAL_HISTORY_ENV = "MRAG_DASHBOARD_EVAL_HISTORY"
_DEFAULT_EVAL_HISTORY = "data/evaluations/history.jsonl"


def _resolve_settings_path(path: str | Path | None) -> str | Path | None:
    return path or os.environ.get(_DASHBOARD_CONFIG_ENV)


class EvaluationService:
    """Run deterministic evaluations and store history entries locally."""

    def __init__(
        self,
        settings_path: str | Path | None = None,
        *,
        history_path: str | Path | None = None,
    ) -> None:
        self._settings_path = _resolve_settings_path(settings_path)
        self.history_path = resolve_path(
            history_path
            or os.environ.get(_DASHBOARD_EVAL_HISTORY_ENV)
            or _DEFAULT_EVAL_HISTORY
        )
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def get_default_fixtures(self) -> dict[str, str]:
        """Return default retrieval and answer fixture paths."""

        return default_fixture_paths()

    def run(
        self,
        mode: str,
        *,
        retrieval_fixtures: str | Path | None = None,
        answer_fixtures: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run the selected evaluation mode and persist a history entry."""

        normalized_mode = mode.strip().lower()
        defaults = self.get_default_fixtures()
        retrieval_path = str(retrieval_fixtures or defaults["retrieval"])
        answer_path = str(answer_fixtures or defaults["answer"])

        if normalized_mode == "retrieval":
            payload = run_retrieval_evaluation(self._settings_path, retrieval_path).to_dict()
        elif normalized_mode == "answer":
            payload = run_answer_evaluation(self._settings_path, answer_path).to_dict()
        elif normalized_mode == "all":
            payload = run_all_evaluations(self._settings_path, retrieval_path, answer_path)
        else:
            raise ValueError("mode must be one of: retrieval, answer, all")

        history_entry = {
            "run_at": datetime.now(UTC).isoformat(),
            "mode": normalized_mode,
            "retrieval_fixtures": retrieval_path,
            "answer_fixtures": answer_path,
            "report": payload,
        }
        self._append_history(history_entry)
        return payload

    def list_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return evaluation history entries, newest first."""

        if not self.history_path.exists():
            return []

        entries: list[dict[str, Any]] = []
        with self.history_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                entries.append(json.loads(raw))
        entries.sort(key=lambda item: str(item.get("run_at", "")), reverse=True)
        return entries[:limit]

    def _append_history(self, payload: dict[str, Any]) -> None:
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
