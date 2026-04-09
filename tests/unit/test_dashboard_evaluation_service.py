from __future__ import annotations

import pytest

from src.observability.dashboard.services.evaluation_service import EvaluationService


@pytest.mark.unit
def test_evaluation_service_runs_and_persists_history(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    history_path = tmp_path / "history.jsonl"

    monkeypatch.setattr(
        "src.observability.dashboard.services.evaluation_service.run_retrieval_evaluation",
        lambda *_args, **_kwargs: type(
            "Report",
            (),
            {
                "to_dict": lambda self: {
                    "kind": "retrieval_eval",
                    "total_cases": 1,
                    "passed_cases": 1,
                    "pass_rate": 1.0,
                    "average_hit_at_k": 1.0,
                    "average_recall_at_k": 1.0,
                    "cases": [],
                }
            },
        )(),
    )

    service = EvaluationService(history_path=history_path)
    payload = service.run("retrieval", retrieval_fixtures="tests/fixtures/evaluation/retrieval_cases.json")
    history = service.list_history()

    assert payload["kind"] == "retrieval_eval"
    assert len(history) == 1
    assert history[0]["mode"] == "retrieval"


@pytest.mark.unit
def test_evaluation_service_rejects_unknown_mode(tmp_path) -> None:
    service = EvaluationService(history_path=tmp_path / "history.jsonl")

    with pytest.raises(ValueError, match="mode must be one of"):
        service.run("invalid")
