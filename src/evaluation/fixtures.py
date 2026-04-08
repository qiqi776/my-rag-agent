"""Fixture loading helpers for retrieval and answer regression."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.core.settings import REPO_ROOT, resolve_path
from src.evaluation.models import AnswerEvalCase, RetrievalEvalCase

DEFAULT_RETRIEVAL_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "evaluation" / "retrieval_cases.json"
DEFAULT_ANSWER_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "evaluation" / "answer_cases.json"


def _load_cases(path: str | Path) -> list[dict[str, Any]]:
    fixture_path = resolve_path(path)
    with fixture_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError(f"Fixture file must contain a 'cases' list: {fixture_path}")
    return raw_cases


def load_retrieval_cases(path: str | Path) -> list[RetrievalEvalCase]:
    """Load retrieval eval cases from JSON."""

    cases: list[RetrievalEvalCase] = []
    for raw in _load_cases(path):
        top_k = int(raw.get("top_k", 3))
        if top_k <= 0:
            raise ValueError("retrieval fixture top_k must be > 0")
        cases.append(
            RetrievalEvalCase(
                name=str(raw["name"]),
                query=str(raw["query"]),
                collection=str(raw["collection"]),
                top_k=top_k,
                mode=str(raw["mode"]) if raw.get("mode") is not None else None,
                expected_doc_ids=[str(item) for item in raw.get("expected_doc_ids", [])],
                expected_chunk_ids=[str(item) for item in raw.get("expected_chunk_ids", [])],
                expected_source_paths=[
                    str(item) for item in raw.get("expected_source_paths", [])
                ],
            )
        )
    return cases


def load_answer_cases(path: str | Path) -> list[AnswerEvalCase]:
    """Load answer eval cases from JSON."""

    cases: list[AnswerEvalCase] = []
    for raw in _load_cases(path):
        top_k = int(raw.get("top_k", 2))
        if top_k <= 0:
            raise ValueError("answer fixture top_k must be > 0")
        cases.append(
            AnswerEvalCase(
                name=str(raw["name"]),
                query=str(raw["query"]),
                collection=str(raw["collection"]),
                top_k=top_k,
                mode=str(raw["mode"]) if raw.get("mode") is not None else None,
                expected_keywords=[str(item) for item in raw.get("expected_keywords", [])],
                expected_source_paths=[
                    str(item) for item in raw.get("expected_source_paths", [])
                ],
            )
        )
    return cases
