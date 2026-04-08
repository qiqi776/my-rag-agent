"""Shared models for evaluation fixtures and reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from src.response.answer_builder import AnswerOutput
from src.response.response_builder import SearchOutput


@dataclass(slots=True)
class RetrievalEvalCase:
    """Expected retrieval behavior for a single query."""

    name: str
    query: str
    collection: str
    top_k: int = 3
    mode: str | None = None
    expected_doc_ids: list[str] = field(default_factory=list)
    expected_chunk_ids: list[str] = field(default_factory=list)
    expected_source_paths: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RetrievalEvalCaseResult:
    """Measured retrieval behavior for a single query."""

    name: str
    passed: bool
    hit_at_k: bool
    recall_at_k: float
    returned_doc_ids: list[str] = field(default_factory=list)
    returned_chunk_ids: list[str] = field(default_factory=list)
    returned_source_paths: list[str] = field(default_factory=list)
    matched_doc_ids: list[str] = field(default_factory=list)
    matched_chunk_ids: list[str] = field(default_factory=list)
    matched_source_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalEvalReport:
    """Aggregate retrieval regression summary."""

    total_cases: int
    passed_cases: int
    average_hit_at_k: float
    average_recall_at_k: float
    cases: list[RetrievalEvalCaseResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "retrieval_eval",
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "pass_rate": round(self.passed_cases / self.total_cases, 2) if self.total_cases else 0.0,
            "average_hit_at_k": self.average_hit_at_k,
            "average_recall_at_k": self.average_recall_at_k,
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(slots=True)
class AnswerEvalCase:
    """Expected answer behavior for a single query."""

    name: str
    query: str
    collection: str
    top_k: int = 2
    mode: str | None = None
    expected_keywords: list[str] = field(default_factory=list)
    expected_source_paths: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AnswerEvalCaseResult:
    """Measured answer behavior for a single query."""

    name: str
    passed: bool
    answer_nonempty: bool
    keyword_coverage: float
    source_coverage: float
    citation_count: int
    matched_keywords: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)
    matched_source_paths: list[str] = field(default_factory=list)
    missing_source_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnswerEvalReport:
    """Aggregate answer regression summary."""

    total_cases: int
    passed_cases: int
    average_keyword_coverage: float
    average_source_coverage: float
    average_citation_count: float
    cases: list[AnswerEvalCaseResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "answer_eval",
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "pass_rate": round(self.passed_cases / self.total_cases, 2) if self.total_cases else 0.0,
            "average_keyword_coverage": self.average_keyword_coverage,
            "average_source_coverage": self.average_source_coverage,
            "average_citation_count": self.average_citation_count,
            "cases": [case.to_dict() for case in self.cases],
        }


class SearchServiceLike(Protocol):
    """Minimal search-service contract required by retrieval evaluation."""

    def search(
        self,
        query: str,
        collection: str | None = None,
        top_k: int | None = None,
        mode: str | None = None,
    ) -> SearchOutput: ...


class AnswerServiceLike(Protocol):
    """Minimal answer-service contract required by answer evaluation."""

    def answer(
        self,
        query: str,
        collection: str | None = None,
        top_k: int | None = None,
        mode: str | None = None,
    ) -> AnswerOutput: ...
