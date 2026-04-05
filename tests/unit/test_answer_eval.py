from __future__ import annotations

import pytest

from src.evaluation.answer_eval import AnswerEvalRunner
from src.evaluation.models import AnswerEvalCase
from src.response.answer_builder import AnswerCitation, AnswerOutput
from src.response.response_builder import SearchResultItem


class StubAnswerService:
    def answer(
        self,
        query: str,
        collection: str | None = None,
        top_k: int | None = None,
        mode: str | None = None,
    ) -> AnswerOutput:
        return AnswerOutput(
            query=query,
            normalized_query=query,
            collection=collection or "knowledge",
            retrieval_mode=mode or "hybrid",
            answer="Semantic embeddings help Python retrieval systems answer questions.",
            citations=[
                AnswerCitation(
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    source_path="/tmp/python.txt",
                    collection=collection or "knowledge",
                    chunk_index=0,
                    score=0.9,
                )
            ],
            supporting_results=[
                SearchResultItem(
                    rank=1,
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    score=0.9,
                    text="semantic embeddings help python systems",
                    source_path="/tmp/python.txt",
                    collection=collection or "knowledge",
                    chunk_index=0,
                    metadata={},
                )
            ],
        )


@pytest.mark.unit
def test_answer_eval_runner_checks_keywords_and_sources() -> None:
    report = AnswerEvalRunner(StubAnswerService()).run(
        [
            AnswerEvalCase(
                name="answer",
                query="semantic embeddings",
                collection="knowledge",
                expected_keywords=["semantic", "python"],
                expected_source_paths=["python.txt"],
            )
        ]
    )

    assert report.passed_cases == 1
    assert report.average_keyword_coverage == 1.0
    assert report.average_source_coverage == 1.0


@pytest.mark.unit
def test_answer_eval_runner_reports_missing_keyword_or_source() -> None:
    report = AnswerEvalRunner(StubAnswerService()).run(
        [
            AnswerEvalCase(
                name="missing",
                query="semantic embeddings",
                collection="knowledge",
                expected_keywords=["vector"],
                expected_source_paths=["missing.txt"],
            )
        ]
    )

    assert report.passed_cases == 0
    assert report.cases[0].keyword_coverage == 0.0
    assert report.cases[0].source_coverage == 0.0
