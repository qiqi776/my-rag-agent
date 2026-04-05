from __future__ import annotations

import pytest

from src.evaluation.models import RetrievalEvalCase
from src.evaluation.retrieval_eval import RetrievalEvalRunner
from src.response.response_builder import Citation, SearchOutput, SearchResultItem


class StubSearchService:
    def search(
        self,
        query: str,
        collection: str | None = None,
        top_k: int | None = None,
        mode: str | None = None,
    ) -> SearchOutput:
        return SearchOutput(
            query=query,
            normalized_query=query,
            collection=collection or "knowledge",
            retrieval_mode=mode or "dense",
            result_count=2,
            results=[
                SearchResultItem(
                    rank=1,
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    score=0.9,
                    text="semantic embeddings",
                    source_path="/tmp/python.txt",
                    collection=collection or "knowledge",
                    chunk_index=0,
                    metadata={},
                ),
                SearchResultItem(
                    rank=2,
                    chunk_id="chunk-2",
                    doc_id="doc-2",
                    score=0.5,
                    text="cooking recipes",
                    source_path="/tmp/cooking.txt",
                    collection=collection or "knowledge",
                    chunk_index=0,
                    metadata={},
                ),
            ],
            citations=[
                Citation(
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    source_path="/tmp/python.txt",
                    collection=collection or "knowledge",
                    chunk_index=0,
                    score=0.9,
                )
            ],
        )


@pytest.mark.unit
def test_retrieval_eval_runner_computes_hit_and_recall() -> None:
    report = RetrievalEvalRunner(StubSearchService()).run(
        [
            RetrievalEvalCase(
                name="semantic",
                query="semantic embeddings",
                collection="knowledge",
                top_k=2,
                expected_doc_ids=["doc-1"],
                expected_chunk_ids=["chunk-1"],
            )
        ]
    )

    assert report.passed_cases == 1
    assert report.average_hit_at_k == 1.0
    assert report.average_recall_at_k == 1.0
    assert report.cases[0].matched_doc_ids == ["doc-1"]


@pytest.mark.unit
def test_retrieval_eval_runner_reports_failure_when_expected_items_missing() -> None:
    report = RetrievalEvalRunner(StubSearchService()).run(
        [
            RetrievalEvalCase(
                name="missing",
                query="semantic embeddings",
                collection="knowledge",
                expected_doc_ids=["doc-9"],
            )
        ]
    )

    assert report.passed_cases == 0
    assert report.average_hit_at_k == 0.0
    assert report.average_recall_at_k == 0.0
