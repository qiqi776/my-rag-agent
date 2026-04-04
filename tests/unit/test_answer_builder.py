from __future__ import annotations

import pytest

from src.response.answer_builder import AnswerBuilder
from src.response.response_builder import Citation, SearchOutput, SearchResultItem


def _search_output() -> SearchOutput:
    return SearchOutput(
        query="semantic embeddings",
        normalized_query="semantic embeddings",
        collection="knowledge",
        retrieval_mode="hybrid",
        result_count=1,
        results=[
            SearchResultItem(
                rank=1,
                chunk_id="chunk-1",
                doc_id="doc-1",
                score=0.9,
                text="semantic embeddings support retrieval",
                source_path="/tmp/python.txt",
                collection="knowledge",
                chunk_index=0,
                metadata={},
            )
        ],
        citations=[
            Citation(
                chunk_id="chunk-1",
                doc_id="doc-1",
                source_path="/tmp/python.txt",
                collection="knowledge",
                chunk_index=0,
                score=0.9,
            )
        ],
    )


@pytest.mark.unit
def test_answer_builder_builds_answer_output_with_citations() -> None:
    builder = AnswerBuilder()
    search_output = _search_output()

    output = builder.build(
        search_output,
        search_output.results,
        "semantic embeddings support retrieval",
    )

    assert output.query == "semantic embeddings"
    assert output.retrieval_mode == "hybrid"
    assert output.answer == "semantic embeddings support retrieval"
    assert output.citations[0].source_path == "/tmp/python.txt"
    assert output.supporting_results[0].chunk_id == "chunk-1"
