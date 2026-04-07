from __future__ import annotations

import pytest

from src.agent.registry import ToolRegistry
from src.agent.tools import AnswerQuestionTool, SearchKnowledgeTool
from src.agent.workflows import WorkflowRunner, create_workflow_runner
from src.response.answer_builder import AnswerOutput
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
            result_count=1,
            results=[
                SearchResultItem(
                    rank=1,
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    score=0.9,
                    text="semantic embeddings support retrieval",
                    source_path="/tmp/python.txt",
                    collection=collection or "knowledge",
                    chunk_index=0,
                    metadata={},
                )
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
            retrieval_mode=mode or "dense",
            answer=f"answer for {query}",
            citations=[],
            supporting_results=[],
        )


@pytest.mark.unit
def test_workflow_runner_executes_research_and_answer_workflow() -> None:
    registry = ToolRegistry(
        [
            SearchKnowledgeTool(StubSearchService()),
            AnswerQuestionTool(StubAnswerService()),
        ]
    )
    runner = create_workflow_runner(registry)

    state = runner.run(
        "research_and_answer",
        {
            "query": "semantic embeddings",
            "collection": "knowledge",
            "mode": "dense",
            "search_top_k": 2,
            "answer_top_k": 1,
        },
    )

    assert state.status == "completed"
    assert state.tools_used == ["search_knowledge", "answer_question"]
    assert state.final_output["answer"] == "answer for semantic embeddings"


@pytest.mark.unit
def test_workflow_runner_rejects_unknown_workflow() -> None:
    runner = WorkflowRunner(ToolRegistry())

    with pytest.raises(ValueError, match="Unknown workflow"):
        runner.run("missing", {"query": "semantic embeddings"})
