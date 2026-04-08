from __future__ import annotations

import sys

import pytest

from src.interfaces.cli.chat import main as chat_main
from src.response.answer_builder import AnswerCitation, AnswerOutput
from src.response.response_builder import Citation, SearchOutput, SearchResultItem


def _build_answer_output(query: str) -> AnswerOutput:
    search_output = SearchOutput(
        query=query,
        normalized_query=query,
        collection="knowledge",
        retrieval_mode="dense",
        result_count=1,
        results=[
            SearchResultItem(
                rank=1,
                chunk_id="chunk-1",
                doc_id="doc-1",
                score=0.9,
                text="Virtual memory gives processes isolated address spaces.",
                source_path="/tmp/ostep.pdf",
                collection="knowledge",
                chunk_index=0,
                metadata={},
            )
        ],
        citations=[
            Citation(
                chunk_id="chunk-1",
                doc_id="doc-1",
                source_path="/tmp/ostep.pdf",
                collection="knowledge",
                chunk_index=0,
                score=0.9,
            )
        ],
    )
    return AnswerOutput(
        query=query,
        normalized_query=query,
        collection="knowledge",
        retrieval_mode="dense",
        answer=f"Answer for {query}",
        citations=[
            AnswerCitation(
                chunk_id="chunk-1",
                doc_id="doc-1",
                source_path="/tmp/ostep.pdf",
                collection="knowledge",
                chunk_index=0,
                score=0.9,
            )
        ],
        supporting_results=search_output.results,
    )


class _StubAnswerService:
    def __init__(self) -> None:
        self.queries: list[dict[str, object]] = []

    def answer(
        self,
        query: str,
        collection: str | None = None,
        top_k: int | None = None,
        mode: str | None = None,
    ) -> AnswerOutput:
        self.queries.append(
            {
                "query": query,
                "collection": collection,
                "top_k": top_k,
                "mode": mode,
            }
        )
        return _build_answer_output(query)


@pytest.mark.unit
def test_chat_cli_runs_multiple_questions_and_exits(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = _StubAnswerService()
    inputs = iter(["what is virtual memory?", "/exit"])

    monkeypatch.setattr(sys, "argv", ["mrag-chat", "--collection", "knowledge", "--mode", "hybrid"])
    monkeypatch.setattr("src.interfaces.cli.chat.build_answer_service", lambda config_path: service)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    assert chat_main() == 0

    output = capsys.readouterr().out
    assert "Interactive RAG chat started" in output
    assert "Answer for what is virtual memory?" in output
    assert "Citations:" in output
    assert service.queries == [
        {
            "query": "what is virtual memory?",
            "collection": "knowledge",
            "top_k": None,
            "mode": "hybrid",
        }
    ]


@pytest.mark.unit
def test_chat_cli_help_command_does_not_call_answer_service(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = _StubAnswerService()
    inputs = iter(["/help", "/exit"])

    monkeypatch.setattr(sys, "argv", ["mrag-chat"])
    monkeypatch.setattr("src.interfaces.cli.chat.build_answer_service", lambda config_path: service)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    assert chat_main() == 0

    output = capsys.readouterr().out
    assert "Commands:" in output
    assert service.queries == []
