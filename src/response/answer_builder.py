"""Stable answer-level response models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from src.core.types import Metadata
from src.response.response_builder import SearchOutput, SearchResultItem


@dataclass(slots=True)
class AnswerCitation:
    """Citation payload for generated answers."""

    chunk_id: str
    doc_id: str
    source_path: str
    collection: str
    chunk_index: int | None
    score: float
    page: int | None = None

    def to_dict(self) -> Metadata:
        return asdict(self)


@dataclass(slots=True)
class AnswerOutput:
    """Stable answer-level output shared across interfaces."""

    query: str
    normalized_query: str
    collection: str
    retrieval_mode: str
    answer: str
    citations: list[AnswerCitation] = field(default_factory=list)
    supporting_results: list[SearchResultItem] = field(default_factory=list)

    def to_dict(self) -> Metadata:
        return {
            "query": self.query,
            "normalized_query": self.normalized_query,
            "collection": self.collection,
            "retrieval_mode": self.retrieval_mode,
            "answer": self.answer,
            "citations": [citation.to_dict() for citation in self.citations],
            "supporting_results": [result.to_dict() for result in self.supporting_results],
        }


class AnswerBuilder:
    """Build stable answer outputs from reranked supporting results."""

    def build(
        self,
        search_output: SearchOutput,
        supporting_results: list[SearchResultItem],
        answer: str,
    ) -> AnswerOutput:
        citations = [
            AnswerCitation(
                chunk_id=result.chunk_id,
                doc_id=result.doc_id,
                source_path=result.source_path,
                collection=result.collection,
                chunk_index=result.chunk_index,
                score=result.score,
                page=result.page,
            )
            for result in supporting_results
        ]
        return AnswerOutput(
            query=search_output.query,
            normalized_query=search_output.normalized_query,
            collection=search_output.collection,
            retrieval_mode=search_output.retrieval_mode,
            answer=answer,
            citations=citations,
            supporting_results=supporting_results,
        )
