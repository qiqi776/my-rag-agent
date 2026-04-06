"""Stable response models for query outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from src.core.types import Metadata, ProcessedQuery, RetrievalResult


def _source_path_for(result: RetrievalResult) -> str:
    return str(result.metadata.get("source_path", ""))


def _collection_for(result: RetrievalResult, query: ProcessedQuery) -> str:
    value = result.metadata.get("collection")
    return str(value) if value is not None else query.collection


def _chunk_index_for(result: RetrievalResult) -> int | None:
    value = result.metadata.get("chunk_index")
    return value if isinstance(value, int) else None


def _page_for(result: RetrievalResult) -> int | None:
    value = result.metadata.get("page")
    return value if isinstance(value, int) else None


@dataclass(slots=True)
class Citation:
    """Stable citation payload for downstream interfaces."""

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
class SearchResultItem:
    """Stable result item payload for downstream interfaces."""

    rank: int
    chunk_id: str
    doc_id: str
    score: float
    text: str
    source_path: str
    collection: str
    chunk_index: int | None
    page: int | None = None
    metadata: Metadata = field(default_factory=dict)

    def to_dict(self) -> Metadata:
        return asdict(self)


@dataclass(slots=True)
class SearchOutput:
    """Normalized query output shared across interfaces."""

    query: str
    normalized_query: str
    collection: str
    retrieval_mode: str
    result_count: int
    results: list[SearchResultItem] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)

    def to_dict(self) -> Metadata:
        return {
            "query": self.query,
            "normalized_query": self.normalized_query,
            "collection": self.collection,
            "retrieval_mode": self.retrieval_mode,
            "result_count": self.result_count,
            "results": [item.to_dict() for item in self.results],
            "citations": [citation.to_dict() for citation in self.citations],
        }


class ResponseBuilder:
    """Build stable query outputs from retrieval results."""

    def build(
        self,
        query: ProcessedQuery,
        results: list[RetrievalResult],
        retrieval_mode: str,
    ) -> SearchOutput:
        items: list[SearchResultItem] = []
        citations: list[Citation] = []

        for rank, result in enumerate(results, start=1):
            source_path = _source_path_for(result)
            collection = _collection_for(result, query)
            chunk_index = _chunk_index_for(result)
            page = _page_for(result)

            items.append(
                SearchResultItem(
                    rank=rank,
                    chunk_id=result.chunk_id,
                    doc_id=result.doc_id,
                    score=result.score,
                    text=result.text,
                    source_path=source_path,
                    collection=collection,
                    chunk_index=chunk_index,
                    page=page,
                    metadata=result.metadata.copy(),
                )
            )
            citations.append(
                Citation(
                    chunk_id=result.chunk_id,
                    doc_id=result.doc_id,
                    source_path=source_path,
                    collection=collection,
                    chunk_index=chunk_index,
                    score=result.score,
                    page=page,
                )
            )

        return SearchOutput(
            query=query.original_query,
            normalized_query=query.normalized_query,
            collection=query.collection,
            retrieval_mode=retrieval_mode,
            result_count=len(items),
            results=items,
            citations=citations,
        )
