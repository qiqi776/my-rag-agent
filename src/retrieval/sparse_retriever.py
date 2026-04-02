"""Sparse retrieval using a lightweight BM25-style index."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from src.adapters.vector_store.base_vector_store import BaseVectorStore
from src.core.types import ChunkRecord, RetrievalResult

_TOKEN_PATTERN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


@dataclass(slots=True)
class SparseIndex:
    """In-memory sparse index for a single collection."""

    records: dict[str, ChunkRecord]
    term_frequencies: dict[str, Counter[str]]
    document_frequencies: dict[str, int]
    document_lengths: dict[str, int]
    average_document_length: float


class SparseRetriever:
    """Build and query a lightweight BM25-style sparse index."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be > 0")
        if b < 0 or b > 1:
            raise ValueError("b must be between 0 and 1")

        self.vector_store = vector_store
        self.k1 = k1
        self.b = b

    def build_index(self, collection: str) -> SparseIndex:
        """Build a sparse index from chunk records in a collection."""

        records = self.vector_store.list_records(collection)
        record_map: dict[str, ChunkRecord] = {}
        term_frequencies: dict[str, Counter[str]] = {}
        document_frequencies: Counter[str] = Counter()
        document_lengths: dict[str, int] = {}
        total_terms = 0

        for record in records:
            tokens = _tokenize(record.text)
            frequencies = Counter(tokens)

            record_map[record.id] = record
            term_frequencies[record.id] = frequencies
            document_lengths[record.id] = sum(frequencies.values())
            total_terms += document_lengths[record.id]

            for term in frequencies:
                document_frequencies[term] += 1

        average_document_length = total_terms / len(records) if records else 0.0

        return SparseIndex(
            records=record_map,
            term_frequencies=term_frequencies,
            document_frequencies=dict(document_frequencies),
            document_lengths=document_lengths,
            average_document_length=average_document_length,
        )

    def retrieve(self, collection: str, query: str, top_k: int) -> list[RetrievalResult]:
        """Run sparse retrieval against a collection."""

        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        query_terms = _tokenize(query)
        if not query_terms:
            return []

        index = self.build_index(collection)
        if not index.records:
            return []

        total_docs = len(index.records)
        average_length = max(index.average_document_length, 1.0)

        scored_results: list[RetrievalResult] = []
        for chunk_id, record in index.records.items():
            term_frequencies = index.term_frequencies[chunk_id]
            document_length = max(index.document_lengths.get(chunk_id, 0), 1)
            score = 0.0

            for term in query_terms:
                tf = term_frequencies.get(term, 0)
                if tf == 0:
                    continue

                df = index.document_frequencies.get(term, 0)
                if df == 0:
                    continue

                idf = math.log(1.0 + ((total_docs - df + 0.5) / (df + 0.5)))
                denominator = tf + self.k1 * (
                    1.0 - self.b + self.b * (document_length / average_length)
                )
                score += idf * ((tf * (self.k1 + 1.0)) / denominator)

            if score <= 0:
                continue

            metadata = record.metadata.copy()
            metadata["sparse_query_terms"] = sorted(set(query_terms))

            scored_results.append(
                RetrievalResult(
                    chunk_id=record.id,
                    doc_id=record.doc_id,
                    score=round(score, 8),
                    text=record.text,
                    metadata=metadata,
                )
            )

        scored_results.sort(key=lambda item: (-item.score, item.chunk_id))
        return scored_results[:top_k]
