from __future__ import annotations

import pytest

from src.retrieval.query_processor import QueryProcessor


@pytest.mark.unit
def test_query_processor_normalizes_keywords_and_filters() -> None:
    processor = QueryProcessor(default_collection="default")

    processed = processor.process(
        "  Virtual   Memory  Paging  ",
        collection="knowledge",
        top_k=4,
        filters={"doc_type": "PDF", "collection": "ignored"},
    )

    assert processed.normalized_query == "Virtual Memory Paging"
    assert processed.collection == "knowledge"
    assert processed.keywords == ["virtual", "memory", "paging"]
    assert processed.filters == {
        "doc_type": "pdf",
        "collection": "ignored",
    }


@pytest.mark.unit
def test_query_processor_uses_default_collection_when_none_provided() -> None:
    processor = QueryProcessor(default_collection="default")

    processed = processor.process(
        "semantic embeddings",
        collection=None,
        top_k=3,
        filters={"doc_type": "text"},
    )

    assert processed.collection == "default"
    assert processed.filters == {"doc_type": "text"}
