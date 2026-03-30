from __future__ import annotations

import pytest

from src.core.types import Chunk, ChunkRecord, Document, ProcessedQuery, RetrievalResult


@pytest.mark.unit
def test_document_requires_source_path() -> None:
    with pytest.raises(ValueError, match="source_path"):
        Document(id="doc-1", text="hello", metadata={})


@pytest.mark.unit
def test_chunk_requires_chunk_index() -> None:
    with pytest.raises(ValueError, match="chunk_index"):
        Chunk(
            id="chunk-1",
            doc_id="doc-1",
            text="hello",
            metadata={"source_path": "/tmp/a.txt"},
        )


@pytest.mark.unit
def test_chunk_record_requires_collection() -> None:
    with pytest.raises(ValueError, match="collection"):
        ChunkRecord(
            id="chunk-1",
            doc_id="doc-1",
            text="hello",
            embedding=[0.1, 0.2],
            metadata={"source_path": "/tmp/a.txt"},
        )


@pytest.mark.unit
def test_types_serialize_roundtrip() -> None:
    document = Document(
        id="doc-1",
        text="hello world",
        metadata={"source_path": "/tmp/a.txt"},
    )
    chunk = Chunk(
        id="chunk-1",
        doc_id="doc-1",
        text="hello",
        metadata={"source_path": "/tmp/a.txt", "chunk_index": 0},
    )
    record = ChunkRecord(
        id="chunk-1",
        doc_id="doc-1",
        text="hello",
        embedding=[0.1, 0.2],
        metadata={
            "source_path": "/tmp/a.txt",
            "chunk_index": 0,
            "collection": "default",
        },
    )
    query = ProcessedQuery(
        original_query=" hello ",
        normalized_query="hello",
        collection="default",
        top_k=5,
    )
    result = RetrievalResult(
        chunk_id="chunk-1",
        doc_id="doc-1",
        score=0.9,
        text="hello",
        metadata={"source_path": "/tmp/a.txt"},
    )

    assert Document.from_dict(document.to_dict()) == document
    assert Chunk.from_dict(chunk.to_dict()) == chunk
    assert ChunkRecord.from_dict(record.to_dict()) == record
    assert query.to_dict()["normalized_query"] == "hello"
    assert result.to_dict()["score"] == 0.9

