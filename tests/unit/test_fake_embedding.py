from __future__ import annotations

import pytest

from src.adapters.embedding.fake_embedding import FakeEmbedding


@pytest.mark.unit
def test_fake_embedding_is_deterministic() -> None:
    embedding = FakeEmbedding(dimensions=8)

    first = embedding.embed_text("python rag system")
    second = embedding.embed_text("python rag system")

    assert first == second


@pytest.mark.unit
def test_fake_embedding_distinguishes_texts() -> None:
    embedding = FakeEmbedding(dimensions=8)

    first = embedding.embed_text("python rag system")
    second = embedding.embed_text("java compiler design")

    assert first != second

