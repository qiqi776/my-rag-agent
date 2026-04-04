from __future__ import annotations

import pytest

from src.adapters.llm.fake_llm import FakeLLM


@pytest.mark.unit
def test_fake_llm_generates_deterministic_answer() -> None:
    llm = FakeLLM()

    first = llm.generate_answer(
        "semantic embeddings",
        ["semantic embeddings support retrieval"],
        max_chars=200,
    )
    second = llm.generate_answer(
        "semantic embeddings",
        ["semantic embeddings support retrieval"],
        max_chars=200,
    )

    assert first == second
    assert "semantic embeddings" in first


@pytest.mark.unit
def test_fake_llm_handles_missing_context() -> None:
    llm = FakeLLM()

    answer = llm.generate_answer("semantic embeddings", [], max_chars=200)

    assert "No supporting context available" in answer
