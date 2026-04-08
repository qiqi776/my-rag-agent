"""LLM-backed reranker using an OpenAI-compatible chat completion API."""

from __future__ import annotations

import importlib
import importlib.util
import json
from dataclasses import replace
from typing import Any

from src.adapters.reranker.base_reranker import BaseReranker
from src.core.errors import ConfigError
from src.response.response_builder import SearchResultItem


def _load_openai_client_class(class_name: str) -> type[Any]:
    if importlib.util.find_spec("openai") is None:
        raise ConfigError(
            "The 'openai' package is required for the llm reranker provider."
        )

    module = importlib.import_module("openai")
    client_class = getattr(module, class_name, None)
    if not isinstance(client_class, type):
        raise ConfigError(f"openai.{class_name} is not available in the installed package")
    return client_class


class LLMReranker(BaseReranker):
    """Rerank results by asking an LLM to order candidate chunk ids."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        azure_endpoint: str | None = None,
        deployment_name: str | None = None,
        api_version: str | None = None,
    ) -> None:
        self._model = model
        self._deployment_name = deployment_name

        if azure_endpoint is not None:
            client_class = _load_openai_client_class("AzureOpenAI")
            self._client = client_class(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
            )
            return

        client_class = _load_openai_client_class("OpenAI")
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self._client = client_class(**client_kwargs)

    @property
    def provider(self) -> str:
        return "llm"

    def rerank(
        self,
        query: str,
        results: list[SearchResultItem],
        top_k: int | None = None,
    ) -> list[SearchResultItem]:
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not results:
            return []

        response = self._client.chat.completions.create(
            model=self._deployment_name or self._model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=self._build_prompt(query, results),
        )
        content = self._extract_content(response)
        rankings = self._parse_rankings(content)

        items_by_id = {item.chunk_id: item for item in results}
        reranked: list[SearchResultItem] = []
        seen: set[str] = set()
        for index, ranking in enumerate(rankings, start=1):
            chunk_id = ranking["chunk_id"]
            item = items_by_id.get(chunk_id)
            if item is None or chunk_id in seen:
                continue
            seen.add(chunk_id)
            metadata = item.metadata.copy()
            metadata["original_rank"] = item.rank
            metadata["rerank_score"] = ranking["score"]
            reranked.append(replace(item, rank=index, metadata=metadata))

        for item in results:
            if item.chunk_id in seen:
                continue
            metadata = item.metadata.copy()
            metadata["original_rank"] = item.rank
            metadata["rerank_score"] = None
            reranked.append(replace(item, rank=len(reranked) + 1, metadata=metadata))

        return reranked[: top_k or len(reranked)]

    def _build_prompt(
        self,
        query: str,
        results: list[SearchResultItem],
    ) -> list[dict[str, str]]:
        candidate_block = "\n".join(
            (
                f"- chunk_id: {item.chunk_id}\n"
                f"  source: {item.source_path}\n"
                f"  text: {' '.join(item.text.split())[:600]}"
            )
            for item in results
        )
        return [
            {
                "role": "system",
                "content": (
                    "You rerank retrieval candidates. "
                    "Return JSON with a 'ranking' array. "
                    "Each array item must contain 'chunk_id' and numeric 'score'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query.strip()}\n\n"
                    f"Candidates:\n{candidate_block}\n\n"
                    "Return the candidates ordered from most relevant to least relevant."
                ),
            },
        ]

    def _extract_content(self, response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or not choices:
            raise ValueError("LLM reranker response did not include any choices")
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM reranker response did not include text content")
        return content.strip()

    def _parse_rankings(self, content: str) -> list[dict[str, float]]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM reranker returned invalid JSON") from exc

        rankings = payload.get("ranking")
        if not isinstance(rankings, list) or not rankings:
            raise ValueError("LLM reranker response must contain a non-empty 'ranking' array")

        parsed: list[dict[str, float]] = []
        for item in rankings:
            if not isinstance(item, dict):
                raise ValueError("Each reranker ranking entry must be an object")
            chunk_id = item.get("chunk_id")
            score = item.get("score")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError("Each reranker ranking entry requires a non-empty chunk_id")
            if not isinstance(score, int | float):
                raise ValueError("Each reranker ranking entry requires a numeric score")
            parsed.append({"chunk_id": chunk_id, "score": float(score)})
        return parsed
