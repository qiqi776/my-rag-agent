"""Chroma-backed vector store with optional runtime dependency loading."""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from src.adapters.vector_store.base_vector_store import BaseVectorStore
from src.core.errors import ConfigError
from src.core.settings import resolve_path
from src.core.types import ChunkRecord, Metadata, RetrievalResult

_INTERNAL_DOC_ID_KEY = "__doc_id"
_JSON_PREFIX = "__json__:"


def _load_chromadb_persistent_client() -> type[Any]:
    if importlib.util.find_spec("chromadb") is None:
        raise ConfigError(
            "The 'chromadb' package is required for the chroma vector store provider."
        )

    module = importlib.import_module("chromadb")
    client_class = getattr(module, "PersistentClient", None)
    if not callable(client_class):
        raise ConfigError("chromadb.PersistentClient is not available in the installed package")
    return client_class


def _encode_metadata(metadata: Metadata, doc_id: str) -> dict[str, str | int | float | bool]:
    encoded: dict[str, str | int | float | bool] = {_INTERNAL_DOC_ID_KEY: doc_id}
    for key, value in metadata.items():
        if isinstance(value, bool | int | float | str):
            encoded[key] = value
            continue
        encoded[key] = f"{_JSON_PREFIX}{json.dumps(value, ensure_ascii=False, sort_keys=True)}"
    return encoded


def _decode_metadata(metadata: dict[str, Any] | None) -> tuple[str, Metadata]:
    raw_metadata = metadata or {}
    doc_id = raw_metadata.get(_INTERNAL_DOC_ID_KEY)
    if not isinstance(doc_id, str) or not doc_id:
        raise ValueError("Chroma record metadata is missing the internal doc_id field")

    decoded: Metadata = {}
    for key, value in raw_metadata.items():
        if key == _INTERNAL_DOC_ID_KEY:
            continue
        if isinstance(value, str) and value.startswith(_JSON_PREFIX):
            decoded[key] = json.loads(value[len(_JSON_PREFIX) :])
            continue
        decoded[key] = value
    return doc_id, decoded


def _distance_to_score(distance: Any) -> float:
    if not isinstance(distance, int | float):
        return 0.0
    return 1.0 / (1.0 + float(distance))


class ChromaVectorStore(BaseVectorStore):
    """Persist chunk embeddings in a local Chroma database."""

    def __init__(self, storage_path: str | Path) -> None:
        self.storage_path = resolve_path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        client_class = _load_chromadb_persistent_client()
        self._client = client_class(path=str(self.storage_path))

    def upsert(self, collection: str, records: list[ChunkRecord]) -> int:
        if not records:
            return 0

        chroma_collection = self._get_collection(collection, create=True)
        chroma_collection.upsert(
            ids=[record.id for record in records],
            embeddings=[record.embedding for record in records],
            documents=[record.text for record in records],
            metadatas=[_encode_metadata(record.metadata, record.doc_id) for record in records],
        )
        return len(records)

    def query(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        filters: Metadata | None = None,
    ) -> list[RetrievalResult]:
        chroma_collection = self._get_collection(collection, create=False)
        if chroma_collection is None:
            return []
        response = chroma_collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=filters or None,
            include=["documents", "metadatas", "distances"],
        )

        ids = self._first_result_row(response.get("ids", []))
        documents = self._first_result_row(response.get("documents", []))
        metadatas = self._first_result_row(response.get("metadatas", []))
        distances = self._first_result_row(response.get("distances", []))

        results: list[RetrievalResult] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            if not isinstance(chunk_id, str) or not isinstance(text, str):
                continue
            doc_id, decoded_metadata = _decode_metadata(metadata if isinstance(metadata, dict) else {})
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    score=_distance_to_score(distance),
                    text=text,
                    metadata=decoded_metadata,
                )
            )

        results.sort(key=lambda item: (-item.score, item.chunk_id))
        return results

    def list_collections(self) -> list[str]:
        collections = self._client.list_collections()
        names: list[str] = []
        for item in collections:
            if isinstance(item, str):
                names.append(item)
                continue
            name = getattr(item, "name", None)
            if isinstance(name, str):
                names.append(name)
        return sorted(names)

    def list_records(self, collection: str) -> list[ChunkRecord]:
        chroma_collection = self._get_collection(collection, create=False)
        if chroma_collection is None:
            return []
        response = chroma_collection.get(include=["documents", "metadatas", "embeddings"])

        ids = response.get("ids", [])
        documents = response.get("documents", [])
        metadatas = response.get("metadatas", [])
        embeddings = response.get("embeddings", [])

        records: list[ChunkRecord] = []
        for chunk_id, text, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            if not isinstance(chunk_id, str) or not isinstance(text, str):
                continue
            doc_id, decoded_metadata = _decode_metadata(metadata if isinstance(metadata, dict) else {})
            if not isinstance(embedding, list) or not all(
                isinstance(value, int | float) for value in embedding
            ):
                raise ValueError("Chroma record is missing a numeric embedding vector")
            records.append(
                ChunkRecord(
                    id=chunk_id,
                    doc_id=doc_id,
                    text=text,
                    embedding=[float(value) for value in embedding],
                    metadata=decoded_metadata,
                )
            )
        return records

    def delete_doc(self, collection: str, doc_id: str) -> int:
        chroma_collection = self._get_collection(collection, create=False)
        if chroma_collection is None:
            return 0
        response = chroma_collection.get(where={_INTERNAL_DOC_ID_KEY: doc_id}, include=[])
        ids = response.get("ids", [])
        if not ids:
            return 0
        chroma_collection.delete(ids=ids)
        return len(ids)

    def _get_collection(self, collection: str, *, create: bool) -> Any | None:
        if create:
            return self._client.get_or_create_collection(
                name=collection,
                metadata={"hnsw:space": "cosine"},
            )

        get_collection = getattr(self._client, "get_collection", None)
        if callable(get_collection):
            try:
                return get_collection(name=collection)
            except Exception:
                return None

        if collection not in self.list_collections():
            return None
        return self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _first_result_row(rows: Any) -> list[Any]:
        if isinstance(rows, list) and rows and isinstance(rows[0], list):
            return rows[0]
        if isinstance(rows, list):
            return rows
        return []
