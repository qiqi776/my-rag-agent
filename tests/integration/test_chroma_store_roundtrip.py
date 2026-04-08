from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.adapters.vector_store.chroma_store import ChromaVectorStore
from src.core.types import ChunkRecord

_FAKE_CHROMA_REGISTRY: dict[str, dict[str, dict[str, dict[str, object]]]] = {}


def _cosine_distance(left: list[float], right: list[float]) -> float:
    dot = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 1.0
    similarity = dot / (left_norm * right_norm)
    return 1.0 - similarity


class _FakeChromaCollection:
    def __init__(self, storage: dict[str, dict[str, object]]) -> None:
        self._storage = storage

    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, object]],
    ) -> None:
        for chunk_id, embedding, document, metadata in zip(
            ids,
            embeddings,
            documents,
            metadatas,
            strict=True,
        ):
            self._storage[chunk_id] = {
                "embedding": embedding,
                "document": document,
                "metadata": metadata,
            }

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, object] | None,
        include: list[str],
    ) -> dict[str, list[list[object]]]:
        del include
        query_vector = query_embeddings[0]
        rows: list[tuple[float, str, dict[str, object]]] = []
        for chunk_id, payload in self._storage.items():
            metadata = payload["metadata"]
            if where and any(metadata.get(key) != value for key, value in where.items()):
                continue
            distance = _cosine_distance(query_vector, payload["embedding"])
            rows.append((distance, chunk_id, payload))
        rows.sort(key=lambda item: (item[0], item[1]))
        selected = rows[:n_results]
        return {
            "ids": [[item[1] for item in selected]],
            "documents": [[item[2]["document"] for item in selected]],
            "metadatas": [[item[2]["metadata"] for item in selected]],
            "distances": [[item[0] for item in selected]],
        }

    def get(
        self,
        *,
        include: list[str],
        where: dict[str, object] | None = None,
    ) -> dict[str, list[object]]:
        del include
        rows = []
        for chunk_id, payload in sorted(self._storage.items()):
            metadata = payload["metadata"]
            if where and any(metadata.get(key) != value for key, value in where.items()):
                continue
            rows.append((chunk_id, payload))
        return {
            "ids": [item[0] for item in rows],
            "documents": [item[1]["document"] for item in rows],
            "metadatas": [item[1]["metadata"] for item in rows],
            "embeddings": [item[1]["embedding"] for item in rows],
        }

    def delete(self, *, ids: list[str]) -> None:
        for chunk_id in ids:
            self._storage.pop(chunk_id, None)


class _FakePersistentClient:
    def __init__(self, *, path: str) -> None:
        self._collections = _FAKE_CHROMA_REGISTRY.setdefault(path, {})

    def get_or_create_collection(self, name: str, metadata: dict[str, object]) -> _FakeChromaCollection:
        del metadata
        collection = self._collections.setdefault(name, {})
        return _FakeChromaCollection(collection)

    def get_collection(self, *, name: str) -> _FakeChromaCollection:
        if name not in self._collections:
            raise KeyError(name)
        return _FakeChromaCollection(self._collections[name])

    def list_collections(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=name) for name in sorted(self._collections)]


def _record(chunk_id: str, doc_id: str, source_path: str, *, vector: list[float]) -> ChunkRecord:
    return ChunkRecord(
        id=chunk_id,
        doc_id=doc_id,
        text=f"text for {chunk_id}",
        embedding=vector,
        metadata={
            "source_path": source_path,
            "collection": "knowledge",
            "chunk_index": 0 if chunk_id.endswith("1") else 1,
            "doc_type": "text",
        },
    )


@pytest.mark.integration
def test_chroma_store_roundtrip_and_restart(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "src.adapters.vector_store.chroma_store.importlib.util.find_spec",
        lambda name: object(),
    )
    monkeypatch.setattr(
        "src.adapters.vector_store.chroma_store.importlib.import_module",
        lambda name: SimpleNamespace(PersistentClient=_FakePersistentClient),
    )
    _FAKE_CHROMA_REGISTRY.clear()

    storage_path = tmp_path / "chroma-store"
    store = ChromaVectorStore(storage_path)
    store.upsert(
        "knowledge",
        [
            _record("chunk-1", "doc-1", "/tmp/a.txt", vector=[1.0, 0.0]),
            _record("chunk-2", "doc-2", "/tmp/b.txt", vector=[0.0, 1.0]),
        ],
    )
    store.upsert(
        "other",
        [
            _record("chunk-3", "doc-3", "/tmp/c.txt", vector=[0.5, 0.5]),
        ],
    )

    queried = store.query("knowledge", [1.0, 0.0], top_k=2, filters={"doc_type": "text"})
    reopened = ChromaVectorStore(storage_path)
    records = reopened.list_records("knowledge")

    assert store.list_collections() == ["knowledge", "other"]
    assert [result.chunk_id for result in queried] == ["chunk-1", "chunk-2"]
    assert queried[0].metadata["source_path"] == "/tmp/a.txt"
    assert len(records) == 2
    assert records[0].metadata["collection"] == "knowledge"
    assert reopened.delete_doc("knowledge", "doc-1") == 1
    assert [record.doc_id for record in reopened.list_records("knowledge")] == ["doc-2"]
    assert reopened.query("missing", [1.0, 0.0], top_k=1) == []
    assert "missing" not in reopened.list_collections()
