"""Ingestion application service for the M1 MVP."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from src.adapters.embedding.base_embedding import BaseEmbedding
from src.adapters.loader.base_loader import BaseLoader
from src.adapters.vector_store.base_vector_store import BaseVectorStore
from src.core.settings import Settings
from src.core.trace import TraceContext
from src.core.types import Chunk, ChunkRecord
from src.ingestion.models import IngestionUnit
from src.ingestion.pipeline import IngestionPipeline, create_ingestion_pipeline
from src.observability.logger import get_logger
from src.observability.trace_store import TraceStore


@dataclass(slots=True)
class IngestedDocument:
    """Summary of a processed document."""

    source_path: str
    doc_id: str
    collection: str
    chunk_count: int


class IngestService:
    """Coordinate file discovery, loading, splitting, embedding, and storage."""

    def __init__(
        self,
        settings: Settings,
        loader: BaseLoader,
        embedding: BaseEmbedding,
        vector_store: BaseVectorStore,
        trace_store: TraceStore | None = None,
        pipeline: IngestionPipeline | None = None,
    ) -> None:
        self.settings = settings
        self.loader = loader
        self.embedding = embedding
        self.vector_store = vector_store
        self.trace_store = trace_store
        self.pipeline = pipeline or create_ingestion_pipeline(settings)
        self.logger = get_logger("minimal-rag.ingest", settings.observability.log_level)

    def ingest_path(self, path: str | Path, collection: str | None = None) -> list[IngestedDocument]:
        """Ingest a single file or all supported files within a directory."""

        effective_collection = collection or self.settings.ingestion.default_collection
        files = self._discover_files(path)
        results = [self._ingest_file(file_path, effective_collection) for file_path in files]
        return results

    def _discover_files(self, path: str | Path) -> list[Path]:
        source = Path(path).resolve()
        if source.is_file():
            return [source]

        discovered: list[Path] = []
        for extension in self.settings.ingestion.supported_extensions:
            discovered.extend(source.rglob(f"*{extension}"))
            discovered.extend(source.rglob(f"*{extension.upper()}"))
        return sorted(set(discovered))

    def _ingest_file(self, file_path: Path, collection: str) -> IngestedDocument:
        trace = TraceContext(
            trace_type="ingestion",
            metadata={
                "collection": collection,
                "source_path": str(file_path),
            },
        )

        load_started = time.monotonic()
        document = self.loader.load(file_path)
        trace.record_stage(
            "load",
            {
                "doc_id": document.id,
                "source_path": document.metadata["source_path"],
                "page_count": document.metadata.get("page_count", 1),
            },
            elapsed_ms=(time.monotonic() - load_started) * 1000.0,
        )

        transform_started = time.monotonic()
        prepared = self.pipeline.prepare(document, collection)
        trace.record_stage(
            "transform",
            prepared.to_trace_payload(),
            elapsed_ms=(time.monotonic() - transform_started) * 1000.0,
        )

        split_started = time.monotonic()
        chunks = list(self._split_units(prepared.units, collection))
        trace.record_stage(
            "split",
            {
                "chunk_count": len(chunks),
                "chunk_size": self.settings.ingestion.chunk_size,
                "source_unit_count": prepared.output_unit_count,
            },
            elapsed_ms=(time.monotonic() - split_started) * 1000.0,
        )

        embed_started = time.monotonic()
        embeddings = self.embedding.embed_texts([chunk.text for chunk in chunks])
        records = [
            ChunkRecord(
                id=chunk.id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                embedding=embedding,
                metadata=chunk.metadata.copy(),
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        trace.record_stage(
            "embed",
            {"chunk_count": len(records), "dimensions": self.embedding.dimensions},
            elapsed_ms=(time.monotonic() - embed_started) * 1000.0,
        )

        store_started = time.monotonic()
        upserted = self.vector_store.upsert(collection, records)
        trace.record_stage(
            "store",
            {"upserted_records": upserted, "collection": collection},
            elapsed_ms=(time.monotonic() - store_started) * 1000.0,
        )

        trace.finish()
        if self.trace_store is not None:
            self.trace_store.append(trace)

        return IngestedDocument(
            source_path=document.metadata["source_path"],
            doc_id=document.id,
            collection=collection,
            chunk_count=len(records),
        )

    def _split_units(self, units: Iterable[IngestionUnit], collection: str) -> Iterable[Chunk]:
        chunk_size = self.settings.ingestion.chunk_size
        chunk_overlap = self.settings.ingestion.chunk_overlap
        step = chunk_size - chunk_overlap
        chunk_index = 0

        for unit in units:
            text = unit.text
            if not text.strip():
                continue

            for start in range(0, len(text), step):
                chunk_text = text[start : start + chunk_size]
                if not chunk_text.strip():
                    continue
                end = min(start + chunk_size, len(text))
                content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:12]
                chunk_id = f"{unit.doc_id}_{chunk_index:04d}_{content_hash}"
                metadata = {
                    **unit.metadata,
                    "collection": collection,
                    "doc_id": unit.doc_id,
                    "chunk_index": chunk_index,
                    "start_offset": start,
                    "end_offset": end,
                }
                yield Chunk(
                    id=chunk_id,
                    doc_id=unit.doc_id,
                    text=chunk_text,
                    metadata=metadata,
                )
                chunk_index += 1
                if end >= len(text):
                    break
