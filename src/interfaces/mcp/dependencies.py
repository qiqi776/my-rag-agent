"""Shared dependency assembly for MCP tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.adapters.embedding.base_embedding import BaseEmbedding
from src.adapters.embedding.factory import create_embedding
from src.adapters.vector_store.base_vector_store import BaseVectorStore
from src.adapters.vector_store.factory import create_vector_store
from src.application.document_service import DocumentService
from src.application.search_service import SearchService
from src.core.settings import Settings, load_settings
from src.observability.trace_store import TraceStore
from src.retrieval.sparse_retriever import SparseRetriever


@dataclass(frozen=True, slots=True)
class MCPDependencies:
    """Concrete services and providers shared by MCP tools."""

    settings: Settings
    trace_store: TraceStore | None
    embedding: BaseEmbedding
    vector_store: BaseVectorStore
    search_service: SearchService
    document_service: DocumentService


def build_dependencies(config_path: str | Path | None = None) -> MCPDependencies:
    """Load settings and assemble application services for MCP use."""

    settings = load_settings(config_path)
    trace_store = (
        TraceStore(settings.observability.trace_file)
        if settings.observability.trace_enabled
        else None
    )
    embedding = create_embedding(settings)
    vector_store = create_vector_store(settings)
    search_service = SearchService(
        settings=settings,
        embedding=embedding,
        vector_store=vector_store,
        sparse_retriever=SparseRetriever(vector_store),
        trace_store=trace_store,
    )
    document_service = DocumentService(
        settings=settings,
        vector_store=vector_store,
    )
    return MCPDependencies(
        settings=settings,
        trace_store=trace_store,
        embedding=embedding,
        vector_store=vector_store,
        search_service=search_service,
        document_service=document_service,
    )
