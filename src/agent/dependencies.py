"""Dependency assembly for agent-ready tools and workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.adapters.embedding.base_embedding import BaseEmbedding
from src.adapters.llm.factory import create_llm
from src.adapters.loader.base_loader import BaseLoader
from src.adapters.loader.factory import create_loader
from src.adapters.reranker.factory import create_reranker
from src.adapters.vector_store.base_vector_store import BaseVectorStore
from src.application.answer_service import AnswerService
from src.application.document_service import DocumentService
from src.application.ingest_service import IngestService
from src.application.search_service import SearchService
from src.core.settings import Settings
from src.interfaces.mcp.dependencies import build_dependencies
from src.observability.trace_store import TraceStore


@dataclass(frozen=True, slots=True)
class AgentDependencies:
    """Concrete services and adapters shared by agent-ready tools."""

    settings: Settings
    trace_store: TraceStore | None
    embedding: BaseEmbedding
    vector_store: BaseVectorStore
    loader: BaseLoader
    search_service: SearchService
    answer_service: AnswerService
    document_service: DocumentService
    ingest_service: IngestService


def build_agent_dependencies(config_path: str | Path | None = None) -> AgentDependencies:
    """Load settings and assemble all services needed by the agent layer."""

    base = build_dependencies(config_path)
    loader = create_loader(base.settings)
    answer_service = AnswerService(
        settings=base.settings,
        search_service=base.search_service,
        reranker=create_reranker(base.settings),
        llm=create_llm(base.settings),
        trace_store=base.trace_store,
    )
    ingest_service = IngestService(
        settings=base.settings,
        loader=loader,
        embedding=base.embedding,
        vector_store=base.vector_store,
        trace_store=base.trace_store,
    )
    return AgentDependencies(
        settings=base.settings,
        trace_store=base.trace_store,
        embedding=base.embedding,
        vector_store=base.vector_store,
        loader=loader,
        search_service=base.search_service,
        answer_service=answer_service,
        document_service=base.document_service,
        ingest_service=ingest_service,
    )
