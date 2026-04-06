"""CLI entry point for dense-only or hybrid query."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.adapters.embedding.factory import create_embedding
from src.adapters.vector_store.factory import create_vector_store
from src.application.search_service import SearchService
from src.core.errors import (
    ConfigError,
    EmptyQueryError,
    UnsupportedRetrievalModeError,
)
from src.core.settings import load_settings
from src.observability.trace_store import TraceStore
from src.retrieval.sparse_retriever import SparseRetriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query the RAG vector store.")
    parser.add_argument("query", help="Query text.")
    parser.add_argument("--collection", default=None, help="Target collection.")
    parser.add_argument("--top-k", type=int, default=None, help="Maximum number of results.")
    parser.add_argument(
        "--mode",
        choices=("dense", "hybrid"),
        default=None,
        help="Override retrieval mode from config.",
    )
    parser.add_argument(
        "--config",
        default=str(Path("config/settings.yaml.example")),
        help="Path to configuration file.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        settings = load_settings(args.config)
        vector_store = create_vector_store(settings)
        service = SearchService(
            settings=settings,
            embedding=create_embedding(settings),
            vector_store=vector_store,
            sparse_retriever=SparseRetriever(vector_store),
            trace_store=TraceStore(settings.observability.trace_file)
            if settings.observability.trace_enabled
            else None,
        )
        response = service.search(
            query=args.query,
            collection=args.collection,
            top_k=args.top_k,
            mode=args.mode,
        )
    except (
        ConfigError,
        EmptyQueryError,
        UnsupportedRetrievalModeError,
    ) as exc:
        print(f"[ERROR] {exc}")
        return 1

    if not response.results:
        print("[INFO] No results found.")
        return 0

    print(
        f"[OK] mode={response.retrieval_mode} "
        f"query='{response.normalized_query}' "
        f"collection={response.collection} "
        f"returned={response.result_count}"
    )
    for result in response.results:
        snippet = result.text.replace("\n", " ")[:120]
        page_suffix = f" page={result.page}" if result.page is not None else ""
        print(
            f"{result.rank:02d}. score={result.score:.4f} chunk_id={result.chunk_id} "
            f"source={result.source_path}{page_suffix}"
        )
        print(f"    {snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
