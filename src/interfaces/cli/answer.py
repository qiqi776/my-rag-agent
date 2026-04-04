"""CLI entry point for answer generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.adapters.embedding.factory import create_embedding
from src.adapters.llm.factory import create_llm
from src.adapters.reranker.factory import create_reranker
from src.adapters.vector_store.factory import create_vector_store
from src.application.answer_service import AnswerService
from src.application.search_service import SearchService
from src.core.errors import ConfigError, EmptyQueryError, UnsupportedRetrievalModeError
from src.core.settings import load_settings
from src.observability.trace_store import TraceStore
from src.retrieval.sparse_retriever import SparseRetriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an answer from retrieved context.")
    parser.add_argument("query", help="Query text.")
    parser.add_argument("--collection", default=None, help="Target collection.")
    parser.add_argument("--top-k", type=int, default=None, help="Maximum supporting results.")
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
        trace_store = (
            TraceStore(settings.observability.trace_file)
            if settings.observability.trace_enabled
            else None
        )
        vector_store = create_vector_store(settings)
        search_service = SearchService(
            settings=settings,
            embedding=create_embedding(settings),
            vector_store=vector_store,
            sparse_retriever=SparseRetriever(vector_store),
            trace_store=trace_store,
        )
        service = AnswerService(
            settings=settings,
            search_service=search_service,
            reranker=create_reranker(settings),
            llm=create_llm(settings),
            trace_store=trace_store,
        )
        output = service.answer(
            query=args.query,
            collection=args.collection,
            top_k=args.top_k,
            mode=args.mode,
        )
    except (ConfigError, EmptyQueryError, UnsupportedRetrievalModeError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(
        f"[OK] mode={output.retrieval_mode} "
        f"query='{output.normalized_query}' "
        f"collection={output.collection} "
        f"supporting={len(output.supporting_results)}"
    )
    print(output.answer)
    if output.citations:
        print("")
        print("Citations:")
        for index, citation in enumerate(output.citations, start=1):
            print(
                f"{index:02d}. chunk_id={citation.chunk_id} "
                f"source={citation.source_path} score={citation.score:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
