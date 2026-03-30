"""CLI entry point for dense-only query."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.adapters.embedding.fake_embedding import FakeEmbedding
from src.adapters.vector_store.in_memory_store import InMemoryVectorStore
from src.application.search_service import SearchService
from src.core.errors import ConfigError, EmptyQueryError
from src.core.settings import load_settings
from src.observability.trace_store import TraceStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query the MVP vector store.")
    parser.add_argument("query", help="Query text.")
    parser.add_argument("--collection", default=None, help="Target collection.")
    parser.add_argument("--top-k", type=int, default=None, help="Maximum number of results.")
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
        service = SearchService(
            settings=settings,
            embedding=FakeEmbedding(settings.adapters.embedding.dimensions),
            vector_store=InMemoryVectorStore(settings.adapters.vector_store.storage_path),
            trace_store=TraceStore(settings.observability.trace_file)
            if settings.observability.trace_enabled
            else None,
        )
        response = service.search(
            query=args.query,
            collection=args.collection,
            top_k=args.top_k,
        )
    except (ConfigError, EmptyQueryError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    if not response.results:
        print("[INFO] No results found.")
        return 0

    print(
        f"[OK] query='{response.query.normalized_query}' "
        f"collection={response.query.collection} returned={len(response.results)}"
    )
    for index, result in enumerate(response.results, start=1):
        snippet = result.text.replace("\n", " ")[:120]
        print(
            f"{index:02d}. score={result.score:.4f} chunk_id={result.chunk_id} "
            f"source={result.metadata.get('source_path', '')}"
        )
        print(f"    {snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
