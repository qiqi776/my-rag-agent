"""CLI entry point for dense-only query."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.adapters.embedding.factory import create_embedding
from src.adapters.vector_store.factory import create_vector_store
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
            embedding=create_embedding(settings),
            vector_store=create_vector_store(settings),
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
