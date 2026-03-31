"""CLI entry point for ingestion."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.adapters.embedding.factory import create_embedding
from src.adapters.loader.factory import create_loader
from src.adapters.vector_store.factory import create_vector_store
from src.application.ingest_service import IngestService
from src.core.errors import ConfigError, UnsupportedFileTypeError
from src.core.settings import load_settings
from src.observability.trace_store import TraceStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest local text documents into the MVP store.")
    parser.add_argument("path", help="File or directory to ingest.")
    parser.add_argument("--collection", default=None, help="Target collection name.")
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
        service = IngestService(
            settings=settings,
            loader=create_loader(settings),
            embedding=create_embedding(settings),
            vector_store=create_vector_store(settings),
            trace_store=TraceStore(settings.observability.trace_file)
            if settings.observability.trace_enabled
            else None,
        )
        results = service.ingest_path(args.path, collection=args.collection)
    except (ConfigError, FileNotFoundError, UnsupportedFileTypeError, UnicodeDecodeError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    if not results:
        print("[INFO] No supported files found.")
        return 0

    print(f"[OK] Ingested {len(results)} document(s).")
    for result in results:
        print(
            f"- collection={result.collection} doc_id={result.doc_id[:12]} "
            f"chunks={result.chunk_count} source={result.source_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
