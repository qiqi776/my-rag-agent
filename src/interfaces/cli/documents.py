"""CLI entry point for document lifecycle operations."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.adapters.vector_store.factory import create_vector_store
from src.application.document_service import DocumentService
from src.core.errors import ConfigError
from src.core.settings import load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage ingested documents.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List ingested documents.")
    list_parser.add_argument("--collection", default=None, help="Filter by collection.")
    list_parser.add_argument(
        "--config",
        default=str(Path("config/settings.yaml.example")),
        help="Path to configuration file.",
    )

    delete_parser = subparsers.add_parser("delete", help="Delete a document from a collection.")
    delete_parser.add_argument("doc_id", help="Document id to delete.")
    delete_parser.add_argument("--collection", default=None, help="Target collection.")
    delete_parser.add_argument(
        "--config",
        default=str(Path("config/settings.yaml.example")),
        help="Path to configuration file.",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        settings = load_settings(args.config)
        service = DocumentService(
            settings=settings,
            vector_store=create_vector_store(settings),
        )
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1

    if args.command == "list":
        documents = service.list_documents(collection=args.collection)
        if not documents:
            print("[INFO] No documents found.")
            return 0

        print(f"[OK] Found {len(documents)} document(s).")
        for document in documents:
            print(
                f"- collection={document.collection} doc_id={document.doc_id[:12]} "
                f"chunks={document.chunk_count} source={document.source_path}"
            )
        return 0

    result = service.delete_document(doc_id=args.doc_id, collection=args.collection)
    if not result.deleted:
        print(
            f"[INFO] Document not found: collection={result.collection} "
            f"doc_id={result.doc_id[:12]}"
        )
        return 0

    print(
        f"[OK] Deleted document collection={result.collection} "
        f"doc_id={result.doc_id[:12]} chunks={result.deleted_chunks}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
