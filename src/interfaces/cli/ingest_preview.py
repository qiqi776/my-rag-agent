"""CLI entry point for previewing extracted ingestion content."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.adapters.loader.factory import create_loader
from src.core.errors import ConfigError, UnsupportedFileTypeError
from src.core.settings import load_settings
from src.ingestion.pipeline import create_ingestion_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview extracted document content before ingestion."
    )
    parser.add_argument("path", help="File or directory to preview.")
    parser.add_argument("--collection", default=None, help="Target collection name.")
    parser.add_argument(
        "--config",
        default=str(Path("config/settings.yaml.example")),
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=240,
        help="Maximum preview characters per document.",
    )
    return parser


def _discover_files(path: str | Path, supported_extensions: list[str]) -> list[Path]:
    source = Path(path).resolve()
    if source.is_file():
        return [source]

    discovered: list[Path] = []
    for extension in supported_extensions:
        discovered.extend(source.rglob(f"*{extension}"))
        discovered.extend(source.rglob(f"*{extension.upper()}"))
    return sorted(set(discovered))


def _first_non_empty_excerpt(texts: list[str], max_chars: int) -> str:
    for text in texts:
        normalized = " ".join(text.split()).strip()
        if not normalized:
            continue
        if len(normalized) <= max_chars:
            return normalized
        return f"{normalized[: max_chars - 3].rstrip()}..."
    return "<empty>"


def main() -> int:
    args = build_parser().parse_args()
    try:
        settings = load_settings(args.config)
        loader = create_loader(settings)
        pipeline = create_ingestion_pipeline(settings)
        collection = args.collection or settings.ingestion.default_collection
        files = _discover_files(args.path, settings.ingestion.supported_extensions)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    if not files:
        print("[INFO] No supported files found.")
        return 0

    try:
        previews: list[dict[str, object]] = []
        for file_path in files:
            document = loader.load(file_path)
            prepared = pipeline.prepare(document, collection)
            preview_text = _first_non_empty_excerpt(
                [unit.text for unit in prepared.units] or [document.text],
                max_chars=args.max_chars,
            )
            quality_status = document.metadata.get("quality_status")
            will_ingest = preview_text != "<empty>" and quality_status != "bad"
            previews.append(
                {
                    "source_path": document.metadata["source_path"],
                    "doc_id": document.id[:12],
                    "collection": collection,
                    "page_count": document.metadata.get("page_count", 1),
                    "quality_status": quality_status or "n/a",
                    "warnings": document.metadata.get("quality_warnings", []),
                    "suggested_loader": settings.adapters.loader.provider,
                    "will_ingest": will_ingest,
                    "transforms": prepared.transforms_applied,
                    "preview": preview_text,
                }
            )
    except (ConfigError, FileNotFoundError, UnsupportedFileTypeError, UnicodeDecodeError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(f"[OK] Previewed {len(previews)} document(s).")
    for preview in previews:
        print(
            f"- source={preview['source_path']} doc_id={preview['doc_id']} "
            f"collection={preview['collection']}"
        )
        print(
            f"  page_count={preview['page_count']} quality_status={preview['quality_status']} "
            f"suggested_loader={preview['suggested_loader']} "
            f"will_ingest={str(preview['will_ingest']).lower()}"
        )
        transforms = preview["transforms"] or ["<none>"]
        print(f"  transforms={','.join(str(item) for item in transforms)}")
        warnings = preview["warnings"] or ["<none>"]
        print(f"  warnings={'; '.join(str(item) for item in warnings)}")
        print(f"  preview={preview['preview']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
