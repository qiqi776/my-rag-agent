"""Interactive CLI entry point for repeated question answering."""

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
from src.response.answer_builder import AnswerOutput
from src.retrieval.sparse_retriever import SparseRetriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start an interactive RAG chat session.")
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


def _display_source_path(source_path: str) -> str:
    name = Path(source_path).name
    return name or source_path


def build_answer_service(config_path: str) -> AnswerService:
    settings = load_settings(config_path)
    trace_store = (
        TraceStore(settings.observability.trace_file) if settings.observability.trace_enabled else None
    )
    vector_store = create_vector_store(settings)
    search_service = SearchService(
        settings=settings,
        embedding=create_embedding(settings),
        vector_store=vector_store,
        sparse_retriever=SparseRetriever(vector_store),
        trace_store=trace_store,
    )
    return AnswerService(
        settings=settings,
        search_service=search_service,
        reranker=create_reranker(settings),
        llm=create_llm(settings),
        trace_store=trace_store,
    )


def print_answer(output: AnswerOutput) -> None:
    print("")
    print(
        f"[OK] collection={output.collection} mode={output.retrieval_mode} "
        f"supporting={len(output.supporting_results)}"
    )
    print("Answer:")
    print(output.answer)
    if not output.citations:
        print("")
        return

    print("")
    print("Citations:")
    for index, citation in enumerate(output.citations, start=1):
        details = [f"source={_display_source_path(citation.source_path)}"]
        if citation.page is not None:
            details.append(f"page={citation.page}")
        details.append(f"score={citation.score:.4f}")
        print(f"{index:02d}. {' '.join(details)}")
        print(f"    chunk_id={citation.chunk_id}")
    print("")


def print_help() -> None:
    print("Commands:")
    print("  /help  Show available commands")
    print("  /exit  Exit the chat session")
    print("  <text> Ask a question against the configured collection")


def main() -> int:
    args = build_parser().parse_args()
    try:
        service = build_answer_service(args.config)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("Interactive RAG chat started. Type /help for commands, /exit to quit.")

    while True:
        try:
            user_input = input("mrag> ")
        except EOFError:
            print("")
            return 0
        except KeyboardInterrupt:
            print("")
            return 0

        command = user_input.strip()
        if not command:
            continue
        if command == "/exit":
            print("Bye.")
            return 0
        if command == "/help":
            print_help()
            continue

        try:
            output = service.answer(
                query=command,
                collection=args.collection,
                top_k=args.top_k,
                mode=args.mode,
            )
        except (EmptyQueryError, UnsupportedRetrievalModeError, ValueError) as exc:
            print(f"[ERROR] {exc}")
            continue

        print_answer(output)


if __name__ == "__main__":
    raise SystemExit(main())
