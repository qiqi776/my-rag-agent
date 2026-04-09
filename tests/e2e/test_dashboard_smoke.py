from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.observability.dashboard.services.config_service import (
    CollectionStat,
    OverviewSnapshot,
    ProviderCard,
)

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest


def _collect_text(app_test) -> str:
    parts: list[str] = []
    for attr in ("header", "subheader", "caption", "text", "markdown", "info", "warning", "success", "error"):
        for element in getattr(app_test, attr, []):
            parts.append(str(getattr(element, "value", "")))
    return "\n".join(parts)


@pytest.mark.e2e
def test_overview_page_renders() -> None:
    snapshot = OverviewSnapshot(
        provider_cards=[
            ProviderCard(name="Loader", provider="text", model="-"),
            ProviderCard(name="Embedding", provider="fake", model="-"),
        ],
        collections=[CollectionStat(name="knowledge", document_count=1, chunk_count=2)],
        collection_count=1,
        document_count=1,
        chunk_count=2,
        trace_file="/tmp/trace.jsonl",
        trace_exists=True,
        trace_line_count=3,
    )

    def page_script():
        from src.observability.dashboard.pages.overview import render

        render()

    app_test = AppTest.from_function(page_script, default_timeout=10)
    with patch(
        "src.observability.dashboard.pages.overview.ConfigService",
        return_value=MagicMock(get_overview_snapshot=lambda: snapshot),
    ):
        app_test.run()

    assert not app_test.exception
    assert "overview" in _collect_text(app_test).lower()


@pytest.mark.e2e
def test_data_browser_page_renders() -> None:
    service = MagicMock()
    service.list_collections.return_value = ["knowledge"]
    service.list_documents.return_value = []

    def page_script():
        from src.observability.dashboard.pages.data_browser import render

        render()

    app_test = AppTest.from_function(page_script, default_timeout=10)
    with patch(
        "src.observability.dashboard.pages.data_browser.DataService",
        return_value=service,
    ):
        app_test.run()

    assert not app_test.exception
    assert "data browser" in _collect_text(app_test).lower()


@pytest.mark.e2e
def test_ingestion_manager_page_renders() -> None:
    service = MagicMock()
    service.settings.ingestion.default_collection = "knowledge"

    def page_script():
        from src.observability.dashboard.pages.ingestion_manager import render

        render()

    app_test = AppTest.from_function(page_script, default_timeout=10)
    with patch(
        "src.observability.dashboard.pages.ingestion_manager.DashboardIngestionService",
        return_value=service,
    ):
        app_test.run()

    assert not app_test.exception
    assert "ingestion manager" in _collect_text(app_test).lower()


@pytest.mark.e2e
def test_ingestion_traces_page_renders() -> None:
    service = MagicMock()
    service.list_traces.return_value = [
        {
            "trace_id": "ing-1",
            "started_at": "2026-01-01T00:00:00",
            "metadata": {"source_path": "/tmp/doc.pdf"},
        }
    ]
    service.get_trace.return_value = {
        "trace_id": "ing-1",
        "total_elapsed_ms": 12.0,
        "metadata": {"source_path": "/tmp/doc.pdf"},
    }
    service.get_stage_rows.return_value = [
        {"stage_name": "load", "elapsed_ms": 5.0, "data": {"page_count": 1, "quality_status": "good"}},
    ]

    def page_script():
        from src.observability.dashboard.pages.ingestion_traces import render

        render()

    app_test = AppTest.from_function(page_script, default_timeout=10)
    with patch(
        "src.observability.dashboard.pages.ingestion_traces.TraceService",
        return_value=service,
    ):
        app_test.run()

    assert not app_test.exception
    assert "ingestion traces" in _collect_text(app_test).lower()


@pytest.mark.e2e
def test_query_traces_page_renders() -> None:
    service = MagicMock()
    service.list_traces.return_value = [
        {
            "trace_id": "q-1",
            "started_at": "2026-01-01T00:00:00",
            "metadata": {"query": "semantic embeddings", "collection": "knowledge"},
        }
    ]
    service.get_trace.return_value = {
        "trace_id": "q-1",
        "total_elapsed_ms": 15.0,
        "metadata": {"query": "semantic embeddings", "collection": "knowledge"},
    }
    service.get_stage_rows.return_value = [
        {"stage_name": "dense_retrieve", "elapsed_ms": 5.0, "data": {"candidate_result_count": 2}},
        {"stage_name": "rrf_fuse", "elapsed_ms": 2.0, "data": {"final_result_count": 1}},
    ]

    def page_script():
        from src.observability.dashboard.pages.query_traces import render

        render()

    app_test = AppTest.from_function(page_script, default_timeout=10)
    with patch(
        "src.observability.dashboard.pages.query_traces.TraceService",
        return_value=service,
    ):
        app_test.run()

    assert not app_test.exception
    assert "query traces" in _collect_text(app_test).lower()


@pytest.mark.e2e
def test_evaluation_panel_page_renders() -> None:
    service = MagicMock()
    service.get_default_fixtures.return_value = {
        "retrieval": "tests/fixtures/evaluation/retrieval_cases.json",
        "answer": "tests/fixtures/evaluation/answer_cases.json",
    }
    service.list_history.return_value = []

    def page_script():
        from src.observability.dashboard.pages.evaluation_panel import render

        render()

    app_test = AppTest.from_function(page_script, default_timeout=10)
    with patch(
        "src.observability.dashboard.pages.evaluation_panel.EvaluationService",
        return_value=service,
    ):
        app_test.run()

    assert not app_test.exception
    assert "evaluation panel" in _collect_text(app_test).lower()
