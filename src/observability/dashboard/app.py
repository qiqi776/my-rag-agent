"""Streamlit dashboard entry point."""

from __future__ import annotations

import streamlit as st


def _page_overview() -> None:
    from src.observability.dashboard.pages.overview import render

    render()


def _page_data_browser() -> None:
    from src.observability.dashboard.pages.data_browser import render

    render()


def _page_ingestion_manager() -> None:
    from src.observability.dashboard.pages.ingestion_manager import render

    render()


def _page_ingestion_traces() -> None:
    from src.observability.dashboard.pages.ingestion_traces import render

    render()


def _page_query_traces() -> None:
    from src.observability.dashboard.pages.query_traces import render

    render()


def _page_evaluation_panel() -> None:
    from src.observability.dashboard.pages.evaluation_panel import render

    render()


def main() -> None:
    """Render the dashboard navigation and selected page."""

    st.set_page_config(
        page_title="Minimal Modular RAG Dashboard",
        layout="wide",
    )

    pages = [
        st.Page(_page_overview, title="Overview", default=True),
        st.Page(_page_data_browser, title="Data Browser"),
        st.Page(_page_ingestion_manager, title="Ingestion Manager"),
        st.Page(_page_ingestion_traces, title="Ingestion Traces"),
        st.Page(_page_query_traces, title="Query Traces"),
        st.Page(_page_evaluation_panel, title="Evaluation Panel"),
    ]

    if hasattr(st, "navigation"):
        st.navigation(pages).run()
        return

    fallback_pages = {
        "Overview": _page_overview,
        "Data Browser": _page_data_browser,
        "Ingestion Manager": _page_ingestion_manager,
        "Ingestion Traces": _page_ingestion_traces,
        "Query Traces": _page_query_traces,
        "Evaluation Panel": _page_evaluation_panel,
    }
    selected = st.sidebar.radio("Pages", list(fallback_pages))
    fallback_pages[selected]()


if __name__ == "__main__":
    main()

