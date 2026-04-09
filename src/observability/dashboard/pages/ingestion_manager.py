"""Ingestion manager page for preview and ingest operations."""

from __future__ import annotations

import streamlit as st

from src.observability.dashboard.services.ingestion_service import DashboardIngestionService


def _render_preview_results(results: list[dict[str, object]]) -> None:
    if not results:
        st.info("No supported files found for preview.")
        return

    st.subheader("Preview Results")
    for item in results:
        with st.container(border=True):
            st.markdown(f"**{item['source_path']}**")
            st.caption(
                f"doc_id=`{item['doc_id']}` · collection=`{item['collection']}` · "
                f"page_count={item['page_count']} · quality_status={item['quality_status']}"
            )
            st.caption(f"Transforms: {', '.join(item['transforms']) or '<none>'}")
            warnings = item["warnings"] or ["<none>"]
            st.caption(f"Warnings: {', '.join(str(value) for value in warnings)}")
            st.text_area(
                "Preview",
                value=str(item["preview"]),
                height=120,
                disabled=True,
                key=f"preview_result_{item['doc_id']}",
            )


def _render_ingest_results(results: list[dict[str, object]]) -> None:
    if not results:
        st.info("No documents were ingested.")
        return

    st.subheader("Ingestion Summary")
    st.dataframe(results, use_container_width=True, hide_index=True)


def render() -> None:
    """Render preview and ingest controls."""

    st.header("Ingestion Manager")
    st.caption("Run preview and ingest directly against a local file or directory.")

    try:
        service = DashboardIngestionService()
        default_collection = service.settings.ingestion.default_collection
    except Exception as exc:
        st.error(f"Failed to initialise ingestion service: {exc}")
        return

    path = st.text_input("Local Path", value="", help="File or directory to preview or ingest.")
    collection = st.text_input("Collection", value=default_collection)
    max_chars = st.number_input("Preview Characters", min_value=80, max_value=2000, value=240)

    preview_col, ingest_col = st.columns(2)
    preview_clicked = preview_col.button("Preview", use_container_width=True)
    ingest_clicked = ingest_col.button("Ingest", type="primary", use_container_width=True)

    if preview_clicked:
        if not path.strip():
            st.warning("Provide a local file or directory path first.")
        else:
            with st.spinner("Generating preview..."):
                _render_preview_results(
                    service.preview_path(path.strip(), collection.strip() or None, max_chars=int(max_chars))
                )

    if ingest_clicked:
        if not path.strip():
            st.warning("Provide a local file or directory path first.")
        else:
            with st.spinner("Running ingestion..."):
                _render_ingest_results(
                    service.ingest_path(path.strip(), collection.strip() or None, max_chars=int(max_chars))
                )

