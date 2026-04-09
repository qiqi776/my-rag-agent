"""Overview page for the local dashboard."""

from __future__ import annotations

import streamlit as st

from src.observability.dashboard.services.config_service import ConfigService


def render() -> None:
    """Render provider and data overview information."""

    st.header("Overview")
    st.caption("Current providers, collections, chunks, and trace file status.")

    try:
        snapshot = ConfigService().get_overview_snapshot()
    except Exception as exc:
        st.error(f"Failed to load dashboard overview: {exc}")
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric("Collections", snapshot.collection_count)
    metric_cols[1].metric("Documents", snapshot.document_count)
    metric_cols[2].metric("Chunks", snapshot.chunk_count)
    metric_cols[3].metric("Traces", snapshot.trace_line_count)

    st.subheader("Providers")
    provider_cols = st.columns(max(1, min(3, len(snapshot.provider_cards))))
    for index, card in enumerate(snapshot.provider_cards):
        with provider_cols[index % len(provider_cols)]:
            st.markdown(f"**{card.name}**")
            st.caption(f"Provider: `{card.provider}`")
            st.caption(f"Model: `{card.model}`")
            if card.details:
                st.json(card.details)

    st.subheader("Collections")
    if snapshot.collections:
        st.dataframe(
            [collection.to_dict() for collection in snapshot.collections],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No collections found yet.")

    st.subheader("Trace File")
    st.code(snapshot.trace_file)
    if snapshot.trace_exists:
        st.success("Trace file is available.")
    else:
        st.info("Trace file does not exist yet.")

