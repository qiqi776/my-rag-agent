"""Ingestion trace browser page."""

from __future__ import annotations

import streamlit as st

from src.observability.dashboard.services.trace_service import TraceService


def render() -> None:
    """Render ingestion trace history and selected trace details."""

    st.header("Ingestion Traces")

    try:
        service = TraceService()
        traces = service.list_traces(trace_type="ingestion", limit=100)
    except Exception as exc:
        st.error(f"Failed to load ingestion traces: {exc}")
        return

    if not traces:
        st.info("No ingestion traces recorded yet.")
        return

    selected_trace = st.selectbox(
        "Trace",
        options=traces,
        format_func=lambda item: (
            f"{item['trace_id']} · {item['started_at']} · "
            f"{item['metadata'].get('source_path', '<unknown>')}"
        ),
    )
    detail = service.get_trace(str(selected_trace["trace_id"]))
    if detail is None:
        st.warning("Selected trace is no longer available.")
        return

    stage_rows = service.get_stage_rows(detail)
    load_stage = next((row for row in stage_rows if row["stage_name"] == "load"), None)

    top_cols = st.columns(4)
    top_cols[0].metric("Stages", len(stage_rows))
    top_cols[1].metric("Elapsed (ms)", round(float(detail["total_elapsed_ms"]), 2))
    top_cols[2].metric("Page Count", (load_stage or {}).get("data", {}).get("page_count", 0))
    top_cols[3].metric(
        "Quality",
        (load_stage or {}).get("data", {}).get("quality_status", "n/a"),
    )

    chart_data = {row["stage_name"]: row["elapsed_ms"] for row in stage_rows}
    if chart_data:
        st.bar_chart(chart_data, horizontal=True)

    st.subheader("Stage Timings")
    st.table(
        [
            {
                "stage": row["stage_name"],
                "elapsed_ms": round(row["elapsed_ms"], 2),
            }
            for row in stage_rows
        ]
    )

    if load_stage is not None:
        st.subheader("PDF Quality Metrics")
        st.json(load_stage["data"])

