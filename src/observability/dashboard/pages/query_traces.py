"""Query and answer trace browser page."""

from __future__ import annotations

import streamlit as st

from src.observability.dashboard.services.trace_service import TraceService


def render() -> None:
    """Render query or answer trace history with stage detail."""

    st.header("Query Traces")

    try:
        service = TraceService()
    except Exception as exc:
        st.error(f"Failed to initialise trace service: {exc}")
        return

    trace_type = st.selectbox("Trace Type", options=["query", "answer"], index=0)
    traces = service.list_traces(trace_type=trace_type, limit=100)
    if not traces:
        st.info(f"No {trace_type} traces recorded yet.")
        return

    selected_trace = st.selectbox(
        "Trace",
        options=traces,
        format_func=lambda item: (
            f"{item['trace_id']} · {item['started_at']} · "
            f"{item['metadata'].get('query', '<no query>')}"
        ),
    )
    detail = service.get_trace(str(selected_trace["trace_id"]))
    if detail is None:
        st.warning("Selected trace is no longer available.")
        return

    st.caption(f"Collection: `{detail['metadata'].get('collection', '<none>')}`")
    if detail["metadata"].get("query"):
        st.markdown(f"> {detail['metadata']['query']}")

    stage_rows = service.get_stage_rows(detail)
    stage_lookup = {row["stage_name"]: row for row in stage_rows}

    if trace_type == "query":
        dense = stage_lookup.get("dense_retrieve", {}).get("data", {})
        sparse = stage_lookup.get("sparse_retrieve", {}).get("data", {})
        fused = stage_lookup.get("rrf_fuse", {}).get("data", {})
        metric_cols = st.columns(4)
        metric_cols[0].metric("Dense Candidates", dense.get("candidate_result_count", 0))
        metric_cols[1].metric("Sparse Candidates", sparse.get("candidate_result_count", 0))
        metric_cols[2].metric("Final Results", fused.get("final_result_count", 0))
        metric_cols[3].metric("Elapsed (ms)", round(float(detail["total_elapsed_ms"]), 2))
    else:
        retrieve = stage_lookup.get("retrieve", {}).get("data", {})
        rerank = stage_lookup.get("rerank", {}).get("data", {})
        assemble = stage_lookup.get("assemble_context", {}).get("data", {})
        metric_cols = st.columns(4)
        metric_cols[0].metric("Retrieved", retrieve.get("result_count", 0))
        metric_cols[1].metric("Reranked", rerank.get("selected_result_count", 0))
        metric_cols[2].metric("Context Chars", assemble.get("char_count", 0))
        metric_cols[3].metric("Elapsed (ms)", round(float(detail["total_elapsed_ms"]), 2))

    chart_data = {row["stage_name"]: row["elapsed_ms"] for row in stage_rows}
    if chart_data:
        st.bar_chart(chart_data, horizontal=True)

    st.subheader("Stage Timings")
    st.table(
        [
            {
                "stage": row["stage_name"],
                "elapsed_ms": round(row["elapsed_ms"], 2),
                "data": row["data"],
            }
            for row in stage_rows
        ]
    )

