"""Evaluation panel page for deterministic retrieval and answer regressions."""

from __future__ import annotations

import streamlit as st

from src.observability.dashboard.services.evaluation_service import EvaluationService


def _render_report(payload: dict[str, object]) -> None:
    kind = str(payload.get("kind", ""))
    if kind == "retrieval_eval":
        cols = st.columns(4)
        cols[0].metric("Cases", payload["total_cases"])
        cols[1].metric("Passed", payload["passed_cases"])
        cols[2].metric("Pass Rate", payload["pass_rate"])
        cols[3].metric("Avg Recall@K", payload["average_recall_at_k"])
        st.dataframe(payload["cases"], use_container_width=True, hide_index=True)
        return

    if kind == "answer_eval":
        cols = st.columns(4)
        cols[0].metric("Cases", payload["total_cases"])
        cols[1].metric("Passed", payload["passed_cases"])
        cols[2].metric("Pass Rate", payload["pass_rate"])
        cols[3].metric("Avg Source Coverage", payload["average_source_coverage"])
        st.dataframe(payload["cases"], use_container_width=True, hide_index=True)
        return

    retrieval = payload.get("retrieval", {})
    answer = payload.get("answer", {})
    st.subheader("Aggregate Metrics")
    cols = st.columns(4)
    cols[0].metric("Retrieval Pass Rate", retrieval.get("pass_rate", 0.0))
    cols[1].metric("Retrieval Avg Recall@K", retrieval.get("average_recall_at_k", 0.0))
    cols[2].metric("Answer Pass Rate", answer.get("pass_rate", 0.0))
    cols[3].metric("Answer Avg Source Coverage", answer.get("average_source_coverage", 0.0))

    st.subheader("Retrieval Cases")
    st.dataframe(retrieval.get("cases", []), use_container_width=True, hide_index=True)
    st.subheader("Answer Cases")
    st.dataframe(answer.get("cases", []), use_container_width=True, hide_index=True)


def render() -> None:
    """Render evaluation controls and recent history."""

    st.header("Evaluation Panel")

    try:
        service = EvaluationService()
        defaults = service.get_default_fixtures()
    except Exception as exc:
        st.error(f"Failed to initialise evaluation service: {exc}")
        return

    mode = st.selectbox("Mode", options=["retrieval", "answer", "all"], index=2)
    retrieval_fixtures = st.text_input("Retrieval Fixtures", value=defaults["retrieval"])
    answer_fixtures = st.text_input("Answer Fixtures", value=defaults["answer"])

    if st.button("Run Evaluation", type="primary"):
        try:
            with st.spinner("Running evaluation..."):
                payload = service.run(
                    mode,
                    retrieval_fixtures=retrieval_fixtures.strip() or None,
                    answer_fixtures=answer_fixtures.strip() or None,
                )
            st.session_state["dashboard_last_evaluation"] = payload
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")

    payload = st.session_state.get("dashboard_last_evaluation")
    if payload:
        _render_report(payload)

    history = service.list_history(limit=10)
    st.subheader("History")
    if not history:
        st.info("No evaluation history recorded yet.")
        return

    st.dataframe(history, use_container_width=True, hide_index=True)
