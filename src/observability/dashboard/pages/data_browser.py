"""Data browser page for documents, chunks, and metadata."""

from __future__ import annotations

import streamlit as st

from src.observability.dashboard.services.data_service import DataService


def render() -> None:
    """Render document list, chunk details, metadata, and delete controls."""

    st.header("Data Browser")

    try:
        service = DataService()
        collections = service.list_collections()
    except Exception as exc:
        st.error(f"Failed to load data browser: {exc}")
        return

    selected_collection = st.selectbox(
        "Collection",
        options=collections,
        index=0 if collections else None,
    )
    documents = service.list_documents(selected_collection)

    if not documents:
        st.info("No documents found in the selected collection.")
        return

    st.caption(f"Documents in `{selected_collection}`: {len(documents)}")

    for index, document in enumerate(documents):
        title = f"{document['doc_id']} · {document['source_path']}"
        with st.expander(title, expanded=len(documents) == 1 and index == 0):
            top_cols = st.columns([4, 1])
            with top_cols[0]:
                st.caption(
                    f"Chunks: {document['chunk_count']} · Collection: `{document['collection']}`"
                )
            with top_cols[1]:
                if st.button("Delete", key=f"delete_doc_{document['doc_id']}_{index}"):
                    result = service.delete_document(document["doc_id"], selected_collection)
                    if result["deleted"]:
                        st.success(f"Deleted {result['deleted_chunks']} chunk(s).")
                        st.rerun()
                    st.warning("Document not found in the selected collection.")

            detail = service.get_document_detail(document["doc_id"], selected_collection)
            if detail is not None:
                st.text_area(
                    "Preview",
                    value=str(detail["preview"]),
                    height=120,
                    disabled=True,
                    key=f"detail_preview_{document['doc_id']}_{index}",
                )
                st.json(detail["metadata"])

            chunks = service.get_document_chunks(document["doc_id"], selected_collection)
            if not chunks:
                st.info("No chunks recorded for this document.")
                continue

            for chunk in chunks:
                label = f"Chunk {chunk['chunk_index']} · {chunk['chunk_id']}"
                with st.expander(label):
                    st.text_area(
                        "Chunk Text",
                        value=str(chunk["text"]),
                        height=160,
                        disabled=True,
                        key=f"chunk_text_{chunk['chunk_id']}",
                    )
                    st.json(chunk["metadata"])

