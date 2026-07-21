"""
src/e_web_interface.py
=======================
Literal e) Interfaz conversacional.

Este módulo contiene la lógica de PRESENTACIÓN reutilizable para la
interfaz web tipo chat (Streamlit): inicialización del estado de la
sesión y funciones de renderizado de mensajes/evidencias. El punto de
entrada real de la app (`streamlit run`) es `app.py`, en la raíz del
proyecto, que se mantiene deliberadamente delgado e importa todo desde
aquí y desde los demás módulos (d_rag_pipeline, extras/*).

Requisitos mínimos que cubre este módulo:
    - realizar consultas conversacionales;
    - visualizar la respuesta del asistente;
    - visualizar los documentos e imágenes usados como contexto (evidencias),
      junto con su puntaje de similitud.
"""

import streamlit as st

from src.c_vector_store import RetrievedDocument
from src.d_rag_pipeline import RAGResult


def init_session_state() -> None:
    """Inicializa las estructuras de estado de Streamlit usadas en toda
    la app. Se llama una única vez al inicio de app.py."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # lista de dicts {role, content, result?}
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = set()  # (query, doc_id) ya calificados


def render_chat_message(role: str, content: str) -> None:
    with st.chat_message(role):
        st.markdown(content)


def render_evidence_card(rank: int, doc: RetrievedDocument, query: str,
                         feedback_store=None) -> None:
    """Renderiza una evidencia (documento recuperado) con su imagen,
    texto y puntaje de similitud. Si se inyecta un `feedback_store`
    (extras/relevance_feedback.py) también muestra los botones
    👍 / 👎 de la funcionalidad de excelencia correspondiente."""
    cols = st.columns([1, 3])
    with cols[0]:
        if doc.image_path:
            try:
                st.image(doc.image_path, use_container_width=True)
            except Exception:
                st.caption("(imagen no disponible)")
        else:
            st.caption("(sin imagen)")
    with cols[1]:
        st.markdown(f"**[{rank}] {doc.product_title}**")
        st.caption(f"Similitud: {doc.score:.3f} · doc_id: `{doc.doc_id}`")
        st.write(doc.text[:280] + ("..." if len(doc.text) > 280 else ""))

        if feedback_store is not None:
            key_base = f"fb_{query}_{doc.doc_id}"
            fcol1, fcol2, _ = st.columns([1, 1, 4])
            already_rated = (query, doc.doc_id) in st.session_state.feedback_given
            with fcol1:
                if st.button("👍", key=key_base + "_up", disabled=already_rated):
                    feedback_store.record_feedback(query, doc.doc_id, +1)
                    st.session_state.feedback_given.add((query, doc.doc_id))
                    st.rerun()
            with fcol2:
                if st.button("👎", key=key_base + "_down", disabled=already_rated):
                    feedback_store.record_feedback(query, doc.doc_id, -1)
                    st.session_state.feedback_given.add((query, doc.doc_id))
                    st.rerun()
    st.divider()


def render_rag_result(result: RAGResult, feedback_store=None) -> None:
    """Renderiza la respuesta del asistente y, debajo, la sección de
    evidencias (trazabilidad recuperación vs. generación)."""
    render_chat_message("assistant", result.answer)

    if result.expanded_queries:
        with st.expander("🔎 Consultas expandidas (Query Expansion)"):
            for q in result.expanded_queries:
                st.write(f"- {q}")

    with st.expander(f"📎 Evidencias utilizadas (Top-{len(result.evidences)})", expanded=False):
        if not result.evidences:
            st.info("No se recuperó ningún documento relevante del corpus.")
        for i, doc in enumerate(result.evidences, start=1):
            render_evidence_card(i, doc, result.query, feedback_store=feedback_store)