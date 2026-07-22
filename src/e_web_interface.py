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
      junto con su puntaje de similitud;
    - visualizar de forma EXPLÍCITA cuándo se dispararon las 3
      funcionalidades de excelencia (Memoria Conversacional, Query
      Expansion y Relevance Feedback / Rocchio), a partir de las
      banderas de trazabilidad que trae `RAGResult`.
"""

import uuid
from typing import Optional
from pathlib import Path
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


def render_extras_traceability(result: RAGResult) -> None:
    """Dibuja componentes visuales de Streamlit que sirven de EVIDENCIA
    directa (para presentación/evaluación) de que las 3 funcionalidades
    de excelencia se ejecutaron sobre esta consulta en particular:

        1. Memoria Conversacional  -> st.info con la consulta reformulada
        2. Query Expansion (RRF)   -> st.expander con las variantes
        3. Relevance Feedback      -> st.success si se aplicó Rocchio

    Se llama SIEMPRE antes de mostrar las evidencias recuperadas; cada
    bloque solo se dibuja si el extra correspondiente realmente se
    disparó para esta consulta.
    """
    # --- 1) Memoria Conversacional ---
    if result.contextualized_query and result.contextualized_query != result.query:
        st.info(
            "🧠 **Memoria Conversacional:** La consulta se contexto-reformuló a: "
            f"*'{result.contextualized_query}'*"
        )

    # --- 2) Query Expansion (RRF) ---
    if result.expanded_queries:
        with st.expander(
            f"✨ **Query Expansion (RRF):** Se generaron {len(result.expanded_queries)} "
            "variantes de búsqueda"
        ):
            st.caption(
                "La consulta original se enriqueció con estas variantes para mejorar "
                "el recall; los resultados se fusionan con Reciprocal Rank Fusion:"
            )
            for q in result.expanded_queries:
                st.caption(f"- *{q}*")

    # --- 3) Relevance Feedback (Rocchio) ---
    if result.feedback_applied:
        st.success(
            "🎯 **Relevance Feedback Activo:** El vector de búsqueda fue modificado "
            "matemáticamente usando el Algoritmo de Rocchio según tus likes/dislikes previos."
        )


def render_rag_result(result: RAGResult, feedback_store=None) -> None:
    """Renderiza el resultado del RAG en la interfaz web."""
    if not result:
        return

    # Trazabilidad visual de los extras (memoria / expansión / feedback)
    render_extras_traceability(result)

    st.markdown(result.answer)

    with st.expander(f"📎 Evidencias utilizadas (Top-{len(result.evidences)})"):
        # Generamos un ID único para este bloque de resultados para que Streamlit no se queje
        # de llaves duplicadas si la misma búsqueda aparece varias veces en el historial.
        turn_id = str(uuid.uuid4())[:8]

        for idx, doc in enumerate(result.evidences, start=1):
            st.markdown(f"**[{idx}] {doc.product_title}**")
            st.caption(f"Similitud: {doc.score:.3f} &middot; `doc_id: {doc.doc_id}`")

            # Columnas: imagen y texto
            col1, col2 = st.columns([1, 3])

            with col1:
                # CORRECCIÓN: leemos la propiedad image_path directamente, usando getattr
                # para que no falle si en algún punto no existe.
                img_path = getattr(doc, "image_path", None)
                if img_path and Path(img_path).exists():
                    st.image(str(img_path), use_container_width=True)
                else:
                    st.caption("(sin imagen)")

            with col2:
                # Texto truncado para no ensuciar la interfaz
                snippet = doc.text[:350] + "..." if len(doc.text) > 350 else doc.text
                st.write(snippet)

                # --- Botones de Relevance Feedback ---
                if feedback_store is not None:
                    # Usamos el turn_id en la llave para garantizar que sea única en toda la pantalla
                    btn_up_key = f"fb_up_{doc.doc_id}_{turn_id}_{idx}"
                    btn_down_key = f"fb_down_{doc.doc_id}_{turn_id}_{idx}"

                    f_col1, f_col2, _ = st.columns([1, 1, 8])

                    with f_col1:
                        # Si el feedback_store tiene add_feedback (como en la última versión de extras)
                        if hasattr(feedback_store, 'add_feedback'):
                            if st.button("👍", key=btn_up_key, help="Marcar como relevante"):
                                feedback_store.add_feedback(result.query, doc.doc_id, True)
                                st.toast("✅ Preferencia guardada para refinar próximas búsquedas.")
                        # Si tiene record_feedback (como en la versión original)
                        elif hasattr(feedback_store, 'record_feedback'):
                            if st.button("👍", key=btn_up_key, help="Marcar como relevante"):
                                feedback_store.record_feedback(result.query, doc.doc_id, +1)
                                st.toast("✅ Preferencia guardada.")

                    with f_col2:
                        if hasattr(feedback_store, 'add_feedback'):
                            if st.button("👎", key=btn_down_key, help="Marcar como irrelevante"):
                                feedback_store.add_feedback(result.query, doc.doc_id, False)
                                st.toast("🚫 Preferencia guardada para filtrar estos resultados.")
                        elif hasattr(feedback_store, 'record_feedback'):
                            if st.button("👎", key=btn_down_key, help="Marcar como irrelevante"):
                                feedback_store.record_feedback(result.query, doc.doc_id, -1)
                                st.toast("🚫 Preferencia guardada.")

            st.divider()