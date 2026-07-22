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

    # Visualizar trazabilidad de extras si está disponible
    if hasattr(result, "contextualized_query") or hasattr(result, "feedback_applied"):
        if hasattr(result, "contextualized_query") and result.contextualized_query and result.contextualized_query != result.query:
            st.info(f"🧠 **Memoria Conversacional:** Consulta reformulada a: *'{result.contextualized_query}'*")
        
        if hasattr(result, "feedback_applied") and result.feedback_applied:
            st.success("🎯 **Relevance Feedback Activo:** Vector de consulta ajustado con algoritmo de Rocchio.")

    st.markdown(result.answer)

    if result.expanded_queries:
        with st.expander(f"✨ **Query Expansion (RRF):** Se generaron {len(result.expanded_queries)} variantes"):
            st.caption("Variantes de búsqueda generadas para mejorar la precisión:")
            for q in result.expanded_queries:
                st.caption(f"- *{q}*")

    with st.expander(f"📎 Evidencias utilizadas (Top-{len(result.evidences)})"):
        turn_id = str(uuid.uuid4())[:8]

        for idx, doc in enumerate(result.evidences, start=1):
            st.markdown(f"**[{idx}] {doc.product_title}**")
            st.caption(f"Similitud: {doc.score:.3f} &middot; `doc_id: {doc.doc_id}`")

            col1, col2 = st.columns([1, 3])

            with col1:
                # --- SOLUCIÓN: Carga directa desde GitHub Raw ---
                raw_path = getattr(doc, "image_path", None)
                img_url = getattr(doc, "image_url", None)
                
                resolved_img = None

                if raw_path:
                    # 1. Normalizar barras y extraer solo el nombre del archivo (ej: B08G7V2X1N.jpg)
                    filename = raw_path.replace('\\', '/').split('/')[-1]
                    
                    # 2. Construir la URL raw directa a tu repositorio de GitHub
                    resolved_img = f"https://raw.githubusercontent.com/ImYasid/ProyectoIIB/main/data/corpus/images/{filename}"

                # 3. Intentar renderizar la imagen
                if resolved_img:
                    try:
                        st.image(resolved_img, use_container_width=True)
                    except Exception:
                        # Si la imagen no está en GitHub, usar la de Amazon como respaldo
                        if img_url:
                            st.image(img_url, use_container_width=True)
                        else:
                            st.caption("(imagen rota en repo)")
                elif img_url:
                    # Si no había ruta local, pero sí URL en la BD
                    st.image(img_url, use_container_width=True)
                else:
                    st.caption("(sin imagen)")

            with col2:
                snippet = doc.text[:350] + "..." if len(doc.text) > 350 else doc.text
                st.write(snippet)

                if feedback_store is not None:
                    btn_up_key = f"fb_up_{doc.doc_id}_{turn_id}_{idx}"
                    btn_down_key = f"fb_down_{doc.doc_id}_{turn_id}_{idx}"

                    f_col1, f_col2, _ = st.columns([1, 1, 8])

                    with f_col1:
                        if hasattr(feedback_store, "record_feedback"):
                            if st.button("👍", key=btn_up_key):
                                feedback_store.record_feedback(result.query, doc.doc_id, +1)
                                st.toast("✅ Voto positivo registrado.")

                    with f_col2:
                        if hasattr(feedback_store, "record_feedback"):
                            if st.button("👎", key=btn_down_key):
                                feedback_store.record_feedback(result.query, doc.doc_id, -1)
                                st.toast("🚫 Voto negativo registrado.")

            st.divider()
