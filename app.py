"""
app.py
======
Punto de entrada de la interfaz web conversacional (literal e).

Este archivo se mantiene deliberadamente delgado: toda la lógica de
recuperación/generación vive en src/d_rag_pipeline.py, todo el
renderizado reutilizable vive en src/e_web_interface.py, y cada
funcionalidad de excelencia vive en su propio módulo bajo src/extras/.
app.py solo los conecta según lo que el usuario active en la barra
lateral.

Ejecución:
    streamlit run app.py
"""

import streamlit as st

import config
from src.b_embeddings import ClipEmbedder
from src.c_vector_store import VectorStore
from src.d_rag_pipeline import RAGPipeline
from src.e_web_interface import init_session_state, render_chat_message, render_rag_result

st.set_page_config(page_title="RAG Multimodal", page_icon="🔎", layout="wide")


# ---------------------------------------------------------------------
# Recursos pesados (modelo CLIP, cliente ChromaDB): se cargan una sola
# vez por proceso gracias a @st.cache_resource.
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo CLIP y base de datos vectorial...")
def load_core_resources():
    embedder = ClipEmbedder()
    store = VectorStore()
    return embedder, store


def build_pipeline(embedder, store, use_reranking, use_query_expansion,
                   use_memory, use_feedback) -> tuple[RAGPipeline, object | None]:
    """Construye el pipeline RAG activando solo los extras marcados en
    la barra lateral. Cada extra se importa de forma perezosa (lazy)
    para no cargar modelos pesados que el usuario no pidió."""
    reranker = None
    if use_reranking:
        from src.extras.reranking import CrossEncoderReranker
        reranker = st.session_state.setdefault("_reranker", CrossEncoderReranker())

    query_expander = None
    if use_query_expansion:
        from src.extras.query_expansion import QueryExpander
        query_expander = QueryExpander()

    memory = None
    if use_memory:
        from src.extras.conversational_memory import ConversationMemory
        if "_memory" not in st.session_state:
            st.session_state["_memory"] = ConversationMemory()
        memory = st.session_state["_memory"]

    feedback_store = None
    if use_feedback:
        from src.extras.relevance_feedback import RelevanceFeedbackStore
        feedback_store = st.session_state.setdefault(
            "_feedback_store", RelevanceFeedbackStore(store, embedder)
        )

    pipeline = RAGPipeline(
        vector_store=store, embedder=embedder,
        reranker=reranker, query_expander=query_expander, memory=memory,
    )
    return pipeline, feedback_store


def main() -> None:
    init_session_state()
    st.title("🔎 Sistema de Recuperación de Información Multimodal con RAG")
    st.caption("Recuperación multimodal (CLIP + ChromaDB) + generación (Gemini)")

    if not config.CORPUS_MANIFEST_PATH.exists():
        st.error(
            "No se encontró el corpus. Ejecuta primero:\n\n"
            "```\npython scripts/build_corpus.py\npython scripts/index_corpus.py\n```"
        )
        st.stop()

    embedder, store = load_core_resources()
    if store.count() == 0:
        st.error("La base vectorial está vacía. Ejecuta: `python scripts/index_corpus.py`")
        st.stop()

    with st.sidebar:
        st.header("⚙️ Configuración")
        top_k = st.slider("Top-k documentos a recuperar", 1, 10, config.TOP_K_DEFAULT)

        st.subheader("Funcionalidades de excelencia")
        use_reranking = st.checkbox("Re-ranking (cross-encoder)", value=False)
        use_query_expansion = st.checkbox("Query Expansion", value=False)
        use_memory = st.checkbox("Memoria conversacional", value=True)
        use_feedback = st.checkbox("Relevance Feedback (👍/👎)", value=True)

        st.divider()
        st.caption(f"Documentos indexados: {store.count()}")
        if st.button("🗑️ Limpiar conversación"):
            st.session_state.chat_history = []
            st.session_state.pop("_memory", None)
            st.rerun()

    pipeline, feedback_store = build_pipeline(
        embedder, store, use_reranking, use_query_expansion, use_memory, use_feedback
    )

    # --- Historial ---
    for turn in st.session_state.chat_history:
        if turn["role"] == "user":
            render_chat_message("user", turn["content"])
        else:
            render_rag_result(turn["result"], feedback_store=feedback_store)

    # --- Nueva consulta ---
    query = st.chat_input("Pregunta algo sobre el catálogo de productos...")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        render_chat_message("user", query)

        with st.spinner("Recuperando evidencias y generando respuesta..."):
            result = pipeline.run(query, top_k=top_k)

        st.session_state.chat_history.append({"role": "assistant", "result": result})
        render_rag_result(result, feedback_store=feedback_store)


if __name__ == "__main__":
    main()