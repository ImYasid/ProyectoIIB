"""
app.py
======
Punto de entrada de la interfaz web conversacional.

Ejecución:
    streamlit run app.py
"""

import json
from pathlib import Path
import streamlit as st

import config
from src.b_embeddings import ClipEmbedder
from src.c_vector_store import VectorStore
from src.d_rag_pipeline import RAGPipeline
from src.e_web_interface import init_session_state, render_chat_message, render_rag_result

# Configuración base de la página
st.set_page_config(page_title="RAG Multimodal", page_icon="🔎", layout="wide", initial_sidebar_state="expanded")

# CSS mínimo solo para ocultar marcas de agua y mejorar un poco los botones
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stButton>button {
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Carga de consultas de ejemplo reales del corpus
# ---------------------------------------------------------------------
@st.cache_data
def load_example_queries(limit: int = 6) -> list[str]:
    """Carga una lista de consultas reales extraídas del dataset."""
    queries_path = Path("data/qrels/queries.json")
    if queries_path.exists():
        try:
            with open(queries_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return list(data.values())[:limit]
                elif isinstance(data, list):
                    return [item.get("query", "") for item in data[:limit]]
        except Exception:
            return []
    return []

# ---------------------------------------------------------------------
# Recursos pesados (modelo CLIP, cliente ChromaDB)
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo CLIP y base de datos vectorial...")
def load_core_resources():
    embedder = ClipEmbedder()
    store = VectorStore()
    return embedder, store


def build_pipeline(embedder, store, use_reranking, use_query_expansion,
                   use_memory, use_feedback) -> tuple[RAGPipeline, object | None]:
    """Construye el pipeline RAG activando solo los extras marcados."""
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
        # IMPORTANTE: se inyecta el feedback_store para que el pipeline
        # pueda aplicar el algoritmo de Rocchio y marcar
        # `RAGResult.feedback_applied` cuando corresponda.
        feedback_store=feedback_store,
    )
    return pipeline, feedback_store


def main() -> None:
    init_session_state()

    # Validaciones iniciales tempranas
    if not config.CORPUS_MANIFEST_PATH.exists():
        st.error("No se encontró el corpus. Ejecuta primero: `python main.py build-corpus`")
        st.stop()

    embedder, store = load_core_resources()
    if store.count() == 0:
        st.error("La base vectorial está vacía. Ejecuta: `python main.py index`")
        st.stop()

    clicked_query = None

    # -----------------------------------------------------------------
    # Barra Lateral (Sidebar)
    # -----------------------------------------------------------------
    with st.sidebar:
        st.title("⚙️ Configuración")

        # --- 1. Información del Corpus Activo ---
        st.subheader("📂 Corpus Activo")
        st.info(f"**Ruta:** `{config.CORPUS_MANIFEST_PATH}`\n\n**Documentos indexados:** `{store.count()}`")

        # --- 2. Parámetros de Búsqueda ---
        st.subheader("🔍 Parámetros")
        top_k = st.slider("Documentos a recuperar (Top-K)", 1, 10, config.TOP_K_DEFAULT)

        # Configuraciones avanzadas ocultas en un desplegable
        with st.expander("🛠️ Ajustes Avanzados de RAG"):
            use_reranking = st.checkbox("Re-ranking (Cross-Encoder)", value=False)
            use_query_expansion = st.checkbox("Expansión de Query", value=False)
            use_memory = st.checkbox("Memoria conversacional", value=True)
            use_feedback = st.checkbox("Relevance Feedback (👍/👎)", value=True)

        # --- Indicador de extras activos (trazabilidad a simple vista) ---
        active_extras = []
        if use_memory:
            active_extras.append("🧠 Memoria")
        if use_query_expansion:
            active_extras.append("✨ Query Expansion")
        if use_feedback:
            active_extras.append("🎯 Relevance Feedback")
        if use_reranking:
            active_extras.append("📊 Re-ranking")
        if active_extras:
            st.caption("**Extras activos:** " + " · ".join(active_extras))

        st.divider()

        # --- 3. Historial en la barra lateral ---
        st.subheader("🕒 Historial de Consultas")
        # Filtramos solo los mensajes del usuario
        user_queries = [msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user"]

        if user_queries:
            # Mostramos el historial invertido (el más reciente arriba)
            for q in reversed(user_queries):
                st.caption(f"• {q}")
        else:
            st.caption("No hay consultas recientes.")

        st.write("") # Espaciador

        # Botón para limpiar
        if st.button("🗑️ Limpiar conversación", use_container_width=True, type="secondary"):
            st.session_state.chat_history = []
            st.session_state.pop("_memory", None)
            st.rerun()

    # Construir pipeline
    pipeline, feedback_store = build_pipeline(
        embedder, store, use_reranking, use_query_expansion, use_memory, use_feedback
    )

    # -----------------------------------------------------------------
    # Interfaz Principal (Main Layout)
    # -----------------------------------------------------------------

    # Pantalla de inicio si el historial está vacío
    if not st.session_state.chat_history:
        st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>🔎 Catálogo Inteligente</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray; font-size: 1.1rem; margin-bottom: 2rem;'>Encuentra productos utilizando texto e imágenes con inteligencia artificial.</p>", unsafe_allow_html=True)

        example_queries = load_example_queries(limit=6)
        if example_queries:
            st.markdown("### ✨ Sugerencias para empezar")
            cols = st.columns(2)
            for idx, q_text in enumerate(example_queries):
                with cols[idx % 2]:
                    # Botón que actúa como input
                    if st.button(f"{q_text}", key=f"hero_btn_{idx}", use_container_width=True):
                        clicked_query = q_text
            st.markdown("<br><br>", unsafe_allow_html=True)
    else:
        st.title("🔎 Chat Multimodal")

    # --- Cargar visualmente el Historial de Chat ---
    for turn in st.session_state.chat_history:
        if turn["role"] == "user":
            render_chat_message("user", turn["content"])
        else:
            render_rag_result(turn["result"], feedback_store=feedback_store)

    # --- Input de nueva consulta ---
    chat_query = st.chat_input("Pregunta algo sobre el catálogo de productos...")
    active_query = chat_query or clicked_query

    # --- Procesamiento de la consulta ---
    if active_query:
        # Añadir al historial
        st.session_state.chat_history.append({"role": "user", "content": active_query})

        # Si vino de un botón, renderizar el mensaje antes de cargar la respuesta
        if clicked_query:
            render_chat_message("user", active_query)

        # Spinner nativo interactivo de Streamlit
        with st.status("🧠 Analizando catálogo y generando respuesta...", expanded=True) as status:
            try:
                st.write("Buscando evidencias en la base vectorial...")
                result = pipeline.run(active_query, top_k=top_k)

                # Guardar respuesta
                st.session_state.chat_history.append({"role": "assistant", "result": result})
                status.update(label="¡Respuesta lista!", state="complete", expanded=False)

                # Renderizar la respuesta en pantalla (incluye la trazabilidad
                # visual de Memoria / Query Expansion / Relevance Feedback)
                render_rag_result(result, feedback_store=feedback_store)

                # Refrescar UI si el input vino del botón para actualizar la vista principal
                if clicked_query:
                    st.rerun()

            except Exception as e:
                status.update(label="Error en el procesamiento", state="error", expanded=False)
                error_msg = str(e)
                if "503" in error_msg or "UNAVAILABLE" in error_msg:
                    st.error("⚠️ **Google Gemini está temporalmente sobrecargado.** Por favor, espera unos segundos e intenta de nuevo.")
                else:
                    st.error(f"⚠️ **Ocurrió un error técnico:** `{error_msg}`")


if __name__ == "__main__":
    main()