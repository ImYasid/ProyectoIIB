"""
src/d_rag_pipeline.py
======================
Literal d) Sistema RAG (Retrieval-Augmented Generation).

Implementa el pipeline mínimo exigido por la especificación:
    1. recibir una consulta del usuario;
    2. recuperar los documentos más relevantes mediante búsqueda vectorial;
    3. construir automáticamente el contexto para el modelo de lenguaje;
    4. generar una respuesta utilizando dicho contexto.

Además, expone explícitamente las "evidencias" usadas (documentos,
imágenes y puntaje de similitud) para que la interfaz web (literal e)
pueda mostrarlas y así garantizar la trazabilidad del sistema.

Adicionalmente, el resultado (`RAGResult`) transporta banderas de
trazabilidad de las 3 funcionalidades de excelencia, para que la UI
pueda mostrar explícitamente CUÁNDO y CÓMO se dispararon:

    - `contextualized_query`: si la Memoria Conversacional reformuló
      la consulta original antes de recuperar (p. ej. "¿y en negro?"
      -> "almohada VIKTOR JURGEN color negro").
    - `expanded_queries`: las variantes generadas por Query Expansion.
    - `feedback_applied`: True si existía Relevance Feedback (👍/👎)
      previo para esa consulta y por lo tanto el vector de búsqueda
      fue desplazado con el algoritmo de Rocchio.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from google import genai
from google.genai import types  # <-- IMPORTANTE: Para configurar los filtros de seguridad

import config
from src.b_embeddings import ClipEmbedder
from src.c_vector_store import RetrievedDocument, VectorStore

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """Eres un asistente de búsqueda de productos. Responde la
pregunta del usuario ÚNICAMENTE con base en el CONTEXTO proporcionado,
que contiene fichas de producto recuperadas de un catálogo.

Reglas:
- Si el contexto no contiene información suficiente para responder,
  dilo explícitamente en vez de inventar datos.
- Cuando menciones un producto, cita su número de evidencia entre
  corchetes, por ejemplo: "el producto [2] es ideal para...".
- Responde en el mismo idioma de la pregunta del usuario.
- Sé conciso y útil."""


@dataclass
class RAGResult:
    query: str
    answer: str
    evidences: list[RetrievedDocument] = field(default_factory=list)
    expanded_queries: list[str] = field(default_factory=list)
    # --- Trazabilidad de funcionalidades de excelencia (para la UI) ---
    contextualized_query: Optional[str] = None
    feedback_applied: bool = False


class RAGPipeline:
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: ClipEmbedder,
        reranker=None,
        query_expander=None,
        memory=None,
        feedback_store=None,
        gemini_api_key: Optional[str] = config.GEMINI_API_KEY,
        gemini_model: str = config.GEMINI_MODEL,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker
        self.query_expander = query_expander
        self.memory = memory
        self.feedback_store = feedback_store

        if not gemini_api_key:
            logger.warning("GEMINI_API_KEY no configurada.")

        self.gemini_client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None
        self.gemini_model = gemini_model

    # ------------------------------------------------------------------
    # Relevance Feedback: obtiene el vector de consulta, aplicando
    # Rocchio si existe feedback previo registrado para esa consulta
    # exacta. Devuelve (vector, feedback_applied: bool).
    # ------------------------------------------------------------------
    def _embed_with_feedback(self, query: str):
        if self.feedback_store is not None:
            has_feedback = bool(self.feedback_store.get_feedback_for_query(query))
            if has_feedback:
                query_vec = self.feedback_store.refine_query_vector(query)
                return query_vec, True
        return self.embedder.embed_query(query), False

    def retrieve(
        self, query: str, top_k: int = config.TOP_K_DEFAULT
    ) -> tuple[list[RetrievedDocument], list[str], bool]:
        expanded_queries: list[str] = []

        if self.query_expander is not None:
            expanded_queries = self.query_expander.expand(query)
            all_queries = [query] + expanded_queries
        else:
            all_queries = [query]

        fetch_k = top_k * config.RERANK_CANDIDATE_MULTIPLIER if self.reranker else top_k

        if len(all_queries) == 1:
            query_vec, feedback_applied = self._embed_with_feedback(query)
            candidates = self.vector_store.query(query_vec, top_k=fetch_k)
        else:
            candidates, feedback_applied = self._multi_query_retrieve(all_queries, fetch_k)

        if self.reranker is not None:
            candidates = self.reranker.rerank(query, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]

        return candidates, expanded_queries, feedback_applied

    def _multi_query_retrieve(
        self, queries: list[str], fetch_k: int
    ) -> tuple[list[RetrievedDocument], bool]:
        rrf_scores: dict[str, float] = {}
        doc_by_id: dict[str, RetrievedDocument] = {}
        k_constant = 60
        feedback_applied = False

        for q in queries:
            q_vec, q_feedback_applied = self._embed_with_feedback(q)
            feedback_applied = feedback_applied or q_feedback_applied

            results = self.vector_store.query(q_vec, top_k=fetch_k)
            for rank, doc in enumerate(results):
                rrf_scores[doc.doc_id] = rrf_scores.get(doc.doc_id, 0) + 1.0 / (k_constant + rank)
                doc_by_id[doc.doc_id] = doc

        ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:fetch_k]
        return [doc_by_id[i] for i in ranked_ids], feedback_applied

    @staticmethod
    def build_context(evidences: list[RetrievedDocument]) -> str:
        blocks = []
        for i, doc in enumerate(evidences, start=1):
            snippet = doc.text[: config.MAX_CONTEXT_CHARS_PER_DOC]
            blocks.append(
                f"[{i}] Producto: {doc.product_title}\n"
                f"    Descripción: {snippet}\n"
                f"    Similitud con la consulta: {doc.score:.3f}"
            )
        return "\n\n".join(blocks) if blocks else "(sin resultados relevantes en el corpus)"

    def generate_answer(self, query: str, context: str) -> str:
        prompt = (
            "Eres un asistente de compras experto. Responde a la consulta del usuario "
            "basándote ÚNICAMENTE en el siguiente contexto de productos.\n\n"
            f"CONTEXTO:\n{context}\n\n"
            f"PREGUNTA DEL USUARIO: {query}\n\n"
            "RESPUESTA:"
        )

        gemini_config = types.GenerateContentConfig(
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                )
            ]
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt,
                    config=gemini_config
                )

                if not response.text:
                    return "⚠️ **Filtro de seguridad de IA activado:** Gemini detectó vocabulario sensible en la descripción de estos productos y bloqueó la generación de la respuesta. Revisa las evidencias mostradas abajo."

                return response.text
            except Exception as exc:
                if ("503" in str(exc) or "UNAVAILABLE" in str(exc)) and attempt < max_retries - 1:
                    logger.warning("Gemini API ocupada (503). Reintentando en %d segundos...", (attempt + 1) * 2)
                    time.sleep((attempt + 1) * 2)
                else:
                    logger.error("Error al generar respuesta con Gemini: %s", exc)
                    return (
                        "⚠️ **Servidor de Gemini ocupado temporalmente (Error 503).**\n\n"
                        "Google está experimentando alta demanda en este momento."
                    )

    def run(self, query: str, top_k: int = config.TOP_K_DEFAULT) -> RAGResult:
        retrieval_query = query
        contextualized_query: Optional[str] = None

        if self.memory is not None:
            retrieval_query = self.memory.contextualize_query(query)
            if retrieval_query and retrieval_query.strip() != query.strip():
                contextualized_query = retrieval_query

        evidences, expanded_queries, feedback_applied = self.retrieve(retrieval_query, top_k=top_k)
        context = self.build_context(evidences)

        if self.gemini_client is None:
            answer = "⚠️ La clave de API de Gemini no está configurada. Mostrando solo resultados recuperados."
        else:
            answer = self.generate_answer(query, context)

        if self.memory is not None:
            self.memory.add_turn("user", query)
            self.memory.add_turn("assistant", answer)

        return RAGResult(
            query=query,
            answer=answer,
            evidences=evidences,
            expanded_queries=expanded_queries,
            contextualized_query=contextualized_query,
            feedback_applied=feedback_applied,
        )