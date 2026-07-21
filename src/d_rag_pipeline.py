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

Este módulo se apoya en los módulos de extras (re-ranking, expansión
de consultas y memoria conversacional) SOLO si se le inyectan; si no,
funciona con el pipeline base sin ellas (principio de composición,
no de herencia forzada).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from google import genai

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
    """Resultado completo de una consulta al sistema RAG: la respuesta
    generada más las evidencias que la sustentan (para trazabilidad)."""

    query: str
    answer: str
    evidences: list[RetrievedDocument] = field(default_factory=list)
    expanded_queries: list[str] = field(default_factory=list)  # si hubo query expansion


class RAGPipeline:
    """Orquesta recuperación + generación. Las funcionalidades de
    excelencia (re-ranker, expansor de consultas, memoria) son
    opcionales y se inyectan por constructor (inyección de
    dependencias), para que el pipeline base (70 pts) funcione
    exactamente igual con o sin ellas."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: ClipEmbedder,
        reranker=None,          # src.extras.reranking.CrossEncoderReranker
        query_expander=None,    # src.extras.query_expansion.QueryExpander
        memory=None,            # src.extras.conversational_memory.ConversationMemory
        gemini_api_key: Optional[str] = config.GEMINI_API_KEY,
        gemini_model: str = config.GEMINI_MODEL,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker
        self.query_expander = query_expander
        self.memory = memory

        if not gemini_api_key:
            logger.warning(
                "GEMINI_API_KEY no configurada: el sistema podrá recuperar "
                "evidencias pero no generará respuestas."
            )
        self.gemini_client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None
        self.gemini_model = gemini_model

    # ---------------------------------------------------------------
    # Paso 2: recuperación
    # ---------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = config.TOP_K_DEFAULT) -> tuple[list[RetrievedDocument], list[str]]:
        """Recupera los documentos más relevantes. Si hay un
        `query_expander` inyectado, se generan variantes de la consulta
        y se combinan los resultados (fusión por rango recíproco)."""
        expanded_queries: list[str] = []

        if self.query_expander is not None:
            expanded_queries = self.query_expander.expand(query)
            all_queries = [query] + expanded_queries
        else:
            all_queries = [query]

        fetch_k = top_k * config.RERANK_CANDIDATE_MULTIPLIER if self.reranker else top_k

        if len(all_queries) == 1:
            query_vec = self.embedder.embed_query(query)
            candidates = self.vector_store.query(query_vec, top_k=fetch_k)
        else:
            candidates = self._multi_query_retrieve(all_queries, fetch_k)

        if self.reranker is not None:
            candidates = self.reranker.rerank(query, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]

        return candidates, expanded_queries

    def _multi_query_retrieve(self, queries: list[str], fetch_k: int) -> list[RetrievedDocument]:
        """Fusión por rango recíproco (Reciprocal Rank Fusion) de los
        resultados obtenidos con la consulta original y sus variantes
        expandidas."""
        rrf_scores: dict[str, float] = {}
        doc_by_id: dict[str, RetrievedDocument] = {}
        k_constant = 60

        for q in queries:
            q_vec = self.embedder.embed_query(q)
            results = self.vector_store.query(q_vec, top_k=fetch_k)
            for rank, doc in enumerate(results):
                rrf_scores[doc.doc_id] = rrf_scores.get(doc.doc_id, 0) + 1.0 / (k_constant + rank)
                doc_by_id[doc.doc_id] = doc

        ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:fetch_k]
        return [doc_by_id[i] for i in ranked_ids]

    # ---------------------------------------------------------------
    # Paso 3: construcción del contexto
    # ---------------------------------------------------------------
    @staticmethod
    def build_context(evidences: list[RetrievedDocument]) -> str:
        """Construye automáticamente el bloque de contexto a partir de
        las evidencias recuperadas, numerándolas para poder citarlas."""
        blocks = []
        for i, doc in enumerate(evidences, start=1):
            snippet = doc.text[: config.MAX_CONTEXT_CHARS_PER_DOC]
            blocks.append(
                f"[{i}] Producto: {doc.product_title}\n"
                f"    Descripción: {snippet}\n"
                f"    Similitud con la consulta: {doc.score:.3f}"
            )
        return "\n\n".join(blocks) if blocks else "(sin resultados relevantes en el corpus)"

    # ---------------------------------------------------------------
    # Paso 4: generación
    # ---------------------------------------------------------------
import time

def generate_answer(self, query: str, context: str) -> str:
    """Genera la respuesta final con Gemini aplicando el prompt RAG."""
    prompt = (
        "Eres un asistente de compras experto. Responde a la consulta del usuario "
        "basándote ÚNICAMENTE en el siguiente contexto de productos.\n\n"
        f"CONTEXTO:\n{context}\n\n"
        f"PREGUNTA DEL USUARIO: {query}\n\n"
        "RESPUESTA:"
    )

    # Reintentos automáticos si la API de Google está sobrecargada (Error 503)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model, contents=prompt
            )
            return response.text
        except Exception as exc:
            if ("503" in str(exc) or "UNAVAILABLE" in str(exc)) and attempt < max_retries - 1:
                logger.warning("Gemini API ocupada (503). Reintentando en %d segundos...", (attempt + 1) * 2)
                time.sleep((attempt + 1) * 2)
            else:
                logger.error("Error al generar respuesta con Gemini: %s", exc)
                return (
                    "⚠️ **Servidor de Gemini ocupado temporalmente (Error 503).**\n\n"
                    "Google está experimentando alta demanda en este momento. "
                    "Por favor, intenta enviar tu pregunta nuevamente en unos segundos."
                )
            
            
    # ---------------------------------------------------------------
    # Orquestación completa (pipeline mínimo del literal d)
    # ---------------------------------------------------------------
    def run(self, query: str, top_k: int = config.TOP_K_DEFAULT) -> RAGResult:
        # Si hay memoria conversacional, primero reformulamos la
        # consulta para que sea autocontenida (p. ej. "¿y en negro?"
        # -> "mochila impermeable en color negro"), y ES esa consulta
        # reformulada la que se usa para recuperar y para expandir.
        retrieval_query = query
        if self.memory is not None:
            retrieval_query = self.memory.contextualize_query(query)

        evidences, expanded_queries = self.retrieve(retrieval_query, top_k=top_k)
        context = self.build_context(evidences)
        answer = self.generate_answer(query, context)

        if self.memory is not None:
            self.memory.add_turn("user", query)
            self.memory.add_turn("assistant", answer)

        return RAGResult(query=query, answer=answer, evidences=evidences, expanded_queries=expanded_queries)