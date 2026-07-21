"""
src/extras/relevance_feedback.py
=================================
Funcionalidad de excelencia: Relevance Feedback (+15 pts).

Permite que el usuario califique con 👍 / 👎 los documentos recuperados
(ver el botón en src/e_web_interface.py). Esa retroalimentación se
usa de dos maneras:

    1. Se persiste en disco (data/relevance_feedback.json) para
       análisis posterior.
    2. Se aplica el algoritmo clásico de Rocchio para desplazar el
       vector de la consulta hacia los documentos marcados como
       relevantes (👍) y alejarlo de los marcados como no relevantes
       (👎), mejorando búsquedas posteriores dentro de la misma sesión
       (p. ej. si el usuario refina o repite una búsqueda parecida).
"""

import json
import logging
from pathlib import Path

import numpy as np

import config
from src.b_embeddings import ClipEmbedder
from src.c_vector_store import VectorStore

logger = logging.getLogger(__name__)

# Constantes estándar del algoritmo de Rocchio.
ALPHA = 1.0   # peso de la consulta original
BETA = 0.75   # peso de los documentos relevantes (👍)
GAMMA = 0.15  # peso de los documentos no relevantes (👎)


class RelevanceFeedbackStore:
    """Registra retroalimentación del usuario y recalcula el vector de
    consulta aplicando el algoritmo de Rocchio."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: ClipEmbedder,
        path: Path = config.FEEDBACK_PATH,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.path = path
        self.feedback: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.feedback, f, ensure_ascii=False, indent=2)

    def record_feedback(self, query: str, doc_id: str, rating: int) -> None:
        """rating: +1 (👍 relevante) o -1 (👎 no relevante)."""
        assert rating in (1, -1), "rating debe ser +1 o -1"
        self.feedback.append({"query": query, "doc_id": doc_id, "rating": rating})
        self._save()
        logger.info("Feedback registrado: query=%r doc_id=%s rating=%+d", query, doc_id, rating)

    def get_feedback_for_query(self, query: str) -> list[dict]:
        return [f for f in self.feedback if f["query"] == query]

    def refine_query_vector(self, query: str) -> np.ndarray:
        """Aplica el algoritmo de Rocchio para producir un nuevo vector
        de consulta, desplazado según el feedback acumulado para esa
        consulta exacta:

            q' = alpha*q + beta * mean(vectores 👍) - gamma * mean(vectores 👎)

        Si no hay feedback registrado para la consulta, devuelve el
        embedding original sin modificar.
        """
        original_vec = self.embedder.embed_query(query)
        entries = self.get_feedback_for_query(query)
        if not entries:
            return original_vec

        liked_ids = [e["doc_id"] for e in entries if e["rating"] == 1]
        disliked_ids = [e["doc_id"] for e in entries if e["rating"] == -1]

        new_vec = ALPHA * original_vec
        if liked_ids:
            liked_docs = self.vector_store.query_by_ids(liked_ids)
            liked_vecs = self.embedder.embed_text([d.text for d in liked_docs])
            new_vec = new_vec + BETA * liked_vecs.mean(axis=0)
        if disliked_ids:
            disliked_docs = self.vector_store.query_by_ids(disliked_ids)
            disliked_vecs = self.embedder.embed_text([d.text for d in disliked_docs])
            new_vec = new_vec - GAMMA * disliked_vecs.mean(axis=0)

        norm = np.linalg.norm(new_vec)
        return new_vec / norm if norm > 0 else original_vec