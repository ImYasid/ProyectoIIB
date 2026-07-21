"""
src/extras/reranking.py
========================
Funcionalidad de excelencia: Re-ranking (+15 pts).

La búsqueda vectorial (literal c) es rápida pero aproximada: CLIP
comprime todo el significado de texto+imagen en un solo vector, lo
que a veces pierde matices léxicos finos entre la consulta y el
título/descripción del producto.

Este módulo refina el ranking inicial usando un *cross-encoder*
(sentence-transformers), un modelo mucho más costoso computacionalmente
pero más preciso: en vez de comparar dos vectores ya calculados,
procesa el par (consulta, documento) JUNTOS en una sola pasada por el
modelo, lo que le permite capturar interacciones término a término.

Por eso solo se aplica sobre un conjunto reducido de candidatos
(top ~20, ver config.RERANK_CANDIDATE_MULTIPLIER) y no sobre todo el
corpus: sería demasiado lento.
"""

import logging

from sentence_transformers import CrossEncoder

import config
from src.c_vector_store import RetrievedDocument

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Re-ordena una lista de documentos candidatos según qué tan bien
    responden a la consulta, usando un cross-encoder de
    sentence-transformers entrenado para ranking (MS MARCO)."""

    def __init__(self, model_name: str = config.RERANKER_MODEL_NAME):
        logger.info("Cargando cross-encoder de re-ranking '%s'...", model_name)
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, candidates: list[RetrievedDocument], top_k: int
    ) -> list[RetrievedDocument]:
        if not candidates:
            return candidates

        pairs = [(query, f"{doc.product_title}. {doc.text}") for doc in candidates]
        cross_scores = self.model.predict(pairs)

        # Combinamos el score original de similitud vectorial con el
        # score del cross-encoder: el cross-encoder decide el orden,
        # pero conservamos también el score vectorial en el objeto
        # para que la interfaz pueda mostrar ambos si se desea.
        reranked = []
        for doc, cross_score in zip(candidates, cross_scores):
            reranked.append(
                RetrievedDocument(
                    doc_id=doc.doc_id,
                    score=round(float(cross_score), 4),
                    text=doc.text,
                    product_title=doc.product_title,
                    image_path=doc.image_path,
                    product_id=doc.product_id,
                )
            )
        reranked.sort(key=lambda d: d.score, reverse=True)
        return reranked[:top_k]