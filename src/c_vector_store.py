"""
src/c_vector_store.py
======================
Literal c) Base de datos vectorial.

Responsabilidad exclusiva de este módulo: envolver ChromaDB para
    - indexar el corpus (embeddings + metadatos: texto, imagen, título),
    - recuperar los documentos más similares a una consulta,
    - devolver un ranking Top-k con el puntaje de similitud de cada
      resultado.

Este módulo no sabe nada de CLIP ni de RAG: solo recibe vectores ya
calculados (por b_embeddings.py) y los administra.
"""

import logging
from dataclasses import dataclass

import chromadb
import numpy as np

import config

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """Un resultado de la búsqueda vectorial, con su puntaje de similitud."""

    doc_id: str
    score: float  # similitud coseno en [-1, 1]; más alto = más similar
    text: str
    product_title: str
    image_path: str | None
    product_id: str


class VectorStore:
    """Envoltorio delgado sobre `chromadb.PersistentClient` para el
    corpus multimodal del proyecto."""

    def __init__(self, persist_dir: str = str(config.CHROMA_DIR),
                 collection_name: str = config.COLLECTION_NAME):
        self.client = chromadb.PersistentClient(path=persist_dir)
        # "hnsw:space": "cosine" -> ChromaDB reporta distancia coseno,
        # que convertimos a similitud (1 - distancia) al consultar.
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    # ---------------------------------------------------------------
    # Indexación
    # ---------------------------------------------------------------
    def index_documents(
        self,
        doc_ids: list[str],
        embeddings: np.ndarray,
        documents: list[str],
        metadatas: list[dict],
        batch_size: int = 64,
    ) -> None:
        """Indexa el corpus completo en ChromaDB, en lotes."""
        n = len(doc_ids)
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            self.collection.upsert(
                ids=doc_ids[i:j],
                embeddings=embeddings[i:j].tolist(),
                documents=documents[i:j],
                metadatas=metadatas[i:j],
            )
        logger.info("Indexados %d documentos en la colección '%s'.", n, self.collection.name)

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Vacía la colección (útil para reindexar desde cero)."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    # ---------------------------------------------------------------
    # Recuperación Top-k
    # ---------------------------------------------------------------
    def query(self, query_embedding: np.ndarray, top_k: int = config.TOP_K_DEFAULT,
               where: dict | None = None) -> list[RetrievedDocument]:
        """Recupera los `top_k` documentos más similares a un embedding
        de consulta y devuelve el ranking con su puntaje de similitud."""
        result = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where,
        )
        return self._parse_result(result)

    def query_by_ids(self, doc_ids: list[str]) -> list[RetrievedDocument]:
        """Recupera documentos específicos por id (usado por los
        módulos de extras, p. ej. relevance_feedback)."""
        result = self.collection.get(ids=doc_ids, include=["documents", "metadatas", "embeddings"])
        docs = []
        for i, doc_id in enumerate(result["ids"]):
            meta = result["metadatas"][i]
            docs.append(
                RetrievedDocument(
                    doc_id=doc_id, score=1.0, text=result["documents"][i],
                    product_title=meta.get("product_title", ""),
                    image_path=meta.get("image_path"),
                    product_id=meta.get("product_id", ""),
                )
            )
        return docs

    @staticmethod
    def _parse_result(result: dict) -> list[RetrievedDocument]:
        docs = []
        ids = result["ids"][0]
        distances = result["distances"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        for doc_id, dist, text, meta in zip(ids, distances, documents, metadatas):
            similarity = 1.0 - dist  # distancia coseno -> similitud coseno
            docs.append(
                RetrievedDocument(
                    doc_id=doc_id,
                    score=round(float(similarity), 4),
                    text=text,
                    product_title=meta.get("product_title", ""),
                    image_path=meta.get("image_path") or None,
                    product_id=meta.get("product_id", ""),
                )
            )
        return docs


if __name__ == "__main__":
    store = VectorStore()
    print(f"Documentos actualmente indexados: {store.count()}")