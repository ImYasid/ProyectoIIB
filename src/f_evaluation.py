"""
src/f_evaluation.py
====================
Literal f) Evaluación del sistema.

Implementa las métricas de recuperación exigidas por la especificación
(Precision@k, Recall@k, NDCG@k) y la rutina que las calcula sobre un
conjunto de consultas con juicios de relevancia (qrels), evaluando
SOLO el componente de recuperación (búsqueda vectorial), que es lo que
las métricas de RI miden — la calidad de la respuesta generada por el
LLM no se mide con Precision/Recall/NDCG.
"""

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.b_embeddings import ClipEmbedder
from src.c_vector_store import VectorStore

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Métricas
# -------------------------------------------------------------------
def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fracción de los k primeros resultados que son relevantes."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fracción del total de documentos relevantes que fueron
    recuperados dentro de los k primeros resultados."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(retrieved_ids: list[str], relevance_grades: dict[str, int], k: int) -> float:
    """NDCG@k con relevancia graduada (0-3, según config.RELEVANCE_GRADES).
    DCG usa el descuento logarítmico estándar (log2(rank+1))."""
    top_k = retrieved_ids[:k]

    dcg = sum(
        relevance_grades.get(doc_id, 0) / math.log2(rank + 2)  # rank 0-based -> +2
        for rank, doc_id in enumerate(top_k)
    )

    ideal_grades = sorted(relevance_grades.values(), reverse=True)[:k]
    idcg = sum(grade / math.log2(rank + 2) for rank, grade in enumerate(ideal_grades))

    return dcg / idcg if idcg > 0 else 0.0


# -------------------------------------------------------------------
# Evaluación end-to-end sobre el conjunto de consultas de prueba
# -------------------------------------------------------------------
def load_eval_data() -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    with open(config.QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(config.QRELS_PATH, "r", encoding="utf-8") as f:
        qrels = json.load(f)
    return queries, qrels


def evaluate_system(
    vector_store: VectorStore,
    embedder: ClipEmbedder,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    k_values: list[int] = config.EVAL_K_VALUES,
) -> pd.DataFrame:
    """Evalúa el componente de recuperación del sistema (búsqueda
    vectorial) sobre todas las consultas de `queries`, usando los
    juicios de relevancia de `qrels`. Devuelve un DataFrame con una
    fila por consulta y una fila final de promedios (macro-average)."""
    rows = []
    max_k = max(k_values)

    for query_id, query_text in queries.items():
        relevance_grades = qrels.get(query_id, {})
        # CORRECCIÓN: Se actualizó a la llave "Substitute" en lugar de "S"
        relevant_ids = {
            doc_id for doc_id, grade in relevance_grades.items()
            if grade >= config.RELEVANCE_GRADES.get("Substitute", 2) 
        }
        if not relevance_grades:
            continue

        query_vec = embedder.embed_query(query_text)
        results = vector_store.query(query_vec, top_k=max_k)
        retrieved_ids = [doc.doc_id for doc in results]

        row = {"query_id": query_id, "query": query_text}
        for k in k_values:
            row[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
            row[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
            row[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevance_grades, k)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No hay consultas evaluables (revisa qrels.json / queries.json).")
        return df

    numeric_cols = [c for c in df.columns if c not in ("query_id", "query")]
    mean_row = {"query_id": "PROMEDIO", "query": ""}
    mean_row.update(df[numeric_cols].mean().to_dict())
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    return df


def save_report(df: pd.DataFrame, out_dir: Path = config.EVAL_RESULTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Reporte de evaluación guardado en %s", csv_path)
    return csv_path


if __name__ == "__main__":
    # Asegúrate de agregar EVAL_K_VALUES = [5, 10, 20] y EVAL_RESULTS_DIR = Path("reports") a config.py
    queries, qrels = load_eval_data()
    store = VectorStore()
    embedder = ClipEmbedder()

    results_df = evaluate_system(store, embedder, queries, qrels)
    save_report(results_df)

    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(results_df.tail(1).to_string(index=False))