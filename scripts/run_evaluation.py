"""
scripts/run_evaluation.py
==========================
Ejecuta el literal f) Evaluación del sistema: calcula Precision@k,
Recall@k y NDCG@k sobre el conjunto de consultas de prueba (qrels
generados a partir de los juicios ESCI) y guarda el reporte en
data/eval_results/evaluation_results.csv

Uso:
    python scripts/run_evaluation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config
from src.b_embeddings import ClipEmbedder
from src.c_vector_store import VectorStore
from src.f_evaluation import evaluate_system, load_eval_data, save_report


def main() -> None:
    if not config.QUERIES_PATH.exists() or not config.QRELS_PATH.exists():
        raise SystemExit(
            "No se encontraron queries.json / qrels.json. "
            "Ejecuta primero: python scripts/build_corpus.py"
        )

    queries, qrels = load_eval_data()
    store = VectorStore()
    if store.count() == 0:
        raise SystemExit(
            "La colección de ChromaDB está vacía. "
            "Ejecuta primero: python scripts/index_corpus.py"
        )

    embedder = ClipEmbedder()
    df = evaluate_system(store, embedder, queries, qrels)
    save_report(df)

    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("\n=== Resultado promedio (macro-average sobre todas las consultas) ===")
    print(df.tail(1).drop(columns=["query_id", "query"]).to_string(index=False))
    print(f"\nDetalle por consulta guardado en: {config.EVAL_RESULTS_DIR / 'evaluation_results.csv'}")


if __name__ == "__main__":
    main()