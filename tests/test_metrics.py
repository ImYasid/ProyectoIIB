"""
tests/test_metrics.py
======================
Pruebas unitarias de las métricas de evaluación (literal f).
Se prueban de forma aislada (con datos sintéticos), sin necesitar el
corpus real ni la base vectorial indexada.

Ejecución:
    pytest tests/test_metrics.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.f_evaluation import precision_at_k, recall_at_k, ndcg_at_k

RETRIEVED = ["a", "b", "c", "d", "e"]
RELEVANT_BINARY = {"a", "c", "e", "z"}  # 'z' no fue recuperado (para probar recall < 1)
RELEVANCE_GRADES = {"a": 3, "b": 0, "c": 2, "d": 0, "e": 1}


def test_precision_at_k():
    assert precision_at_k(RETRIEVED, RELEVANT_BINARY, 3) == 2 / 3
    assert precision_at_k(RETRIEVED, RELEVANT_BINARY, 5) == 3 / 5
    assert precision_at_k([], RELEVANT_BINARY, 5) == 0.0


def test_recall_at_k():
    assert recall_at_k(RETRIEVED, RELEVANT_BINARY, 3) == 2 / 4
    assert recall_at_k(RETRIEVED, RELEVANT_BINARY, 5) == 3 / 4
    assert recall_at_k(RETRIEVED, set(), 5) == 0.0


def test_ndcg_at_k_perfect_order_is_one():
    # Si el orden recuperado coincide con el orden ideal, NDCG debe ser 1.0
    ideal_order = ["a", "c", "e", "b", "d"]  # ordenado por relevancia descendente
    assert abs(ndcg_at_k(ideal_order, RELEVANCE_GRADES, 5) - 1.0) < 1e-9


def test_ndcg_at_k_worse_order_is_lower():
    good = ndcg_at_k(["a", "c", "e"], RELEVANCE_GRADES, 3)
    bad = ndcg_at_k(["b", "d", "a"], RELEVANCE_GRADES, 3)
    assert good > bad


def test_ndcg_no_relevant_docs_is_zero():
    assert ndcg_at_k(RETRIEVED, {}, 5) == 0.0


if __name__ == "__main__":
    test_precision_at_k()
    test_recall_at_k()
    test_ndcg_at_k_perfect_order_is_one()
    test_ndcg_at_k_worse_order_is_lower()
    test_ndcg_no_relevant_docs_is_zero()
    print("✅ Todas las pruebas de métricas pasaron.")