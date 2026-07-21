"""
scripts/build_corpus.py
========================
Ejecuta el literal a) Preparación del corpus: descarga y procesa la
muestra de ESCI + SQID, y genera:
    data/corpus/corpus.jsonl   (documentos multimodales)
    data/corpus/images/*.jpg   (imágenes descargadas)
    data/qrels/queries.json    (consultas de evaluación)
    data/qrels/qrels.json      (juicios de relevancia)

Uso:
    python scripts/build_corpus.py --n-queries 40
    python scripts/build_corpus.py --n-queries 40 --no-images
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.a_corpus_preparation import prepare_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Paso 1: preparar el corpus multimodal")
    parser.add_argument("--n-queries", type=int, default=config.N_QUERIES_SAMPLE,
                         help="Número de consultas ESCI a muestrear")
    parser.add_argument("--no-images", action="store_true",
                         help="No descargar imágenes (más rápido, útil para pruebas de texto)")
    args = parser.parse_args()

    documents = prepare_corpus(n_queries=args.n_queries, download_images=not args.no_images)

    with_image = sum(1 for d in documents if d.image_path)
    print(f"\nCorpus construido: {len(documents)} documentos "
          f"({with_image} con imagen, {len(documents) - with_image} solo texto).")
    print(f"Guardado en: {config.CORPUS_MANIFEST_PATH}")
    print("Siguiente paso: python scripts/index_corpus.py")


if __name__ == "__main__":
    main()