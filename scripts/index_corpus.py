"""
scripts/index_corpus.py
========================
Ejecuta los literales b) y c): carga el corpus generado por
build_corpus.py, genera los embeddings multimodales con CLIP
(b_embeddings.py) y los indexa en ChromaDB (c_vector_store.py).

Uso:
    python scripts/index_corpus.py
    python scripts/index_corpus.py --reset   # reindexar desde cero
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.a_corpus_preparation import load_corpus
from src.b_embeddings import ClipEmbedder
from src.c_vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Pasos 2-3: generar embeddings e indexar en ChromaDB")
    parser.add_argument("--reset", action="store_true", help="Vaciar la colección antes de indexar")
    args = parser.parse_args()

    if not config.CORPUS_MANIFEST_PATH.exists():
        raise SystemExit(
            f"No se encontró {config.CORPUS_MANIFEST_PATH}. "
            "Ejecuta primero: python scripts/build_corpus.py"
        )

    documents = load_corpus()
    print(f"Cargados {len(documents)} documentos del corpus.")

    embedder = ClipEmbedder()
    store = VectorStore()
    if args.reset:
        store.reset()

    texts = [d.text for d in documents]
    image_paths = [d.image_path for d in documents]
    embeddings = embedder.embed_documents(texts, image_paths)

    doc_ids = [d.doc_id for d in documents]
    metadatas = [
        {
            "product_id": d.product_id,
            "product_title": d.product_title,
            # ChromaDB no admite None en metadatos: usamos "" si no hay imagen.
            "image_path": d.image_path or "",
        }
        for d in documents
    ]

    store.index_documents(doc_ids, embeddings, texts, metadatas)
    print(f"Indexación completa. Documentos en la colección: {store.count()}")
    print("Siguiente paso: streamlit run app.py   (o) python scripts/run_evaluation.py")


if __name__ == "__main__":
    main()