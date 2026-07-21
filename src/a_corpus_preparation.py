"""
src/a_corpus_preparation.py
============================
Literal a) Preparación del corpus.

Responsabilidades de este módulo (y solo de este módulo):
    1. Cargar el corpus multimodal (texto + imágenes) a partir de:
         - ESCI (Amazon Shopping Queries Dataset): consultas, productos
           y juicios de relevancia (esci_label).
         - SQID (Shopping Queries Image Dataset): URLs de imagen por
           producto, que extiende a ESCI con información visual.
    2. Procesar el texto de cada producto (título + descripción +
       bullet points) en un único documento textual limpio.
    3. Asociar correctamente cada imagen con su información textual
       (join por product_id) y descargarla a disco.
    4. Generar el manifiesto del corpus (corpus.jsonl) y los qrels
       de evaluación (queries.json / qrels.json) para el literal f).

Este módulo NO genera embeddings ni indexa nada en la base de datos
vectorial: eso ocurre en b_embeddings.py y c_vector_store.py.
"""

import json
import logging
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

import config

logger = logging.getLogger(__name__)


@dataclass
class CorpusDocument:
    """Un elemento del corpus multimodal: un producto con su texto
    procesado y (opcionalmente) su imagen asociada."""
    doc_id: str
    product_id: str
    product_title: str
    text: str  # texto procesado: título + descripción + bullet points
    image_path: Optional[str]  # ruta local, None si no hay imagen
    image_url: Optional[str]


# -------------------------------------------------------------------
# 1. Carga del corpus (ESCI + SQID)
# -------------------------------------------------------------------
def load_esci_subset(n_queries: int = config.N_QUERIES_SAMPLE,
                     locale: str = config.LOCALE,
                     seed: int = config.RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descarga una muestra del dataset ESCI junto con los productos evaluados."""
    logger.info("Cargando juicios de relevancia ESCI (split=test)...")
    examples = load_dataset(
        config.ESCI_DATASET_ID, config.ESCI_QUERIES_CONFIG, split="test"
    ).to_pandas()

    # CORRECCIÓN 1: Quitamos 'small_version == 1' ya que el split de test no suele tenerlo.
    examples = examples[
        examples["product_locale"] == locale
    ]

    rng = random.Random(seed)
    unique_query_ids = sorted(examples["query_id"].unique().tolist())
    sampled_query_ids = rng.sample(
        unique_query_ids, k=min(n_queries, len(unique_query_ids))
    )
    examples = examples[examples["query_id"].isin(sampled_query_ids)].copy()

    # CORRECCIÓN 2: Usamos .head() nativo sobre el groupby en lugar de .apply()
    # Esto preserva las columnas incluso si el DataFrame llega a estar vacío y es mucho más rápido.
    examples = (
        examples.groupby("query_id")
        .head(config.MAX_PRODUCTS_PER_QUERY)
        .reset_index(drop=True)
    )

    needed_product_ids = set(examples["product_id"].unique())
    if len(needed_product_ids) > config.MAX_TOTAL_PRODUCTS:
        needed_product_ids = set(
            rng.sample(sorted(needed_product_ids), config.MAX_TOTAL_PRODUCTS)
        )
        examples = examples[examples["product_id"].isin(needed_product_ids)]

    logger.info("Cargando catálogo de productos ESCI...")
    products = load_dataset(
        config.ESCI_DATASET_ID, config.ESCI_PRODUCTS_CONFIG, split="test"
    ).to_pandas()
    
    products = products[
        (products["product_locale"] == locale)
        & (products["product_id"].isin(needed_product_ids))
    ].drop_duplicates(subset="product_id")

    logger.info(
        "Muestra final: %d consultas, %d pares (query, producto), %d productos.",
        examples["query_id"].nunique(), len(examples), len(products),
    )
    return examples, products

def load_image_urls(product_ids: set[str]) -> pd.DataFrame:
    """Carga las URLs de imagen (dataset SQID) para el conjunto de product_id dado."""
    logger.info("Cargando URLs de imagen (SQID)...")
    img_urls = load_dataset(
        config.SQID_DATASET_ID, config.SQID_IMAGES_CONFIG, split="train"
    ).to_pandas()
    img_urls = img_urls[img_urls["product_id"].isin(product_ids)]
    img_urls = img_urls.dropna(subset=["image_url"]).drop_duplicates(subset="product_id")
    logger.info("Imágenes disponibles para %d/%d productos.", len(img_urls), len(product_ids))
    return img_urls


# -------------------------------------------------------------------
# 2. Procesamiento de texto
# -------------------------------------------------------------------
def process_product_text(row: pd.Series) -> str:
    """Combina y limpia los campos textuales de un producto en un único documento."""
    parts = []
    for field in ("product_title", "product_brand", "product_color",
                  "product_description", "product_bullet_point"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    text = " . ".join(parts)
    text = re.sub(r"\s+", " ", text)          # colapsar espacios
    text = re.sub(r"<[^>]+>", " ", text)       # quitar posibles tags HTML
    return text.strip()


# -------------------------------------------------------------------
# 3. Asociación y descarga de imágenes
# -------------------------------------------------------------------
def download_image(url: str, dest_path: Path) -> bool:
    """Descarga una imagen y valida que sea legible con PIL."""
    try:
        response = requests.get(url, timeout=config.IMAGE_DOWNLOAD_TIMEOUT)
        response.raise_for_status()
        dest_path.write_bytes(response.content)
        with Image.open(dest_path) as img:
            img.verify()  # descarta archivos corruptos
        return True
    except Exception as exc: 
        logger.warning("No se pudo descargar %s: %s", url, exc)
        dest_path.unlink(missing_ok=True)
        return False


def build_multimodal_corpus(
    df_products: pd.DataFrame,
    df_image_urls: pd.DataFrame,
    download_images: bool = config.DOWNLOAD_IMAGES,
) -> list[CorpusDocument]:
    """Construye la lista de documentos del corpus."""
    url_by_product = dict(zip(df_image_urls["product_id"], df_image_urls["image_url"]))
    documents: list[CorpusDocument] = []

    for _, row in tqdm(df_products.iterrows(), total=len(df_products), desc="Procesando corpus"):
        product_id = row["product_id"]
        text = process_product_text(row)
        if not text:
            continue

        image_url = url_by_product.get(product_id)
        image_path = None
        if image_url and download_images:
            dest = config.IMAGES_DIR / f"{product_id}.jpg"
            if dest.exists() or download_image(image_url, dest):
                image_path = str(dest)

        documents.append(
            CorpusDocument(
                doc_id=f"doc_{product_id}",
                product_id=product_id,
                product_title=str(row.get("product_title", ""))[:200],
                text=text,
                image_path=image_path,
                image_url=image_url,
            )
        )
    return documents


# -------------------------------------------------------------------
# 4. Persistencia: corpus.jsonl + queries.json + qrels.json
# -------------------------------------------------------------------
def save_corpus(documents: list[CorpusDocument], path: Path = config.CORPUS_MANIFEST_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
    logger.info("Corpus guardado en %s (%d documentos).", path, len(documents))


def load_corpus(path: Path = config.CORPUS_MANIFEST_PATH) -> list[CorpusDocument]:
    documents = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            documents.append(CorpusDocument(**json.loads(line)))
    return documents


def save_queries_and_qrels(df_examples: pd.DataFrame, valid_product_ids: set[str]) -> None:
    """A partir de los juicios ESCI, genera queries.json y qrels.json."""
    df = df_examples[df_examples["product_id"].isin(valid_product_ids)]

    queries = (
        df.drop_duplicates(subset="query_id")
        .set_index("query_id")["query"].to_dict()
    )
    queries = {str(k): v for k, v in queries.items()}

    qrels: dict[str, dict[str, int]] = {}
    for _, row in df.iterrows():
        qid = str(row["query_id"])
        doc_id = f"doc_{row['product_id']}"
        grade = config.RELEVANCE_GRADES.get(row["esci_label"], 0)
        qrels.setdefault(qid, {})[doc_id] = grade

    with open(config.QUERIES_PATH, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
    with open(config.QRELS_PATH, "w", encoding="utf-8") as f:
        json.dump(qrels, f, ensure_ascii=False, indent=2)

    logger.info(
        "Guardadas %d consultas y qrels para %d consultas en %s / %s.",
        len(queries), len(qrels), config.QUERIES_PATH, config.QRELS_PATH,
    )


# -------------------------------------------------------------------
# Orquestación del literal a)
# -------------------------------------------------------------------
def prepare_corpus(n_queries: int = config.N_QUERIES_SAMPLE,
                   download_images: bool = config.DOWNLOAD_IMAGES) -> list[CorpusDocument]:
    """Punto de entrada del literal a)."""
    df_examples, df_products = load_esci_subset(n_queries=n_queries)
    df_image_urls = load_image_urls(set(df_products["product_id"]))
    documents = build_multimodal_corpus(df_products, df_image_urls, download_images)

    valid_ids = {d.product_id for d in documents}
    save_corpus(documents)
    save_queries_and_qrels(df_examples, valid_ids)
    return documents


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Literal a) Preparación del corpus")
    parser.add_argument("--n-queries", type=int, default=config.N_QUERIES_SAMPLE)
    parser.add_argument("--no-images", action="store_true", help="No descargar imágenes")
    args = parser.parse_args()

    # Configuración básica de logging para ver los mensajes en consola
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    prepare_corpus(n_queries=args.n_queries, download_images=not args.no_images)