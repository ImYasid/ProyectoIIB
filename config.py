"""
config.py
=========
Configuración centralizada del sistema de Recuperación de Información
Multimodal con RAG. Todos los módulos (a-f y extras) importan sus
parámetros desde aquí, para no repetir "numeros magicos" ni rutas
hardcodeadas en cada archivo.

No contiene lógica de negocio, solo constantes y carga de variables
de entorno (.env).
"""

import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# -----------------------------------------------------------------
# Logging (configurado UNA sola vez aquí; el resto de módulos solo
# hacen `logger = logging.getLogger(__name__)`, ya que
# logging.basicConfig() solo tiene efecto la primera vez que se llama
# en todo el proceso).
# -----------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

# -----------------------------------------------------------------
# Variables de entorno (.env)
# -----------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# -----------------------------------------------------------------
# Rutas del proyecto
# -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CORPUS_DIR = DATA_DIR / "corpus"
IMAGES_DIR = CORPUS_DIR / "images"
CORPUS_MANIFEST_PATH = CORPUS_DIR / "corpus.jsonl"

QRELS_DIR = DATA_DIR / "qrels"
QUERIES_PATH = QRELS_DIR / "queries.json"
QRELS_PATH = QRELS_DIR / "qrels.json"

CHROMA_DIR = DATA_DIR / "chroma_db"
COLLECTION_NAME = "multimodal_corpus"

FEEDBACK_PATH = DATA_DIR / "relevance_feedback.json"
EVAL_RESULTS_DIR = DATA_DIR / "eval_results"

for _dir in (CORPUS_DIR, IMAGES_DIR, QRELS_DIR, CHROMA_DIR, EVAL_RESULTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------
# Fuentes de datos (HuggingFace Hub)
# -----------------------------------------------------------------
# Texto + juicios de relevancia (ESCI): query, product_id, esci_label
ESCI_DATASET_ID = "milistu/amazon-esci-data"
ESCI_PRODUCTS_CONFIG = "products"
ESCI_QUERIES_CONFIG = "queries"

# URLs de imágenes de producto (SQID, extiende a ESCI con imágenes)
SQID_DATASET_ID = "crossingminds/shopping-queries-image-dataset"
SQID_IMAGES_CONFIG = "product_image_urls"

LOCALE = "us"  # SQID solo cubre locale 'us'

# Mapeo de la etiqueta ESCI a un grado de relevancia graduado (para NDCG)
# E = Exact, S = Substitute, C = Complement, I = Irrelevant
RELEVANCE_GRADES = {"E": 3, "S": 2, "C": 1, "I": 0}
# Para Precision/Recall (métricas binarias) consideramos relevante E y S,
# siguiendo el criterio estándar usado en la literatura del benchmark ESCI.
BINARY_RELEVANT_LABELS = {"E", "S"}

# -----------------------------------------------------------------
# Muestreo del corpus (para que el proyecto sea manejable en un
# entorno de estudiante: no se indexan los ~165k productos completos)
# -----------------------------------------------------------------
N_QUERIES_SAMPLE = int(os.environ.get("N_QUERIES_SAMPLE", 40))
MAX_PRODUCTS_PER_QUERY = int(os.environ.get("MAX_PRODUCTS_PER_QUERY", 12))
MAX_TOTAL_PRODUCTS = int(os.environ.get("MAX_TOTAL_PRODUCTS", 300))
DOWNLOAD_IMAGES = os.environ.get("DOWNLOAD_IMAGES", "true").lower() == "true"
IMAGE_DOWNLOAD_TIMEOUT = 8  # segundos
RANDOM_SEED = 42

# -----------------------------------------------------------------
# Embeddings multimodales (CLIP)
# -----------------------------------------------------------------
CLIP_MODEL_NAME = os.environ.get("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
# Peso del texto vs. imagen al fusionar en un solo vector por documento.
# 0.5 = mismo peso para texto e imagen.
TEXT_IMAGE_FUSION_ALPHA = float(os.environ.get("TEXT_IMAGE_FUSION_ALPHA", 0.6))
EMBEDDING_BATCH_SIZE = 16

# -----------------------------------------------------------------
# Recuperación / RAG
# -----------------------------------------------------------------
TOP_K_DEFAULT = int(os.environ.get("TOP_K_DEFAULT", 5))
MAX_CONTEXT_CHARS_PER_DOC = 600  # recorte de cada doc al construir el contexto

# -----------------------------------------------------------------
# Funcionalidades de excelencia (extras)
# -----------------------------------------------------------------
RERANKER_MODEL_NAME = os.environ.get(
    "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
RERANK_CANDIDATE_MULTIPLIER = 4  # cuántos candidatos traer antes de re-rankear (k*4)

QUERY_EXPANSION_N = int(os.environ.get("QUERY_EXPANSION_N", 3))  # nº de variantes

MEMORY_MAX_TURNS = int(os.environ.get("MEMORY_MAX_TURNS", 6))  # turnos a recordar

# -----------------------------------------------------------------
# Evaluación
# -----------------------------------------------------------------
EVAL_K_VALUES = [3, 5, 10]