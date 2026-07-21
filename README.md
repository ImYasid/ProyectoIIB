# Sistema de Recuperación de Información Multimodal con RAG

Proyecto final de la asignatura **Recuperación de Información**. Sistema que
responde consultas conversacionales sobre un catálogo de productos
(texto + imágenes), usando embeddings multimodales (CLIP), una base de
datos vectorial (ChromaDB) y generación aumentada por recuperación (RAG)
con Gemini.

## Corpus utilizado

El corpus se construye automáticamente combinando dos datasets públicos de
Hugging Face:

- **[ESCI / Shopping Queries Dataset](https://huggingface.co/datasets/milistu/amazon-esci-data)**
  (Amazon): consultas reales de usuarios, productos y juicios de relevancia
  graduados (`E`=Exact, `S`=Substitute, `C`=Complement, `I`=Irrelevant).
  Estos juicios se reutilizan directamente como **qrels** para la evaluación
  (literal f).
- **[SQID - Shopping Queries Image Dataset](https://huggingface.co/datasets/crossingminds/shopping-queries-image-dataset)**:
  extiende ESCI con URLs de imagen por producto, lo que permite construir un
  corpus verdaderamente **multimodal** (texto + imagen asociados por
  `product_id`).

Por defecto se muestrea un subconjunto manejable (40 consultas, hasta ~300
productos) para que el proyecto sea reproducible en un computador normal sin
necesidad de descargar los ~165.000 productos completos del dataset.

## Arquitectura y estructura del código

El proyecto es **modular por diseño**: cada literal del enunciado y cada
funcionalidad de excelencia vive en su propio archivo, sin lógica mezclada.

```
rag_multimodal/
├── config.py                          # Configuración central (rutas, modelos, parámetros)
├── main.py                            # CLI orquestador (delega en scripts/, sin lógica propia)
├── app.py                             # Punto de entrada de la interfaz web (Streamlit)
│
├── src/
│   ├── a_corpus_preparation.py        # Literal a) Preparación del corpus
│   ├── b_embeddings.py                # Literal b) Embeddings multimodales (CLIP)
│   ├── c_vector_store.py              # Literal c) Base de datos vectorial (ChromaDB)
│   ├── d_rag_pipeline.py              # Literal d) Sistema RAG + evidencias
│   ├── e_web_interface.py             # Literal e) Componentes de la interfaz conversacional
│   ├── f_evaluation.py                # Literal f) Evaluación (Precision@k, Recall@k, NDCG@k)
│   │
│   └── extras/                        # Funcionalidades de excelencia (+15 c/u)
│       ├── reranking.py               # Re-ranking con cross-encoder
│       ├── query_expansion.py         # Expansión/reformulación de consultas
│       ├── relevance_feedback.py      # Feedback 👍/👎 + algoritmo de Rocchio
│       └── conversational_memory.py   # Memoria conversacional
│
├── scripts/                           # CLIs delgadas, una por etapa del pipeline
│   ├── build_corpus.py                # Ejecuta literal a)
│   ├── index_corpus.py                # Ejecuta literales b) + c)
│   └── run_evaluation.py              # Ejecuta literal f)
│
└── data/                               # Se genera al ejecutar los scripts (no versionar)
    ├── corpus/corpus.jsonl
    ├── corpus/images/*.jpg
    ├── qrels/{queries,qrels}.json
    ├── chroma_db/                      # Persistencia de ChromaDB
    └── eval_results/evaluation_results.csv
```

### Flujo de datos

```
   ESCI + SQID (HuggingFace)
            │
            ▼
 a_corpus_preparation.py   ── corpus.jsonl, queries.json, qrels.json
            │
            ▼
    b_embeddings.py (CLIP) ── vectores texto+imagen fusionados
            │
            ▼
   c_vector_store.py (ChromaDB) ── índice vectorial persistente
            │
            ▼
   d_rag_pipeline.py ── retrieve → build_context → generate (Gemini)
            │                 ▲            ▲
            │           [reranking]  [query_expansion]
            │                 ▲            ▲
            │           [memoria conversacional]
            ▼
   e_web_interface.py / app.py ── chat + evidencias + 👍/👎 (feedback)

   f_evaluation.py ── Precision@k / Recall@k / NDCG@k sobre queries+qrels
```

## Instalación

Requiere **Python 3.10+**.

```bash
# 1. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar la API key de Gemini
cp .env.example .env
# Edita .env y pega tu GEMINI_API_KEY
# (gratuita en https://aistudio.google.com/apikey)
```

## Ejecución

El pipeline se ejecuta en 3 pasos, cada uno independiente y modular:

```bash
# Paso 1 (literal a): descargar y preparar el corpus multimodal
python scripts/build_corpus.py --n-queries 40

# Paso 2 (literales b + c): generar embeddings CLIP e indexar en ChromaDB
python scripts/index_corpus.py

# Paso 3 (literal e): lanzar la interfaz web conversacional
streamlit run app.py
```

También puedes usar el orquestador `main.py` (equivalente, más cómodo):

```bash
python main.py build-corpus --n-queries 40
python main.py index
python main.py evaluate
python main.py chat   # imprime el comando de streamlit
```

### Evaluación del sistema (literal f)

```bash
python scripts/run_evaluation.py
```

Genera `data/eval_results/evaluation_results.csv` con Precision@k, Recall@k
y NDCG@k (k = 3, 5, 10) por consulta y el promedio general.

### Funcionalidades de excelencia

Todas están **desactivadas por defecto excepto memoria y feedback**, y se
activan/desactivan desde la barra lateral de la interfaz web sin reiniciar
el sistema:

| Funcionalidad | Módulo | Cómo activarla |
|---|---|---|
| Re-ranking | `src/extras/reranking.py` | Checkbox "Re-ranking" en la barra lateral |
| Query Expansion | `src/extras/query_expansion.py` | Checkbox "Query Expansion" |
| Relevance Feedback | `src/extras/relevance_feedback.py` | Botones 👍/👎 bajo cada evidencia |
| Memoria conversacional | `src/extras/conversational_memory.py` | Checkbox "Memoria conversacional" |

## Parámetros configurables

Todos los parámetros relevantes están centralizados en `config.py` y pueden
sobreescribirse mediante variables de entorno (ver `.env.example`):
tamaño de la muestra del corpus, modelo CLIP, modelo de re-ranking, top-k
por defecto, número de expansiones de consulta, turnos de memoria a
recordar, etc.

## Notas técnicas

- **Fusión texto+imagen**: cada documento del corpus se representa con un
  único vector CLIP, combinando el embedding de texto y el de imagen con un
  promedio ponderado (`config.TEXT_IMAGE_FUSION_ALPHA`), y renormalizando a
  norma 1. Esto permite indexar en una sola colección de ChromaDB tanto
  productos con imagen como productos que solo tienen texto.
- **Métrica de similitud**: ChromaDB se configura con `hnsw:space: cosine`.
- **Qrels reales, no simulados**: las métricas de evaluación se calculan
  sobre los juicios de relevancia originales del benchmark ESCI, no sobre
  datos sintéticos.
- El **cross-encoder de re-ranking** y la **expansión de consultas** solo
  se aplican sobre los primeros `top_k * 4` candidatos (no sobre todo el
  corpus), ya que son computacionalmente más costosos que la búsqueda
  vectorial.

## Autoría

Desarrollado para el proyecto final de Recuperación de Información. El uso
de herramientas de IA como asistencia en el desarrollo está permitido según
las bases del proyecto; cada integrante del grupo debe poder explicar la
arquitectura y las decisiones técnicas de cada módulo.