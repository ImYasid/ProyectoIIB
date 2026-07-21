"""
src/b_embeddings.py
====================
Literal b) Construcción de representaciones vectoriales.

Responsabilidad exclusiva de este módulo: envolver un modelo CLIP
(HuggingFace `transformers`) para generar embeddings...
    - de texto (título + descripción de producto, y consultas del usuario)
    - de imagen (foto del producto)
    - de documento multimodal (fusión texto+imagen en un solo vector)

Todos los vectores se normalizan (norma L2 = 1) para que la similitud
coseno usada por ChromaDB (ver c_vector_store.py) sea comparable entre
documentos "solo texto" y documentos "texto+imagen".
"""

import logging
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import config

logger = logging.getLogger(__name__)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms[norms == 0] = 1e-8
    return vectors / norms


class ClipEmbedder:
    """Genera embeddings multimodales usando un modelo CLIP preentrenado.

    CLIP proyecta texto e imágenes al mismo espacio vectorial, lo que
    permite comparar una consulta textual directamente contra imágenes
    de producto, o fusionar texto+imagen en un único vector por documento.
    """

    def __init__(self, model_name: str = config.CLIP_MODEL_NAME, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Cargando modelo CLIP '%s' en %s...", model_name, self.device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.embedding_dim = self.model.config.projection_dim

# ---------------------------------------------------------------
    # Embeddings de texto
    # ---------------------------------------------------------------
    @torch.no_grad()
    def embed_text(self, texts: list[str]) -> np.ndarray:
        """Genera embeddings normalizados para una lista de textos.
        Usado tanto para procesar el corpus como para las consultas
        del usuario (mismo método -> mismo espacio vectorial)."""
        all_vectors = []
        for i in range(0, len(texts), config.EMBEDDING_BATCH_SIZE):
            batch = texts[i : i + config.EMBEDDING_BATCH_SIZE]
            inputs = self.processor(
                text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77
            ).to(self.device)
            
            features = self.model.get_text_features(**inputs)
            
            # FIX: Extraer el tensor si HuggingFace devuelve un objeto complejo
            if not isinstance(features, torch.Tensor):
                features = getattr(features, "text_embeds", getattr(features, "pooler_output", features))
                # Si se extrajo el output sin proyectar, aplicamos la proyección de CLIP
                if hasattr(self.model, "text_projection") and features.shape[-1] != self.embedding_dim:
                    features = self.model.text_projection(features)
                    
            all_vectors.append(features.cpu().numpy())
            
        vectors = np.concatenate(all_vectors, axis=0)
        return _l2_normalize(vectors)

    def embed_query(self, query: str) -> np.ndarray:
        """Embedding de una única consulta del usuario. Requisito
        explícito del literal b): 'generar embeddings para las
        consultas del usuario'."""
        return self.embed_text([query])[0]

   # ---------------------------------------------------------------
    # Embeddings de imagen
    # ---------------------------------------------------------------
    @torch.no_grad()
    def embed_images(self, image_paths: list[str]) -> np.ndarray:
        """Genera embeddings normalizados para una lista de rutas de
        imagen. Si una imagen no se puede abrir, se reemplaza por un
        vector de ceros (se descarta su influencia en la fusión)."""
        images = []
        valid_mask = []
        for path in image_paths:
            try:
                images.append(Image.open(path).convert("RGB"))
                valid_mask.append(True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("No se pudo abrir la imagen %s: %s", path, exc)
                valid_mask.append(False)

        vectors = np.zeros((len(image_paths), self.embedding_dim), dtype=np.float32)
        if images:
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            features = self.model.get_image_features(**inputs)
            
            # FIX: Misma lógica de compatibilidad para imágenes
            if not isinstance(features, torch.Tensor):
                features = getattr(features, "image_embeds", getattr(features, "pooler_output", features))
                if hasattr(self.model, "visual_projection") and features.shape[-1] != self.embedding_dim:
                    features = self.model.visual_projection(features)
                    
            features = features.cpu().numpy()
            features = _l2_normalize(features)
            vectors[np.array(valid_mask)] = features
        return vectors

    # ---------------------------------------------------------------
    # Fusión texto + imagen -> un solo vector por documento
    # ---------------------------------------------------------------
    def embed_documents(
        self,
        texts: list[str],
        image_paths: list[Optional[str]],
        alpha: float = config.TEXT_IMAGE_FUSION_ALPHA,
    ) -> np.ndarray:
        """Genera el embedding final de cada documento del corpus,
        fusionando su vector de texto y su vector de imagen (cuando
        existe) mediante un promedio ponderado:

            doc_vec = alpha * text_vec + (1 - alpha) * image_vec

        Si el documento no tiene imagen, se usa únicamente el vector
        de texto. El resultado se vuelve a normalizar (L2 = 1) para
        mantener todas las magnitudes comparables en ChromaDB.
        """
        text_vecs = self.embed_text(texts)

        has_image = [p is not None for p in image_paths]
        image_paths_clean = [p for p in image_paths if p is not None]
        image_vecs_only = (
            self.embed_images(image_paths_clean) if image_paths_clean else np.empty((0, self.embedding_dim))
        )

        image_vecs = np.zeros_like(text_vecs)
        img_iter = iter(image_vecs_only)
        for i, flag in enumerate(has_image):
            if flag:
                image_vecs[i] = next(img_iter)

        fused = np.array(has_image, dtype=np.float32)[:, None] * (
            alpha * text_vecs + (1 - alpha) * image_vecs
        ) + (1 - np.array(has_image, dtype=np.float32)[:, None]) * text_vecs

        return _l2_normalize(fused)


if __name__ == "__main__":
    # Prueba rápida y manual del módulo (no requiere el corpus completo).
    embedder = ClipEmbedder()
    demo_vecs = embedder.embed_text(["a red running shoe", "a wireless keyboard"])
    print("Forma de los embeddings de texto:", demo_vecs.shape)
    print("Norma de cada vector (debe ser ~1.0):", np.linalg.norm(demo_vecs, axis=1))