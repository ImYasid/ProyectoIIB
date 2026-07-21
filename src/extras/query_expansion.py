"""
src/extras/query_expansion.py
==============================
Funcionalidad de excelencia: Query Expansion (+15 pts).

Genera automáticamente variantes/reformulaciones de la consulta
original del usuario (sinónimos, términos relacionados, forma más
específica) usando el mismo LLM (Gemini) que el sistema RAG. Cada
variante se embebe y se recupera por separado; los resultados se
fusionan en d_rag_pipeline.py mediante Reciprocal Rank Fusion (RRF),
lo que mejora el recall cuando la consulta original usa un vocabulario
distinto al de las fichas de producto del corpus.

Si no hay GEMINI_API_KEY configurada, se aplica un mecanismo de
respaldo simple y determinista (sin LLM) para que la funcionalidad
siga siendo demostrable.
"""

import logging

from google import genai

import config

logger = logging.getLogger(__name__)

EXPANSION_PROMPT = """Genera {n} reformulaciones distintas y breves de la
siguiente consulta de búsqueda de productos, usando sinónimos y
términos relacionados que un catálogo de e-commerce podría usar.
No expliques nada, responde solo con las {n} reformulaciones, una por
línea, sin numeración ni comillas.

Consulta original: "{query}"
"""


class QueryExpander:
    """Expande una consulta del usuario en varias variantes semánticamente
    equivalentes para mejorar la cobertura de la recuperación."""

    def __init__(
        self,
        gemini_api_key: str | None = config.GEMINI_API_KEY,
        gemini_model: str = config.GEMINI_MODEL,
        n_expansions: int = config.QUERY_EXPANSION_N,
    ):
        self.client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None
        self.gemini_model = gemini_model
        self.n_expansions = n_expansions

    def expand(self, query: str) -> list[str]:
        if self.client is None:
            return self._fallback_expand(query)

        try:
            prompt = EXPANSION_PROMPT.format(n=self.n_expansions, query=query)
            response = self.client.models.generate_content(
                model=self.gemini_model, contents=prompt
            )
            lines = [line.strip("- ").strip() for line in response.text.splitlines()]
            expansions = [line for line in lines if line and line.lower() != query.lower()]
            return expansions[: self.n_expansions]
        except Exception as exc: 
            logger.warning("Falló la expansión con LLM (%s); usando respaldo simple.", exc)
            return self._fallback_expand(query)

    @staticmethod
    def _fallback_expand(query: str) -> list[str]:
        """Mecanismo de respaldo sin LLM: variantes triviales pero útiles
        (singular/plural, con/sin la palabra "producto"/"comprar")."""
        variants = {f"comprar {query}", f"{query} producto", query.rstrip("s")}
        variants.discard(query)
        return list(variants)