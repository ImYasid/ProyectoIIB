"""
src/extras/conversational_memory.py
====================================
Funcionalidad de excelencia: Memoria conversacional (+15 pts).

Mantiene el historial de la conversación (usuario/asistente) y lo
utiliza en dos puntos del pipeline (ver d_rag_pipeline.py):

    1. Generación: el historial reciente se incluye en el prompt
       enviado al LLM, para que pueda resolver referencias como
       "¿y el segundo?" o "muéstrame algo más barato".
    2. Recuperación (opcional, vía `contextualize_query`): las
       preguntas de seguimiento cortas ("¿y en color negro?") se
       reformulan como una consulta autocontenida antes de buscar en
       la base vectorial, ya que de lo contrario el embedding de una
       consulta como "¿y en negro?" por sí sola es poco informativo.
"""

import logging

from google import genai

import config

logger = logging.getLogger(__name__)

CONTEXTUALIZE_PROMPT = """Dado el siguiente historial de conversación y una
nueva pregunta del usuario, reformula la nueva pregunta como una
consulta de búsqueda autocontenida (que no dependa del historial para
entenderse). Si la pregunta ya es autocontenida, devuélvela igual.
Responde SOLO con la consulta reformulada, sin explicaciones.

HISTORIAL:
{history}

NUEVA PREGUNTA: {query}

CONSULTA REFORMULADA:"""


class ConversationMemory:
    """Historial de turnos de una conversación, con utilidades para
    inyectarlo en el prompt de generación y para reformular consultas
    de seguimiento antes de la recuperación vectorial."""

    def __init__(
        self,
        max_turns: int = config.MEMORY_MAX_TURNS,
        gemini_api_key: str | None = config.GEMINI_API_KEY,
        gemini_model: str = config.GEMINI_MODEL,
    ):
        self.max_turns = max_turns
        self.turns: list[dict] = []  # [{"role": "user"|"assistant", "content": str}, ...]
        self.client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None
        self.gemini_model = gemini_model

    def add_turn(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})
        self.turns = self.turns[-self.max_turns:]

    def get_context_window(self) -> str:
        """Devuelve el historial reciente formateado como texto plano
        para incluirlo en el prompt del LLM."""
        return "\n".join(f"{t['role']}: {t['content']}" for t in self.turns)

    def clear(self) -> None:
        self.turns = []

    def contextualize_query(self, query: str) -> str:
        """Reformula `query` como una consulta autocontenida usando el
        historial reciente. Si no hay historial o no hay LLM
        disponible, devuelve la consulta sin modificar."""
        if not self.turns or self.client is None:
            return query
        try:
            prompt = CONTEXTUALIZE_PROMPT.format(history=self.get_context_window(), query=query)
            response = self.client.models.generate_content(
                model=self.gemini_model, contents=prompt
            )
            return response.text.strip() or query
        except Exception as exc: 
            logger.warning("Falló la contextualización de la consulta (%s); usando original.", exc)
            return query