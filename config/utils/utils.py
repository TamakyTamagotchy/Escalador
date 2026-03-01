# utils.py
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def informacion_formato_codec(configuraciones_codec: Dict[str, Any]) -> str:
    """Formatea la informacion del codec de manera legible."""
    return (
        f"{configuraciones_codec.get('codec', 'unknown')} "
        f"(preset: {configuraciones_codec.get('preset', 'unknown')})"
    )
