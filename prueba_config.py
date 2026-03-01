# prueba_config.py
import logging
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass

from config.models_config import UPSCALERS, detectar_modelos_ncnn

logger = logging.getLogger(__name__)

# Raiz del proyecto es el directorio que contiene este archivo
PROJECT_ROOT = Path(__file__).resolve().parent

# Rutas de binarios relativas a la raiz del proyecto
BIN_DIR = PROJECT_ROOT / "bin"
EXECUTABLE_PATH = BIN_DIR / "realesrgan-ncnn-vulkan.exe"
MODELS_PATH = BIN_DIR / "models"


@dataclass
class UpscalerPaths:
    """Rutas resueltas para un upscaler instalado."""
    upscaler_id: str
    ruta_exe: str
    ruta_modelos: Dict[str, str]  # modelo_id -> ruta al directorio de modelos


def get_realesrgan_path() -> Optional[str]:
    """Retorna la ruta al ejecutable realesrgan si existe."""
    if EXECUTABLE_PATH.is_file():
        return str(EXECUTABLE_PATH)
    logger.error(f"Ejecutable no encontrado: {EXECUTABLE_PATH}")
    return None


def get_models_path() -> Optional[str]:
    """Retorna la ruta al directorio de modelos si existe."""
    if MODELS_PATH.is_dir():
        return str(MODELS_PATH)
    logger.error(f"Directorio de modelos no encontrado: {MODELS_PATH}")
    return None


def detectar_upscalers_instalados() -> Dict[str, UpscalerPaths]:
    """
    Escanea bin/ para detectar que upscalers estan instalados.

    Para realesrgan: auto-detecta modelos NCNN (.param+.bin) en bin/models/.
    Para otros: busca subdirectorios de modelos predefinidos.
    """
    instalados = {}

    for upscaler_id, info in UPSCALERS.items():
        # Buscar ejecutable: primero en bin/ (flat), luego en bin/<nombre>/
        candidatos = [
            BIN_DIR / info.ejecutable,
            BIN_DIR / info.ejecutable.replace(".exe", "") / info.ejecutable,
        ]

        ruta_exe = None
        for candidato in candidatos:
            if candidato.is_file():
                ruta_exe = candidato
                break

        if not ruta_exe:
            continue

        # Buscar modelos
        directorio_exe = ruta_exe.parent
        modelos_encontrados = {}

        if upscaler_id == "realesrgan":
            # Auto-detectar modelos NCNN en bin/models/
            models_dir = BIN_DIR / "models"
            if models_dir.is_dir():
                modelos_detectados = detectar_modelos_ncnn(str(models_dir))
                # Inyectar los modelos detectados en el UpscalerInfo
                info.modelos = modelos_detectados
                for mid in modelos_detectados:
                    modelos_encontrados[mid] = str(models_dir)
        else:
            # Otros upscalers: cada modelo es un subdirectorio
            for modelo_id in info.modelos:
                modelo_dir = directorio_exe / modelo_id
                if modelo_dir.is_dir():
                    modelos_encontrados[modelo_id] = str(modelo_dir)

        if modelos_encontrados:
            instalados[upscaler_id] = UpscalerPaths(
                upscaler_id=upscaler_id,
                ruta_exe=str(ruta_exe),
                ruta_modelos=modelos_encontrados,
            )
            logger.info(
                f"Upscaler detectado: {info.nombre_display} "
                f"({len(modelos_encontrados)} modelos)"
            )

    if not instalados:
        logger.error("No se detecto ningun upscaler instalado en bin/")

    return instalados


def verify_installation() -> bool:
    """Verifica que al menos un upscaler esta instalado."""
    return len(detectar_upscalers_instalados()) > 0
