# config.py
from dataclasses import dataclass
from typing import Dict, Any, List
import os

@dataclass
class Configuracion:
    MAXIMO_USO_CPU: int = 85
    MAXIMO_USO_MEMORIA_GPU: int = 85
    MAXIMO_HILOS: int = min(os.cpu_count() - 1, 6)
    TIEMPO_ESPERA: float = 0.1
    TAMANIO_CACHE: int = 256
    TAMANIO_LOTE: int = 4
    CALIDAD_JPEG: int = 95
    COMPRESION_PNG: int = 0

@dataclass
class ModeloConfig:
    nombre: str
    descripcion: str
    escala_predeterminada: int
    permite_escala_personalizada: bool
    parametros_gpu: Dict[str, Any]
    optimizaciones: Dict[str, Any]
    tipos_contenido: List[str]