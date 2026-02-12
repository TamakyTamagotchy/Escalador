# config.py
from dataclasses import dataclass
from typing import Dict, Any, List
import os
import psutil

@dataclass
class Configuracion:
    # Recursos del sistema
    MAXIMO_USO_CPU: int = 90  # Aumentado para mejor rendimiento
    MAXIMO_USO_MEMORIA_GPU: int = 90
    MAXIMO_HILOS: int = max(1, min(os.cpu_count() - 1, 8))
    
    # Parámetros de procesamiento
    TIEMPO_ESPERA: float = 0.05  # Reducido para mejor respuesta
    TAMANIO_CACHE: int = 512  # Aumentado para mejor rendimiento
    TAMANIO_LOTE: int = min(8, MAXIMO_HILOS)  # Optimizado según hilos
    
    # Calidad de salida
    CALIDAD_JPEG: int = 95
    COMPRESION_PNG: int = 1  # Compresión ligera para mejor rendimiento
    
    # Monitoreo de recursos
    UMBRAL_MEMORIA: int = 90
    UMBRAL_MEMORIA_GPU: int = 90
    INTERVALO_MONITOREO: float = 0.5
    INTERVALO_VERIFICACION: float = 0.1
    
    # Nuevos parámetros optimizados
    TAMANIO_MOSAICO: int = 256  # Tamaño óptimo para procesamiento
    CACHE_FRAMES: int = 100  # Número de frames en caché
    TIEMPO_MAXIMO_PROCESAMIENTO: int = 300  # Timeout en segundos
    
    @staticmethod
    def obtener_configuracion_optima():
        """Ajusta la configuración según los recursos del sistema"""
        memoria_total = psutil.virtual_memory().total / (1024 ** 3)  # GB
        nucleos_cpu = os.cpu_count()
        
        return {
            'TAMANIO_LOTE': min(8, max(2, nucleos_cpu // 2)),
            'TAMANIO_CACHE': min(1024, int(memoria_total * 128)),
            'MAXIMO_HILOS': max(1, min(nucleos_cpu - 1, 8)),
            'TAMANIO_MOSAICO': 512 if memoria_total > 16 else 256
        }

@dataclass
class ModeloConfig:
    nombre: str
    descripcion: str
    escala_predeterminada: int
    permite_escala_personalizada: bool
    parametros_gpu: Dict[str, Any]
    optimizaciones: Dict[str, Any]
    tipos_contenido: List[str]