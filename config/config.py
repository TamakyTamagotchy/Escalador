# config.py
import os
import psutil
from dataclasses import dataclass


@dataclass
class Configuracion:
    # CPU threading
    MAXIMO_HILOS: int = max(1, min(os.cpu_count() - 1, 8))

    # Procesamiento
    TIEMPO_ESPERA: float = 0.05
    TAMANIO_LOTE: int = min(8, MAXIMO_HILOS)

    # Calidad de salida
    CALIDAD_JPEG: int = 95
    COMPRESION_PNG: int = 1

    # Monitoreo de memoria
    UMBRAL_MEMORIA: int = 90
    INTERVALO_MONITOREO: float = 0.5

    # Timeouts
    TIEMPO_MAXIMO_PROCESAMIENTO: int = 600  # segundos

    @staticmethod
    def obtener_configuracion_optima():
        """Ajusta la configuracion segun los recursos del sistema."""
        memoria_total = psutil.virtual_memory().total / (1024 ** 3)  # GB
        nucleos_cpu = os.cpu_count()
        return {
            'TAMANIO_LOTE': min(8, max(2, nucleos_cpu // 2)),
            'MAXIMO_HILOS': max(1, min(nucleos_cpu - 1, 8)),
        }
