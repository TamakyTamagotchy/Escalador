# gpu_manager.py
import subprocess, logging

logger = logging.getLogger(__name__)


class GestorGPU:
    _cache_gpu = None

    @staticmethod
    def verificar_compatibilidad_gpu():
        """
        Detecta el fabricante de la GPU usando WMIC (built-in en Windows).
        Retorna 'NVIDIA', 'AMD', 'Intel', o None.
        """
        if GestorGPU._cache_gpu is not None:
            return GestorGPU._cache_gpu

        try:
            result = subprocess.run(
                ['wmic', 'path', 'win32_VideoController', 'get', 'Name'],
                capture_output=True, text=True, timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            if result.returncode == 0:
                output = result.stdout.upper()
                if 'NVIDIA' in output or 'GEFORCE' in output or 'RTX' in output or 'GTX' in output:
                    GestorGPU._cache_gpu = 'NVIDIA'
                elif 'AMD' in output or 'RADEON' in output:
                    GestorGPU._cache_gpu = 'AMD'
                elif 'INTEL' in output and ('ARC' in output or 'UHD' in output or 'IRIS' in output):
                    GestorGPU._cache_gpu = 'Intel'

        except Exception as e:
            logger.warning(f"Error en deteccion de GPU via WMIC: {e}")

        if GestorGPU._cache_gpu:
            logger.info(f"GPU detectada: {GestorGPU._cache_gpu}")
        else:
            logger.warning("No se detecto GPU compatible; se usara codificacion por software")

        return GestorGPU._cache_gpu

    @staticmethod
    def reset_cache():
        """Resetea la cache de deteccion de GPU."""
        GestorGPU._cache_gpu = None
