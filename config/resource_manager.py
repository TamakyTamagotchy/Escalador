# resource_manager.py
import gc, psutil, logging

logger = logging.getLogger(__name__)


class GestorRecursos:
    @staticmethod
    def obtener_uso_memoria():
        """Retorna el porcentaje de uso de memoria del sistema."""
        return psutil.virtual_memory().percent

    @staticmethod
    def obtener_uso_cpu():
        """Retorna el porcentaje de uso de CPU."""
        return psutil.cpu_percent(interval=0.1)

    @staticmethod
    def liberar_memoria():
        """Fuerza garbage collection para liberar memoria no usada."""
        gc.collect()
        logger.debug("Memoria liberada via gc.collect()")

    @staticmethod
    def verificar_recursos_disponibles(umbral_memoria=90):
        """Verifica si el sistema tiene suficientes recursos para continuar."""
        mem = psutil.virtual_memory().percent
        if mem > umbral_memoria:
            gc.collect()
            logger.warning(f"Uso de memoria alto: {mem}%")
            return False
        return True
