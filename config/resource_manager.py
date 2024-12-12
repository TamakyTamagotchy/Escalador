#resource_manager
import psutil
import gc
import time
import torch
from .config import Configuracion

class GestorRecursos:
    @staticmethod
    def monitorear_memoria():
        if psutil.virtual_memory().percent > 85:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(1)
    @staticmethod
    def verificar_uso_recursos():
        while psutil.cpu_percent(interval=0.1) > Configuracion.MAXIMO_USO_CPU or \
            psutil.virtual_memory().percent > 90:
            time.sleep(0.2)
            gc.collect()
