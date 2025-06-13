#resource_manager
import psutil
import gc
import time
import torch
from .config import Configuracion
from main_pruebas import logging as logger

class GestorRecursos:
    @staticmethod
    def monitorear_memoria():
        """
        Monitorea el uso de memoria y libera recursos si se supera el umbral configurado.
        """
        while True:
            memoria_virtual = psutil.virtual_memory()
            if memoria_virtual.percent > Configuracion.UMBRAL_MEMORIA:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"Memoria liberada. Uso actual: {memoria_virtual.percent}%")
            time.sleep(Configuracion.INTERVALO_MONITOREO)

    @staticmethod
    def verificar_uso_recursos():
        """
        Verifica el uso de CPU y memoria, y espera si se supera el umbral configurado.
        """
        while True:
            uso_cpu = psutil.cpu_percent(interval=0.1)
            uso_memoria = psutil.virtual_memory().percent
            if uso_cpu > Configuracion.MAXIMO_USO_CPU or uso_memoria > Configuracion.UMBRAL_MEMORIA:
                logger.warning(f"Uso alto de recursos detectado. CPU: {uso_cpu}%, Memoria: {uso_memoria}%")
                time.sleep(Configuracion.INTERVALO_VERIFICACION)
                gc.collect()
            time.sleep(Configuracion.INTERVALO_VERIFICACION)
    
    @staticmethod
    def monitorear_gpu():
        """
        Monitorea el uso de GPU y libera recursos si se supera el umbral configurado.
        """
        if torch.cuda.is_available():
            while True:
                for i in range(torch.cuda.device_count()):
                    gpu_mem = torch.cuda.memory_reserved(i)
                    gpu_total = torch.cuda.get_device_properties(i).total_memory
                    if gpu_mem / gpu_total > Configuracion.UMBRAL_MEMORIA_GPU / 100:
                        torch.cuda.empty_cache()
                        logger.info(f"Memoria GPU liberada en dispositivo {i}. Uso actual: {gpu_mem / gpu_total * 100}%")
                        time.sleep(Configuracion.INTERVALO_MONITOREO)
                time.sleep(Configuracion.INTERVALO_MONITOREO)
    
    @staticmethod
    def monitorear_recursos_dinamico():
        """
        Monitorea dinámicamente el uso de recursos y ajusta configuraciones según el estado del sistema.
        """
        while True:
            uso_cpu = psutil.cpu_percent(interval=0.1)
            uso_memoria = psutil.virtual_memory().percent
            if uso_cpu > Configuracion.MAXIMO_USO_CPU:
                Configuracion.MAXIMO_HILOS = max(1, Configuracion.MAXIMO_HILOS - 1)
                logger.warning(f"Reduciendo hilos debido a alto uso de CPU: {uso_cpu}%")
            if uso_memoria > Configuracion.UMBRAL_MEMORIA:
                gc.collect()
                logger.warning(f"Liberando memoria debido a alto uso: {uso_memoria}%")
            time.sleep(Configuracion.INTERVALO_VERIFICACION)
