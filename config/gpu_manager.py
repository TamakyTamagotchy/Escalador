import torch
import logging
import wmi
from typing import Optional, Dict
import psutil

logger = logging.getLogger(__name__)

class GestorGPU:
    _cache_gpu = None
    _cache_device = None

    @staticmethod
    def verificar_compatibilidad_gpu() -> Optional[str]:
        """
        Detecta y verifica la compatibilidad de GPU con caché.
        """
        if GestorGPU._cache_gpu is not None:
            return GestorGPU._cache_gpu

        try:
            logger.info("Iniciando detección de GPU...")
            
            # Verificación PyTorch CUDA
            if torch.cuda.is_available():
                logger.info("PyTorch CUDA disponible")
                device_count = torch.cuda.device_count()
                
                if device_count > 0:
                    # Verificar capacidades de todas las GPUs
                    for i in range(device_count):
                        gpu_name = torch.cuda.get_device_name(i)
                        compute_capability = torch.cuda.get_device_capability(i)
                        
                        logger.info(f"GPU {i}: {gpu_name} (CC: {compute_capability})")
                        
                        # Verificar capacidad mínima
                        if compute_capability[0] >= 3:
                            GestorGPU._cache_gpu = 'NVIDIA'
                            return 'NVIDIA'
            
            # Detección WMI como respaldo
            try:
                c = wmi.WMI()
                gpu_info = c.Win32_VideoController()
                
                for gpu in gpu_info:
                    if 'NVIDIA' in gpu.AdapterCompatibility:
                        GestorGPU._cache_gpu = 'NVIDIA'
                        return 'NVIDIA'
                    elif 'Advanced Micro Devices' in gpu.AdapterCompatibility:
                        GestorGPU._cache_gpu = 'AMD'
                        return 'AMD'
                    elif 'Intel' in gpu.AdapterCompatibility:
                        GestorGPU._cache_gpu = 'Intel'
                        return 'Intel'
                        
            except Exception as e:
                logger.warning(f"Error en detección WMI: {e}")
        
        except Exception as e:
            logger.error(f"Error en detección de GPU: {e}")
        
        GestorGPU._cache_gpu = None
        return None

    @staticmethod
    def obtener_info_gpu() -> Optional[Dict]:
        """
        Obtiene información detallada de la GPU con monitoreo de recursos.
        """
        try:
            if not torch.cuda.is_available():
                return None

            info = {}
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info[f"GPU_{i}"] = {
                    "nombre": props.name,
                    "capacidad": torch.cuda.get_device_capability(i),
                    "memoria_total": props.total_memory,
                    "memoria_reservada": torch.cuda.memory_reserved(i),
                    "memoria_asignada": torch.cuda.memory_allocated(i),
                    "temperatura": None,  # Se podría implementar con nvidia-smi
                    "uso_gpu": None,
                    "memoria_sistema": psutil.virtual_memory().percent
                }
            return info
        except Exception as e:
            logger.error(f"Error obteniendo info GPU: {e}")
            return None

    @staticmethod
    def seleccionar_dispositivo_gpu() -> str:
        """
        Selecciona el dispositivo GPU más adecuado basado en recursos disponibles.
        """
        if GestorGPU._cache_device is not None:
            return GestorGPU._cache_device

        try:
            if torch.cuda.is_available():
                # Verificar cada GPU disponible
                for i in range(torch.cuda.device_count()):
                    # Verificar capacidad de cómputo
                    if torch.cuda.get_device_capability(i)[0] >= 3:
                        # Verificar memoria disponible
                        memoria_total = torch.cuda.get_device_properties(i).total_memory
                        memoria_usada = torch.cuda.memory_allocated(i)
                        if memoria_usada / memoria_total < 0.9:  # 90% umbral
                            GestorGPU._cache_device = 'cuda'
                            return 'cuda'

            GestorGPU._cache_device = 'cpu'
            return 'cpu'
        except Exception as e:
            logger.error(f"Error seleccionando dispositivo: {e}")
            return 'cpu'