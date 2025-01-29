# gpu_manager
import torch
import logging
import wmi
from typing import Optional

logger = logging.getLogger(__name__)

class GestorGPU:
    @staticmethod
    def verificar_compatibilidad_gpu() -> Optional[str]:
        """
        Detecta y verifica la compatibilidad de GPU en Windows.
    
        Prioriza la detección en el siguiente orden:
        1. Torch CUDA
        2. WMI para detección de GPUs de NVIDIA, AMD e Intel
    
        Returns:
            Optional[str]: Tipo de GPU detectada o None si no se encuentra
        """
        try:
            logger.info("Iniciando detección de GPU...")
            # Verificación primaria usando PyTorch
            if torch.cuda.is_available():
                logger.info("PyTorch CUDA disponible")
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    for i in range(device_count):
                        gpu_name = torch.cuda.get_device_name(i)
                        compute_capability = torch.cuda.get_device_capability(i)
                        
                        # Log de información de la GPU
                        logger.info(f"GPU {i}: {gpu_name}")
                        logger.info(f"Compute Capability: {compute_capability}")
                        
                        # Verificar si la GPU tiene capacidad de computación suficiente
                        if compute_capability[0] < 3:
                            logger.warning(f"GPU {i} puede tener limitaciones de rendimiento")
                
                # Configuraciones optimizadas de PyTorch para GPU NVIDIA
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                logger.info("Configuraciones de PyTorch optimizadas")
                return 'NVIDIA'
            
            # Detección adicional usando WMI para Windows
            try:
                logger.info("Iniciando detección de GPU con WMI...")
                c = wmi.WMI()
                gpu_info = c.Win32_VideoController()
                
                for gpu in gpu_info:
                    gpu_name = gpu.Name
                    if 'NVIDIA' in gpu.AdapterCompatibility:
                        logger.info(f"GPU NVIDIA Detectada: {gpu_name}")
                        return 'NVIDIA'
                    elif 'Advanced Micro Devices' in gpu.AdapterCompatibility:
                        logger.info(f"GPU AMD Detectada: {gpu_name}")
                        return 'AMD'
                    elif 'Intel' in gpu.AdapterCompatibility:
                        logger.info(f"GPU Intel Detectada: {gpu_name}")
                        return 'Intel'
            
            except ImportError:
                logger.warning("WMI no disponible. Asegúrese de instalar 'wmi' con pip.")
            except Exception as e:
                logger.warning(f"Error al detectar GPU con WMI: {e}")
        
        except Exception as e:
            logger.warning(f"Error al detectar GPU: {e}")
    
        logger.info("No se encontró GPU compatible")
        return None

    @staticmethod
    def seleccionar_dispositivo_gpu() -> str:
        """
        Selecciona un dispositivo GPU adecuado para procesamiento en Windows.
        
        Returns:
            str: Dispositivo seleccionado ('cuda', 'cpu')
        """
        try:
            logger.info("Iniciando selección de dispositivo GPU...")
            if torch.cuda.is_available():
                logger.info("PyTorch CUDA disponible")
                # Verificar si hay GPUs con suficiente capacidad de computación
                for i in range(torch.cuda.device_count()):
                    compute_capability = torch.cuda.get_device_capability(i)
                    if compute_capability[0] >= 3:  # Requiere compute capability 3.0+
                        logger.info("Dispositivo GPU seleccionado: cuda")
                        return 'cuda'
            
            # Fallback a CPU si no hay GPU adecuada
            logger.info("Dispositivo GPU seleccionado: cpu")
            return 'cpu'
        
        except Exception as e:
            logger.warning(f"Error al seleccionar dispositivo GPU: {e}")
            logger.info("Dispositivo GPU seleccionado: cpu")
            return 'cpu'

    @staticmethod
    def obtener_info_gpu() -> Optional[dict]:
        """
        Obtiene información detallada de la GPU si está disponible.
        
        Returns:
            Optional[dict]: Información de la GPU o None si no se encuentra
        """
        if torch.cuda.is_available():
            info = {}
            for i in range(torch.cuda.device_count()):
                info[f"GPU_{i}"] = {
                    "name": torch.cuda.get_device_name(i),
                    "capability": torch.cuda.get_device_capability(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory,
                    "memory_reserved": torch.cuda.memory_reserved(i),
                    "memory_allocated": torch.cuda.memory_allocated(i)
                }
            return info
        return None