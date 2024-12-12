#gpu_manager
import torch
import logging
import wmi
from typing import Optional

class GestorGPU:
    @staticmethod
    def verificar_compatibilidad_gpu() -> Optional[str]:
        """
        Detecta y verifica la compatibilidad de GPU en Windows.
    
        Prioriza la detección en el siguiente orden:
        1. Torch CUDA
        2. WMI para detección de GPUs de NVIDIA
    
        Returns:
            Optional[str]: Tipo de GPU detectada o None si no se encuentra
        """
        try:
            logging.info("Iniciando detección de GPU...")
            # Verificación primaria usando PyTorch
            if torch.cuda.is_available():
                logging.info("PyTorch CUDA disponible")
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    for i in range(device_count):
                        gpu_name = torch.cuda.get_device_name(i)
                        compute_capability = torch.cuda.get_device_capability(i)
                        
                        # Log de información de la GPU
                        logging.info(f"GPU {i}: {gpu_name}")
                        logging.info(f"Compute Capability: {compute_capability}")
                        
                        # Verificar si la GPU tiene capacidad de computación suficiente
                        if compute_capability[0] < 3:
                            logging.warning(f"GPU {i} puede tener limitaciones de rendimiento")
                
                # Configuraciones optimizadas de PyTorch para GPU NVIDIA
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                logging.info("Configuraciones de PyTorch optimizadas")
                return 'NVIDIA'
            
            # Detección adicional usando WMI para Windows
            try:
                logging.info("Iniciando detección de GPU con WMI...")
                c = wmi.WMI()
                gpu_info = c.Win32_VideoController(AdapterCompatibility='NVIDIA')
                
                if gpu_info:
                    for gpu in gpu_info:
                        logging.info(f"GPU Detectada: {gpu.Name}")
                        return 'NVIDIA'
                
                # Verificación para GPUs AMD o Intel
                gpu_info_amd = c.Win32_VideoController(AdapterCompatibility='Advanced Micro Devices')
                if gpu_info_amd:
                    for gpu in gpu_info_amd:
                        logging.info(f"GPU AMD Detectada: {gpu.Name}")
                        return 'AMD'
                
                gpu_info_intel = c.Win32_VideoController(AdapterCompatibility='Intel')
                if gpu_info_intel:
                    for gpu in gpu_info_intel:
                        logging.info(f"GPU Intel Detectada: {gpu.Name}")
                        return 'Intel'
            
            except ImportError:
                logging.warning("WMI no disponible. Asegúrese de instalar 'wmi' con pip.")
            except Exception as e:
                logging.warning(f"Error al detectar GPU con WMI: {e}")
        
        except Exception as e:
            logging.warning(f"Error al detectar GPU: {e}")
    
        logging.info("No se encontró GPU compatible")
        return None

    @staticmethod
    def seleccionar_dispositivo_gpu() -> str:
        """
        Selecciona un dispositivo GPU adecuado para procesamiento en Windows.
        
        Returns:
            str: Dispositivo seleccionado ('cuda', 'cpu')
        """
        try:
            logging.info("Iniciando selección de dispositivo GPU...")
            if torch.cuda.is_available():
                logging.info("PyTorch CUDA disponible")
                # Verificar si hay GPUs con suficiente capacidad de computación
                for i in range(torch.cuda.device_count()):
                    compute_capability = torch.cuda.get_device_capability(i)
                    if compute_capability[0] >= 3:  # Requiere compute capability 3.0+
                        logging.info("Dispositivo GPU seleccionado: cuda")
                        return 'cuda'
            
            # Fallback a CPU si no hay GPU adecuada
            logging.info("Dispositivo GPU seleccionado: cpu")
            return 'cpu'
        
        except Exception as e:
            logging.warning(f"Error al seleccionar dispositivo GPU: {e}")
            logging.info("Dispositivo GPU seleccionado: cpu")
            return 'cpu'