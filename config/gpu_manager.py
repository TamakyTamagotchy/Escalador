from typing import Optional
import torch
import subprocess
import logging
import shutil
from functools import lru_cache

class GestorGPU:
    @staticmethod
    @lru_cache(maxsize=1)
    def verificar_compatibilidad_gpu() -> Optional[str]:
        """
        Detecta y verifica la compatibilidad de GPU con múltiples métodos.
       
        Prioriza la detección en el siguiente orden:
        1. Torch CUDA
        2. Comandos de sistema para NVIDIA/AMD
       
        Returns:
            Optional[str]: Tipo de GPU detectada o None si no se encuentra
        """
        try:
            # Verificación primaria usando PyTorch
            if torch.cuda.is_available():
                # Verificar si hay GPUs CUDA disponibles
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    # Obtener información de la GPU
                    for i in range(device_count):
                        gpu_name = torch.cuda.get_device_name(i)
                        compute_capability = torch.cuda.get_device_capability(i)
                        
                        # Log de información de la GPU
                        logging.info(f"GPU {i}: {gpu_name}")
                        logging.info(f"Compute Capability: {compute_capability}")
                        
                        # Verificar si la GPU tiene capacidad de computación suficiente
                        # Las GPUs GeForce GT 710 suelen tener compute capability de 2.1 o menor
                        # Esto puede causar problemas con algunas bibliotecas modernas
                        if compute_capability[0] < 3:
                            logging.warning(f"GPU {i} puede tener limitaciones de rendimiento")
                
                # Configuraciones optimizadas de PyTorch para GPU NVIDIA
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                return 'NVIDIA'
           
            # Verificaciones alternativas usando comandos del sistema
            gpu_comandos = {
                'NVIDIA': ['nvidia-smi', 'nvidia-detect'],
                'AMD': ['rocm-smi', 'amdgpu-pro-info']
            }
           
            for tipo_gpu, comandos in gpu_comandos.items():
                for comando in comandos:
                    if shutil.which(comando):
                        try:
                            resultado = subprocess.run([comando],
                                                       stdout=subprocess.PIPE,
                                                       stderr=subprocess.PIPE,
                                                       timeout=2)
                            if resultado.returncode == 0:
                                return tipo_gpu
                        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                            continue
           
            # Verificación de sistema basada en lscpu para Intel
            try:
                cpu_info = subprocess.check_output(['lscpu']).decode().lower()
                if 'intel' in cpu_info and 'graphics' in cpu_info:
                    return 'Intel'
            except Exception:
                pass
           
        except Exception as e:
            logging.warning(f"Error al detectar GPU: {e}")
       
        return None

    @staticmethod
    def seleccionar_dispositivo_gpu() -> str:
        """
        Selecciona un dispositivo GPU adecuado para procesamiento.
        
        Returns:
            str: Dispositivo seleccionado ('cuda', 'cpu')
        """
        try:
            if torch.cuda.is_available():
                # Verificar si hay GPUs con suficiente capacidad de computación
                for i in range(torch.cuda.device_count()):
                    compute_capability = torch.cuda.get_device_capability(i)
                    if compute_capability[0] >= 3:  # Requiere compute capability 3.0+
                        return 'cuda'
            
            # Fallback a CPU si no hay GPU adecuada
            return 'cpu'
        
        except Exception as e:
            logging.warning(f"Error al seleccionar dispositivo GPU: {e}")
            return 'cpu'