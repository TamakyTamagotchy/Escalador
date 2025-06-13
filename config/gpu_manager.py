# gpu_manager
import torch
import logging
import wmi
from typing import Optional, Dict, List
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

@dataclass
class GPUInfo:
    """Clase para almacenar información detallada de una GPU"""
    name: str
    vendor: str
    vram_mb: int
    device_id: str
    bus_id: str
    is_integrated: bool

class GestorGPU:
    _instance = None
    _gpus_cache: List[GPUInfo] = None
    _selected_gpu: Optional[GPUInfo] = None

    # Base de datos de GPUs conocidas y su VRAM (en MB)
    NVIDIA_GPUS = {
        # Serie RTX 40
        'RTX 4090': 24576,  # 24GB
        'RTX 4080': 16384,  # 16GB
        'RTX 4070 Ti': 12288,  # 12GB
        'RTX 4070': 12288,  # 12GB
        'RTX 4060 Ti': 8192,  # 8GB
        'RTX 4060': 8192,  # 8GB
        # Serie RTX 30
        'RTX 3090 Ti': 24576,  # 24GB
        'RTX 3090': 24576,  # 24GB
        'RTX 3080 Ti': 12288,  # 12GB
        'RTX 3080': 10240,  # 10GB
        'RTX 3070 Ti': 8192,  # 8GB
        'RTX 3070': 8192,  # 8GB
        'RTX 3060 Ti': 8192,  # 8GB
        'RTX 3060': 12288,  # 12GB
        'RTX 3050': 6144,   # 6GB
        # Serie RTX 20
        'RTX 2080 Ti': 11264,  # 11GB
        'RTX 2080 SUPER': 8192,  # 8GB
        'RTX 2080': 8192,  # 8GB
        'RTX 2070 SUPER': 8192,  # 8GB
        'RTX 2070': 8192,  # 8GB
        'RTX 2060 SUPER': 8192,  # 8GB
        'RTX 2060': 6144,  # 6GB
        # Serie GTX 16
        'GTX 1660 Ti': 6144,  # 6GB
        'GTX 1660 SUPER': 6144,  # 6GB
        'GTX 1660': 6144,  # 6GB
        'GTX 1650 SUPER': 4096,  # 4GB
        'GTX 1650': 4096,  # 4GB
    }

    AMD_GPUS = {
        # Serie RX 7000
        'RX 7900 XTX': 24576,  # 24GB
        'RX 7900 XT': 20480,  # 20GB
        'RX 7800 XT': 16384,  # 16GB
        'RX 7700 XT': 12288,  # 12GB
        'RX 7600': 8192,  # 8GB
        # Serie RX 6000
        'RX 6950 XT': 16384,  # 16GB
        'RX 6900 XT': 16384,  # 16GB
        'RX 6800 XT': 16384,  # 16GB
        'RX 6800': 16384,  # 16GB
        'RX 6750 XT': 12288,  # 12GB
        'RX 6700 XT': 12288,  # 12GB
        'RX 6650 XT': 8192,  # 8GB
        'RX 6600 XT': 8192,  # 8GB
        'RX 6600': 8192,  # 8GB
        'RX 6500 XT': 4096,  # 4GB
        # Serie RX 5000
        'RX 5700 XT': 8192,  # 8GB
        'RX 5700': 8192,  # 8GB
        'RX 5600 XT': 6144,  # 6GB
        'RX 5500 XT': 8192,  # 8GB
    }

    @staticmethod
    def _get_gpu_vram(name: str, vendor: str) -> int:
        """
        Obtiene la VRAM de una GPU basándose en su nombre y fabricante.
        """
        if vendor == 'NVIDIA':
            # Buscar coincidencias en la base de datos de NVIDIA
            for model, vram in GestorGPU.NVIDIA_GPUS.items():
                if model in name:
                    return vram
            # Si no se encuentra, buscar patrones comunes
            if 'RTX' in name or 'GTX' in name:
                # Intentar extraer la VRAM del nombre (ej: "8GB" en el nombre)
                vram_match = re.search(r'(\d+)\s*GB', name)
                if vram_match:
                    return int(vram_match.group(1)) * 1024
        elif vendor == 'AMD':
            # Buscar coincidencias en la base de datos de AMD
            for model, vram in GestorGPU.AMD_GPUS.items():
                if model in name:
                    return vram
            # Si no se encuentra, buscar patrones comunes
            if 'RX' in name:
                vram_match = re.search(r'(\d+)\s*GB', name)
                if vram_match:
                    return int(vram_match.group(1)) * 1024
        
        # Valores por defecto si no se puede determinar
        if vendor == 'NVIDIA':
            return 4096  # 4GB mínimo para GPUs NVIDIA modernas
        elif vendor == 'AMD':
            return 4096  # 4GB mínimo para GPUs AMD modernas
        elif vendor == 'Intel':
            return 2048  # 2GB para GPUs Intel integradas
        return 0

    @staticmethod
    def _get_instance():
        if GestorGPU._instance is None:
            GestorGPU._instance = GestorGPU()
        return GestorGPU._instance

    @staticmethod
    def _detect_gpus() -> List[GPUInfo]:
        """
        Detecta todas las GPUs disponibles en el sistema usando múltiples métodos.
        """
        gpus = []
        
        # 1. Detección mediante torch.cuda
        try:
            if torch.cuda.is_available():
                logger.info("PyTorch CUDA disponible")
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)
                    vram = props.total_memory // (1024*1024)  # Convertir a MB
                    gpus.append(GPUInfo(
                        name=gpu_name,
                        vendor='NVIDIA',
                        vram_mb=vram,
                        device_id=f'cuda:{i}',
                        bus_id=f'GPU{i}',
                        is_integrated=False
                    ))
                    logger.info(f"GPU CUDA detectada: {gpu_name} ({vram}MB VRAM)")
        except Exception as e:
            logger.warning(f"Error en detección CUDA: {e}")

        # 2. Detección mediante WMI
        try:
            c = wmi.WMI()
            wmi_gpus = c.Win32_VideoController()
            for gpu in wmi_gpus:
                try:
                    name = gpu.Name
                    pci_id = gpu.PNPDeviceID
                    vendor = ''
                    
                    # Determinar el fabricante
                    if 'NVIDIA' in gpu.AdapterCompatibility:
                        vendor = 'NVIDIA'
                    elif 'AMD' in gpu.AdapterCompatibility or 'ATI' in gpu.AdapterCompatibility:
                        vendor = 'AMD'
                    elif 'Intel' in gpu.AdapterCompatibility:
                        vendor = 'Intel'
                    
                    # Obtener VRAM
                    # Orden de prioridad mejorado
                    vram = 0
                    name = gpu.Name.strip()

                    # 1. Intentar obtener de WMI AdapterRAM si es válido
                    if hasattr(gpu, 'AdapterRAM') and gpu.AdapterRAM and gpu.AdapterRAM > 0:
                        try:
                            vram = int(gpu.AdapterRAM / (1024*1024))  # Convertir a MB
                            logger.debug(f"VRAM obtenida de WMI AdapterRAM: {vram}MB")
                        except Exception as e:
                            logger.debug(f"Error al obtener VRAM de WMI: {e}")
                    
                    # 2. Si no hay valor válido, buscar en la base de datos de GPUs
                    if vram <= 0:
                        if 'NVIDIA' in gpu.AdapterCompatibility:
                            vendor = 'NVIDIA'
                            for model, known_vram in GestorGPU.NVIDIA_GPUS.items():
                                if model in name:
                                    vram = known_vram
                                    logger.debug(f"VRAM obtenida de base de datos NVIDIA: {vram}MB")
                                    break
                        elif 'AMD' in gpu.AdapterCompatibility or 'ATI' in gpu.AdapterCompatibility:
                            vendor = 'AMD'
                            for model, known_vram in GestorGPU.AMD_GPUS.items():
                                if model in name:
                                    vram = known_vram
                                    logger.debug(f"VRAM obtenida de base de datos AMD: {vram}MB")
                                    break
                        elif 'Intel' in gpu.AdapterCompatibility:
                            vendor = 'Intel'
                    
                    # 3. Si aún no hay valor, intentar extraer del nombre
                    if vram <= 0:
                        vram_match = re.search(r'(\d+)\s*GB', name)
                        if vram_match:
                            vram = int(vram_match.group(1)) * 1024
                            logger.debug(f"VRAM extraída del nombre: {vram}MB")
                    
                    # 4. Si todo lo anterior falla, usar valores por defecto según el fabricante
                    if vram <= 0:
                        if 'NVIDIA' in gpu.AdapterCompatibility:
                            if 'RTX 3050' in name:
                                vram = 6 * 1024  # 6GB para RTX 3050
                            else:
                                vram = 4 * 1024  # 4GB mínimo para otras NVIDIA
                        elif 'AMD' in gpu.AdapterCompatibility:
                            vram = 4 * 1024  # 4GB mínimo para AMD
                        elif 'Intel' in gpu.AdapterCompatibility:
                            vram = 2 * 1024  # 2GB para Intel
                        logger.debug(f"VRAM establecida por valor por defecto: {vram}MB")

                    # Asegurar que el valor final sea válido
                    vram = max(1024, vram)  # Mínimo 1GB
                    
                    logger.info(f"Detección final de VRAM para {name}: {vram}MB")
                    
                    # Determinar si es integrada
                    is_integrated = ('Intel' in name or 
                                   'UHD' in name or 
                                   'Integrated' in name or
                                   'Graphics 730' in name)
                    
                    # Solo agregar si no es una GPU ya detectada
                    if not any(g.name == name for g in gpus):
                        gpu_info = GPUInfo(
                            name=name,
                            vendor=vendor,
                            vram_mb=max(0, vram),
                            device_id=pci_id,
                            bus_id=pci_id.split('\\')[-1] if '\\' in pci_id else '',
                            is_integrated=is_integrated
                        )
                        gpus.append(gpu_info)
                        logger.info(f"GPU WMI detectada: {name} ({vendor}, {vram}MB VRAM)")
                except Exception as e:
                    logger.warning(f"Error procesando GPU WMI: {e}")
        except Exception as e:
            logger.warning(f"Error en detección WMI: {e}")

        return gpus

    @staticmethod
    def _select_best_gpu(gpus: List[GPUInfo]) -> Optional[GPUInfo]:
        """
        Selecciona la mejor GPU disponible según varios criterios.
        """
        if not gpus:
            return None

        # Ordenar GPUs según prioridad
        def gpu_score(gpu: GPUInfo) -> float:
            score = 0.0
            # Prioridad por fabricante
            if gpu.vendor == 'NVIDIA':
                score += 1000
            elif gpu.vendor == 'AMD':
                score += 500
            elif gpu.vendor == 'Intel':
                score += 100
            
            # Penalizar GPUs integradas
            if gpu.is_integrated:
                score -= 300
            
            # Bonus por VRAM
            score += gpu.vram_mb / 1024  # Puntos adicionales por GB de VRAM
            
            return score

        # Ordenar GPUs por puntuación
        sorted_gpus = sorted(gpus, key=gpu_score, reverse=True)
        selected = sorted_gpus[0]
        
        logger.info(f"GPU seleccionada: {selected.name} ({selected.vendor})")
        return selected

    @staticmethod
    def verificar_compatibilidad_gpu() -> Optional[str]:
        """
        Detecta y verifica la compatibilidad de GPU.
        Returns:
            Optional[str]: Tipo de GPU detectada ('NVIDIA', 'AMD', 'Intel') o None
        """
        instance = GestorGPU._get_instance()
        if instance._gpus_cache is None:
            instance._gpus_cache = GestorGPU._detect_gpus()
        
        selected = GestorGPU._select_best_gpu(instance._gpus_cache)
        if selected:
            return selected.vendor
        return None

    @staticmethod
    def seleccionar_dispositivo_gpu() -> str:
        """
        Selecciona y retorna el tipo de GPU a utilizar.
        """
        instance = GestorGPU._get_instance()
        if instance._selected_gpu is None:
            if instance._gpus_cache is None:
                instance._gpus_cache = GestorGPU._detect_gpus()
            instance._selected_gpu = GestorGPU._select_best_gpu(instance._gpus_cache)

        if instance._selected_gpu:
            return instance._selected_gpu.vendor
        return "CPU"

    @staticmethod
    def get_gpu_info() -> Optional[Dict]:
        """
        Retorna información detallada de la GPU seleccionada.
        """
        instance = GestorGPU._get_instance()
        if instance._selected_gpu:
            return {
                'name': instance._selected_gpu.name,
                'vendor': instance._selected_gpu.vendor,
                'vram_mb': instance._selected_gpu.vram_mb,
                'is_integrated': instance._selected_gpu.is_integrated
            }
        return None