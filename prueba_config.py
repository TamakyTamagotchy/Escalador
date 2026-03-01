import os, logging
from dataclasses import dataclass
from typing import Dict, List, Optional

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',

)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Clase para almacenar la configuración de cada modelo"""
    name: str
    description: str
    param_file: str
    bin_file: str
    default_scale: int
    supports_custom_scale: bool
    best_for: List[str]
    available_scales: Optional[List[int]] = None
    min_scale: Optional[int] = None
    max_scale: Optional[int] = None

class ConfigHandler:
    """Clase para manejar la configuración y verificación de modelos"""
    
    # Rutas base
    BASE_PATH = r"D:\Escritorio\Carpetas con cosas\Programas\Python\Prueba2_test\realesrgan-ncnn-vulkann"
    EXECUTABLE_PATH = os.path.join(BASE_PATH, "realesrgan-ncnn-vulkan.exe")
    MODELS_PATH = os.path.join(BASE_PATH, "models")

    # Diccionario de modelos que se llenará automáticamente
    MODELS: Dict[str, ModelConfig] = {}

    @classmethod
    def discover_models(cls) -> Dict[str, ModelConfig]:
        """Descubre automáticamente los modelos disponibles en el directorio de modelos"""
        discovered_models = {}
        
        # Verifica que el directorio de modelos existe
        if not os.path.isdir(cls.MODELS_PATH):
            logger.error(f"Directorio de modelos no encontrado: {cls.MODELS_PATH}")
            return discovered_models

        # Obtiene todos los archivos .param en el directorio
        param_files = [f for f in os.listdir(cls.MODELS_PATH) if f.endswith('.param')]
        
        for param_file in param_files:
            model_name = param_file[:-6]  # Elimina la extensión '.param'
            bin_file = f"{model_name}.bin"
            
            # Verifica que existe el archivo .bin correspondiente
            if not os.path.exists(os.path.join(cls.MODELS_PATH, bin_file)):
                logger.warning(f"Archivo bin faltante para {model_name}")
                continue
                
            # Determina la configuración del modelo basada en su nombre
            config = cls._get_model_config(model_name, param_file, bin_file)
            discovered_models[model_name] = config
            
        return discovered_models

    @classmethod
    def _get_model_config(cls, model_name: str, param_file: str, bin_file: str) -> ModelConfig:
        """Determina la configuración apropiada para un modelo basado en su nombre"""
        
        # Configuraciones predeterminadas
        default_config = {
            'name': model_name,
            'param_file': param_file,
            'bin_file': bin_file,
            'default_scale': 4,
            'supports_custom_scale': False,
            'best_for': ['general'],
        }
        
        # Configuraciones específicas basadas en el nombre del modelo
        if 'anime' in model_name.lower():
            default_config.update({
                'description': 'Optimizado para contenido anime/manga',
                'best_for': ['anime', 'manga', 'dibujos animados']
            })
        elif 'remacri' in model_name.lower():
            default_config.update({
                'description': 'Optimizado para remasterización de imágenes',
                'best_for': ['remasterización', 'fotografía']
            })
        elif 'ultramix_balanced' in model_name.lower():
            default_config.update({
                'description': 'Optimizado para una mezcla equilibrada de mejoras',
                'best_for': ['general', 'fotografía']
            })
        elif 'ultrasharp' in model_name.lower():
            default_config.update({
                'description': 'Optimizado para mejorar la nitidez de las imágenes',
                'best_for': ['nitidez', 'detalles']
            })

        return ModelConfig(**default_config)

    @classmethod
    def verify_file_exists(cls, filepath: str) -> bool:
        """Verifica si un archivo existe y registra el resultado"""
        exists = os.path.isfile(filepath)
        if not exists:
            logger.warning(f"Archivo no encontrado: {filepath}")
        return exists

    @classmethod
    def get_realesrgan_path(cls) -> Optional[str]:
        """Obtiene y verifica la ruta del ejecutable"""
        if cls.verify_file_exists(cls.EXECUTABLE_PATH):
            return cls.EXECUTABLE_PATH
        return None

    @classmethod
    def get_models_path(cls) -> Optional[str]:
        """Verifica el directorio de modelos y sus archivos"""
        if not os.path.isdir(cls.MODELS_PATH):
            logger.error(f"Directorio de modelos no encontrado: {cls.MODELS_PATH}")
            return None

        all_files_exist = True
        for model_config in cls.MODELS.values():
            param_path = os.path.join(cls.MODELS_PATH, model_config.param_file)
            bin_path = os.path.join(cls.MODELS_PATH, model_config.bin_file)
            
            if not (cls.verify_file_exists(param_path) and cls.verify_file_exists(bin_path)):
                logger.error(f"Archivos faltantes para el modelo {model_config.name}")
                all_files_exist = False

        return cls.MODELS_PATH if all_files_exist else None

    @classmethod
    def verify_model(cls, model_name: str) -> bool:
        """Verifica la disponibilidad de un modelo específico"""
        if model_name not in cls.MODELS:
            logger.error(f"Modelo no reconocido: {model_name}")
            return False

        model_config = cls.MODELS[model_name]
        param_path = os.path.join(cls.MODELS_PATH, model_config.param_file)
        bin_path = os.path.join(cls.MODELS_PATH, model_config.bin_file)

        if not (cls.verify_file_exists(param_path) and cls.verify_file_exists(bin_path)):
            return False

        logger.info(f"Modelo {model_name} verificado y disponible")
        return True

    @classmethod
    def get_recommended_model(cls, content_type: str) -> str:
        """Recomienda el modelo más apropiado según el tipo de contenido"""
        content_type = content_type.lower()
        
        if 'video' in content_type and ('anime' in content_type or 'animación' in content_type):
            return 'realesr-animevideov3-x2'  # Escala x2 por defecto para video
        elif 'anime' in content_type or 'manga' in content_type:
            return 'realesrgan-x4plus-anime'
        elif 'remasterización' in content_type:
            return 'remacri'
        elif 'mezcla equilibrada' in content_type:
            return 'ultramix_balanced'
        elif 'nitidez' in content_type:
            return 'ultrasharp'
        else:
            return 'realesrgan-x4plus'

    @classmethod
    def validate_scale(cls, model_name: str, scale: int) -> int:
        """Valida que la escala sea apropiada para el modelo seleccionado"""
        if model_name not in cls.MODELS:
            raise ValueError(f"Modelo no reconocido: {model_name}")
            
        model_config = cls.MODELS[model_name]
        
        if not model_config.supports_custom_scale:
            return model_config.default_scale
            
        if model_config.available_scales:
            if scale not in model_config.available_scales:
                raise ValueError(
                    f"Escala {scale} no válida para {model_name}. "
                    f"Escalas disponibles: {model_config.available_scales}"
                )
        else:
            if not (model_config.min_scale <= scale <= model_config.max_scale):
                raise ValueError(
                    f"Escala {scale} fuera de rango para {model_name}. "
                    f"Rango válido: {model_config.min_scale}-{model_config.max_scale}"
                )
                
        return scale

    @classmethod
    def initialize(cls):
        """Inicializa la configuración y descubre los modelos disponibles"""
        cls.MODELS = cls.discover_models()
        logger.info(f"Modelos descubiertos: {list(cls.MODELS.keys())}")

# Funciones de ayuda para acceso externo
def get_realesrgan_path() -> Optional[str]:
    """Obtiene la ruta del ejecutable realesrgan"""
    return ConfigHandler.get_realesrgan_path()

def get_models_path() -> Optional[str]:
    """Obtiene la ruta del directorio de modelos"""
    return ConfigHandler.get_models_path()

def get_recommended_model(content_type: str) -> str:
    """Obtiene el modelo recomendado según el tipo de contenido"""
    return ConfigHandler.get_recommended_model(content_type)

def validate_scale(model_name: str, scale: int) -> int:
    """Valida la escala para un modelo específico"""
    return ConfigHandler.validate_scale(model_name, scale)

def verify_model(model_name: str) -> bool:
    """Verifica la disponibilidad de un modelo específico"""
    return ConfigHandler.verify_model(model_name)