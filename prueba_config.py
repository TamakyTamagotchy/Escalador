import os
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ModelConfig:
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
    # Rutas absolutas basadas en la ubicación del script
    BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "realesrgan-ncnn-vulkan"))
    EXECUTABLE_PATH = os.path.join(BASE_PATH, "realesrgan-ncnn-vulkan.exe")
    MODELS_PATH = os.path.join(BASE_PATH, "models")
    MODELS: Dict[str, ModelConfig] = {}

    @classmethod
    def discover_models(cls) -> Dict[str, ModelConfig]:
        discovered = {}
        if not os.path.isdir(cls.MODELS_PATH):
            return discovered
        param_files = [f for f in os.listdir(cls.MODELS_PATH) if f.endswith('.param')]
        for param_file in param_files:
            model_name = param_file[:-6]
            bin_file = f"{model_name}.bin"
            if not os.path.exists(os.path.join(cls.MODELS_PATH, bin_file)):
                continue
            config = cls._get_model_config(model_name, param_file, bin_file)
            discovered[model_name] = config
        return discovered

    @classmethod
    def _get_model_config(cls, model_name: str, param_file: str, bin_file: str) -> ModelConfig:
        default_config = {
            'name': model_name,
            'param_file': param_file,
            'bin_file': bin_file,
            'default_scale': 4,
            'supports_custom_scale': False,
            'best_for': ['general'],
        }
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
    def get_realesrgan_path(cls) -> Optional[str]:
        return cls.EXECUTABLE_PATH if os.path.isfile(cls.EXECUTABLE_PATH) else None

    @classmethod
    def get_models_path(cls) -> Optional[str]:
        if not os.path.isdir(cls.MODELS_PATH):
            return None
        return cls.MODELS_PATH

    @classmethod
    def get_recommended_model(cls, content_type: str) -> str:
        content_type = content_type.lower()
        if 'video' in content_type and ('anime' in content_type or 'animación' in content_type):
            return 'realesr-animevideov3-x2'
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
        cls.MODELS = cls.discover_models()

# Funciones de ayuda para acceso externo
def get_realesrgan_path() -> Optional[str]:
    return ConfigHandler.get_realesrgan_path()

def get_models_path() -> Optional[str]:
    return ConfigHandler.get_models_path()

def get_recommended_model(content_type: str) -> str:
    return ConfigHandler.get_recommended_model(content_type)

def validate_scale(model_name: str, scale: int) -> int:
    return ConfigHandler.validate_scale(model_name, scale)