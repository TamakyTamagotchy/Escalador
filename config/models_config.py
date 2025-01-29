#models_config
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from asyncio.log import logger
from config.config import Configuracion
import os

@dataclass
class ModeloConfig:
    nombre: str
    descripcion: str
    escala_predeterminada: int
    permite_escala_personalizada: bool
    parametros_gpu: Dict = field(default_factory=dict)
    optimizaciones: Dict = field(default_factory=dict)
    tipos_contenido: List[str] = field(default_factory=list)

class ConfiguracionModelos:
    CONFIGURACION_DEFAULT = {
        'parametros_gpu': {
            'tile_size': 0,  # Modo automático
            'gpu_threads': Configuracion.MAXIMO_HILOS,
            'gpu_block': '128,1,1'
        },
        'optimizaciones': {
            'denoise_level': 0,
            'sharpening': 1.0,
            'color_correction': False,
            'fp32': True
        }
    }
    
    MODELOS = {
        'realesrgan-x4plus': ModeloConfig(
            nombre='realesrgan-x4plus',
            descripcion='Mejora de imágenes realistas',
            escala_predeterminada=4,
            permite_escala_personalizada=True,
            parametros_gpu={'tile_size': 0},
            tipos_contenido=['fotografías', 'paisajes']
        ),
        'realesrgan-x4plus-anime': ModeloConfig(
            nombre='realesrgan-x4plus-anime', 
            descripcion='Mejora de imágenes de anime',
            escala_predeterminada=4,
            permite_escala_personalizada=True,
            parametros_gpu={'tile_size': 0},
            tipos_contenido=['anime', 'dibujos animados']
        ),
        'realesr-animevideov3': ModeloConfig(
            nombre='realesr-animevideov3',
            descripcion='Mejora de video de anime',
            escala_predeterminada=4,
            permite_escala_personalizada=True,
            parametros_gpu={'tile_size': 0},
            tipos_contenido=['anime', 'video']
        )
    }

    @classmethod
    def obtener_configuracion_modelo(cls, nombre_modelo: str) -> Optional[ModeloConfig]:
        config = cls.MODELOS.get(nombre_modelo)
        if not config:
            # Registro de modelo no encontrado
            logger.warning(f"Modelo no encontrado: {nombre_modelo}")
            return None

        # Usar método más seguro de fusión de diccionarios
        config.parametros_gpu = {**cls.CONFIGURACION_DEFAULT['parametros_gpu'], **config.parametros_gpu}
        config.optimizaciones = {**cls.CONFIGURACION_DEFAULT['optimizaciones'], **config.optimizaciones}
        return config

    @classmethod
    def aplicar_optimizaciones_modelo(cls, comando: List[str], modelo: str) -> List[str]:
        """Aplica optimizaciones al comando de procesamiento"""
        config = cls.obtener_configuracion_modelo(modelo)
        if not config:
            return comando

        try:
            # Limpiar comandos duplicados
            comando = [c for c in comando if c not in ['-s', '--sharpening', '-d', '-x']]

            # Parámetros de tile y threads
            num_hilos = min(Configuracion.MAXIMO_HILOS, os.cpu_count())
            comando.extend([
                "-t", str(config.parametros_gpu.get('tile_size', 0)),
                "-j", f"1:{num_hilos}:{num_hilos}"
            ])

            # Optimizaciones generales
            comando.extend([
                "--fp32",  # Precisión de punto flotante
            ])

            # Denoise y nitidez
            if config.optimizaciones.get('denoise_level', 0) > 0:
                comando.extend(["-d", str(config.optimizaciones['denoise_level'])])

        except Exception as e:
            logger.error(f"Error al aplicar optimizaciones: {str(e)}")
        
        return comando

    @classmethod
    def listar_modelos_disponibles(cls) -> List[str]:
        """Devuelve lista de modelos disponibles"""
        return list(cls.MODELOS.keys())