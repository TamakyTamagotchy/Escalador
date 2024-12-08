from typing import List, Dict, Optional
from .config import ModeloConfig

class ConfiguracionModelos:
    """Configuración específica para cada tipo de modelo"""
    # Configuración por defecto que se usará si algún modelo no especifica ciertos parámetros
    CONFIGURACION_DEFAULT = {
        'parametros_gpu': {
            'tile_size': 200,
            'gpu_threads': 1,
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
            descripcion='Optimizado para fotos reales y rostros humanos',
            escala_predeterminada=4,
            permite_escala_personalizada=True,
            parametros_gpu={
                'tile_size': 400,  # Reducido para mejor estabilidad
                'gpu_threads': 2,
                'gpu_block': '128,1,1'  # Reducido para evitar errores CUDA
            },
            optimizaciones={
                'denoise_level': 0,
                'sharpening': 1.0,
                'color_correction': True,
                'fp32': True  # Usar precisión completa
            },
            tipos_contenido=['fotografías', 'retratos', 'paisajes']
        ),
        
        'ultrasharp': ModeloConfig(
            nombre='ultrasharp',
            descripcion='Máximo detalle y nitidez',
            escala_predeterminada=2,  # Reducido para mejor estabilidad
            permite_escala_personalizada=True,
            parametros_gpu={
                'tile_size': 200,  # Reducido significativamente
                'gpu_threads': 2,
                'gpu_block': '128,1,1'
            },
            optimizaciones={
                'denoise_level': 0,
                'sharpening': 1.2,
                'color_correction': True,
                'fp32': True,
                'alpha_mode': 'no-alpha'  # Evitar problemas con canal alfa
            },
            tipos_contenido=['fotografías detalladas', 'texturas']
        ),
        
        'ultramix_balanced': ModeloConfig(
            nombre='ultramix_balanced',
            descripcion='Balance óptimo entre calidad y rendimiento',
            escala_predeterminada=3,
            permite_escala_personalizada=True,
            parametros_gpu={
                'tile_size': 300,
                'gpu_threads': 2,
                'gpu_block': '128,1,1'
            },
            optimizaciones={
                'denoise_level': 1,
                'sharpening': 1.1,
                'color_correction': True,
                'fp32': True,
                'pre_pad': 10  # Añadir padding para evitar artefactos
            },
            tipos_contenido=['uso general', 'fotografías', 'arte digital']
        ),
        
        'remacri': ModeloConfig(
            nombre='remacri',
            descripcion='Mejora de detalles y color',
            escala_predeterminada=4,
            permite_escala_personalizada=False,
            parametros_gpu={
                'tile_size': 250,  # Restablecer a 250
                'gpu_threads': 2,
                'gpu_block': '128,1,1'
            },
            optimizaciones={
                'denoise_level': 0,
                'sharpening': 4,  #1.15 defecto
                'color_correction': True,
                'fp32': True,
                'alpha_mode': 'no-alpha',
                'pre_pad': 8
            },
            tipos_contenido=['fotografías', 'arte digital', 'mejora de color']
        )
    }
    
    @classmethod
    def obtener_configuracion_modelo(cls, nombre_modelo: str) -> Optional[ModeloConfig]:
        """Obtiene la configuración de un modelo específico o None si no existe"""
        config = cls.MODELOS.get(nombre_modelo)
        if config:
            # Asegurarse de que todos los parámetros necesarios estén presentes
            config.parametros_gpu = {**cls.CONFIGURACION_DEFAULT['parametros_gpu'], 
                                   **config.parametros_gpu}
            config.optimizaciones = {**cls.CONFIGURACION_DEFAULT['optimizaciones'], 
                                   **config.optimizaciones}
        return config

    @classmethod
    def aplicar_optimizaciones_modelo(cls, comando: List[str], modelo: str) -> List[str]:
        """Aplica las optimizaciones del modelo al comando"""
        config = cls.obtener_configuracion_modelo(modelo)
        if not config:
            return comando

        try:
            # Parámetros GPU optimizados
            comando.extend([
                "-t", str(config.parametros_gpu['tile_size']),
                "-j", f"{config.parametros_gpu['gpu_threads']}:{config.parametros_gpu['gpu_block']}"
            ])

            # Optimizaciones específicas
            if config.optimizaciones.get('fp32', False):
                comando.append("--fp32")

            if config.optimizaciones.get('pre_pad'):
                comando.extend(["--pre-pad", str(config.optimizaciones['pre_pad'])])

            if config.optimizaciones.get('alpha_mode'):
                comando.extend(["--alpha", config.optimizaciones['alpha_mode']])

            if config.optimizaciones['denoise_level'] > 0:
                comando.extend(["-d", str(config.optimizaciones['denoise_level'])])

            if config.optimizaciones['sharpening'] != 1.0:
                comando.extend(["-s", str(config.optimizaciones['sharpening'])])

            if config.optimizaciones['color_correction']:
                comando.append("--color-correction")

        except Exception as e:
            print(f"Error al aplicar optimizaciones: {str(e)}")
            return comando

        return comando

    @classmethod
    def listar_modelos_disponibles(cls) -> List[str]:
        """Devuelve una lista de nombres de modelos disponibles"""
        return list(cls.MODELOS.keys())

    @classmethod
    def obtener_info_modelo(cls, nombre_modelo: str) -> Dict:
        """Devuelve información básica sobre un modelo específico"""
        config = cls.obtener_configuracion_modelo(nombre_modelo)
        if not config:
            return {}
        
        return {
            'nombre': config.nombre,
            'descripcion': config.descripcion,
            'escala': config.escala_predeterminada,
            'tipos_contenido': config.tipos_contenido
        }