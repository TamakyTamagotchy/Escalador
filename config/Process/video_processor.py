from asyncio.log import logger
import glob
import os
from typing import Any, Dict, Optional
import cv2
import shutil
import subprocess
import logging
from pathlib import Path
from config.gpu_manager import GestorGPU
from config.config import Configuracion
from config.models_config import ConfiguracionModelos
from config.resource_manager import GestorRecursos

class ProcesadorVideo:
    def __init__(self, ruta_entrada: str, ruta_salida: str, directorio_temp: str):
        self.ruta_entrada = Path(ruta_entrada)
        self.ruta_salida = Path(ruta_salida)
        self.directorio_temp = Path(directorio_temp)
        self.tipo_gpu = GestorGPU.verificar_compatibilidad_gpu()
    @staticmethod
    def obtener_codec_optimal(tipo_gpu: Optional[str] = None) -> Dict[str, Any]:
        if tipo_gpu is None:
            tipo_gpu = GestorGPU.verificar_compatibilidad_gpu()
            
        parametros_base = {
            'hilos': Configuracion.MAXIMO_HILOS,
            'bitrate': '50000k',
            'parametros_extra': {
                'profile:v': 'high',
                'level': '5.2',
                'pix_fmt': 'yuv444p',
                'color_primaries': 'bt709',
                'color_trc': 'bt709',
                'colorspace': 'bt709',
                'color_range': 'tv',
            }
        }
        configuraciones_codec = {
            'NVIDIA': {
                'codec': 'h264_nvenc',
                'preset': 'p7',
                'parametros_extra': {**parametros_base['parametros_extra'], 
                            'rc': 'constqp', 'qp': '15'}
            },
            'AMD': {
                'codec': 'h264_amf',
                'preset': 'quality',
                'parametros_extra': {**parametros_base['parametros_extra'],
                            'quality': 'quality'}
            },
            'Intel': {
                'codec': 'h264_qsv',
                'preset': 'veryslow',
                'parametros_extra': {**parametros_base['parametros_extra'],
                            'look_ahead': '1'}
            }
        }
        
        return configuraciones_codec.get(tipo_gpu, {
            'codec': 'libx264',
            'preset': 'veryslow',
            'parametros_extra': {**parametros_base['parametros_extra'],
                        'crf': '15'}
        })
def procesar_frame(self, datos_frame: tuple) -> bool:
    numero_frame, frame = datos_frame
    try:
        GestorRecursos.verificar_uso_recursos()
        ruta_frame = self.directorio_temp / f"frame_{numero_frame:08d}.png"
        return cv2.imwrite(str(ruta_frame), frame, 
                        [cv2.IMWRITE_PNG_COMPRESSION, Configuracion.COMPRESION_PNG])
    except Exception as e:
        logging.error(f"Error en frame {numero_frame}: {e}")
        return False

def procesar_lote_frames(args):
    """
    Procesa un lote de frames usando GPU cuando está disponible.
    """
    lote_frames, realesrgan_path, models_path, modelo, escala, indice_lote = args
    frames_procesados = []
    
    # Obtener configuración del modelo
    config_modelo = ConfiguracionModelos.obtener_configuracion_modelo(modelo)
    if not config_modelo:
        logger.error(f"Modelo no reconocido: {modelo}")
        return []
    
    # Crear directorio temporal para el lote
    directorio_base = os.path.dirname(lote_frames[0])
    directorio_lote = os.path.join(directorio_base, f"batch_{indice_lote}")
    directorio_salida = os.path.join(directorio_base, f"output_{indice_lote}")
    
    os.makedirs(directorio_lote, exist_ok=True)
    os.makedirs(directorio_salida, exist_ok=True)
    
    try:
        # Crear lista de frames de entrada y salida
        frames_entrada = [frame for frame in lote_frames if os.path.exists(frame)]
        if not frames_entrada:
            logger.error(f"No se encontraron frames de entrada válidos para el lote {indice_lote}")
            return []

        # Limpiar cualquier archivo residual
        for archivo in glob.glob(os.path.join(directorio_lote, "*.png")):
            os.remove(archivo)
        for archivo in glob.glob(os.path.join(directorio_salida, "*.png")):
            os.remove(archivo)

        GestorRecursos.verificar_uso_recursos()

        # Copiar frames al directorio del lote
        for i, frame in enumerate(frames_entrada):
            nuevo_nombre = f"frame_{i:08d}.png"
            ruta_destino = os.path.join(directorio_lote, nuevo_nombre)
            try:
                shutil.copy2(frame, ruta_destino)
                if not os.path.exists(ruta_destino):
                    raise FileNotFoundError(f"No se pudo copiar el frame {i} al directorio del lote")
            except Exception as e:
                logger.error(f"Error copiando frame {i} en lote {indice_lote}: {str(e)}")
                continue

        # Verificar frames copiados
        frames_copiados = glob.glob(os.path.join(directorio_lote, "*.png"))
        if not frames_copiados:
            raise FileNotFoundError(f"No se pudieron copiar frames al directorio del lote {indice_lote}")
        
        logger.info(f"Lote {indice_lote}: Copiados {len(frames_copiados)} frames al directorio temporal")

        # Construir comando base
        comando = [
            os.path.normpath(realesrgan_path),
            "-i", os.path.normpath(directorio_lote),
            "-o", os.path.normpath(directorio_salida),
            "-n", modelo,
            "-f", "png",
            "-m", os.path.normpath(models_path),
            "-g", "1"
        ]

        # Aplicar optimizaciones específicas del modelo
        comando = ConfiguracionModelos.aplicar_optimizaciones_modelo(comando, modelo)

        # Añadir escala si el modelo lo permite
        if config_modelo.permite_escala_personalizada:
            comando.extend(["-s", str(escala)])

        logger.info(f"Ejecutando comando para lote {indice_lote}: {' '.join(comando)}")

        # Configurar entorno para GPU
        env = os.environ.copy()
        env["PATH"] = f"{os.path.dirname(realesrgan_path)};{env['PATH']}"

        # Ejecutar proceso
        process = subprocess.Popen(
            comando,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=env,
            cwd=os.path.dirname(realesrgan_path)
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Error en el proceso del lote {indice_lote}: {stderr}")
            raise subprocess.CalledProcessError(
                process.returncode, comando, output=stdout, stderr=stderr
            )

        # Verificar y mover los frames mejorados
        frames_mejorados = glob.glob(os.path.join(directorio_salida, "*.png"))
        if not frames_mejorados:
            raise FileNotFoundError(f"No se encontraron frames mejorados en el directorio de salida para el lote {indice_lote}")
        
        logger.info(f"Lote {indice_lote}: Generados {len(frames_mejorados)} frames mejorados")

        # Mover y renombrar los frames mejorados
        for i, frame_mejorado in enumerate(sorted(frames_mejorados)):
            indice_global = indice_lote * len(lote_frames) + i
            nuevo_nombre = f"enhanced_frame_{indice_global:08d}.png"
            ruta_destino = os.path.join(directorio_base, nuevo_nombre)
            try:
                shutil.move(frame_mejorado, ruta_destino)
                frames_procesados.append(indice_global)
            except Exception as e:
                logger.error(f"Error moviendo frame mejorado {i} en lote {indice_lote}: {str(e)}")
                continue

        if not frames_procesados:
            raise FileNotFoundError(f"No se pudieron mover los frames mejorados para el lote {indice_lote}")
        
        logger.info(f"Lote {indice_lote} completado exitosamente con {len(frames_procesados)} frames")

    except Exception as e:
        logger.error(f"Error procesando lote {indice_lote}: {str(e)}")
        return []
    finally:
        # Limpiar directorios temporales
        try:
            shutil.rmtree(directorio_lote, ignore_errors=True)
            shutil.rmtree(directorio_salida, ignore_errors=True)
        except Exception as e:
            logger.error(f"Error limpiando directorios temporales del lote {indice_lote}: {str(e)}")
        
    return frames_procesados