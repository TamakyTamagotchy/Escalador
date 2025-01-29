import os
import glob
import shutil
import subprocess
import logging
from config.gpu_manager import GestorGPU
from config.config import Configuracion
from prueba_config import get_recommended_model
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("ProcesadorVideo")

class ProcesadorVideo:
    @staticmethod
    def obtener_codec_optimal(tipo_gpu=None):
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
                'tune': 'film',  # Mejora para contenido de alta calidad
                'qmin': '10',    # Calidad mínima
                'qmax': '51',    # Calidad máxima
                'qcomp': '0.6'   # Compresión de calidad
            }
        }
        configuraciones_codec = {
            'NVIDIA': {
                'codec': 'h264_nvenc',
                'preset': 'p7',
                'parametros_extra': {**parametros_base['parametros_extra'], 'rc': 'constqp', 'qp': '15'}
            },
            'AMD': {
                'codec': 'h264_amf',
                'preset': 'quality',
                'parametros_extra': {**parametros_base['parametros_extra'], 'quality': 'quality'}
            },
            'Intel': {
                'codec': 'h264_qsv',
                'preset': 'veryslow',
                'parametros_extra': {**parametros_base['parametros_extra'], 'look_ahead': '1'}
            }
        }

        return configuraciones_codec.get(tipo_gpu, {
            'codec': 'libx264',
            'preset': 'veryslow',
            'parametros_extra': {**parametros_base['parametros_extra'], 'crf': '15'}
        })

    @staticmethod
    def procesar_lote_frames(args):
        """
        Procesa un lote de frames usando RealesrGAN con manejo de errores mejorado y optimización de códecs.
        """
        lote_frames, realesrgan_path, models_path, tipo_contenido, escala, indice_lote = args
        frames_procesados = []

        directorio_base = os.path.dirname(lote_frames[0])
        directorio_lote = os.path.join(directorio_base, f"batch_{indice_lote}")
        directorio_salida = os.path.join(directorio_base, f"output_{indice_lote}")

        os.makedirs(directorio_lote, exist_ok=True)
        os.makedirs(directorio_salida, exist_ok=True)

        try:
            frames_entrada = [frame for frame in lote_frames if os.path.exists(frame)]
            if not frames_entrada:
                logger.error(f"No se encontraron frames de entrada válidos para el lote {indice_lote}")
                return []

            # Limpiar directorios
            for archivo in glob.glob(os.path.join(directorio_lote, "*.png")):
                os.remove(archivo)
            for archivo in glob.glob(os.path.join(directorio_salida, "*.png")):
                os.remove(archivo)

            # Copiar frames
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(shutil.copy2, frame, os.path.join(directorio_lote, f"frame_{i:08d}.png")) for i, frame in enumerate(frames_entrada)]
                for future in as_completed(futures):
                    future.result()

            # Obtener el modelo recomendado
            modelo = get_recommended_model(tipo_contenido)
            
            # Construir comando base
            comando = [
                os.path.normpath(realesrgan_path),
                "-i", os.path.normpath(directorio_lote),
                "-o", os.path.normpath(directorio_salida),
                "-n", modelo,
                "-m", os.path.normpath(models_path),
                "-s", str(escala),
                "-f", "png",
                "-g", "0","1",
                "-t", "0","0",
                "-j", "1:2:2", # Ajustar los hilos de procesamiento
                "-x"
                ]

            logger.info(f"Ejecutando comando: {' '.join(comando)}")

            # Ejecutar proceso
            process = subprocess.Popen(
                comando,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            stdout, stderr = process.communicate(timeout=600)  # Timeout de 10 minutos

            if process.returncode != 0:
                logger.error(f"Error procesando lote {indice_lote}: Código {process.returncode}")
                logger.error(f"Salida estándar:\n{stdout}")
                logger.error(f"Error estándar:\n{stderr}")
                return []

            # Verificar frames mejorados
            frames_mejorados = sorted(glob.glob(os.path.join(directorio_salida, "*.png")))
            if not frames_mejorados:
                logger.warning(f"No se encontraron frames mejorados en el lote {indice_lote}")
                return []

            # Obtener configuración del codec
            codec_config = ProcesadorVideo.obtener_codec_optimal()
            logger.info(f"Configuración de códec: {codec_config}")

            # Mover frames mejorados
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(shutil.move, frame_mejorado, os.path.join(directorio_base, f"enhanced_frame_{indice_lote * len(lote_frames) + i:08d}.png")) for i, frame_mejorado in enumerate(frames_mejorados)]
                for future in as_completed(futures):
                    frames_procesados.append(future.result())

            logger.info(f"Lote {indice_lote} completado: {len(frames_procesados)} frames procesados.")

        except subprocess.TimeoutExpired:
            process.kill()
            logger.error(f"Timeout expirado procesando lote {indice_lote}")
        except Exception as e:
            logger.error(f"Error inesperado procesando lote {indice_lote}: {str(e)}")
        finally:
            # Limpiar directorios temporales
            shutil.rmtree(directorio_lote, ignore_errors=True)
            shutil.rmtree(directorio_salida, ignore_errors=True)

        return frames_procesados