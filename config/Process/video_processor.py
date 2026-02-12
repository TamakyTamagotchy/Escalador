import os
import glob
import shutil
import subprocess
import logging
from config.gpu_manager import GestorGPU
from config.config import Configuracion
from prueba_config import get_recommended_model

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
        lote_frames, realesrgan_path, models_path, tipo_contenido, escala, indice_lote, indice_gpu = args
        frames_procesados = []

        directorio_base = os.path.dirname(lote_frames[0])
        directorio_lote = os.path.join(directorio_base, f"batch_{indice_lote}")
        directorio_salida = os.path.join(directorio_base, f"output_{indice_lote}")

        try:
            # Crear directorios si no existen
            os.makedirs(directorio_lote, exist_ok=True)
            os.makedirs(directorio_salida, exist_ok=True)

            # Validar frames de entrada
            frames_entrada = [frame for frame in lote_frames if os.path.exists(frame)]
            if not frames_entrada:
                logger.error(f"No se encontraron frames válidos en el lote {indice_lote}")
                return []

            # Limpiar directorios antes de procesar
            for directorio in [directorio_lote, directorio_salida]:
                for archivo in glob.glob(os.path.join(directorio, "*.png")):
                    os.remove(archivo)

            # Copiar y preparar frames
            for i, frame in enumerate(frames_entrada):
                nuevo_nombre = f"frame_{i:08d}.png"
                shutil.copy2(frame, os.path.join(directorio_lote, nuevo_nombre))

            # Obtener modelo y configurar parámetros
            modelo = get_recommended_model(tipo_contenido)
            num_hilos = min(Configuracion.MAXIMO_HILOS, os.cpu_count())
            batch_size = max(1, len(frames_entrada) // num_hilos)

            # Construir y ejecutar comando
            comando = [
                os.path.normpath(realesrgan_path),
                "-i", os.path.normpath(directorio_lote),
                "-o", os.path.normpath(directorio_salida),
                "-n", modelo,
                "-m", os.path.normpath(models_path),
                "-s", str(escala),
                "-f", "png",
                "-g", str(indice_gpu),
                "-t", "0",  # Tamaño de mosaico optimizado
                "-j", f"1:{num_hilos}:{batch_size}"
            ]

            logger.info(f"Ejecutando comando: {' '.join(comando)}")
            proceso = subprocess.Popen(
                comando,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            stdout, stderr = proceso.communicate(timeout=300)

            if proceso.returncode != 0:
                raise subprocess.CalledProcessError(
                    proceso.returncode, comando, stdout, stderr
                )

            # Procesar frames mejorados
            frames_mejorados = sorted(glob.glob(os.path.join(directorio_salida, "*.png")))
            if not frames_mejorados:
                raise FileNotFoundError(f"No se generaron frames mejorados en el lote {indice_lote}")

            # Mover frames mejorados
            for i, frame_mejorado in enumerate(frames_mejorados):
                indice_global = indice_lote * len(lote_frames) + i
                nuevo_nombre = f"enhanced_frame_{indice_global:08d}.png"
                ruta_destino = os.path.join(directorio_base, nuevo_nombre)
                shutil.move(frame_mejorado, ruta_destino)
                frames_procesados.append(ruta_destino)

            logger.info(f"Lote {indice_lote} completado: {len(frames_procesados)} frames")
            return frames_procesados

        except Exception as e:
            logger.error(f"Error en lote {indice_lote}: {str(e)}")
            return []

        finally:
            # Limpiar recursos
            shutil.rmtree(directorio_lote, ignore_errors=True)
            shutil.rmtree(directorio_salida, ignore_errors=True)