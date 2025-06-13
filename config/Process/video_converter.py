import logging
import cv2
import os
import gc
import torch
from moviepy import VideoFileClip, ImageSequenceClip
from .video_processor import ProcesadorVideo
from config.config import Configuracion
from pathlib import Path

logger = logging.getLogger(__name__)

def limpiar_archivos_temp(*rutas):
    import shutil
    for ruta in rutas:
        if os.path.exists(ruta):
            try:
                shutil.rmtree(ruta, ignore_errors=True)
            except Exception:
                pass

def convertir_frames_a_video(archivo_entrada, archivo_salida, directorio_temp, barra_progreso=None, 
                            callback_finalizar=None, verbose=False):
    archivo_entrada = Path(archivo_entrada).resolve()
    archivo_salida = Path(archivo_salida).resolve()
    directorio_temp = Path(directorio_temp).resolve()

    try:
        logging.getLogger('moviepy').setLevel(logging.ERROR)
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)

        if not archivo_entrada.exists():
            logger.error(f"Archivo de entrada no encontrado: {archivo_entrada}")
            return

        # Mejorado el ordenamiento de frames usando una clave numérica
        def extraer_numero_frame(nombre_archivo):
            try:
                return int(str(nombre_archivo).split('enhanced_frame_')[1].split('.')[0])
            except (ValueError, IndexError):
                return -1

        # Obtener y ordenar frames verificando la secuencia
        imagenes = sorted(directorio_temp.glob("enhanced_frame_*.png"), key=extraer_numero_frame)
        if not imagenes:
            logger.error(f"No se encontraron imágenes en el directorio temporal: {directorio_temp}")
            return

        # Verificar secuencia continua de frames y manejar frames faltantes
        numeros_frames = [extraer_numero_frame(img) for img in imagenes]
        frame_count_original = max(numeros_frames) + 1  # Total esperado de frames

        # Si faltan frames, duplicar el frame anterior para mantener la sincronización
        frames_completos = []
        for i in range(frame_count_original):
            if i in numeros_frames:
                frames_completos.append(str(imagenes[numeros_frames.index(i)]))
            else:
                # Si es el primer frame el que falta, usar el siguiente. Si no, usar el anterior
                idx_referencia = numeros_frames.index(min(n for n in numeros_frames if n > i)) if i == 0 else numeros_frames.index(max(n for n in numeros_frames if n < i))
                frames_completos.append(str(imagenes[idx_referencia]))
                logger.warning(f"Frame {i} faltante, duplicando frame {numeros_frames[idx_referencia]}")

        cap = cv2.VideoCapture(str(archivo_entrada))
        if not cap.isOpened():
            logger.error(f"No se pudo abrir el archivo de video: {archivo_entrada}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duracion = frame_count / fps
        cap.release()

        # Crear el clip de video con la duración correcta
        clip = ImageSequenceClip(frames_completos, fps=fps)
        
        # Cargar el video original para el audio
        video_original = VideoFileClip(str(archivo_entrada))
        audio_original = video_original.audio

        configuracion_codec = ProcesadorVideo.obtener_codec_optimal()
        num_hilos = min(Configuracion.MAXIMO_HILOS, os.cpu_count())

        # Escribir el video y el audio por separado si hay audio
        if audio_original is not None:
            clip.write_videofile(
                str(archivo_salida),
                codec=configuracion_codec['codec'],
                audio=str(archivo_entrada),  # Usar el archivo original como fuente de audio
                audio_codec='aac',
                threads=num_hilos,
                preset=configuracion_codec.get('preset', 'veryslow'),
                ffmpeg_params=['-pix_fmt', 'rgb24']
            )
        else:
            clip.write_videofile(
                str(archivo_salida),
                codec=configuracion_codec['codec'],
                audio=None,
                threads=num_hilos,
                preset=configuracion_codec.get('preset', 'veryslow'),
                ffmpeg_params=['-pix_fmt', 'rgb24']
            )


        if barra_progreso:
            barra_progreso.set(1.0)

    except Exception as error_general:
        logger.error(f"Error general: {error_general}")
        raise  # Re-lanzar la excepción para manejo superior
    finally:
        logger.info(f"Video convertido y guardado en: {archivo_salida}")
        # Cerrar recursos correctamente
        if 'clip' in locals():
            try:
                clip.close()
            except Exception:
                pass
        if 'video_original' in locals():
            try:
                video_original.close()
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as error_gpu:
                logging.warning(f"Error limpiando caché de GPU: {error_gpu}")

# Configuración de logging global con mejoras
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('aplicacion.log', encoding='utf-8', mode='a')
    ]
)