import logging
import cv2
import glob
import os
import gc
import torch
from moviepy import VideoFileClip, ImageSequenceClip
from .video_processor import ProcesadorVideo
from config.config import Configuracion

logger = logging.getLogger(__name__)

def convertir_frames_a_video(archivo_entrada, archivo_salida, directorio_temp, barra_progreso=None, 
                            callback_finalizar=None, verbose=False):
    """
    Convierte los frames mejorados en un video manteniendo el audio original con mejoras en manejo de errores y recursos.
    
    Mejoras implementadas:
    - Gestión robusta de recursos
    - Manejo detallado de errores
    - Optimización de procesamiento de video
    
    Args:
        archivo_entrada (str): Ruta del video original
        archivo_salida (str): Ruta del video de salida
        directorio_temp (str): Directorio con los frames mejorados
        barra_progreso (optional): Objeto para actualizar progreso
        callback_finalizar (optional): Función a llamar al finalizar
        verbose (optional): Modo detallado para más información de logging
        tipo_contenido (str): Tipo de contenido para seleccionar el modelo adecuado
    """
    video_original = None
    clip = None
    clip_final = None

    try:
        # Configuración de logging mejorada
        logging.getLogger('moviepy').setLevel(logging.ERROR)
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)

        # Validación de archivos de entrada
        if not os.path.exists(archivo_entrada):
            raise FileNotFoundError(f"El archivo de video de entrada no existe: {archivo_entrada}")

        # Obtener lista de imágenes mejoradas con manejo de errores
        imagenes = sorted(glob.glob(os.path.join(directorio_temp, "enhanced_frame_*.png")))
        if not imagenes:
            raise FileNotFoundError("No se encontraron imágenes mejoradas para procesar")

        # Obtención de metadatos de video con manejo de errores
        cap = cv2.VideoCapture(archivo_entrada)
        if not cap.isOpened():
            raise Exception(f"No se pudo abrir el archivo de video: {archivo_entrada}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        clip = ImageSequenceClip(imagenes, fps=fps)
        
        video_original = VideoFileClip(archivo_entrada)
        if video_original.audio is not None:
            clip = clip.with_audio(video_original.audio)
            clip = clip.with_duration(video_original.audio.duration)
        
        configuracion_codec = ProcesadorVideo.obtener_codec_optimal()

        # Ajustar el número de hilos de procesamiento
        num_hilos = min(Configuracion.MAXIMO_HILOS, os.cpu_count())
        
        clip.write_videofile(
            archivo_salida,
            codec=configuracion_codec['codec'],
            audio_codec='aac',
            threads=num_hilos  # Ajustar los hilos de procesamiento
        )
        
        clip.close()
        video_original.close()
        
        if barra_progreso:
            barra_progreso.set(0.9)
        if callback_finalizar:
            callback_finalizar()
        logging.info(f"Video generado exitosamente: {archivo_salida}")

    except Exception as error_general:
        logging.error(f"Error crítico en procesamiento de video: {error_general}")
        raise
    
    finally:
        # Liberación segura y exhaustiva de recursos
        def liberar_recurso(recurso):
            try:
                if recurso:
                    recurso.close()
            except Exception as error_recurso:
                logging.warning(f"Error liberando recurso: {error_recurso}")

        liberar_recurso(clip)
        liberar_recurso(video_original)
        liberar_recurso(clip_final)
        
        # Gestión de memoria optimizada
        gc.collect()
        
        # Limpieza de memoria de GPU
        if torch.cuda.is_available():
            try:
                with torch.cuda.device('cuda'):
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