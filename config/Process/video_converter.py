#video_converter
from asyncio.log import logger
from moviepy import ImageSequenceClip, VideoFileClip
import cv2
import glob
import os
import gc
import torch
from .video_processor import ProcesadorVideo

def convertir_frames_a_video(archivo_entrada, archivo_salida, directorio_temp, barra_progreso, callback_finalizar):
    """
    Convierte los frames mejorados en un video manteniendo el audio original.
    """
    try:
        imagenes = sorted(glob.glob(os.path.join(directorio_temp, "enhanced_frame_*.png")))
        if not imagenes:
            raise FileNotFoundError("No se encontraron imágenes mejoradas para procesar")
        
        cap = cv2.VideoCapture(archivo_entrada)
        if not cap.isOpened():
            raise Exception(f"No se pudo abrir el archivo de video: {archivo_entrada}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        clip = ImageSequenceClip(imagenes, fps=fps)
        
        video_original = VideoFileClip(archivo_entrada)
        if video_original.audio is not None:
            clip = clip.set_audio(video_original.audio)
            clip = clip.set_duration(video_original.audio.duration)
        
        configuracion_codec = ProcesadorVideo.obtener_codec_optimal()
        
        clip.write_videofile(
            archivo_salida,
            codec=configuracion_codec['codec'],
            audio_codec='aac'
        )
        
        clip.close()
        video_original.close()
        
        barra_progreso.set(0.9)
        if callback_finalizar:
            callback_finalizar()

    except Exception as e:
        logger.error(f"Error al crear video: {e}")
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
