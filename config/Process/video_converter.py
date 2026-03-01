# video_converter.py
import logging
import os
import subprocess

import cv2

from config.gpu_manager import GestorGPU

logger = logging.getLogger(__name__)


def _encontrar_ffmpeg():
    """Busca el ejecutable ffmpeg - primero en bin/, luego en PATH del sistema."""
    # Buscar en carpeta bin/ relativa al proyecto
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ffmpeg_local = os.path.join(project_root, 'bin', 'ffmpeg.exe')
    if os.path.isfile(ffmpeg_local):
        return ffmpeg_local

    # Fallback a PATH del sistema
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        if result.returncode == 0:
            return 'ffmpeg'
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def _detectar_encoder_hw(ffmpeg_path):
    """
    Detecta el mejor encoder HW disponible segun el fabricante de GPU.
    Retorna (nombre_encoder, lista_params_extra).
    """
    gpu = GestorGPU.verificar_compatibilidad_gpu()

    encoders = {
        'NVIDIA': ('h264_nvenc', ['-preset', 'p7', '-rc', 'constqp', '-qp', '18']),
        'AMD':    ('h264_amf',   ['-quality', 'quality']),
        'Intel':  ('h264_qsv',   ['-preset', 'veryslow']),
    }

    if gpu in encoders:
        encoder, params = encoders[gpu]
        # Verificar que el encoder existe en ffmpeg
        try:
            result = subprocess.run(
                [ffmpeg_path, '-hide_banner', '-encoders'],
                capture_output=True, text=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            if encoder in result.stdout:
                logger.info(f"Usando encoder HW: {encoder}")
                return encoder, params
        except Exception:
            pass

    logger.info("Usando encoder software: libx264")
    return 'libx264', ['-preset', 'slow', '-crf', '18']


def convertir_frames_a_video(archivo_entrada, archivo_salida, directorio_temp,
                              barra_progreso=None, callback_finalizar=None):
    """
    Ensambla frames mejorados en un video preservando el audio original.
    Usa FFmpeg directamente en lugar de moviepy.
    """
    ffmpeg = _encontrar_ffmpeg()
    if not ffmpeg:
        raise FileNotFoundError(
            "FFmpeg no encontrado. Coloque ffmpeg.exe en la carpeta bin/ "
            "o instalelo en el PATH del sistema."
        )

    # Obtener FPS del video original
    cap = cv2.VideoCapture(archivo_entrada)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video original: {archivo_entrada}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0:
        fps = 30.0
        logger.warning(f"FPS no detectado, usando valor por defecto: {fps}")

    # Detectar mejor encoder
    encoder, encoder_params = _detectar_encoder_hw(ffmpeg)

    # Construir comando FFmpeg
    frame_pattern = os.path.join(directorio_temp, 'enhanced_frame_%08d.png')

    comando = [
        ffmpeg, '-y',
        '-framerate', str(fps),
        '-i', frame_pattern,
        '-i', archivo_entrada,
        '-map', '0:v',          # video de los frames
        '-map', '1:a?',         # audio del original (si existe)
        '-c:v', encoder,
        *encoder_params,
        '-c:a', 'copy',         # copiar audio sin re-encodear
        '-pix_fmt', 'yuv420p',  # maxima compatibilidad
        '-movflags', '+faststart',
        archivo_salida
    ]

    logger.info(f"Ejecutando FFmpeg: {' '.join(comando)}")

    proceso = subprocess.Popen(
        comando,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        creationflags=subprocess.CREATE_NO_WINDOW
    )

    stdout, stderr = proceso.communicate(timeout=600)

    if proceso.returncode != 0:
        # Si el encoder HW fallo, reintentar con libx264
        if encoder != 'libx264':
            logger.warning(f"Encoder {encoder} fallo, reintentando con libx264")
            comando_fallback = [
                ffmpeg, '-y',
                '-framerate', str(fps),
                '-i', frame_pattern,
                '-i', archivo_entrada,
                '-map', '0:v',
                '-map', '1:a?',
                '-c:v', 'libx264',
                '-preset', 'slow', '-crf', '18',
                '-c:a', 'copy',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                archivo_salida
            ]
            proceso2 = subprocess.Popen(
                comando_fallback,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            stdout2, stderr2 = proceso2.communicate(timeout=600)
            if proceso2.returncode != 0:
                raise RuntimeError(f"FFmpeg fallo: {stderr2}")
        else:
            raise RuntimeError(f"FFmpeg fallo: {stderr}")

    if not os.path.exists(archivo_salida):
        raise FileNotFoundError(f"FFmpeg no genero el archivo de salida: {archivo_salida}")

    logger.info(f"Video generado exitosamente: {archivo_salida}")

    if barra_progreso:
        barra_progreso.set(0.95)
    if callback_finalizar:
        callback_finalizar()
