# frame_processor.py
import glob
import logging
import os
import subprocess

import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.models_config import construir_comando

logger = logging.getLogger(__name__)


def extraer_frames(archivo_seleccionado, directorio_temp, barra_progreso=None, after=None):
    """
    Extrae todos los frames de un archivo de video como imagenes PNG.
    Usa escritura en hilos para paralelismo de I/O.
    """
    cap = cv2.VideoCapture(archivo_seleccionado)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el archivo de video: {archivo_seleccionado}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Video sin frames: {archivo_seleccionado}")

    logger.info(f"Extrayendo {frame_count} frames de {archivo_seleccionado}")

    def save_frame(frame_data, frame_number):
        frame_path = os.path.join(directorio_temp, f"frame_{frame_number:08d}.png")
        cv2.imwrite(frame_path, frame_data)
        return frame_number

    current_frame = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            futures.append(executor.submit(save_frame, frame.copy(), current_frame))
            current_frame += 1

        # Rastrear progreso
        completed = 0
        for future in as_completed(futures):
            future.result()  # lanza excepcion si fallo la escritura
            completed += 1
            if barra_progreso and after and completed % 10 == 0:
                progress = 0.1 + (completed / frame_count) * 0.2
                after(0, lambda p=progress: barra_progreso.set(p))

    cap.release()
    logger.info(f"Extraccion completada: {current_frame} frames")


def mejorar_frames(
    directorio_temp,
    ruta_exe,
    ruta_modelos,
    escala=4,
    upscaler_id="realesrgan",
    modelo_id="realesr-animevideov3",
    denoise=-1,
    barra_progreso=None,
):
    """
    Mejora todos los frames usando UNA SOLA llamada al upscaler NCNN-Vulkan.
    El binario maneja su propio batching interno y paralelismo GPU.
    """
    directorio_temp = os.path.normpath(directorio_temp)

    frames = sorted(glob.glob(os.path.join(directorio_temp, "frame_*.png")))
    if not frames:
        raise FileNotFoundError("No se encontraron frames para mejorar")

    total_frames = len(frames)
    logger.info(f"Mejorando {total_frames} frames con {upscaler_id}/{modelo_id}, escala={escala}")

    # Crear subdirectorio de salida
    directorio_salida = os.path.join(directorio_temp, "enhanced")
    os.makedirs(directorio_salida, exist_ok=True)

    # Construir comando usando el builder centralizado
    num_hilos = min(max(1, os.cpu_count() - 1), 8)
    comando = construir_comando(
        upscaler_id=upscaler_id,
        modelo_id=modelo_id,
        ruta_exe=ruta_exe,
        entrada=directorio_temp,
        salida=directorio_salida,
        ruta_modelos=ruta_modelos,
        escala=escala,
        gpu_id=0,
        denoise=denoise,
        hilos=f"1:{num_hilos}:{num_hilos}",
    )

    logger.info(f"Ejecutando: {' '.join(comando)}")

    proceso = subprocess.Popen(
        comando,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=os.path.dirname(ruta_exe),
        creationflags=subprocess.CREATE_NO_WINDOW
    )

    # Parsear stderr para progreso (los upscalers NCNN emiten lineas con porcentaje)
    for line in proceso.stderr:
        line = line.strip()
        if '%' in line:
            try:
                for part in line.split():
                    if '%' in part:
                        pct = float(part.replace('%', '').strip())
                        if barra_progreso:
                            progress = 0.3 + (pct / 100.0) * 0.4
                            barra_progreso.set(progress)
                        break
            except (ValueError, IndexError):
                pass
        if line:
            logger.debug(f"upscaler: {line}")

    proceso.wait()

    if proceso.returncode != 0:
        raise RuntimeError(
            f"Upscaler fallo con codigo {proceso.returncode}. "
            f"Revise el log para mas detalles."
        )

    # Renombrar archivos de salida al patron esperado
    output_files = sorted(glob.glob(os.path.join(directorio_salida, "*.png")))
    if not output_files:
        raise FileNotFoundError("El upscaler no genero ningun frame mejorado")

    for i, src in enumerate(output_files):
        dst = os.path.join(directorio_temp, f"enhanced_frame_{i:08d}.png")
        os.rename(src, dst)

    # Limpiar subdirectorio
    try:
        os.rmdir(directorio_salida)
    except OSError:
        pass

    enhanced = sorted(glob.glob(os.path.join(directorio_temp, "enhanced_frame_*.png")))
    logger.info(f"Frames mejorados: {len(enhanced)} de {total_frames}")

    if len(enhanced) == 0:
        raise FileNotFoundError("No se genero ningun frame mejorado")

    return {
        'total_frames': total_frames,
        'enhanced_frames': len(enhanced),
    }
