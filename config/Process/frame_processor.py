import os
import glob
import gc
from asyncio.log import logger
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import torch
import cv2
from config.config import Configuracion
from config.models_config import ConfiguracionModelos
from .video_processor import ProcesadorVideo

def extraer_frames(archivo_seleccionado, directorio_temp, barra_progreso, after):
    cap = cv2.VideoCapture(archivo_seleccionado)
    if not cap.isOpened():
        raise Exception(f"No se pudo abrir el archivo de video: {archivo_seleccionado}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    
    def save_frame(frame, frame_number):
        frame_path = os.path.join(directorio_temp, f"frame_{frame_number:08d}.png")
        cv2.imwrite(frame_path, frame)
        
    with ThreadPoolExecutor() as executor:
        futures = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            futures.append(executor.submit(save_frame, frame, current_frame))
            current_frame += 1
            
        for future in as_completed(futures):
            progress = 0.1 + (current_frame / frame_count) * 0.2
            after(0, lambda p=progress: barra_progreso.set(p))
    
    cap.release()

def mejorar_frames(directorio_temp, ruta_realesrgan, ruta_modelos, barra_progreso, escala, modelo=None, indice_gpu=0):
    """
    Mejora los frames usando procesamiento por lotes y GPU.
    """
    # Validar y configurar el modelo
    modelos_disponibles = ConfiguracionModelos.listar_modelos_disponibles()
    
    if not modelo:
        # Usar un modelo predeterminado si no se especifica
        modelo = modelos_disponibles[0] if modelos_disponibles else None
    
    if not modelo or modelo not in modelos_disponibles:
        raise ValueError(f"Modelo no válido. Modelos disponibles: {modelos_disponibles}")
    
    # Logging de información de modelo
    logger.info(f"Usando modelo de mejora: {modelo}")
    
    directorio_temp = os.path.normpath(directorio_temp)
    ruta_realesrgan = os.path.normpath(ruta_realesrgan)
    ruta_modelos = os.path.normpath(ruta_modelos)
    
    frames = sorted(glob.glob(os.path.join(directorio_temp, "frame_*.png")))
    logger.info(f"Frames encontrados: {len(frames)}")
    
    if not frames:
        raise Exception("No se encontraron frames para mejorar")
    
    total_frames = len(frames)
    processed_frames = set()
    failed_frames = []
    batch_size = min(Configuracion.TAMANIO_LOTE, max(1, total_frames // os.cpu_count()))
    
    lotes = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]
    logger.info(f"Iniciando mejora de {total_frames} frames en {len(lotes)} lotes de tamaño {batch_size}")
    
    lote_args = [
        (lote, ruta_realesrgan, ruta_modelos, modelo, escala, i, indice_gpu)
        for i, lote in enumerate(lotes)
    ]
    
    max_workers = min(Configuracion.MAXIMO_HILOS, os.cpu_count())
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(ProcesadorVideo.procesar_lote_frames, args): i 
                for i, args in enumerate(lote_args)}
        
        for future in as_completed(futures):
            try:
                frames_completados = future.result()
                processed_frames.update(frames_completados)
                progress = 0.3 + (len(processed_frames) / total_frames) * 0.4
                barra_progreso.set(progress)
                
                lote_actual = futures[future]
                logger.info(f"Lote {lote_actual} completado. Progreso: {len(processed_frames)}/{total_frames}")
                
                if not frames_completados:
                    failed_frames.extend(range(
                        lote_actual * batch_size,
                        min((lote_actual + 1) * batch_size, total_frames)
                    ))
            except Exception as e:
                logger.error(f"Error en lote {futures[future]}: {str(e)}")
                failed_frames.extend(range(
                    futures[future] * batch_size,
                    min((futures[future] + 1) * batch_size, total_frames)
                ))

    imagenes_mejoradas = glob.glob(os.path.join(directorio_temp, "enhanced_frame_*.png"))
    logger.info(f"Imágenes mejoradas encontradas: {len(imagenes_mejoradas)}")
    logger.info(f"Frames procesados exitosamente: {len(processed_frames)}")
    logger.info(f"Frames fallidos: {len(failed_frames)}")
    
    if len(imagenes_mejoradas) == 0:
        raise FileNotFoundError("No se generó ningún frame mejorado")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'total_frames': total_frames,
        'processed_frames': len(processed_frames),
        'failed_frames': len(failed_frames),
        'improved_frames': len(imagenes_mejoradas)
    }
