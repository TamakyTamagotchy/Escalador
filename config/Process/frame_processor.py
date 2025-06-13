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
    import shutil
    # Limpia el directorio temporal antes de extraer
    if os.path.exists(directorio_temp):
        limpiar_directorio(directorio_temp)
    else:
        os.makedirs(directorio_temp, exist_ok=True)
    cap = cv2.VideoCapture(archivo_seleccionado)
    if not cap.isOpened():
        raise Exception(f"No se pudo abrir el archivo de video: {archivo_seleccionado}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    def save_frame(frame, frame_number):
        cv2.imwrite(os.path.join(directorio_temp, f"frame_{frame_number:08d}.png"), frame)
        
    with ThreadPoolExecutor() as executor:
        futures = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            futures.append(executor.submit(save_frame, frame, current_frame))
            current_frame += 1
        for future in as_completed(futures):
            # Actualización de progreso basada en el índice actual (aproximado)
            after(0, lambda p=(0.1 + (current_frame / frame_count) * 0.2): barra_progreso.set(p))
    
    cap.release()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def extraer_frames_optimizado(archivo_seleccionado, directorio_temp, barra_progreso, after):
    import shutil
    if os.path.exists(directorio_temp):
        limpiar_directorio(directorio_temp)
    else:
        os.makedirs(directorio_temp, exist_ok=True)
    cap = cv2.VideoCapture(archivo_seleccionado)
    if not cap.isOpened():
        raise Exception(f"No se pudo abrir el archivo de video: {archivo_seleccionado}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    def save_frame(frame, frame_number):
        cv2.imwrite(os.path.join(directorio_temp, f"frame_{frame_number:08d}.png"), frame)

    with ProcessPoolExecutor() as executor:
        futures = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            futures.append(executor.submit(save_frame, frame, current_frame))
            current_frame += 1
        for future in as_completed(futures):
            after(0, lambda p=(0.1 + (current_frame / frame_count) * 0.2): barra_progreso.set(p))

    cap.release()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def detectar_configuracion_gpu():
    """
    Detecta la GPU y ajusta tile_size, batch_size y hilos según la VRAM disponible.
    Utiliza el nuevo sistema de detección de GPUs para asegurar el uso de NVIDIA cuando esté disponible.
    """
    from config.gpu_manager import GestorGPU

    # Obtener información detallada de la GPU
    gestor = GestorGPU._get_instance()
    if gestor._gpus_cache is None:
        gestor._gpus_cache = GestorGPU._detect_gpus()

    selected_gpu = None
    for gpu in gestor._gpus_cache:
        if gpu.vendor == 'NVIDIA':
            selected_gpu = gpu
            break
    
    if not selected_gpu and gestor._gpus_cache:
        selected_gpu = gestor._gpus_cache[0]

    # Configuración optimizada según VRAM y capacidades de la GPU
    def calcular_parametros_gpu(vram_gb):
        # Cálculo dinámico del tile_size basado en VRAM
        # Aseguramos que sea múltiplo de 32 para mejor rendimiento
        tile_size = min(512, max(32, ((vram_gb * 1024) // 8) // 32 * 32))
        
        # Cálculo dinámico del batch_size basado en VRAM disponible
        # Reservamos 20% de VRAM para el sistema y otros procesos
        vram_disponible = vram_gb * 0.8
        # Estimamos memoria por frame basado en tile_size
        mem_por_frame = (tile_size ** 2 * 4) / (1024 * 1024)  # En GB
        batch_size = max(1, min(8, int(vram_disponible / mem_por_frame)))
        
        # Cálculo de workers basado en CPU cores y VRAM
        max_workers = max(1, min(
            os.cpu_count() - 1,  # Dejamos un núcleo libre para el sistema
            int(vram_gb / 2),    # Aproximadamente 2GB por worker
            4                    # Máximo razonable para evitar sobrecarga
        ))
        
        return tile_size, batch_size, max_workers

    if not selected_gpu:
        # Configuración por defecto si no se detecta GPU
        return {
            'gpu_name': "CPU",
            'vram_gb': 4,
            'tile_size': 64,
            'batch_size': 1,
            'max_workers': 1,
            'dispositivo': "cpu"
        }

    # Obtener parámetros optimizados
    vram_gb = selected_gpu.vram_mb / 1024
    tile_size, batch_size, max_workers = calcular_parametros_gpu(vram_gb)

    # Ajustar dispositivo según el vendor
    if selected_gpu.vendor == 'NVIDIA':
        dispositivo = "cuda:0"
    elif selected_gpu.vendor == 'AMD':
        dispositivo = "vulkan"
    elif selected_gpu.vendor == 'Intel':
        dispositivo = "intel:0"
    else:
        dispositivo = "cpu"

    logger.info(f"GPU detectada: {selected_gpu.name}")
    logger.info(f"VRAM: {vram_gb:.1f}GB")
    logger.info(f"Configuración optimizada: tile_size={tile_size}, batch_size={batch_size}, max_workers={max_workers}")

    return {
        'gpu_name': selected_gpu.name,
        'vram_gb': vram_gb,
        'tile_size': tile_size,
        'batch_size': batch_size,
        'max_workers': max_workers,
        'dispositivo': dispositivo
    }

def limpiar_directorio(directorio, patron="*.png"):
    for archivo in glob.glob(os.path.join(directorio, patron)):
        os.remove(archivo)

def copiar_archivos_paralelo(archivos, destino):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import shutil
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(shutil.copy2, archivo, os.path.join(destino, f"frame_{i:08d}.png"))
                   for i, archivo in enumerate(archivos)]
        for future in as_completed(futures):
            future.result()

def mejorar_frames(directorio_temp, ruta_realesrgan, ruta_modelos, barra_progreso, escala, modelo=None, dispositivo_gpu=None):
    from config.gpu_manager import GestorGPU
    modelos_disponibles = ConfiguracionModelos.listar_modelos_disponibles()
    if not modelo:
        modelo = modelos_disponibles[0] if modelos_disponibles else None
    if not modelo or modelo not in modelos_disponibles:
        raise ValueError(f"Modelo no válido. Modelos disponibles: {modelos_disponibles}")
    logger.info(f"Usando modelo de mejora: {modelo}")
    directorio_temp = os.path.normpath(directorio_temp)
    ruta_realesrgan = os.path.normpath(ruta_realesrgan)
    ruta_modelos = os.path.normpath(ruta_modelos)
    frames = sorted(glob.glob(os.path.join(directorio_temp, "frame_*.png")))
    logger.info(f"Frames encontrados: {len(frames)}")
    if not frames:
        raise Exception("No se encontraron frames para mejorar")
    # --- SOLO UNA VEZ: Detectar GPU y batch config ---
    config_gpu = detectar_configuracion_gpu()
    total_frames = len(frames)
    # Selección de batch_size: máximo permitido por VRAM, nunca menor a 1, nunca mayor a total_frames
    batch_size = min(max(1, config_gpu['batch_size']), total_frames)
    tile_size = config_gpu['tile_size']
    max_workers = min(Configuracion.MAXIMO_HILOS, config_gpu['max_workers'], os.cpu_count())
    # Forzar uso de NVIDIA si está disponible
    dispositivo_gpu = GestorGPU.seleccionar_dispositivo_gpu()
    if dispositivo_gpu and 'intel' in dispositivo_gpu.lower():
        logger.warning('Se detectó GPU Intel, pero se priorizará NVIDIA si está disponible.')
        dispositivo_gpu = 'NVIDIA' if GestorGPU.verificar_compatibilidad_gpu() == 'NVIDIA' else dispositivo_gpu
        logger.info(f'Usando GPU: {dispositivo_gpu}')
    processed_frames = set()
    failed_frames = []
    lotes = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]
    logger.info(f"Iniciando mejora de {total_frames} frames en {len(lotes)} lotes de tamaño {batch_size}")
    # --- PASAR SIEMPRE EL MISMO dispositivo_gpu ---
    lote_args = [
        (lote, ruta_realesrgan, ruta_modelos, modelo, escala, i, tile_size, dispositivo_gpu)
        for i, lote in enumerate(lotes)
    ]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(ProcesadorVideo.procesar_lote_frames, args): i for i, args in enumerate(lote_args)}
        for future in as_completed(futures):
            try:
                frames_completados = future.result()
                processed_frames.update(frames_completados)
                barra_progreso.set(0.3 + (len(processed_frames) / total_frames) * 0.4)
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
    imagenes_mejoradas = sorted(glob.glob(os.path.join(directorio_temp, "enhanced_frame_*.png")))
    # Ordenar por el número de frame extraído del nombre
    imagenes_mejoradas.sort(key=lambda x: int(os.path.basename(x).replace("enhanced_frame_", "").replace(".png", "")))
    logger.info(f"Imágenes mejoradas encontradas: {len(imagenes_mejoradas)}")
    logger.info(f"Frames procesados exitosamente: {len(processed_frames)}")
    logger.info(f"Frames fallidos: {len(failed_frames)}")
    if not imagenes_mejoradas:
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
