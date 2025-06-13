import os
import glob
import shutil
import subprocess
import logging
from config.config import Configuracion
from prueba_config import get_recommended_model
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("ProcesadorVideo")

def limpiar_directorio(directorio, patron="*.png"):
    for archivo in glob.glob(os.path.join(directorio, patron)):
        os.remove(archivo)

def copiar_archivos_paralelo(archivos, destino):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(shutil.copy2, archivo, os.path.join(destino, f"frame_{i:08d}.png"))
                   for i, archivo in enumerate(archivos)]
        for future in as_completed(futures):
            future.result()

def mover_archivos_paralelo(archivos, destino, offset=0):
    resultados = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(shutil.move, archivo,
                                  os.path.join(destino, f"enhanced_frame_{offset + i:08d}.png"))
                   for i, archivo in enumerate(archivos)]
        for future in as_completed(futures):
            resultados.append(future.result())
    return resultados

def get_gpu_flag(dispositivo_gpu):
    """
    Devuelve el flag correcto para realesrgan-ncnn-vulkan según el dispositivo.
    Prioriza NVIDIA si está disponible.
    """
    logger.info(f"Configurando GPU para procesamiento: {dispositivo_gpu}")
    
    if dispositivo_gpu:
        if isinstance(dispositivo_gpu, str):
            if 'NVIDIA' in dispositivo_gpu.upper():
                logger.info("Usando GPU NVIDIA")
                return ["-g", "gpu"]  # Usar GPU NVIDIA
            elif 'cuda' in dispositivo_gpu.lower():
                logger.info("Usando CUDA GPU")
                return ["-g", "gpu"]  # Usar GPU NVIDIA via CUDA
            elif 'AMD' in dispositivo_gpu.upper():
                logger.info("Usando GPU AMD")
                return ["-g", "gpu"]  # Usar GPU AMD
        else:
            # Si es un objeto GPUInfo
            if hasattr(dispositivo_gpu, 'vendor'):
                if dispositivo_gpu.vendor == 'NVIDIA':
                    logger.info("Usando GPU NVIDIA")
                    return ["-g", "gpu"]
                elif dispositivo_gpu.vendor == 'AMD':
                    logger.info("Usando GPU AMD")
                    return ["-g", "gpu"]
    
    logger.warning("No se detectó GPU compatible, usando CPU")
    return ["-g", "cpu"]

class ProcesadorVideo:
    @staticmethod
    def procesar_lote_frames(args):
        """
        Procesa un lote de frames usando RealesrGAN con manejo de errores y optimización de recursos.
        """
        if len(args) == 8:
            lote_frames, realesrgan_path, models_path, tipo_contenido, escala, indice_lote, tile_size, dispositivo_gpu = args
        elif len(args) == 7:
            lote_frames, realesrgan_path, models_path, tipo_contenido, escala, indice_lote, tile_size = args
            dispositivo_gpu = None
        else:
            lote_frames, realesrgan_path, models_path, tipo_contenido, escala, indice_lote = args
            tile_size = 200
            dispositivo_gpu = None
        
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

            for directorio in [directorio_lote, directorio_salida]:
                limpiar_directorio(directorio)

            copiar_archivos_paralelo(frames_entrada, directorio_lote)

            modelo = get_recommended_model(tipo_contenido)
            
            # Configuración dinámica para GPU
            def calcular_parametros_gpu(dispositivo_gpu):
                # Valores base
                num_hilos_gpu = 2
                gpu_blocks = 8
                num_hilos_cpu = min(Configuracion.MAXIMO_HILOS, os.cpu_count() - 1)

                # Ajustar según el tipo de GPU y VRAM disponible
                if isinstance(dispositivo_gpu, str):
                    if 'NVIDIA' in dispositivo_gpu.upper():
                        # Para GPUs NVIDIA, usar configuración más agresiva
                        from config.gpu_manager import GestorGPU
                        gestor = GestorGPU._get_instance()
                        if gestor._gpus_cache:
                            for gpu in gestor._gpus_cache:
                                if gpu.vendor == 'NVIDIA':
                                    vram_gb = gpu.vram_mb / 1024
                                    # Ajustar parámetros según VRAM
                                    num_hilos_gpu = max(2, min(8, int(vram_gb)))
                                    gpu_blocks = max(8, min(16, int(vram_gb * 2)))
                                    break
                    elif 'AMD' in dispositivo_gpu.upper():
                        # Configuración específica para AMD
                        from config.gpu_manager import GestorGPU
                        gestor = GestorGPU._get_instance()
                        if gestor._gpus_cache:
                            for gpu in gestor._gpus_cache:
                                if gpu.vendor == 'AMD':
                                    vram_gb = gpu.vram_mb / 1024
                                    # AMD tiene mejor rendimiento con más hilos y menos bloques
                                    # debido a su arquitectura GCN/RDNA
                                    num_hilos_gpu = max(4, min(12, int(vram_gb * 1.5)))  # AMD maneja bien más hilos
                                    gpu_blocks = max(4, min(10, int(vram_gb)))  # Menos bloques pero más eficientes
                                    break
                    elif 'Intel' in dispositivo_gpu.upper():
                        # Configuración optimizada para Intel
                        from config.gpu_manager import GestorGPU
                        gestor = GestorGPU._get_instance()
                        if gestor._gpus_cache:
                            for gpu in gestor._gpus_cache:
                                if gpu.vendor == 'Intel':
                                    vram_gb = gpu.vram_mb / 1024
                                    # Intel tiene mejor rendimiento con configuración balanceada
                                    # debido a que comparte recursos con CPU
                                    shared_memory = vram_gb * 0.8  # Consideramos memoria compartida
                                    num_hilos_gpu = max(2, min(6, int(shared_memory)))  # Menos hilos por memoria compartida
                                    gpu_blocks = max(4, min(8, int(shared_memory)))  # Bloques balanceados
                                    # Ajustar hilos CPU para no sobrecargar el sistema
                                    num_hilos_cpu = max(1, min(os.cpu_count() - 2, int(shared_memory)))
                                    break
                return num_hilos_gpu, gpu_blocks, num_hilos_cpu

            # Obtener configuración optimizada
            gpu_flags = get_gpu_flag(dispositivo_gpu)
            num_hilos_gpu, gpu_blocks, num_hilos_cpu = calcular_parametros_gpu(dispositivo_gpu)
            
            comando = [
                realesrgan_path,
                "-i", directorio_lote,
                "-o", directorio_salida,
                "-n", modelo,
                "-m", models_path,
                "-s", str(escala),
                "-f", "png",
                "-t", str(tile_size),
                "-j", f"{num_hilos_gpu}:{gpu_blocks}:{num_hilos_cpu}",  # gpu_threads:gpu_blocks:cpu_threads
            ]
            comando.extend(gpu_flags)

            logger.info(f"Configuración GPU optimizada:")
            logger.info(f"GPU threads: {num_hilos_gpu}, GPU blocks: {gpu_blocks}, CPU threads: {num_hilos_cpu}")
            logger.info(f"Comando completo: {' '.join(comando)}")
            
            resultado = subprocess.run(
                comando,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            if resultado.returncode == 0:
                frames_mejorados = glob.glob(os.path.join(directorio_salida, "*.png"))
                if frames_mejorados:
                    frames_procesados = mover_archivos_paralelo(
                        frames_mejorados,
                        directorio_base,
                        indice_lote * len(lote_frames)
                    )
                    logger.info(f"Lote {indice_lote} completado: {len(frames_mejorados)} frames procesados.")
            else:
                logger.error(f"Error en el procesamiento del lote {indice_lote}: {resultado.stderr}")

        except Exception as e:
            logger.error(f"Error procesando lote {indice_lote}: {str(e)}")
        finally:
            # Limpiar directorios temporales
            for dir_temp in [directorio_lote, directorio_salida]:
                try:
                    shutil.rmtree(dir_temp, ignore_errors=True)
                except Exception as e:
                    logger.error(f"Error limpiando directorio temporal {dir_temp}: {str(e)}")

        return frames_procesados

    @staticmethod
    def obtener_codec_optimal(tipo_gpu=None):
        # --- NUEVO: tipo_gpu es obligatorio, no volver a detectar dentro ---
        if tipo_gpu is None:
            tipo_gpu = "NVIDIA"  # Por defecto, priorizar NVIDIA si no se pasa
        parametros_base = {
            'hilos': Configuracion.MAXIMO_HILOS,
            'crf': '18',
            'parametros_extra': {
                'profile:v': 'high',
                'level': '5.2',
                'pix_fmt': 'rgb24',  # SOLO RGB
                'color_primaries': 'bt709',
                'color_trc': 'bt709',
                'colorspace': 'bt709',
                'color_range': 'tv',
                'tune': 'film',
                'qmin': '10',
                'qmax': '51',
                'qcomp': '0.6'
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