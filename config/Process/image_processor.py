import logging
import subprocess
import gc
import pathlib
import tempfile
from PIL import Image, ImageFilter  # Se importa ImageFilter para aplicar un suavizado
import torch
from typing import Callable, Optional
import os  # Se importa os para obtener el conteo de CPUs

logger = logging.getLogger(__name__)

def validate_image(image_path: str) -> bool:
    try:
        if not pathlib.Path(image_path).exists():
            logger.error(f"Archivo no existe: {image_path}")
            return False
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        logger.error(f"Error validando imagen {image_path}: {e}")
        return False

def run_command(comando: list, timeout: int = 120) -> subprocess.CompletedProcess:
    logger.info(f"Ejecutando comando: {' '.join(comando)}")
    return subprocess.run(
        comando,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=timeout,
        check=True
    )

def limpiar_tempfiles(*paths):
    for path in paths:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass

def limpiar_recursos():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_gpu_flag(dispositivo_gpu):
    if dispositivo_gpu and dispositivo_gpu.startswith("cuda"):
        return ["-g", "0"]
    elif dispositivo_gpu and dispositivo_gpu.startswith("amd"):
        return ["-g", "0"]
    elif dispositivo_gpu and dispositivo_gpu.startswith("intel"):
        return ["-g", "0"]
    return []

def calcular_parametros_gpu(dispositivo_gpu, modelo):
    num_hilos_gpu = 2
    gpu_blocks = 8
    num_hilos_cpu = min(os.cpu_count() - 1, 4) # Dejar al menos un hilo para el sistema

    if isinstance(dispositivo_gpu, str):
        from config.gpu_manager import GestorGPU
        gestor = GestorGPU._get_instance()
        if 'NVIDIA' in dispositivo_gpu.upper():
            if gestor._gpus_cache:
                for gpu in gestor._gpus_cache:
                    if gpu.vendor == 'NVIDIA':
                        vram_gb = gpu.vram_mb / 1024
                        num_hilos_gpu = max(2, min(8, int(vram_gb)))
                        gpu_blocks = max(8, min(16, int(vram_gb * 2)))
                        break
        elif 'AMD' in dispositivo_gpu.upper():
            if gestor._gpus_cache:
                for gpu in gestor._gpus_cache:
                    if gpu.vendor == 'AMD':
                        vram_gb = gpu.vram_mb / 1024
                        num_hilos_gpu = max(4, min(12, int(vram_gb * 1.5)))
                        gpu_blocks = max(4, min(10, int(vram_gb)))
                        break
        elif 'Intel' in dispositivo_gpu.upper():
            if gestor._gpus_cache:
                for gpu in gestor._gpus_cache:
                    if gpu.vendor == 'Intel':
                        vram_gb = gpu.vram_mb / 1024
                        shared_memory = vram_gb * 0.8
                        num_hilos_gpu = max(2, min(6, int(shared_memory)))
                        gpu_blocks = max(4, min(8, int(shared_memory)))
                        num_hilos_cpu = max(1, min(os.cpu_count() - 2, int(shared_memory)))
                        break

    return num_hilos_gpu, gpu_blocks, num_hilos_cpu

def mejorar_imagen(
    archivo_entrada: str,
    archivo_salida: str,
    ruta_realesrgan: str,
    ruta_modelos: str,
    escala: int = 4,
    modelo: str = "realesrgan-x4plus",
    callback_progreso: Optional[Callable[[float], None]] = None,
    max_intentos: int = 2,
    dispositivo_gpu: str = None
) -> bool:
    """
    Mejora la imagen usando Real-ESRGAN, preservando el fondo (transparencia) cuando corresponda
    y aplicando un filtro suave para obtener un resultado natural.
    """
    if not validate_image(archivo_entrada):
        logger.error(f"Imagen inválida: {archivo_entrada}")
        return False

    # Obtener rutas absolutas con pathlib y validarlas
    rutas = {
        "Real-ESRGAN": pathlib.Path(ruta_realesrgan).resolve(),
        "Modelos": pathlib.Path(ruta_modelos).resolve(),
        "Entrada": pathlib.Path(archivo_entrada).resolve(),
        "Salida": pathlib.Path(archivo_salida).resolve()
    }
    for nombre, path in rutas.items():
        if nombre in ["Real-ESRGAN", "Modelos", "Entrada"] and not path.exists():
            raise FileNotFoundError(f"{nombre} no encontrado: {path}")

    intento = 0
    while intento < max_intentos:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_input:
            temp_input_path = pathlib.Path(temp_input.name)
        try:
            with Image.open(rutas["Entrada"]) as img:
                if img.mode in ("RGBA", "LA") or (img.mode == "P" and 'transparency' in img.info):
                    img = img.convert("RGBA")
                else:
                    img = img.convert("RGB")
                img.save(temp_input_path)

            num_hilos_gpu, gpu_blocks, num_hilos_cpu = calcular_parametros_gpu(dispositivo_gpu, modelo)
            gpu_flag = get_gpu_flag(dispositivo_gpu)
            gpu_flag += ["-j", f"{num_hilos_gpu}:{gpu_blocks}:{num_hilos_cpu}"]

            comando = [
                str(rutas["Real-ESRGAN"]),
                "-i", str(temp_input_path),
                "-o", str(rutas["Salida"]),
                "-n", modelo,
                "-m", str(rutas["Modelos"]),
                "-s", str(escala),
                "-f", "png"
            ] + gpu_flag

            logger.info(f"Configuración optimizada para modelo {modelo}:")
            logger.info(f"GPU threads: {num_hilos_gpu}, GPU blocks: {gpu_blocks}, CPU threads: {num_hilos_cpu}")
            logger.info(f"Comando completo: {' '.join(comando)}")

            resultado = run_command(comando)
            logger.info(f"Salida del comando: {resultado.stdout}")

            if not rutas["Salida"].exists():
                raise FileNotFoundError(f"No se generó el archivo de salida: {rutas['Salida']}")

            with Image.open(rutas["Salida"]) as img_salida:
                img_salida.verify()
                if img_salida.size == (0, 0):
                    raise Exception("Imagen de salida vacía")

            # Abrir imagen de salida para aplicar un filtro suave (suavizado) y reinyectar alfa si es necesario
            with Image.open(rutas["Salida"]) as img_mejorada:
                img_mejorada = img_mejorada.convert("RGBA") if img_mejorada.mode != "RGBA" else img_mejorada
                img_mejorada = img_mejorada.filter(ImageFilter.SMOOTH)
                img_mejorada.save(rutas["Salida"])

            if callback_progreso:
                callback_progreso(1.0)
            limpiar_tempfiles(temp_input_path)
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Error en el comando (intento {intento + 1}): {e.stderr}")
            intento += 1

        except Exception as e:
            logger.error(f"Intento {intento + 1} fallido: {str(e)}")
            intento += 1

        finally:
            limpiar_recursos()
            limpiar_tempfiles(temp_input_path)

    logger.error(f"Procesamiento fallido para {archivo_entrada}")
    return False