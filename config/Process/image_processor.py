import logging
import cv2
import os
import shutil
import subprocess
import gc
import torch
import numpy as np
from PIL import Image
from typing import Callable, Optional

logger = logging.getLogger(__name__)

def validate_image(image_path: str) -> bool:
    try:
        if not os.path.exists(image_path):
            logger.error(f"Archivo no existe: {image_path}")
            return False

        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            logger.error(f"No se pudo leer la imagen: {image_path}")
            return False

        height, width = img.shape[:2]
        if height == 0 or width == 0:
            logger.error(f"Imagen con dimensiones inválidas: {image_path}")
            return False

        if len(img.shape) not in [2, 3, 4]:  # Modificado para aceptar imágenes RGBA
            logger.error(f"Formato de imagen no soportado: {image_path}")
            return False

        return True
    except Exception as e:
        logger.error(f"Error validando imagen {image_path}: {e}")
        return False

def detectar_indice_gpu_nvidia():
    """
    Devuelve el índice de la GPU NVIDIA si existe, si no, devuelve 0.
    Si no hay GPU NVIDIA, devuelve -1 (CPU).
    """
    try:
        # Ejecuta el binario realesrgan-ncnn-vulkan con --list-gpu para obtener la lista de GPUs
        # Si no existe ese parámetro, se puede usar torch o simplemente asumir 0
        # Aquí usamos torch si está disponible
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                if "NVIDIA" in name.upper():
                    return i
            return 0  # Si no hay NVIDIA, usar la primera GPU
        else:
            return -1  # CPU
    except Exception:
        return 0  # Fallback


def _ejecutar_realesrgan(ruta_realesrgan, archivo_entrada, archivo_salida, modelo, ruta_modelos, escala, indice_g):
    comando = [
        os.path.normpath(ruta_realesrgan),
        "-i", os.path.normpath(archivo_entrada),
        "-o", os.path.normpath(archivo_salida),
        "-n", modelo,
        "-m", os.path.normpath(ruta_modelos),
        "-s", str(escala),
        "-g", str(indice_g),
        "-t", "0",
        "-j", "1:2:2"
    ]
    logger.info(f"Ejecutando comando: {' '.join(comando)}")
    proceso = subprocess.Popen(
        comando,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=os.path.dirname(ruta_realesrgan)
    )
    salida, error = proceso.communicate()
    if proceso.returncode != 0:
        raise subprocess.CalledProcessError(
            proceso.returncode,
            comando,
            output=salida,
            stderr=error
        )
def _validar_rutas_existentes(rutas_nombres):
    for path, nombre in rutas_nombres:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{nombre} no encontrado: {path}")

def _procesar_por_extension(ext, archivo_entrada, archivo_salida, ruta_realesrgan, ruta_modelos, modelo, escala, indice_g, callback_progreso):
    if ext == ".gif":
        return _procesar_gif(archivo_entrada, archivo_salida, ruta_realesrgan, ruta_modelos, modelo, escala, indice_g, callback_progreso)
    return _procesar_imagen(archivo_entrada, archivo_salida, ruta_realesrgan, ruta_modelos, modelo, escala, indice_g, callback_progreso)

def _procesar_gif(archivo_entrada, archivo_salida, ruta_realesrgan, ruta_modelos, modelo, escala, indice_g, callback_progreso):
    ret, frames = cv2.imreadmulti(archivo_entrada, [], cv2.IMREAD_UNCHANGED)
    if not ret or not frames:
        raise ValueError("No se pudo leer el GIF de entrada")
    
    directorio_temp = os.path.join(os.path.dirname(archivo_salida), "temp_processing")
    os.makedirs(directorio_temp, exist_ok=True)
    
    # Obtener información del GIF original para preservar la duración
    gif_original = Image.open(archivo_entrada)
    durations = []
    try:
        while True:
            durations.append(gif_original.info.get('duration', 100))
            gif_original.seek(gif_original.tell() + 1)
    except EOFError:
        pass
    gif_original.close()
    
    frames_mejorados_pil = []
    total = len(frames)
    
    for idx, frame in enumerate(frames):
        archivo_temp = os.path.join(directorio_temp, f"input_{idx}.png")
        cv2.imwrite(archivo_temp, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        archivo_salida_frame = os.path.join(directorio_temp, f"output_{idx}.png")
        _ejecutar_realesrgan(ruta_realesrgan, archivo_temp, archivo_salida_frame, modelo, ruta_modelos, escala, indice_g)
        
        img_mejorado = cv2.imread(archivo_salida_frame, cv2.IMREAD_UNCHANGED)
        if img_mejorado is None:
            raise ValueError(f"El frame de salida {archivo_salida_frame} está corrupto o vacío")
        
        # Convertir de BGR a RGB para PIL
        if len(img_mejorado.shape) == 3:
            img_rgb = cv2.cvtColor(img_mejorado, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_mejorado
            
        frame_pil = Image.fromarray(img_rgb)
        frames_mejorados_pil.append(frame_pil)
        
        if callback_progreso:
            callback_progreso((idx+1)/total)
    
    # Guardar como GIF animado usando PIL
    if frames_mejorados_pil:
        # Asegurar que todas las duraciones estén disponibles
        if len(durations) == 1:
            durations = durations * len(frames_mejorados_pil)
        elif len(durations) < len(frames_mejorados_pil):
            durations.extend([100] * (len(frames_mejorados_pil) - len(durations)))
        
        frames_mejorados_pil[0].save(
            archivo_salida,
            save_all=True,
            append_images=frames_mejorados_pil[1:],
            duration=durations,
            loop=0,
            format='GIF'
        )
        logger.info(f"GIF animado mejorado guardado exitosamente en: {archivo_salida}")
    else:
        raise ValueError("No se generaron frames mejorados")
    
    shutil.rmtree(directorio_temp, ignore_errors=True)
    return True

def _procesar_imagen(archivo_entrada, archivo_salida, ruta_realesrgan, ruta_modelos, modelo, escala, indice_g, callback_progreso):
    directorio_temp = os.path.join(os.path.dirname(archivo_salida), "temp_processing")
    os.makedirs(directorio_temp, exist_ok=True)
    archivo_temp = os.path.join(directorio_temp, "input.png")
    img = cv2.imread(archivo_entrada, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("No se pudo leer la imagen de entrada")
    cv2.imwrite(archivo_temp, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    _ejecutar_realesrgan(ruta_realesrgan, archivo_temp, archivo_salida, modelo, ruta_modelos, escala, indice_g)
    if not os.path.exists(archivo_salida):
        raise FileNotFoundError(f"No se generó el archivo de salida: {archivo_salida}")
    img_salida = cv2.imread(archivo_salida, cv2.IMREAD_UNCHANGED)
    if img_salida is None:
        raise ValueError("La imagen de salida está corrupta o vacía")
    # Fusionar if anidados
    if len(img.shape) == 4 and img.shape[3] == 4 and len(img_salida.shape) == 3:
        alpha = cv2.resize(img[:,:,3], 
                        (img_salida.shape[1], img_salida.shape[0]), 
                        interpolation=cv2.INTER_LANCZOS4)
        img_salida = cv2.cvtColor(img_salida, cv2.COLOR_BGR2BGRA)
        img_salida[:,:,3] = alpha
    cv2.imwrite(archivo_salida, img_salida)
    if callback_progreso:
        callback_progreso(1.0)
    logger.info(f"Imagen mejorada guardada exitosamente en: {archivo_salida}")
    shutil.rmtree(directorio_temp, ignore_errors=True)
    return True

def mejorar_imagen(archivo_entrada, archivo_salida, ruta_realesrgan, ruta_modelos: str, 
    escala: int = 4, 
    modelo: str = "realesrgan-x4plus", 
    usar_cpu: bool = False,
    indice_gpu: int = 0,
    callback_progreso: Optional[Callable[[float], None]] = None,
    max_intentos: int = 2
) -> bool:
    archivo_entrada = os.path.abspath(archivo_entrada)
    archivo_salida = os.path.abspath(archivo_salida)
    ruta_realesrgan = os.path.abspath(ruta_realesrgan)
    ruta_modelos = os.path.abspath(ruta_modelos)

    if not validate_image(archivo_entrada):
        logger.error(f"Imagen inválida: {archivo_entrada}")
        return False

    indice_g = 0 if (usar_cpu or not torch.cuda.is_available()) else indice_gpu
    rutas_nombres = [
        (ruta_realesrgan, "Ejecutable Real-ESRGAN"),
        (ruta_modelos, "Directorio de modelos"),
        (archivo_entrada, "Archivo de entrada")
    ]
    ext = os.path.splitext(archivo_entrada)[1].lower()
    intentos = 0
    resultado = None
    while intentos < max_intentos:
        try:
            _validar_rutas_existentes(rutas_nombres)
            resultado = _procesar_por_extension(ext, archivo_entrada, archivo_salida, ruta_realesrgan, ruta_modelos, modelo, escala, indice_g, callback_progreso)
            break
        except subprocess.CalledProcessError as e:
            logger.error(f"Error en el proceso: {str(e)}")
            logger.error(f"Salida de error: {e.stderr}")
            intentos += 1
            usar_cpu = True
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Error esperado: {str(e)}")
            intentos += 1
        except Exception as e:
            logger.error(f"Error inesperado: {str(e)}")
            intentos += 1
        finally:
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"Error al limpiar caché de GPU: {str(e)}")
    if resultado is not None:
        return resultado
    logger.error(f"Procesamiento fallido después de {max_intentos} intentos para {archivo_entrada}")
    return False