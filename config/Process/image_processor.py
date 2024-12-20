import logging
import cv2
import os
import shutil
import subprocess
import gc
import torch
from typing import Callable, Optional
import pathlib

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
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

        if len(img.shape) not in [2, 3]:
            logger.error(f"Formato de imagen no soportado: {image_path}")
            return False

        return True
    except Exception as e:
        logger.error(f"Error validando imagen {image_path}: {e}")
        return False

def mejorar_imagen(archivo_entrada, archivo_salida, ruta_realesrgan, ruta_modelos: str, 
    escala: int = 4, 
    modelo: str = "realesrgan-x4plus", 
    usar_cpu: bool = False,
    callback_progreso: Optional[Callable[[float], None]] = None,
    max_intentos: int = 2
) -> bool:
    """
    Mejora una imagen individual con validaciones y reintentos más robustos.
    """
    if not validate_image(archivo_entrada):
        logger.error(f"Imagen inválida: {archivo_entrada}")
        return False

    intentos = 0
    while intentos < max_intentos:
        try:
            # Asegurar rutas absolutas
            paths = map(pathlib.Path, [
                ruta_realesrgan, ruta_modelos, 
                archivo_entrada, archivo_salida
            ])
            ruta_realesrgan, ruta_modelos, archivo_entrada, archivo_salida = [
                str(path.resolve()) for path in paths
            ]

            for path, nombre in [
                (ruta_realesrgan, "Ejecutable Real-ESRGAN"),
                (ruta_modelos, "Directorio de modelos"),
                (archivo_entrada, "Archivo de entrada")
            ]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{nombre} no encontrado: {path}")

            # Crear directorio temporal para procesamiento
            directorio_temp = os.path.join(os.path.dirname(archivo_salida), "temp_processing")
            os.makedirs(directorio_temp, exist_ok=True)

            archivo_temp = os.path.join(directorio_temp, "input.png")
            
            # Convertir imagen de entrada a PNG para asegurar compatibilidad
            img = cv2.imread(archivo_entrada)
            if img is None:
                raise ValueError("No se pudo leer la imagen de entrada")
            cv2.imwrite(archivo_temp, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            # Construir comando base
            comando = [
                ruta_realesrgan,
                "-i", archivo_temp,
                "-o", archivo_salida,
                "-n", modelo,
                "-m", ruta_modelos,
                "-s", str(escala),
                "-g", "-1" if usar_cpu else "0"
            ]

            logger.info(f"Ejecutando comando: {' '.join(comando)}")

            entorno = os.environ.copy()
            entorno["PATH"] = f"{os.path.dirname(ruta_realesrgan)};{entorno['PATH']}"

            # Ejecutar proceso
            proceso = subprocess.Popen(
                comando,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                env=entorno,
                cwd=os.path.dirname(ruta_realesrgan)
            )

            salida, error = proceso.communicate()

            if proceso.returncode != 0:
                logger.error(f"Error en el comando: {error}")
                raise subprocess.CalledProcessError(
                    proceso.returncode,
                    comando,
                    output=salida,
                    stderr=error
                )

            # Verificar archivo de salida
            if not os.path.exists(archivo_salida):
                raise FileNotFoundError(f"No se generó el archivo de salida: {archivo_salida}")

            # Verificar que la imagen de salida no esté corrupta
            img_salida = cv2.imread(archivo_salida)
            if img_salida is None:
                raise ValueError("La imagen de salida está corrupta o vacía")

            if callback_progreso:
                callback_progreso(1.0)

            return True

        except Exception as e:
            logger.error(f"Intento {intentos + 1} fallido: {str(e)}")
            intentos += 1
            usar_cpu = True

        finally:
            # Limpiar recursos
            if 'directorio_temp' in locals():
                shutil.rmtree(directorio_temp, ignore_errors=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.error(f"Procesamiento fallido para {archivo_entrada}")
    return False