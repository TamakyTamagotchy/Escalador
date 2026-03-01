# image_processor.py
import logging
import cv2
import gc
import os
import shutil
import subprocess
from typing import Callable, Optional

from config.models_config import construir_comando

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
            logger.error(f"Imagen con dimensiones invalidas: {image_path}")
            return False

        if len(img.shape) not in [2, 3]:  # 2D = grayscale, 3D = color/RGBA
            logger.error(f"Formato de imagen no soportado: {image_path}")
            return False

        return True
    except Exception as e:
        logger.error(f"Error validando imagen {image_path}: {e}")
        return False


def mejorar_imagen(
    archivo_entrada,
    archivo_salida,
    ruta_exe,
    ruta_modelos: str,
    escala: int = 4,
    upscaler_id: str = "realesrgan",
    modelo_id: str = "realesrgan-x4plus",
    denoise: int = -1,
    usar_cpu: bool = False,
    callback_progreso: Optional[Callable[[float], None]] = None,
    max_intentos: int = 2,
) -> bool:
    """
    Mejora una imagen individual usando un upscaler NCNN-Vulkan.
    Reintenta con CPU si falla en GPU.
    """
    if not validate_image(archivo_entrada):
        logger.error(f"Imagen invalida: {archivo_entrada}")
        return False

    intentos = 0
    while intentos < max_intentos:
        try:
            # Verificar existencia de archivos
            for path, nombre in [
                (ruta_exe, "Ejecutable upscaler"),
                (ruta_modelos, "Directorio de modelos"),
                (archivo_entrada, "Archivo de entrada"),
            ]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{nombre} no encontrado: {path}")

            # Crear directorio temporal para procesamiento
            directorio_temp = os.path.join(
                os.path.dirname(archivo_salida), "temp_processing"
            )
            os.makedirs(directorio_temp, exist_ok=True)

            archivo_temp = os.path.join(directorio_temp, "input.png")

            # Convertir imagen de entrada a PNG para compatibilidad
            img = cv2.imread(archivo_entrada, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("No se pudo leer la imagen de entrada")

            cv2.imwrite(archivo_temp, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            # Construir comando usando el builder centralizado
            gpu_id = -1 if usar_cpu else 0
            comando = construir_comando(
                upscaler_id=upscaler_id,
                modelo_id=modelo_id,
                ruta_exe=ruta_exe,
                entrada=archivo_temp,
                salida=archivo_salida,
                ruta_modelos=ruta_modelos,
                escala=escala,
                gpu_id=gpu_id,
                denoise=denoise,
                hilos="1:2:2",
            )

            logger.info(f"Ejecutando comando: {' '.join(comando)}")

            proceso = subprocess.Popen(
                comando,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                cwd=os.path.dirname(ruta_exe),
                creationflags=subprocess.CREATE_NO_WINDOW,
            )

            salida, error = proceso.communicate()

            if proceso.returncode != 0:
                raise subprocess.CalledProcessError(
                    proceso.returncode, comando, output=salida, stderr=error
                )

            # Verificar que el archivo de salida existe y es valido
            if not os.path.exists(archivo_salida):
                raise FileNotFoundError(
                    f"No se genero el archivo de salida: {archivo_salida}"
                )

            img_salida = cv2.imread(archivo_salida, cv2.IMREAD_UNCHANGED)
            if img_salida is None:
                raise ValueError("La imagen de salida esta corrupta o vacia")

            # Preservar canal alfa si existe (3D con 4 canales = RGBA)
            if len(img.shape) == 3 and img.shape[2] == 4:
                if len(img_salida.shape) == 3 and img_salida.shape[2] == 3:
                    # Redimensionar el canal alfa a las nuevas dimensiones
                    alpha = cv2.resize(
                        img[:, :, 3],
                        (img_salida.shape[1], img_salida.shape[0]),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                    img_salida = cv2.cvtColor(img_salida, cv2.COLOR_BGR2BGRA)
                    img_salida[:, :, 3] = alpha

                cv2.imwrite(archivo_salida, img_salida)

            if callback_progreso:
                callback_progreso(1.0)

            logger.info(f"Imagen mejorada guardada exitosamente en: {archivo_salida}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Error en el proceso (intento {intentos + 1}): {e}")
            logger.error(f"Salida de error: {e.stderr}")
            intentos += 1
            usar_cpu = True  # Reintentar con CPU si falla con GPU
        except Exception as e:
            logger.error(f"Error inesperado (intento {intentos + 1}): {e}")
            intentos += 1

        finally:
            # Limpiar directorio temporal
            if "directorio_temp" in locals():
                shutil.rmtree(directorio_temp, ignore_errors=True)
            gc.collect()

    logger.error(
        f"Procesamiento fallido despues de {max_intentos} intentos para {archivo_entrada}"
    )
    return False
