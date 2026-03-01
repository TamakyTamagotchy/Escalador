import logging, cv2, os, shutil, subprocess, gc, torch
from typing import Callable, Optional

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime=s - %(levelname=s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('aplicacion.log', encoding='utf-8', mode='a')
    ]
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

        if len(img.shape) not in [2, 3, 4]:  # Modificado para aceptar imágenes RGBA
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
            # Asegurar rutas absolutas y verificar existencia
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
            img = cv2.imread(archivo_entrada, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("No se pudo leer la imagen de entrada")
            
            # Guardar imagen temporal con la configuración adecuada
            cv2.imwrite(archivo_temp, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
            # Construir comando base con rutas normalizadas
            comando = [
                os.path.normpath(ruta_realesrgan),
                "-i", os.path.normpath(archivo_temp),
                "-o", os.path.normpath(archivo_salida),
                "-n", modelo,
                "-m", os.path.normpath(ruta_modelos),
                "-s", str(escala),
                "-g", "2",
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

            # Verificar que el archivo de salida existe y es válido
            if not os.path.exists(archivo_salida):
                raise FileNotFoundError(f"No se generó el archivo de salida: {archivo_salida}")

            img_salida = cv2.imread(archivo_salida, cv2.IMREAD_UNCHANGED)
            if img_salida is None:
                raise ValueError("La imagen de salida está corrupta o vacía")
            
            # Preservar canal alfa si existe
            if len(img.shape) == 4 and img.shape[3] == 4:
                if len(img_salida.shape) == 3:
                    # Redimensionar el canal alfa a las nuevas dimensiones
                    alpha = cv2.resize(img[:,:,3], 
                                    (img_salida.shape[1], img_salida.shape[0]), 
                                    interpolation=cv2.INTER_LANCZOS4)
                    img_salida = cv2.cvtColor(img_salida, cv2.COLOR_BGR2BGRA)
                    img_salida[:,:,3] = alpha
                
            cv2.imwrite(archivo_salida, img_salida)

            if callback_progreso:
                callback_progreso(1.0)

            logger.info(f"Imagen mejorada guardada exitosamente en: {archivo_salida}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Error en el proceso: {str(e)}")
            logger.error(f"Salida de error: {e.stderr}")
            intentos += 1
            usar_cpu = True  # Intentar con CPU si falla con GPU
        except Exception as e:
            logger.error(f"Error inesperado: {str(e)}")
            intentos += 1

        finally:
            # Limpiar recursos
            if 'directorio_temp' in locals():
                try:
                    shutil.rmtree(directorio_temp, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"Error al limpiar directorio temporal: {str(e)}")
            
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"Error al limpiar caché de GPU: {str(e)}")

    logger.error(f"Procesamiento fallido después de {max_intentos} intentos para {archivo_entrada}")
    return False