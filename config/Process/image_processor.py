# image_processor.py
import logging
import cv2
import os
import shutil
import subprocess
import gc
import torch

logging.basicConfig(level=logging.DEBUG)

def mejorar_imagen(
    archivo_entrada, 
    archivo_salida, 
    ruta_realesrgan, 
    ruta_modelos, 
    escala=4, 
    modelo="realesrgan-x4plus", 
    usar_cpu=False,
    callback_progreso=None
):
    """
    Mejora una imagen individual utilizando Real-ESRGAN con optimizaciones y manejo de errores.

    Args:
        archivo_entrada (str): Ruta del archivo de imagen de entrada
        archivo_salida (str): Ruta para guardar la imagen mejorada
        ruta_realesrgan (str): Ruta al ejecutable de Real-ESRGAN
        ruta_modelos (str): Ruta al directorio de modelos
        escala (int, optional): Factor de escalado. Defaults to 4.
        modelo (str, optional): Modelo de mejora a utilizar. Defaults to "realesrgan-x4plus".
        usar_cpu (bool, optional): Forzar procesamiento con CPU. Defaults to False.
        callback_progreso (callable, optional): Función de callback para progreso. Defaults to None.
    
    Raises:
        FileNotFoundError: Si los archivos o directorios requeridos no existen
        ValueError: Si la imagen no puede ser procesada
    """
    try:
        # Rutas absolutas
        ruta_realesrgan = os.path.abspath(ruta_realesrgan)
        ruta_modelos = os.path.abspath(ruta_modelos)
        archivo_entrada = os.path.abspath(archivo_entrada)
        archivo_salida = os.path.abspath(archivo_salida)

        # Verificación de archivos
        if not os.path.exists(archivo_entrada):
            raise FileNotFoundError(f"Archivo de entrada no encontrado: {archivo_entrada}")

        if not os.path.exists(ruta_realesrgan):
            raise FileNotFoundError(f"Ejecutable Real-ESRGAN no encontrado: {ruta_realesrgan}")

        if not os.path.exists(ruta_modelos):
            raise FileNotFoundError(f"Directorio de modelos no encontrado: {ruta_modelos}")

        # Directorio temporal para procesamiento
        directorio_temp = os.path.join(os.path.dirname(archivo_salida), "temp_processing")
        os.makedirs(directorio_temp, exist_ok=True)

        archivo_temp = os.path.join(directorio_temp, "input.png")

        # Conversión de imagen a PNG
        img = cv2.imread(archivo_entrada)
        if img is None:
            raise ValueError("No se pudo leer la imagen de entrada")
        
        cv2.imwrite(archivo_temp, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        # Construcción del comando
        comando = [
            ruta_realesrgan,
            "-i", archivo_temp,
            "-o", archivo_salida,
            "-n", modelo,
            "-m", ruta_modelos,
            "-s", str(int(escala))
        ]

        # Configurar GPU o CPU
        if usar_cpu:
            comando.extend(["-g", "-1"])  # Forzar CPU
        else:
            comando.extend(["-g", "0"])   # Usar GPU por defecto

        logging.debug(f"Comando Real-ESRGAN: {' '.join(comando)}")

        # Configuración del entorno
        entorno = os.environ.copy()
        entorno["PATH"] = f"{os.path.dirname(ruta_realesrgan)};{entorno['PATH']}"

        # Ejecución del proceso
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
            mensaje_error = f"Error al procesar imagen: {salida.strip()}\n{error.strip()}"
            logging.error(mensaje_error)
            
            # Si no se usa CPU, intentar forzar CPU
            if not usar_cpu:
                logging.info("Intentando procesar con CPU...")
                return mejorar_imagen(
                    archivo_entrada, 
                    archivo_salida, 
                    ruta_realesrgan, 
                    ruta_modelos, 
                    escala, 
                    modelo, 
                    usar_cpu=True,
                    callback_progreso=callback_progreso
                )
            
            raise subprocess.CalledProcessError(
                proceso.returncode,
                comando,
                output=salida,
                stderr=error
            )

        # Verificación del archivo de salida
        if not os.path.exists(archivo_salida):
            raise FileNotFoundError(f"No se generó el archivo de salida: {archivo_salida}")

        img_salida = cv2.imread(archivo_salida)
        if img_salida is None:
            raise ValueError("La imagen de salida está corrupta o vacía")

        # Callback de progreso
        if callback_progreso:
            callback_progreso(1.0)

    except Exception as e:
        logging.error(f"Error al mejorar imagen: {str(e)}")
        raise

    finally:
        # Limpieza de recursos
        shutil.rmtree(directorio_temp, ignore_errors=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def mejorar_lista_imagenes(
    lista_imagenes, 
    directorio_salida, 
    ruta_realesrgan, 
    ruta_modelos, 
    escala=4, 
    modelo="realesrgan-x4plus", 
    usar_cpu=False,
    callback_progreso=None
):
    """
    Mejora múltiples imágenes de manera secuencial.

    Args:
        lista_imagenes (list): Lista de rutas de imágenes de entrada
        directorio_salida (str): Directorio para guardar imágenes mejoradas
        ruta_realesrgan (str): Ruta al ejecutable de Real-ESRGAN
        ruta_modelos (str): Ruta al directorio de modelos
        escala (int, optional): Factor de escalado. Defaults to 4.
        modelo (str, optional): Modelo de mejora. Defaults to "realesrgan-x4plus".
        usar_cpu (bool, optional): Forzar procesamiento con CPU. Defaults to False.
        callback_progreso (callable, optional): Función de callback para progreso. Defaults to None.
    
    Returns:
        list: Lista de rutas de imágenes mejoradas
    """
    os.makedirs(directorio_salida, exist_ok=True)
    imagenes_mejoradas = []
    total_imagenes = len(lista_imagenes)

    for idx, imagen in enumerate(lista_imagenes, 1):
        try:
            nombre_archivo = os.path.basename(imagen)
            archivo_salida = os.path.join(directorio_salida, f"enhanced_{nombre_archivo}")
            
            def progreso_parcial(p):
                if callback_progreso:
                    callback_progreso(idx / total_imagenes)

            mejorar_imagen(
                imagen, 
                archivo_salida, 
                ruta_realesrgan, 
                ruta_modelos, 
                escala, 
                modelo, 
                usar_cpu,
                callback_progreso=progreso_parcial
            )
            
            imagenes_mejoradas.append(archivo_salida)
        
        except Exception as e:
            logging.error(f"Error procesando {imagen}: {e}")
            continue

    return imagenes_mejoradas