from asyncio.log import logger
from typing import Dict, Any
from tkinter import messagebox
from Process.video_processor import ProcesadorVideo
from config.gpu_manager import GestorGPU


def finalizar_mejora(archivo_salida, barra_progreso):
    """Finaliza el proceso de mejora"""
    try:
        messagebox.showinfo(
            "Proceso completado",
            f"Video mejorado guardado en:\n{archivo_salida}"
        )
        barra_progreso.set(1.0)
    except Exception as e:
        logger.error(f"Error al finalizar: {e}")

def informacion_formato_codec(configuraciones_codec: Dict[str, Any]) -> str:
    """Formatea la información del codec de manera legible"""
    return (f"{configuraciones_codec['codec']} "
            f"(preset: {configuraciones_codec['preset']}, "
            f"bitrate: {configuraciones_codec.get('bitrate', 'auto')})")

# Inicialización y logging
try:
    tipo_gpu = GestorGPU.verificar_compatibilidad_gpu()
    configuraciones_codec = ProcesadorVideo.obtener_codec_optimal(tipo_gpu)
except Exception as e:
    logger.error(f"Error al inicializar configuraciones: {e}")