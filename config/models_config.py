# models_config.py
import logging, os, re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ModeloInfo:
    id: str
    nombre_display: str
    descripcion: str
    escalas: List[int]
    escala_predeterminada: int
    tipos_contenido: List[str] = field(default_factory=list)
    denoise_niveles: Optional[List[int]] = None


@dataclass
class UpscalerInfo:
    id: str
    nombre_display: str
    ejecutable: str
    modelos: Dict[str, ModeloInfo] = field(default_factory=dict)
    soporta_directorio: bool = True


# --- Metadatos de modelos conocidos (nombre_display, descripcion) ---
# Si el modelo detectado esta aqui, usa descripcion bonita.
# Si no, se genera automaticamente desde el nombre del archivo.

MODELOS_CONOCIDOS: Dict[str, tuple] = {
    "realesr-animevideov3": ("Anime Video v3", "Optimizado para video y frames de anime"),
    "realesrgan-x4plus": ("General x4+", "Fotos, paisajes y contenido general"),
    "realesrgan-x4plus-anime": ("Anime x4+", "Imagenes anime estilo ilustracion"),
    "realesr-general-x4v3": ("General v3 (Ligero)", "Modelo ligero para contenido general, rapido"),
    "realesr-general-wdn-x4v3": ("General Denoise v3", "Contenido general con reduccion de ruido"),
    "RealESRGAN_x2plus": ("General x2+", "Fotos y contenido general a escala 2x"),
    "ESRGAN_SRx4_DF2KOST_official": ("ESRGAN DF2KOST", "Modelo ESRGAN oficial entrenado en DF2K+OST"),
    "realesrnet-x4plus": ("ESRNet x4+", "Modelo base sin GAN, resultados mas suaves"),
    "4x-UltraSharp": ("UltraSharp 4x", "Modelo comunitario para maxima nitidez"),
    "4x-AnimeSharp": ("AnimeSharp 4x", "Modelo comunitario para anime con nitidez"),
    "2x_Compact_Pretrain": ("Compact 2x", "Modelo compacto rapido a 2x"),
    "4x_NMKD-Superscale-SP_178000_G": ("NMKD Superscale 4x", "Super-escalado de alta calidad"),
}


def detectar_modelos_ncnn(directorio_modelos: str) -> Dict[str, ModeloInfo]:
    """
    Escanea un directorio para detectar modelos NCNN por pares .param+.bin.

    Patron de nombres:
      - Modelo simple: {nombre}.param + {nombre}.bin
      - Modelo multi-escala: {nombre}-x{escala}.param + {nombre}-x{escala}.bin

    Retorna un dict de modelo_id -> ModeloInfo con las escalas detectadas.
    """
    modelos = {}
    models_dir = Path(directorio_modelos)

    if not models_dir.is_dir():
        return modelos

    # Paso 1: Agrupar archivos .param por nombre base de modelo
    modelos_raw: Dict[str, set] = {}
    for param_file in sorted(models_dir.glob("*.param")):
        bin_file = param_file.with_suffix(".bin")
        if not bin_file.exists():
            logger.debug(f"Ignorando {param_file.name}: falta archivo .bin")
            continue

        name = param_file.stem

        # Detectar si tiene sufijo -x{N} (multi-escala)
        match = re.match(r'^(.+)-x(\d+)$', name)
        if match:
            base_name = match.group(1)
            scale = int(match.group(2))
        else:
            base_name = name
            # Inferir escala del nombre (ej: "realesrgan-x4plus" -> 4)
            scale_match = re.search(r'x(\d+)', name)
            scale = int(scale_match.group(1)) if scale_match else 4

        if base_name not in modelos_raw:
            modelos_raw[base_name] = set()
        modelos_raw[base_name].add(scale)

    # Paso 2: Crear ModeloInfo para cada modelo detectado
    for base_name, escalas in modelos_raw.items():
        escalas_sorted = sorted(escalas)
        known = MODELOS_CONOCIDOS.get(base_name)

        modelos[base_name] = ModeloInfo(
            id=base_name,
            nombre_display=known[0] if known else base_name,
            descripcion=known[1] if known else f"Modelo detectado: {base_name}",
            escalas=escalas_sorted,
            escala_predeterminada=escalas_sorted[0],
        )
        logger.info(f"Modelo NCNN detectado: {base_name} (escalas: {escalas_sorted})")

    return modelos


# --- Registro de upscalers ---
# Los modelos de realesrgan se dejan vacios aqui porque se llenan
# dinamicamente en prueba_config.detectar_upscalers_instalados()
# escaneando los archivos .param+.bin en bin/models/.

UPSCALERS: Dict[str, UpscalerInfo] = {
    "realesrgan": UpscalerInfo(
        id="realesrgan",
        nombre_display="Real-ESRGAN",
        ejecutable="realesrgan-ncnn-vulkan.exe",
        soporta_directorio=True,
        modelos={},  # Se llena dinamicamente por auto-deteccion
    ),
    "realcugan": UpscalerInfo(
        id="realcugan",
        nombre_display="Real-CUGAN",
        ejecutable="realcugan-ncnn-vulkan.exe",
        soporta_directorio=True,
        modelos={
            "models-se": ModeloInfo(
                id="models-se",
                nombre_display="SE (Rapido)",
                descripcion="Modelo rapido para anime con buena calidad",
                escalas=[2, 3, 4],
                escala_predeterminada=2,
                tipos_contenido=["anime"],
                denoise_niveles=[-1, 0, 1, 2, 3],
            ),
            "models-pro": ModeloInfo(
                id="models-pro",
                nombre_display="Pro (Calidad)",
                descripcion="Mayor calidad, mas lento",
                escalas=[2, 3],
                escala_predeterminada=2,
                tipos_contenido=["anime"],
                denoise_niveles=[-1, 0, 3],
            ),
            "models-nose": ModeloInfo(
                id="models-nose",
                nombre_display="No-SE",
                descripcion="Sin skip-connection, diferente estilo visual",
                escalas=[2, 3, 4],
                escala_predeterminada=2,
                tipos_contenido=["anime"],
            ),
        },
    ),
    "waifu2x": UpscalerInfo(
        id="waifu2x",
        nombre_display="Waifu2x",
        ejecutable="waifu2x-ncnn-vulkan.exe",
        soporta_directorio=True,
        modelos={
            "models-cunet": ModeloInfo(
                id="models-cunet",
                nombre_display="CUNet",
                descripcion="Modelo principal para anime e ilustraciones",
                escalas=[1, 2],
                escala_predeterminada=2,
                tipos_contenido=["anime", "dibujos animados"],
                denoise_niveles=[-1, 0, 1, 2, 3],
            ),
            "models-upconv_7_anime_style_art_rgb": ModeloInfo(
                id="models-upconv_7_anime_style_art_rgb",
                nombre_display="Upconv7 Anime",
                descripcion="Modelo clasico para arte anime RGB",
                escalas=[1, 2],
                escala_predeterminada=2,
                tipos_contenido=["anime"],
                denoise_niveles=[-1, 0, 1, 2, 3],
            ),
        },
    ),
    "realsr": UpscalerInfo(
        id="realsr",
        nombre_display="RealSR",
        ejecutable="realsr-ncnn-vulkan.exe",
        soporta_directorio=True,
        modelos={
            "models-DF2K": ModeloInfo(
                id="models-DF2K",
                nombre_display="DF2K",
                descripcion="Fotos realistas de alta calidad",
                escalas=[4],
                escala_predeterminada=4,
                tipos_contenido=["fotografias", "paisajes"],
            ),
            "models-DF2K_JPEG": ModeloInfo(
                id="models-DF2K_JPEG",
                nombre_display="DF2K JPEG",
                descripcion="Fotos JPEG con artefactos de compresion",
                escalas=[4],
                escala_predeterminada=4,
                tipos_contenido=["fotografias", "jpeg"],
            ),
        },
    ),
}


def construir_comando(
    upscaler_id: str,
    modelo_id: str,
    ruta_exe: str,
    entrada: str,
    salida: str,
    ruta_modelos: str,
    escala: int,
    gpu_id: int = 0,
    denoise: int = -1,
    formato: str = "png",
    tile: int = 0,
    hilos: Optional[str] = None,
) -> List[str]:
    """Construye la lista de argumentos CLI segun el tipo de upscaler."""
    # Resolver rutas a absolutas para que funcionen correctamente
    # independientemente del CWD del subproceso
    ruta_exe = os.path.abspath(ruta_exe)
    entrada = os.path.abspath(entrada)
    salida = os.path.abspath(salida)
    ruta_modelos = os.path.abspath(ruta_modelos)

    cmd = [ruta_exe, "-i", entrada, "-o", salida, "-s", str(escala)]

    if upscaler_id == "realesrgan":
        cmd += ["-n", modelo_id, "-m", ruta_modelos]
    elif upscaler_id == "realcugan":
        cmd += ["-m", ruta_modelos]
        if denoise >= 0:
            cmd += ["-n", str(denoise)]
    elif upscaler_id == "waifu2x":
        cmd += ["-n", str(denoise), "-m", ruta_modelos]
    elif upscaler_id == "realsr":
        cmd += ["-m", ruta_modelos]

    cmd += ["-g", str(gpu_id), "-f", formato, "-t", str(tile)]

    if hilos:
        cmd += ["-j", hilos]

    return cmd


def obtener_upscaler(upscaler_id: str) -> Optional[UpscalerInfo]:
    return UPSCALERS.get(upscaler_id)


def obtener_modelo(upscaler_id: str, modelo_id: str) -> Optional[ModeloInfo]:
    upscaler = UPSCALERS.get(upscaler_id)
    if upscaler:
        return upscaler.modelos.get(modelo_id)
    return None
