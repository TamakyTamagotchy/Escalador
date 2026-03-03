# main.py
import sys
import os
import logging

# Establecer el directorio de trabajo al directorio del proyecto/ejecutable
# para que todas las rutas relativas (bin/, config/, etc.) funcionen correctamente
if getattr(sys, 'frozen', False):
    # Ejecutando como .exe compilado (cx_Freeze)
    _base_dir = os.path.dirname(sys.executable)
else:
    # Ejecutando como script .py
    _base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_base_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('aplicacion.log', encoding='utf-8', mode='a')
    ]
)

logger = logging.getLogger(__name__)

from ui_pruebas import AplicacionMejoradorVideo

if __name__ == "__main__":
    logger.info('Aplicacion iniciada')
    app = AplicacionMejoradorVideo()
    app.mainloop()
    logger.info('Aplicacion finalizada')
