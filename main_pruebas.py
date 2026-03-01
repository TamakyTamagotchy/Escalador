# main.py
import logging

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
