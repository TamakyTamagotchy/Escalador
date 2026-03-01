#main.py
from ui_pruebas import AplicacionMejoradorVideo
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('aplicacion.log', encoding='utf-8', mode='a')
    ]
)

if __name__ == "__main__":
    logging.info('Aplicación iniciada')

    app = AplicacionMejoradorVideo()
    app.mainloop()
    logging.info('Aplicación finalizada')
