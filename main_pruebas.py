#main.py
from ui_pruebas import AplicacionMejoradorVideo
import logging

if __name__ == "__main__":
    logging.info('Aplicación iniciada')

    app = AplicacionMejoradorVideo()
    app.mainloop()
    logging.info('Aplicación finalizada')
