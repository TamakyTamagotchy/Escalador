import sys
import os

# Asegura rutas absolutas para imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from ui_pruebas import AplicacionMejoradorVideo

if __name__ == "__main__":
    app = AplicacionMejoradorVideo()
    app.mainloop()
