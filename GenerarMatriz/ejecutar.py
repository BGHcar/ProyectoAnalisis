import os
import sys

# Añadir el directorio actual al path para poder importar los módulos
sys.path.insert(0, os.path.abspath('.'))

from src.main import main

if __name__ == "__main__":
    main()