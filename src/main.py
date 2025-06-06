# src/main.py
from src.controllers.manager import Manager
from src.strategies.force import BruteForce
from src.strategies.geometry import Geometric
from src.strategies.q_nodes import QNodes
import time

def iniciar():
    """Punto de entrada principal para comparación de estrategias."""
    estado_inicial = "1000000000"  # Solo A=1, resto=0
    condiciones =    "1111111111"  # Solo A,B,C condicionadas
    alcance =        "1111111110"  # ABCDEFGHI en futuro (excluye J)
    mecanismo =      "0101010101"  # BDFHJ en presente (posiciones 1,3,5,7,9)

    gestor_sistema = Manager(estado_inicial)

    # BruteForce
    inicio = time.time()
    analizador_bf =QNodes(gestor_sistema)
    resultado_bf = analizador_bf.aplicar_estrategia(condiciones, alcance, mecanismo)
    tiempo_bf = time.time() - inicio
    
    print(resultado_bf)

    # GeometricSIA
    inicio = time.time()
    analizador_geo = Geometric(gestor_sistema)
    resultado_geo = analizador_geo.aplicar_estrategia(condiciones, alcance, mecanismo)
    tiempo_geo = time.time() - inicio
    
    print(resultado_geo)

if __name__ == "__main__":
    iniciar()
