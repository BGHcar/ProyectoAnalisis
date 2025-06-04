import numpy as np
import random
from collections import deque

def _camino_existe(matriz, origen, destino, max_pasos=None):
    """
    Verifica si existe un camino entre el origen y el destino
    
    Args:
        matriz: Matriz de adyacencia
        origen: Nodo de origen
        destino: Nodo de destino
        max_pasos: Número máximo de pasos (None para no limitar)
        
    Returns:
        bool: True si existe un camino, False en caso contrario
    """
    n = len(matriz)
    visitados = [False] * n
    
    cola = deque([(origen, 0)])  # (nodo, distancia)
    visitados[origen] = True
    
    while cola:
        nodo, distancia = cola.popleft()
        
        if nodo == destino:
            return True
        
        if max_pasos is not None and distancia >= max_pasos:
            continue
        
        for vecino in range(n):
            if matriz[nodo, vecino] > 0 and not visitados[vecino]:
                visitados[vecino] = True
                cola.append((vecino, distancia + 1))
    
    return False

def estimar_conexiones(matriz, num_muestras):
    """
    Estima el número total de conexiones en la matriz usando Monte Carlo
    
    Args:
        matriz: Matriz de adyacencia
        num_muestras: Número de pares (origen, destino) a muestrear
        
    Returns:
        tuple: (estimación de conexiones, lista de pares muestreados)
    """
    n = len(matriz)
    total_posibles = n * n  # Total de posibles conexiones
    
    # Seleccionar pares aleatorios de nodos
    pares_muestreados = set()
    while len(pares_muestreados) < min(num_muestras, total_posibles):
        origen = random.randint(0, n-1)
        destino = random.randint(0, n-1)
        if (origen != destino):  # Evitar auto-conexiones
            pares_muestreados.add((origen, destino))
    
    # Comprobar si existe un camino entre cada par
    conexiones_encontradas = 0
    pares_con_camino = []
    
    for origen, destino in pares_muestreados:
        if _camino_existe(matriz, origen, destino):
            conexiones_encontradas += 1
            pares_con_camino.append((origen, destino))
    
    # Estimar el total basado en la proporción
    proporcion = conexiones_encontradas / len(pares_muestreados)
    estimacion = proporcion * (n * (n-1))  # Excluimos auto-conexiones
    
    return estimacion, pares_con_camino