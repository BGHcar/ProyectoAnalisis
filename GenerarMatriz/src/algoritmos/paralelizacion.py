import numpy as np
from multiprocessing import Pool
from .alcanzabilidad import matriz_alcanzabilidad

def _calcular_fila_alcanzabilidad(args):
    """
    Función auxiliar para calcular una fila de la matriz de alcanzabilidad
    
    Args:
        args: Tupla (fila, matriz)
        
    Returns:
        tuple: (índice de fila, fila de la matriz de alcanzabilidad)
    """
    idx, matriz = args
    n = len(matriz)
    fila_resultado = np.zeros(n, dtype=int)
    
    # Realizar BFS desde el nodo idx
    visitados = [False] * n
    cola = [idx]
    visitados[idx] = True
    
    while cola:
        nodo = cola.pop(0)
        fila_resultado[nodo] = 1
        
        for vecino in range(n):
            if matriz[nodo, vecino] > 0 and not visitados[vecino]:
                visitados[vecino] = True
                cola.append(vecino)
    
    return idx, fila_resultado

def procesar_paralelo(matriz, num_procesos):
    """
    Calcula la matriz de alcanzabilidad en paralelo
    
    Args:
        matriz: Matriz de adyacencia
        num_procesos: Número de procesos a utilizar
        
    Returns:
        numpy.ndarray: Matriz de alcanzabilidad
    """
    n = len(matriz)
    resultado = np.zeros((n, n), dtype=int)
    
    # Preparar argumentos para cada proceso
    args = [(i, matriz) for i in range(n)]
    
    # Ejecutar en paralelo
    with Pool(processes=num_procesos) as pool:
        resultados = pool.map(_calcular_fila_alcanzabilidad, args)
    
    # Ensamblar resultados
    for idx, fila in resultados:
        resultado[idx] = fila
    
    return resultado