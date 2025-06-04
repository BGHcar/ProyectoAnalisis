import numpy as np
from collections import deque
import scipy.sparse as sp

def nodos_alcanzables(matriz, origen):
    """
    Encuentra todos los nodos alcanzables desde un nodo origen
    utilizando BFS (Breadth-First Search).
    
    Funciona tanto con matrices densas como sparse.
    
    Args:
        matriz: Matriz de adyacencia (puede ser numpy.ndarray o scipy.sparse)
        origen: Nodo de origen
        
    Returns:
        list: Lista de nodos alcanzables desde origen
    """
    if sp.issparse(matriz):
        return _nodos_alcanzables_sparse(matriz, origen)
    else:
        return _nodos_alcanzables_denso(matriz, origen)

def _nodos_alcanzables_denso(matriz, origen):
    """
    Implementación para matrices densas
    """
    n = len(matriz)
    visitados = [False] * n
    alcanzables = []
    
    cola = deque([origen])
    visitados[origen] = True
    
    while cola:
        nodo = cola.popleft()
        alcanzables.append(nodo)
        
        for vecino in range(n):
            if matriz[nodo, vecino] > 0 and not visitados[vecino]:
                visitados[vecino] = True
                cola.append(vecino)
    
    return alcanzables

def _nodos_alcanzables_sparse(matriz, origen):
    """
    Implementación optimizada para matrices sparse
    """
    n = matriz.shape[0]
    visitados = [False] * n
    alcanzables = []
    
    cola = deque([origen])
    visitados[origen] = True
    
    while cola:
        nodo = cola.popleft()
        alcanzables.append(nodo)
        
        # Obtener índices de los elementos no cero en la fila
        fila = matriz[nodo].toarray().flatten()
        vecinos = np.where(fila > 0)[0]
        
        for vecino in vecinos:
            if not visitados[vecino]:
                visitados[vecino] = True
                cola.append(vecino)
    
    return alcanzables

def matriz_alcanzabilidad(matriz):
    """
    Calcula la matriz de alcanzabilidad (Warshall)
    
    Args:
        matriz: Matriz de adyacencia
        
    Returns:
        numpy.ndarray: Matriz de alcanzabilidad
    """
    n = len(matriz)
    alcanzabilidad = np.array(matriz, copy=True)
    
    # Convertir valores positivos a 1 (para tener una matriz booleana)
    alcanzabilidad = (alcanzabilidad > 0).astype(int)
    
    # Algoritmo de Warshall para calcular la clausura transitiva
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if alcanzabilidad[i, k] and alcanzabilidad[k, j]:
                    alcanzabilidad[i, j] = 1
    
    return alcanzabilidad