import numpy as np
from .alcanzabilidad import matriz_alcanzabilidad

def procesar_por_bloques(matriz, tamano_bloque):
    """
    Procesa una matriz por bloques para calcular la matriz de alcanzabilidad
    
    Args:
        matriz: Matriz de adyacencia
        tamano_bloque: Tamaño de los bloques a procesar
        
    Returns:
        numpy.ndarray: Matriz de alcanzabilidad
    """
    n = len(matriz)
    resultado = np.zeros((n, n), dtype=int)
    
    # Definir los bloques
    bloques = []
    for i in range(0, n, tamano_bloque):
        filas = list(range(i, min(i + tamano_bloque, n)))
        for j in range(0, n, tamano_bloque):
            columnas = list(range(j, min(j + tamano_bloque, n)))
            bloques.append((filas, columnas))
    
    # Procesar cada bloque
    for filas, columnas in bloques:
        # Extraer submatriz
        submatriz = matriz[np.ix_(filas, columnas)]
        
        # Procesar submatriz (calcular alcanzabilidad para el bloque)
        if len(filas) == 1 or len(columnas) == 1:
            # Si es un bloque 1x1 o Nx1 o 1xN, no necesitamos Warshall
            subresultado = (submatriz > 0).astype(int)
        else:
            # Para bloques más grandes calculamos la clausura transitiva
            subresultado = matriz_alcanzabilidad(submatriz)
        
        # Almacenar resultado en la matriz final
        resultado[np.ix_(filas, columnas)] = subresultado
    
    # Fase de combinación: necesitamos unir los resultados
    # Aplicamos Warshall pero solo en los bordes de los bloques
    for k in range(n):
        for i in range(n):
            if resultado[i, k]:
                for j in range(n):
                    if resultado[k, j]:
                        resultado[i, j] = 1
    
    return resultado