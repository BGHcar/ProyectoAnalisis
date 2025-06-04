import numpy as np
import scipy.sparse as sp
import json
import os
import csv

def cargar_matriz(ruta_archivo):
    """
    Carga una matriz de adyacencia desde un archivo CSV
    
    Args:
        ruta_archivo: Ruta al archivo CSV
        
    Returns:
        numpy.ndarray: Matriz de adyacencia como array denso
    """
    try:
        # Intentar cargar con numpy
        return np.loadtxt(ruta_archivo, delimiter=',', dtype=int)
    except Exception as e:
        # Si falla, intentar cargar manualmente línea por línea
        print(f"Error al cargar con numpy: {e}")
        try:
            filas = []
            with open(ruta_archivo, 'r', newline='') as archivo:
                reader = csv.reader(archivo)
                for fila in reader:
                    if fila:  # Evitar filas vacías
                        filas.append([int(celda) for celda in fila])
            return np.array(filas)
        except Exception as e2:
            print(f"Error al cargar manualmente: {e2}")
            return None

def matriz_a_sparse(matriz_densa):
    """
    Convierte una matriz densa a formato sparse (CSR)
    
    Args:
        matriz_densa: Matriz densa (numpy.ndarray)
        
    Returns:
        scipy.sparse.csr_matrix: Matriz en formato sparse
    """
    return sp.csr_matrix(matriz_densa)

def guardar_resultados(resultados, nombre_archivo):
    """
    Guarda los resultados en un archivo JSON
    
    Args:
        resultados: Diccionario con los resultados
        nombre_archivo: Nombre del archivo donde guardar
    """
    directorio = os.path.join('resultados')
    os.makedirs(directorio, exist_ok=True)
    
    ruta_completa = os.path.join(directorio, nombre_archivo)
    
    # Convertir arrays numpy a listas para que sean serializables
    resultados_serializable = {}
    for clave, valor in resultados.items():
        if isinstance(valor, dict):
            resultados_serializable[clave] = {}
            for subclave, subvalor in valor.items():
                if isinstance(subvalor, np.ndarray):
                    resultados_serializable[clave][subclave] = subvalor.tolist()
                else:
                    resultados_serializable[clave][subclave] = subvalor
        else:
            if isinstance(valor, np.ndarray):
                resultados_serializable[clave] = valor.tolist()
            else:
                resultados_serializable[clave] = valor
    
    with open(ruta_completa, 'w') as archivo:
        json.dump(resultados_serializable, archivo, indent=4)

def generar_matriz_aleatoria(n, densidad=0.5):
    """
    Genera una matriz de adyacencia aleatoria con una densidad específica
    
    Args:
        n: Tamaño de la matriz (nxn)
        densidad: Probabilidad de que un elemento sea 1
        
    Returns:
        numpy.ndarray: Matriz generada
    """
    matriz = np.random.random((n, n)) < densidad
    # Asegurar que la diagonal principal sea 0 (sin auto-ciclos)
    np.fill_diagonal(matriz, 0)
    return matriz.astype(int)