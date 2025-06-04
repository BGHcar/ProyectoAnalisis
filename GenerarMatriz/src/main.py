import os
import time
import numpy as np
import scipy.sparse as sp
import argparse
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

from utils.matriz_utils import cargar_matriz, matriz_a_sparse, guardar_resultados
from utils.visualizador import visualizar_matriz, graficar_comparacion
from algoritmos.alcanzabilidad import nodos_alcanzables, matriz_alcanzabilidad
from algoritmos.bloques import procesar_por_bloques
from algoritmos.paralelizacion import procesar_paralelo
from algoritmos.montecarlo import estimar_conexiones

def main():
    parser = argparse.ArgumentParser(description='Análisis de matrices de adyacencia con optimización de memoria')
    parser.add_argument('--matriz', type=str, default='N2A.csv', 
                      help='Archivo de matriz a analizar')
    parser.add_argument('--ejercicio', type=int, choices=[0, 1, 2, 3, 4], default=0,
                      help='Ejercicio a ejecutar (0 para todos)')
    parser.add_argument('--visualizar', action='store_true',
                      help='Visualizar resultados gráficamente')
    
    args = parser.parse_args()
    
    # Determinar ruta de la matriz
    matriz_path = os.path.join('..', 'src', '.samples', args.matriz)
    if not os.path.exists(matriz_path):
        matriz_path = os.path.join('src', '.samples', args.matriz)
        if not os.path.exists(matriz_path):
            print(f"Error: No se encontró la matriz {args.matriz}")
            return
    
    print(f"Analizando matriz: {matriz_path}")
    
    # Cargar matriz
    matriz_densa = cargar_matriz(matriz_path)
    n = len(matriz_densa)
    print(f"Tamaño de la matriz: {n}x{n}")
    
    # Convertir a formato sparse para comparación
    matriz_sparse = matriz_a_sparse(matriz_densa)
    
    # Calcular y mostrar uso de memoria
    memoria_densa = matriz_densa.nbytes / (1024 * 1024)  # MB
    memoria_sparse = (matriz_sparse.data.nbytes + matriz_sparse.indices.nbytes + 
                     matriz_sparse.indptr.nbytes) / (1024 * 1024)  # MB
    
    print(f"Uso de memoria (matriz densa): {memoria_densa:.2f} MB")
    print(f"Uso de memoria (matriz sparse): {memoria_sparse:.2f} MB")
    print(f"Reducción de memoria: {(1 - memoria_sparse/memoria_densa) * 100:.2f}%")
    
    resultados = {}
    
    # Ejercicio 1: Representación compacta con matrices dispersas
    if args.ejercicio == 0 or args.ejercicio == 1:
        print("\n=== Ejercicio 1: Representación Compacta de Datos ===")
        
        # Medir tiempo con matriz densa
        start_time = time.time()
        alcanzables_densa = nodos_alcanzables(matriz_densa, 0)
        tiempo_densa = time.time() - start_time
        
        # Medir tiempo con matriz sparse
        start_time = time.time()
        alcanzables_sparse = nodos_alcanzables(matriz_sparse, 0)
        tiempo_sparse = time.time() - start_time
        
        print(f"Nodos alcanzables desde 0: {alcanzables_densa}")
        print(f"Tiempo (matriz densa): {tiempo_densa:.6f} segundos")
        print(f"Tiempo (matriz sparse): {tiempo_sparse:.6f} segundos")
        print(f"Speedup: {tiempo_densa/tiempo_sparse:.2f}x")
        
        if args.visualizar:
            visualizar_matriz(matriz_densa, "Matriz Original")
            matriz_alc = matriz_alcanzabilidad(matriz_densa)
            visualizar_matriz(matriz_alc, "Matriz de Alcanzabilidad")
        
        resultados['ejercicio1'] = {
            'nodos_alcanzables': alcanzables_densa,
            'tiempo_densa': tiempo_densa,
            'tiempo_sparse': tiempo_sparse,
            'memoria_densa': memoria_densa,
            'memoria_sparse': memoria_sparse
        }
    
    # Ejercicio 2: Procesamiento por bloques
    if args.ejercicio == 0 or args.ejercicio == 2:
        print("\n=== Ejercicio 2: Procesamiento por Bloques ===")
        
        # Definir tamaños de bloque para pruebas
        if n <= 10:
            tamanos_bloque = [1, 2, n]
        else:
            tamanos_bloque = [n//4, n//2, n]
        
        tiempos_bloques = []
        for tamano in tamanos_bloque:
            start_time = time.time()
            resultado_bloques = procesar_por_bloques(matriz_densa, tamano)
            tiempo_bloque = time.time() - start_time
            tiempos_bloques.append(tiempo_bloque)
            
            print(f"Bloque tamaño {tamano}: {tiempo_bloque:.6f} segundos")
        
        if args.visualizar:
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(tamanos_bloque)), tiempos_bloques, tick_label=tamanos_bloque)
            plt.xlabel('Tamaño de bloque')
            plt.ylabel('Tiempo (segundos)')
            plt.title('Tiempo de procesamiento por tamaño de bloque')
            plt.show()
        
        resultados['ejercicio2'] = {
            'tamanos_bloque': tamanos_bloque,
            'tiempos': tiempos_bloques
        }
    
    # Ejercicio 3: Paralelización
    if args.ejercicio == 0 or args.ejercicio == 3:
        print("\n=== Ejercicio 3: Paralelización ===")
        
        # Comparar rendimiento con diferentes números de procesos
        num_cores = cpu_count()
        procesos_a_probar = [1, 2, min(4, num_cores), num_cores]
        procesos_a_probar = sorted(list(set(procesos_a_probar)))  # Eliminar duplicados
        
        tiempos_paralelo = []
        for num_procesos in procesos_a_probar:
            start_time = time.time()
            resultado_paralelo = procesar_paralelo(matriz_densa, num_procesos)
            tiempo_paralelo = time.time() - start_time
            tiempos_paralelo.append(tiempo_paralelo)
            
            print(f"Procesos: {num_procesos}, Tiempo: {tiempo_paralelo:.6f} segundos")
        
        if args.visualizar:
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(procesos_a_probar)), tiempos_paralelo, tick_label=procesos_a_probar)
            plt.xlabel('Número de procesos')
            plt.ylabel('Tiempo (segundos)')
            plt.title('Tiempo de procesamiento paralelo')
            plt.show()
        
        resultados['ejercicio3'] = {
            'num_procesos': procesos_a_probar,
            'tiempos': tiempos_paralelo
        }
    
    # Ejercicio 4: Reducción de memoria con Monte Carlo
    if args.ejercicio == 0 or args.ejercicio == 4:
        print("\n=== Ejercicio 4: Reducción de Memoria con Monte Carlo ===")
        
        # Comparar precisión y tiempo con diferente número de muestras
        num_muestras = [10, 100, 1000, min(10000, n*n)]
        
        tiempos_mc = []
        precisiones = []
        
        # Calcular valor exacto para comparación
        start_time = time.time()
        conexiones_exactas = np.sum(matriz_alcanzabilidad(matriz_densa))
        tiempo_exacto = time.time() - start_time
        
        print(f"Conexiones exactas: {conexiones_exactas}")
        print(f"Tiempo cálculo exacto: {tiempo_exacto:.6f} segundos")
        
        for muestras in num_muestras:
            start_time = time.time()
            estimacion, _ = estimar_conexiones(matriz_densa, muestras)
            tiempo_mc = time.time() - start_time
            
            # Calcular precisión
            precision = 100 * (1 - abs(estimacion - conexiones_exactas) / conexiones_exactas)
            
            tiempos_mc.append(tiempo_mc)
            precisiones.append(precision)
            
            print(f"Muestras: {muestras}, Estimación: {estimacion:.1f}, Precisión: {precision:.2f}%, Tiempo: {tiempo_mc:.6f} s")
        
        if args.visualizar:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            ax1.plot(num_muestras, tiempos_mc, 'o-')
            ax1.axhline(y=tiempo_exacto, color='r', linestyle='--', label='Tiempo exacto')
            ax1.set_xlabel('Número de muestras')
            ax1.set_ylabel('Tiempo (segundos)')
            ax1.set_title('Tiempo vs Número de muestras')
            ax1.legend()
            
            ax2.plot(num_muestras, precisiones, 'o-')
            ax2.set_xlabel('Número de muestras')
            ax2.set_ylabel('Precisión (%)')
            ax2.set_title('Precisión vs Número de muestras')
            
            plt.tight_layout()
            plt.show()
        
        resultados['ejercicio4'] = {
            'num_muestras': num_muestras,
            'tiempos': tiempos_mc,
            'precisiones': precisiones,
            'conexiones_exactas': int(conexiones_exactas),
            'tiempo_exacto': tiempo_exacto
        }
    
    # Guardar resultados
    if resultados:
        nombre_archivo = f"resultados_{os.path.splitext(args.matriz)[0]}.json"
        guardar_resultados(resultados, nombre_archivo)
        print(f"\nResultados guardados en: {nombre_archivo}")

if __name__ == "__main__":
    main()