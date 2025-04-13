import numpy as np
import scipy.sparse as sp
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import psutil
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial

class ProcesadorBloques:
    """
    Clase para procesar una matriz de espacio de estados por bloques.
    Optimizada para matrices grandes, solo procesa bloques relevantes
    y evita generar archivos individuales por bloque.
    """
    
    def __init__(self, archivo_matriz, tamanio_bloque=64, num_procesos=None):
        """
        Inicializa el procesador de bloques.
        
        Args:
            archivo_matriz: Ruta al archivo CSV con la matriz
            tamanio_bloque: Tamaño de los bloques para el procesamiento
            num_procesos: Número de procesos para paralelización (None = usar todos los núcleos disponibles)
        """
        self.archivo_matriz = archivo_matriz
        self.tamanio_bloque = tamanio_bloque
        
        # Configuración de paralelización
        self.num_procesos = num_procesos if num_procesos is not None else mp.cpu_count()
        print(f"Configurado para utilizar {self.num_procesos} procesos en paralelo")
        
        # Obtenemos el tamaño de la matriz leyendo la primera línea
        with open(archivo_matriz, 'r') as f:
            primera_linea = f.readline()
            self.dim_matriz = len(primera_linea.split(','))
        
        # Calculamos cuántos bloques tendremos
        self.num_bloques = (self.dim_matriz + tamanio_bloque - 1) // tamanio_bloque
        
        print(f"Matriz de tamaño: {self.dim_matriz}x{self.dim_matriz}")
        print(f"Procesando en bloques de: {tamanio_bloque}x{tamanio_bloque}")
        print(f"Número total de bloques: {self.num_bloques}x{self.num_bloques} = {self.num_bloques**2}")
        
        # Directorio para resultados de visualización
        self.dir_resultados = os.path.join(os.path.dirname(archivo_matriz), "resultados_bloques")
        os.makedirs(self.dir_resultados, exist_ok=True)
        
        # Mapa de bloques que contienen valores no cero
        self.mapa_bloques_relevantes = None
        
        # Estructura para almacenar resultados agregados (no por bloque individual)
        # Utilizamos un Manager para compartir datos entre procesos
        self.manager = mp.Manager()
        self.resultados_por_bloque = self.manager.dict()
        
        # Para medir el rendimiento de la paralelización
        self.tiempo_secuencial_estimado = 0
    
    @staticmethod
    def cargar_bloque_estatico(archivo_matriz, fila_inicio, col_inicio, tamanio_bloque, dim_matriz):
        """
        Versión estática del método para cargar un bloque (para uso con multiprocessing)
        
        Args:
            archivo_matriz: Ruta al archivo CSV con la matriz
            fila_inicio: Índice de la fila donde comienza el bloque
            col_inicio: Índice de la columna donde comienza el bloque
            tamanio_bloque: Tamaño del bloque
            dim_matriz: Dimensión total de la matriz
            
        Returns:
            Bloque de la matriz como un array de NumPy
        """
        # Aseguramos que los límites no excedan las dimensiones
        fila_fin = min(fila_inicio + tamanio_bloque, dim_matriz)
        col_fin = min(col_inicio + tamanio_bloque, dim_matriz)
        
        # Usamos loadtxt con usecols y skiprows para cargar solo el bloque necesario
        indices_cols = list(range(col_inicio, col_fin))
        
        # Cargamos solo las filas necesarias
        bloque = np.loadtxt(
            archivo_matriz, 
            delimiter=',',
            usecols=indices_cols,
            skiprows=fila_inicio,
            max_rows=fila_fin - fila_inicio
        )
        
        return bloque
    
    def cargar_bloque(self, fila_inicio, col_inicio):
        """
        Carga un bloque específico de la matriz.
        
        Args:
            fila_inicio: Índice de la fila donde comienza el bloque
            col_inicio: Índice de la columna donde comienza el bloque
            
        Returns:
            Bloque de la matriz como un array de NumPy
        """
        return self.cargar_bloque_estatico(
            self.archivo_matriz, fila_inicio, col_inicio, 
            self.tamanio_bloque, self.dim_matriz
        )
    
    @staticmethod
    def procesar_bloque_estatico(bloque, i_bloque, j_bloque):
        """
        Versión estática del método para procesar un bloque (para uso con multiprocessing)
        
        Args:
            bloque: Matriz NumPy con el bloque a procesar
            i_bloque: Índice de fila del bloque
            j_bloque: Índice de columna del bloque
            
        Returns:
            Resultado del procesamiento del bloque
        """
        # Ejemplo de procesamiento: calculamos estadísticas del bloque
        resultados = {
            'suma': np.sum(bloque),
            'media': np.mean(bloque),
            'desviacion': np.std(bloque),
            'max': np.max(bloque),
            'min': np.min(bloque),
            'densidad': np.count_nonzero(bloque) / bloque.size * 100,
            'tam_bloque': bloque.shape
        }
        
        return resultados
    
    def procesar_bloque(self, bloque, i_bloque, j_bloque):
        """
        Procesa un bloque de la matriz.
        
        Args:
            bloque: Matriz NumPy con el bloque a procesar
            i_bloque: Índice de fila del bloque
            j_bloque: Índice de columna del bloque
            
        Returns:
            Resultado del procesamiento del bloque
        """
        resultado = self.procesar_bloque_estatico(bloque, i_bloque, j_bloque)
        
        # Almacenamos los resultados en nuestra estructura global
        clave_bloque = f"{i_bloque}_{j_bloque}"
        self.resultados_por_bloque[clave_bloque] = resultado
        
        return resultado
    
    @staticmethod
    def analizar_bloque_relevancia(params):
        """
        Función para analizar si un bloque es relevante (para uso con multiprocessing)
        
        Args:
            params: Tupla con (archivo_matriz, fila_inicio, col_inicio, tamanio_bloque, dim_matriz, umbral_relevancia)
            
        Returns:
            Tupla (i_bloque, j_bloque, es_relevante, densidad)
        """
        archivo_matriz, fila_inicio, col_inicio, tamanio_bloque, dim_matriz, umbral_relevancia, i_bloque, j_bloque = params
        
        try:
            # Cargamos el bloque
            bloque = ProcesadorBloques.cargar_bloque_estatico(
                archivo_matriz, fila_inicio, col_inicio, tamanio_bloque, dim_matriz
            )
            
            # Calculamos la densidad
            densidad = np.count_nonzero(bloque) / bloque.size * 100
            
            # Determinamos si es relevante
            es_relevante = densidad > umbral_relevancia
            
            # Liberamos memoria
            del bloque
            gc.collect()
            
            return (i_bloque, j_bloque, es_relevante, densidad)
        except Exception as e:
            print(f"Error al analizar bloque ({i_bloque}, {j_bloque}): {str(e)}")
            return (i_bloque, j_bloque, False, 0.0)
    
    @staticmethod
    def procesar_bloque_completo(params):
        """
        Función para procesar un bloque completo (para uso con multiprocessing)
        
        Args:
            params: Tupla con (archivo_matriz, fila_inicio, col_inicio, tamanio_bloque, dim_matriz, i_bloque, j_bloque)
            
        Returns:
            Tupla (i_bloque, j_bloque, resultado, tiempo_proceso)
        """
        archivo_matriz, fila_inicio, col_inicio, tamanio_bloque, dim_matriz, i_bloque, j_bloque = params
        
        try:
            # Medimos el tiempo
            tiempo_inicio = time.time()
            
            # Cargamos el bloque
            bloque = ProcesadorBloques.cargar_bloque_estatico(
                archivo_matriz, fila_inicio, col_inicio, tamanio_bloque, dim_matriz
            )
            
            # Procesamos el bloque
            resultado = ProcesadorBloques.procesar_bloque_estatico(bloque, i_bloque, j_bloque)
            
            # Liberamos memoria
            del bloque
            gc.collect()
            
            # Calculamos el tiempo total
            tiempo_proceso = time.time() - tiempo_inicio
            
            return (i_bloque, j_bloque, resultado, tiempo_proceso)
        except Exception as e:
            print(f"Error al procesar bloque ({i_bloque}, {j_bloque}): {str(e)}")
            return (i_bloque, j_bloque, None, 0.0)
    
    def detectar_bloques_relevantes_paralelo(self, umbral_relevancia=0.0):
        """
        Detecta qué bloques de la matriz contienen información relevante,
        utilizando procesamiento paralelo para acelerar la detección.
        
        Args:
            umbral_relevancia: Umbral mínimo de densidad para considerar un bloque relevante
                            (0.0 significa que cualquier bloque con al menos un valor no cero es relevante)
        
        Returns:
            Matriz booleana donde True indica que el bloque es relevante
        """
        print(f"Analizando matriz para detectar bloques relevantes (en paralelo con {self.num_procesos} procesos)...")
        
        # Matriz para marcar bloques relevantes
        mapa_bloques = np.zeros((self.num_bloques, self.num_bloques), dtype=bool)
        
        # Mapa de densidad para visualización
        mapa_densidad = np.zeros((self.num_bloques, self.num_bloques))
        
        # Generamos los parámetros para cada bloque
        params_list = []
        for i in range(self.num_bloques):
            fila_inicio = i * self.tamanio_bloque
            for j in range(self.num_bloques):
                col_inicio = j * self.tamanio_bloque
                params_list.append((
                    self.archivo_matriz, fila_inicio, col_inicio, 
                    self.tamanio_bloque, self.dim_matriz, umbral_relevancia,
                    i, j
                ))
        
        # Calculamos cuántos bloques procesar por lote
        bloques_por_lote = max(1, len(params_list) // (self.num_procesos * 4))
        total_lotes = (len(params_list) + bloques_por_lote - 1) // bloques_por_lote
        
        print(f"Dividiendo el análisis en {total_lotes} lotes de ~{bloques_por_lote} bloques cada uno")
        
        # Procesamos en paralelo usando multiprocessing.Pool directamente
        tiempo_inicio = time.time()
        resultados = []
        
        with mp.Pool(processes=self.num_procesos) as pool:
            for lote_inicio in tqdm(range(0, len(params_list), bloques_por_lote), 
                                   desc="Analizando bloques en paralelo"):
                lote_fin = min(lote_inicio + bloques_por_lote, len(params_list))
                lote_params = params_list[lote_inicio:lote_fin]
                
                # Aplicamos el proceso a todo el lote en paralelo
                resultados_lote = pool.map(self.analizar_bloque_relevancia, lote_params)
                resultados.extend(resultados_lote)
                
                # Liberamos memoria entre lotes
                gc.collect()
        
        tiempo_deteccion = time.time() - tiempo_inicio
        print(f"Tiempo de detección en paralelo: {tiempo_deteccion:.2f} segundos")
        
        # Procesamos los resultados
        for i_bloque, j_bloque, es_relevante, densidad in resultados:
            mapa_bloques[i_bloque, j_bloque] = es_relevante
            mapa_densidad[i_bloque, j_bloque] = densidad
        
        # Guardamos el mapa para uso futuro
        self.mapa_bloques_relevantes = mapa_bloques
        
        # Contamos bloques relevantes
        bloques_relevantes = np.count_nonzero(mapa_bloques)
        print(f"Bloques relevantes detectados: {bloques_relevantes} de {self.num_bloques**2} ({bloques_relevantes/self.num_bloques**2*100:.2f}%)")
        
        # Visualizamos el mapa de densidad
        self.visualizar_mapa_densidad(mapa_densidad, "mapa_densidad_inicial_paralelo.png")
        
        return mapa_bloques
    
    def procesar_bloques_relevantes_paralelo(self, umbral_relevancia=0.0):
        """
        Procesa solo los bloques que contienen información relevante utilizando
        procesamiento paralelo para acelerar las operaciones.
        
        Args:
            umbral_relevancia: Umbral mínimo de densidad para considerar un bloque relevante
        
        Returns:
            Resultados agregados de los bloques relevantes
        """
        tiempo_inicio_total = time.time()
        
        # Si no tenemos el mapa de bloques relevantes, lo generamos en paralelo
        if self.mapa_bloques_relevantes is None:
            self.detectar_bloques_relevantes_paralelo(umbral_relevancia)
        
        # Para guardar resultados globales
        resultados_globales = {
            'suma_total': 0,
            'valores_no_cero': 0,
            'bloques_procesados': 0,
            'bloques_relevantes': np.count_nonzero(self.mapa_bloques_relevantes),
            'bloques_ignorados': self.num_bloques**2 - np.count_nonzero(self.mapa_bloques_relevantes),
            'memoria_pico': 0,
            'tiempo_por_bloque': [],
            'densidad_por_bloque': [],
            'speedup': 0,
            'eficiencia': 0
        }
        
        # Matriz para visualización de procesamiento
        mapa_procesamiento = np.zeros((self.num_bloques, self.num_bloques))
        
        # Generamos los parámetros solo para bloques relevantes
        params_list = []
        for i in range(self.num_bloques):
            fila_inicio = i * self.tamanio_bloque
            for j in range(self.num_bloques):
                if not self.mapa_bloques_relevantes[i, j]:
                    continue
                
                col_inicio = j * self.tamanio_bloque
                params_list.append((
                    self.archivo_matriz, fila_inicio, col_inicio, 
                    self.tamanio_bloque, self.dim_matriz, i, j
                ))
        
        print(f"Procesando {len(params_list)} bloques relevantes en paralelo con {self.num_procesos} procesos...")
        
        # Calculamos cuántos bloques procesar por lote para maximizar el uso de CPU
        # Usamos lotes más grandes para reducir la sobrecarga de comunicación
        bloques_por_lote = max(1, len(params_list) // (self.num_procesos * 2))
        total_lotes = (len(params_list) + bloques_por_lote - 1) // bloques_por_lote
        
        print(f"Estrategia de paralelización optimizada: {total_lotes} lotes con ~{bloques_por_lote} bloques por lote")
        
        # Procesamos en paralelo usando multiprocessing.Pool directamente para mayor control
        tiempo_inicio_proc = time.time()
        resultados = []
        
        # Dividimos en lotes para mejor rendimiento y feedback al usuario
        with mp.Pool(processes=self.num_procesos) as pool:
            for lote_inicio in tqdm(range(0, len(params_list), bloques_por_lote), 
                                   desc="Procesando lotes de bloques en paralelo"):
                lote_fin = min(lote_inicio + bloques_por_lote, len(params_list))
                lote_params = params_list[lote_inicio:lote_fin]
                
                # Aplicamos el proceso a todo el lote en paralelo
                resultados_lote = pool.map(self.procesar_bloque_completo, lote_params)
                resultados.extend(resultados_lote)
                
                # Liberamos memoria entre lotes
                gc.collect()
        
        tiempo_procesamiento = time.time() - tiempo_inicio_proc
        
        # Procesamos los resultados
        for i_bloque, j_bloque, resultado, tiempo_bloque in resultados:
            if resultado is None:
                continue
                
            # Guardamos en el diccionario compartido
            clave_bloque = f"{i_bloque}_{j_bloque}"
            self.resultados_por_bloque[clave_bloque] = resultado
            
            # Actualizamos resultados globales
            resultados_globales['suma_total'] += resultado['suma']
            resultados_globales['valores_no_cero'] += int(resultado['densidad'] * resultado['tam_bloque'][0] * resultado['tam_bloque'][1] / 100)
            resultados_globales['bloques_procesados'] += 1
            resultados_globales['tiempo_por_bloque'].append(tiempo_bloque)
            resultados_globales['densidad_por_bloque'].append(resultado['densidad'])
            
            # Guardamos el mapa de procesamiento para visualización
            mapa_procesamiento[i_bloque, j_bloque] = resultado['densidad']
        
        # Calculamos estadísticas finales
        tiempo_total = time.time() - tiempo_inicio_total
        resultados_globales['tiempo_total'] = tiempo_total
        resultados_globales['tiempo_procesamiento'] = tiempo_procesamiento
        
        if resultados_globales['bloques_procesados'] > 0:
            resultados_globales['tiempo_promedio_bloque'] = np.mean(resultados_globales['tiempo_por_bloque'])
            resultados_globales['densidad_promedio_bloques_relevantes'] = np.mean(resultados_globales['densidad_por_bloque'])
            
            # Estimamos el tiempo que habría tomado secuencialmente
            self.tiempo_secuencial_estimado = resultados_globales['tiempo_promedio_bloque'] * resultados_globales['bloques_procesados']
            resultados_globales['tiempo_secuencial_estimado'] = self.tiempo_secuencial_estimado
            
            # Calculamos speedup y eficiencia
            resultados_globales['speedup'] = self.tiempo_secuencial_estimado / tiempo_procesamiento
            resultados_globales['eficiencia'] = resultados_globales['speedup'] / self.num_procesos
        else:
            resultados_globales['tiempo_promedio_bloque'] = 0
            resultados_globales['densidad_promedio_bloques_relevantes'] = 0
            resultados_globales['speedup'] = 0
            resultados_globales['eficiencia'] = 0
            
        resultados_globales['densidad_global'] = resultados_globales['valores_no_cero'] / (self.dim_matriz ** 2) * 100
        
        # Visualización del mapa de procesamiento
        self.visualizar_mapa_densidad(mapa_procesamiento, "mapa_procesamiento_bloques_relevantes_paralelo.png")
        
        # Visualización del speedup
        self.visualizar_speedup(resultados_globales)
        
        # Guardamos un único archivo con resumen de resultados
        self.guardar_resultados_resumen(resultados_globales)
        
        return resultados_globales
    
    def visualizar_speedup(self, resultados_globales):
        """
        Crea una visualización del speedup obtenido con el procesamiento paralelo.
        
        Args:
            resultados_globales: Diccionario con los resultados globales
        """
        if resultados_globales['speedup'] <= 0:
            return
            
        plt.figure(figsize=(10, 6))
        
        # Gráfico de barras con speedup ideal vs real
        x = np.arange(2)
        ancho_barra = 0.35
        
        plt.bar(x[0], self.num_procesos, ancho_barra, label='Speedup Ideal', color='lightblue')
        plt.bar(x[1], resultados_globales['speedup'], ancho_barra, label='Speedup Real', color='blue')
        
        plt.title('Comparación de Speedup Ideal vs Real')
        plt.xticks([])
        plt.ylabel('Speedup (veces más rápido)')
        plt.legend()
        
        # Añadimos texto con los valores
        plt.text(x[0], self.num_procesos + 0.1, f"{self.num_procesos:.1f}x", ha='center')
        plt.text(x[1], resultados_globales['speedup'] + 0.1, f"{resultados_globales['speedup']:.2f}x", ha='center')
        
        # Añadimos información de eficiencia
        plt.figtext(0.5, 0.01, f"Eficiencia: {resultados_globales['eficiencia']*100:.1f}% (Speedup Real / Número de Procesos)", 
                   ha="center", fontsize=10, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        # Guardamos el gráfico
        archivo_speedup = os.path.join(self.dir_resultados, "speedup_paralelo.png")
        plt.savefig(archivo_speedup, dpi=300)
        plt.close()
        
        print(f"Gráfico de speedup guardado en: {archivo_speedup}")
    
    def guardar_resultados_resumen(self, resultados_globales):
        """
        Guarda un único archivo con el resumen de los resultados,
        en lugar de un archivo por bloque.
        
        Args:
            resultados_globales: Diccionario con los resultados globales
        """
        archivo_resumen = os.path.join(self.dir_resultados, "resumen_procesamiento_paralelo.txt")
        
        with open(archivo_resumen, 'w') as f:
            f.write("===== RESUMEN DE PROCESAMIENTO POR BLOQUES (PARALELO) =====\n\n")
            f.write(f"Tamaño de matriz: {self.dim_matriz}x{self.dim_matriz}\n")
            f.write(f"Tamaño de bloques: {self.tamanio_bloque}x{self.tamanio_bloque}\n")
            f.write(f"Número total de bloques: {self.num_bloques**2}\n")
            f.write(f"Número de procesos utilizados: {self.num_procesos}\n\n")
            
            f.write("--- Estadísticas Globales ---\n")
            f.write(f"Bloques relevantes: {resultados_globales['bloques_relevantes']} ({resultados_globales['bloques_relevantes']/self.num_bloques**2*100:.2f}%)\n")
            f.write(f"Bloques procesados: {resultados_globales['bloques_procesados']}\n")
            f.write(f"Bloques ignorados: {resultados_globales['bloques_ignorados']} ({resultados_globales['bloques_ignorados']/self.num_bloques**2*100:.2f}%)\n")
            f.write(f"Suma total de elementos: {resultados_globales['suma_total']:.4f}\n")
            f.write(f"Valores no cero: {resultados_globales['valores_no_cero']}\n")
            f.write(f"Densidad global: {resultados_globales['densidad_global']:.4f}%\n")
            f.write(f"Densidad promedio en bloques relevantes: {resultados_globales['densidad_promedio_bloques_relevantes']:.4f}%\n\n")
            
            f.write("--- Rendimiento Paralelo ---\n")
            f.write(f"Tiempo total de procesamiento: {resultados_globales['tiempo_total']:.2f} segundos\n")
            f.write(f"Tiempo de procesamiento puro: {resultados_globales.get('tiempo_procesamiento', 0):.2f} segundos\n")
            f.write(f"Tiempo promedio por bloque: {resultados_globales['tiempo_promedio_bloque']:.4f} segundos\n")
            
            if 'tiempo_secuencial_estimado' in resultados_globales:
                f.write(f"Tiempo secuencial estimado: {resultados_globales['tiempo_secuencial_estimado']:.2f} segundos\n")
                f.write(f"Speedup: {resultados_globales['speedup']:.2f}x\n")
                f.write(f"Eficiencia: {resultados_globales['eficiencia']*100:.2f}%\n")
                
            f.write("\n--- Bloques con mayor densidad ---\n")
            # Obtenemos los 10 bloques más densos
            bloques_ordenados = sorted(
                [(k, v) for k, v in self.resultados_por_bloque.items()], 
                key=lambda x: x[1]['densidad'], 
                reverse=True
            )
            
            for i, (clave, datos) in enumerate(bloques_ordenados[:10]):
                i_bloque, j_bloque = map(int, clave.split('_'))
                f.write(f"{i+1}. Bloque ({i_bloque}, {j_bloque}):\n")
                f.write(f"   - Densidad: {datos['densidad']:.2f}%\n")
                f.write(f"   - Suma: {datos['suma']:.4f}\n")
                f.write(f"   - Min/Max: {datos['min']:.4f}/{datos['max']:.4f}\n")
                f.write(f"   - Media: {datos['media']:.4f}\n")
        
        print(f"Resumen de resultados guardado en: {archivo_resumen}")
    
    def visualizar_mapa_densidad(self, mapa_densidad, nombre_archivo="mapa_densidad_bloques.png"):
        """
        Crea una visualización del mapa de densidad por bloques.
        
        Args:
            mapa_densidad: Matriz con la densidad por bloques
            nombre_archivo: Nombre del archivo para guardar la visualización
        """
        plt.figure(figsize=(12, 10))
        im = plt.imshow(mapa_densidad, cmap='viridis')
        plt.colorbar(im, label='Densidad (%)')
        plt.title(f'Mapa de densidad por bloques ({self.tamanio_bloque}x{self.tamanio_bloque})')
        plt.xlabel('Columna de bloque')
        plt.ylabel('Fila de bloque')
        
        # Guardar la visualización
        archivo_salida = os.path.join(self.dir_resultados, nombre_archivo)
        plt.savefig(archivo_salida, dpi=300)
        plt.close()
        
        print(f"Mapa de densidad guardado en: {archivo_salida}")
    
    def analizar_distribucion_elementos(self):
        """
        Analiza la distribución de elementos en la matriz y genera visualizaciones.
        """
        if not self.resultados_por_bloque:
            raise ValueError("No hay resultados de bloques disponibles. Ejecuta primero procesar_bloques_relevantes()")
        
        # Extraemos información de densidad por bloque
        datos_densidad = [resultado['densidad'] for resultado in self.resultados_por_bloque.values()]
        
        # Histograma de densidad
        plt.figure(figsize=(10, 6))
        plt.hist(datos_densidad, bins=20, alpha=0.7, color='blue')
        plt.title('Distribución de Densidad por Bloque')
        plt.xlabel('Densidad (%)')
        plt.ylabel('Número de Bloques')
        plt.grid(alpha=0.3)
        
        # Guardamos el histograma
        archivo_histograma = os.path.join(self.dir_resultados, "histograma_densidad.png")
        plt.savefig(archivo_histograma, dpi=300)
        plt.close()
        
        # Gráfico de dispersión de valores máximos vs densidad
        datos_max = [resultado['max'] for resultado in self.resultados_por_bloque.values()]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(datos_densidad, datos_max, alpha=0.6, c=datos_densidad, cmap='viridis')
        plt.colorbar(label='Densidad (%)')
        plt.title('Relación entre Densidad y Valor Máximo por Bloque')
        plt.xlabel('Densidad (%)')
        plt.ylabel('Valor Máximo')
        plt.grid(alpha=0.3)
        
        # Guardamos el gráfico de dispersión
        archivo_dispersion = os.path.join(self.dir_resultados, "dispersion_max_densidad.png")
        plt.savefig(archivo_dispersion, dpi=300)
        plt.close()
        
        print(f"Análisis de distribución guardado en: {self.dir_resultados}")
        return {'densidad_media': np.mean(datos_densidad), 'max_media': np.mean(datos_max)}

    def procesar_bloques_relevantes(self, umbral_relevancia=0.0):
        """
        Procesa solo los bloques que contienen información relevante de manera secuencial.
        
        Args:
            umbral_relevancia: Umbral mínimo de densidad para considerar un bloque relevante
                            (0.0 significa que cualquier bloque con al menos un valor no cero es relevante)
        
        Returns:
            Resultados agregados de los bloques relevantes
        """
        tiempo_inicio_total = time.time()
        
        # Si no tenemos el mapa de bloques relevantes, lo generamos
        if self.mapa_bloques_relevantes is None:
            self.detectar_bloques_relevantes(umbral_relevancia)
        
        # Para guardar resultados globales
        resultados_globales = {
            'suma_total': 0,
            'valores_no_cero': 0,
            'bloques_procesados': 0,
            'bloques_relevantes': np.count_nonzero(self.mapa_bloques_relevantes),
            'bloques_ignorados': self.num_bloques**2 - np.count_nonzero(self.mapa_bloques_relevantes),
            'memoria_pico': 0,
            'tiempo_por_bloque': [],
            'densidad_por_bloque': []
        }
        
        # Matriz para visualización de procesamiento
        mapa_procesamiento = np.zeros((self.num_bloques, self.num_bloques))
        
        print(f"Procesando {resultados_globales['bloques_relevantes']} bloques relevantes secuencialmente...")
        
        # Iteramos sobre todos los bloques
        for i in tqdm(range(self.num_bloques), desc="Procesando filas de bloques"):
            fila_inicio = i * self.tamanio_bloque
            
            for j in range(self.num_bloques):
                # Saltamos bloques no relevantes
                if not self.mapa_bloques_relevantes[i, j]:
                    continue
                    
                col_inicio = j * self.tamanio_bloque
                
                # Medimos el tiempo de procesamiento del bloque
                tiempo_inicio_bloque = time.time()
                
                # Cargamos el bloque
                bloque = self.cargar_bloque(fila_inicio, col_inicio)
                
                # Procesamos el bloque
                resultado = self.procesar_bloque(bloque, i, j)
                
                # Liberamos memoria
                del bloque
                gc.collect()
                
                # Calculamos tiempo de procesamiento
                tiempo_bloque = time.time() - tiempo_inicio_bloque
                
                # Actualizamos resultados globales
                resultados_globales['suma_total'] += resultado['suma']
                resultados_globales['valores_no_cero'] += int(resultado['densidad'] * resultado['tam_bloque'][0] * resultado['tam_bloque'][1] / 100)
                resultados_globales['bloques_procesados'] += 1
                resultados_globales['tiempo_por_bloque'].append(tiempo_bloque)
                resultados_globales['densidad_por_bloque'].append(resultado['densidad'])
                
                # Guardamos el mapa de procesamiento para visualización
                mapa_procesamiento[i, j] = resultado['densidad']
        
        # Calculamos estadísticas finales
        tiempo_total = time.time() - tiempo_inicio_total
        resultados_globales['tiempo_total'] = tiempo_total
        
        if resultados_globales['bloques_procesados'] > 0:
            resultados_globales['tiempo_promedio_bloque'] = np.mean(resultados_globales['tiempo_por_bloque'])
            resultados_globales['densidad_promedio_bloques_relevantes'] = np.mean(resultados_globales['densidad_por_bloque'])
        else:
            resultados_globales['tiempo_promedio_bloque'] = 0
            resultados_globales['densidad_promedio_bloques_relevantes'] = 0
            
        resultados_globales['densidad_global'] = resultados_globales['valores_no_cero'] / (self.dim_matriz ** 2) * 100
        
        # Visualización del mapa de procesamiento
        self.visualizar_mapa_densidad(mapa_procesamiento, "mapa_procesamiento_bloques_relevantes_secuencial.png")
        
        # Guardamos un único archivo con resumen de resultados
        self.guardar_resultados_resumen_secuencial(resultados_globales)
        
        return resultados_globales

    def detectar_bloques_relevantes(self, umbral_relevancia=0.0):
        """
        Detecta qué bloques de la matriz contienen información relevante.
        
        Args:
            umbral_relevancia: Umbral mínimo de densidad para considerar un bloque relevante
                            (0.0 significa que cualquier bloque con al menos un valor no cero es relevante)
        
        Returns:
            Matriz booleana donde True indica que el bloque es relevante
        """
        print(f"Analizando matriz para detectar bloques relevantes (secuencial)...")
        
        # Matriz para marcar bloques relevantes
        mapa_bloques = np.zeros((self.num_bloques, self.num_bloques), dtype=bool)
        
        # Mapa de densidad para visualización
        mapa_densidad = np.zeros((self.num_bloques, self.num_bloques))
        
        # Procesamos los bloques secuencialmente
        tiempo_inicio = time.time()
        
        for i in tqdm(range(self.num_bloques), desc="Analizando filas de bloques"):
            fila_inicio = i * self.tamanio_bloque
            
            for j in range(self.num_bloques):
                col_inicio = j * self.tamanio_bloque
                
                # Cargamos el bloque
                bloque = self.cargar_bloque(fila_inicio, col_inicio)
                
                # Calculamos la densidad
                densidad = np.count_nonzero(bloque) / bloque.size * 100
                
                # Determinamos si es relevante
                es_relevante = densidad > umbral_relevancia
                mapa_bloques[i, j] = es_relevante
                mapa_densidad[i, j] = densidad
                
                # Liberamos memoria
                del bloque
                gc.collect()
        
        tiempo_deteccion = time.time() - tiempo_inicio
        print(f"Tiempo de detección secuencial: {tiempo_deteccion:.2f} segundos")
        
        # Guardamos el mapa para uso futuro
        self.mapa_bloques_relevantes = mapa_bloques
        
        # Contamos bloques relevantes
        bloques_relevantes = np.count_nonzero(mapa_bloques)
        print(f"Bloques relevantes detectados: {bloques_relevantes} de {self.num_bloques**2} ({bloques_relevantes/self.num_bloques**2*100:.2f}%)")
        
        # Visualizamos el mapa de densidad
        self.visualizar_mapa_densidad(mapa_densidad, "mapa_densidad_inicial_secuencial.png")
        
        return mapa_bloques

    def guardar_resultados_resumen_secuencial(self, resultados_globales):
        """
        Guarda un único archivo con el resumen de los resultados secuenciales,
        en lugar de un archivo por bloque.
        
        Args:
            resultados_globales: Diccionario con los resultados globales
        """
        archivo_resumen = os.path.join(self.dir_resultados, "resumen_procesamiento_secuencial.txt")
        
        with open(archivo_resumen, 'w') as f:
            f.write("===== RESUMEN DE PROCESAMIENTO POR BLOQUES (SECUENCIAL) =====\n\n")
            f.write(f"Tamaño de matriz: {self.dim_matriz}x{self.dim_matriz}\n")
            f.write(f"Tamaño de bloques: {self.tamanio_bloque}x{self.tamanio_bloque}\n")
            f.write(f"Número total de bloques: {self.num_bloques**2}\n\n")
            
            f.write("--- Estadísticas Globales ---\n")
            f.write(f"Bloques relevantes: {resultados_globales['bloques_relevantes']} ({resultados_globales['bloques_relevantes']/self.num_bloques**2*100:.2f}%)\n")
            f.write(f"Bloques procesados: {resultados_globales['bloques_procesados']}\n")
            f.write(f"Bloques ignorados: {resultados_globales['bloques_ignorados']} ({resultados_globales['bloques_ignorados']/self.num_bloques**2*100:.2f}%)\n")
            f.write(f"Suma total de elementos: {resultados_globales['suma_total']:.4f}\n")
            f.write(f"Valores no cero: {resultados_globales['valores_no_cero']}\n")
            f.write(f"Densidad global: {resultados_globales['densidad_global']:.4f}%\n")
            f.write(f"Densidad promedio en bloques relevantes: {resultados_globales['densidad_promedio_bloques_relevantes']:.4f}%\n\n")
            
            f.write("--- Rendimiento Secuencial ---\n")
            f.write(f"Tiempo total de procesamiento: {resultados_globales['tiempo_total']:.2f} segundos\n")
            f.write(f"Tiempo promedio por bloque: {resultados_globales['tiempo_promedio_bloque']:.4f} segundos\n")
                
            f.write("\n--- Bloques con mayor densidad ---\n")
            # Obtenemos los 10 bloques más densos
            bloques_ordenados = sorted(
                [(k, v) for k, v in self.resultados_por_bloque.items()], 
                key=lambda x: x[1]['densidad'], 
                reverse=True
            )
            
            for i, (clave, datos) in enumerate(bloques_ordenados[:10]):
                i_bloque, j_bloque = map(int, clave.split('_'))
                f.write(f"{i+1}. Bloque ({i_bloque}, {j_bloque}):\n")
                f.write(f"   - Densidad: {datos['densidad']:.2f}%\n")
                f.write(f"   - Suma: {datos['suma']:.4f}\n")
                f.write(f"   - Min/Max: {datos['min']:.4f}/{datos['max']:.4f}\n")
                f.write(f"   - Media: {datos['media']:.4f}\n")
        
        print(f"Resumen de resultados guardado en: {archivo_resumen}")

def procesar_matriz_por_bloques(archivo_matriz, tamanio_bloque=64, solo_relevantes=True, umbral_relevancia=0.0, paralelo=True, num_procesos=None):
    """
    Función principal para procesar una matriz por bloques, con opción de paralelización.
    
    Args:
        archivo_matriz: Ruta al archivo CSV con la matriz
        tamanio_bloque: Tamaño de los bloques para el procesamiento
        solo_relevantes: Si es True, solo procesa bloques que contienen información (no todos ceros)
        umbral_relevancia: Umbral mínimo de densidad para considerar un bloque relevante
        paralelo: Si es True, utiliza procesamiento paralelo
        num_procesos: Número de procesos a utilizar (None = usar todos los núcleos disponibles)
        
    Returns:
        Resultados del procesamiento
    """
    # Inicializamos el procesador con tamaño de bloque indicado
    procesador = ProcesadorBloques(archivo_matriz, tamanio_bloque, num_procesos)
    
    # Procesamos bloques (todos o solo relevantes, en paralelo o secuencial)
    if solo_relevantes:
        if paralelo:
            print(f"Iniciando procesamiento PARALELO selectivo con {procesador.num_procesos} procesos...")
            resultados = procesador.procesar_bloques_relevantes_paralelo(umbral_relevancia)
        else:
            print("Iniciando procesamiento SECUENCIAL selectivo de bloques relevantes...")
            resultados = procesador.procesar_bloques_relevantes(umbral_relevancia)
            
        print(f"Se procesaron {resultados['bloques_procesados']} bloques relevantes de un total de {procesador.num_bloques**2} bloques")
        print(f"Se ignoraron {resultados['bloques_ignorados']} bloques (ahorro de {resultados['bloques_ignorados']/procesador.num_bloques**2*100:.2f}%)")
    else:
        print("Esta funcionalidad requiere implementación adicional para el procesamiento de todos los bloques.")
        return None
    
    # Mostramos resultados
    print("\n===== RESULTADOS DEL PROCESAMIENTO POR BLOQUES =====")
    print(f"Suma total de elementos: {resultados['suma_total']}")
    print(f"Valores no cero: {resultados['valores_no_cero']}")
    print(f"Densidad global: {resultados['densidad_global']:.4f}%")
    print(f"Tiempo total de procesamiento: {resultados['tiempo_total']:.2f} segundos")
    print(f"Tiempo promedio por bloque: {resultados['tiempo_promedio_bloque']:.4f} segundos")
    
    if paralelo and 'speedup' in resultados:
        print(f"Speedup: {resultados['speedup']:.2f}x (con {procesador.num_procesos} procesos)")
        print(f"Eficiencia: {resultados['eficiencia']*100:.2f}%")
    
    print("====================================================")
    
    # Realizamos análisis adicional de la distribución
    print("\nGenerando visualizaciones de análisis...")
    procesador.analizar_distribucion_elementos()
    
    return resultados