import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import scipy.optimize as opt
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import os

class MatrizEspacioEstados:
    """
    Clase para representar un espacio de estados de tamaño (n * 2^n) usando matrices dispersas.
    """
    
    def __init__(self, n):
        """
        Inicializa una matriz dispersa para representar un espacio de estados.
        
        Args:
            n: Tamaño de la dimensión base
        """
        self.n = n
        self.size = n * (2**n)  # Tamaño de la matriz: n * 2^n
        
        # Inicializamos una matriz dispersa vacía en formato COO (Coordinate)
        # que luego convertiremos a CSR para operaciones eficientes
        self.matriz = None
        
    def generar_matriz_aleatoria(self, densidad=0.1):
        """
        Genera una matriz dispersa aleatoria con una densidad dada.
        
        Args:
            densidad: Porcentaje de elementos no cero (por defecto 10%)
        """
        # Para matrices muy grandes, no podemos usar métodos convencionales
        # así que generamos directamente en formato COO
        size = self.size
        
        # Estimamos el número de elementos no cero
        nnz = int(size * size * densidad)
        
        # Generar índices aleatorios
        filas = np.random.randint(0, size, nnz)
        columnas = np.random.randint(0, size, nnz)
        datos = np.random.randn(nnz)
        
        # Crear matriz en formato COO y convertir a CSR para operaciones eficientes
        self.matriz = sp.coo_matrix((datos, (filas, columnas)), shape=(size, size)).tocsr()
        
        return self.matriz
    
    def aplicar_vector(self, vector):
        """
        Aplica la matriz a un vector (Ax).
        
        Args:
            vector: Vector de entrada
            
        Returns:
            Vector resultado
        """
        if self.matriz is None:
            raise ValueError("La matriz no ha sido inicializada")
        
        return self.matriz.dot(vector)
    
    def resolver_sistema(self, b):
        """
        Resuelve el sistema de ecuaciones Ax = b.
        
        Args:
            b: Vector de términos independientes
            
        Returns:
            Solución del sistema
        """
        if self.matriz is None:
            raise ValueError("La matriz no ha sido inicializada")
            
        # Usamos un método iterativo para matrices grandes
        x, info = opt.bicgstab(self.matriz, b)
        
        if info != 0:
            print(f"Advertencia: el solver convergió con código de salida {info}")
            
        return x
    
    def calcular_autovalores(self, k=6):
        """
        Calcula los k autovalores más grandes en magnitud.
        
        Args:
            k: Número de autovalores a calcular
            
        Returns:
            Autovalores calculados
        """
        if self.matriz is None:
            raise ValueError("La matriz no ha sido inicializada")
            
        # Usamos método de Arnoldi para matrices grandes
        eigenvalues = sp.linalg.eigs(self.matriz, k=k, which='LM', return_eigenvectors=False)
        return eigenvalues
    
    def obtener_info(self):
        """
        Obtiene información sobre la matriz dispersa.
        
        Returns:
            Diccionario con información sobre la matriz
        """
        if self.matriz is None:
            return {"estado": "No inicializada"}
        
        return {
            "tamaño": self.matriz.shape,
            "elementos_no_cero": self.matriz.nnz,
            "densidad": self.matriz.nnz / (self.size**2) * 100,
            "memoria_MB": self.matriz.data.nbytes / 1024**2 + 
                          self.matriz.indices.nbytes / 1024**2 + 
                          self.matriz.indptr.nbytes / 1024**2
        }
    
    def guardar_como_npz(self, archivo_salida):
        """
        Guarda la matriz en formato .npz de NumPy.
        
        Args:
            archivo_salida: Ruta del archivo NPZ de salida
        """
        if self.matriz is None:
            raise ValueError("La matriz no ha sido inicializada")
        
        # Guardar en formato NPZ de NumPy
        sp.save_npz(archivo_salida, self.matriz)
        
        print(f"Matriz guardada en formato NPZ: {archivo_salida}")
        print(f"Tamaño del archivo: {os.path.getsize(archivo_salida)/1024/1024:.2f} MB")
    
    def guardar_como_csv(self, archivo_salida):
        """
        Guarda la matriz en formato CSV plano sin cabeceras.
        La matriz completa se guarda, sin formato especial de matriz dispersa.
        
        Args:
            archivo_salida: Ruta del archivo CSV de salida
        """
        if self.matriz is None:
            raise ValueError("La matriz no ha sido inicializada")
        
        # Para matrices grandes, esto puede consumir mucha memoria,
        # ya que convierte la matriz dispersa a densa
        print(f"Convirtiendo matriz dispersa a formato denso para CSV...")
        matriz_densa = self.matriz.toarray()
        
        print(f"Guardando matriz en {archivo_salida}...")
        np.savetxt(archivo_salida, matriz_densa, delimiter=',', fmt='%.10f')
        
        print(f"Matriz guardada en formato CSV.")
        print(f"Tamaño del archivo: {os.path.getsize(archivo_salida)/1024/1024:.2f} MB")
    
    def guardar_como_csv_columnas(self, archivo_salida):
        """
        Guarda solo los valores no cero de la matriz en formato CSV sin cabeceras.
        Cada línea contiene: fila,columna,valor
        
        Args:
            archivo_salida: Ruta del archivo CSV de salida
        """
        if self.matriz is None:
            raise ValueError("La matriz no ha sido inicializada")
        
        # Convertir a formato COO para exportar solo los elementos no cero
        coo_matriz = self.matriz.tocoo()
        
        print(f"Guardando elementos no cero en {archivo_salida}...")
        
        with open(archivo_salida, 'w') as f:
            for i, j, v in zip(coo_matriz.row, coo_matriz.col, coo_matriz.data):
                f.write(f"{v}\n")  # Solo escribimos el valor sin filas ni columnas
        
        print(f"Matriz guardada: {coo_matriz.nnz} elementos no cero.")
        print(f"Tamaño del archivo: {os.path.getsize(archivo_salida)/1024/1024:.2f} MB")
    
    def guardar_como_csv_completo(self, archivo_salida):
        """
        Guarda la matriz completa en formato CSV, incluyendo los ceros.
        
        Args:
            archivo_salida: Ruta del archivo CSV de salida
        """
        if self.matriz is None:
            raise ValueError("La matriz no ha sido inicializada")
        
        print(f"Convirtiendo matriz dispersa a formato denso para CSV...")
        matriz_densa = self.matriz.toarray()
        
        print(f"Guardando matriz completa en {archivo_salida}...")
        np.savetxt(archivo_salida, matriz_densa, delimiter=',', fmt='%.6f')
        
        print(f"Matriz guardada en formato CSV.")
        print(f"Tamaño del archivo: {os.path.getsize(archivo_salida)/1024/1024:.2f} MB")
    
    def visualizar_matriz(self, max_size=100):
        """
        Visualiza la matriz o una porción de ella si es muy grande.
        
        Args:
            max_size: Tamaño máximo de la porción a visualizar
        """
        if self.matriz is None:
            raise ValueError("La matriz no ha sido inicializada")
        
        # Para matrices grandes, mostrar solo una porción
        if self.size > max_size:
            # Tomamos una submatriz de las esquinas
            size_corner = min(max_size // 2, 50)
            top_left = self.matriz[:size_corner, :size_corner].toarray()
            top_right = self.matriz[:size_corner, -size_corner:].toarray()
            bottom_left = self.matriz[-size_corner:, :size_corner].toarray()
            bottom_right = self.matriz[-size_corner:, -size_corner:].toarray()
            
            # Crear una figura con las cuatro esquinas
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            
            axs[0, 0].matshow(top_left, cmap='viridis')
            axs[0, 0].set_title(f'Esquina superior izquierda ({size_corner}x{size_corner})')
            
            axs[0, 1].matshow(top_right, cmap='viridis')
            axs[0, 1].set_title(f'Esquina superior derecha ({size_corner}x{size_corner})')
            
            axs[1, 0].matshow(bottom_left, cmap='viridis')
            axs[1, 0].set_title(f'Esquina inferior izquierda ({size_corner}x{size_corner})')
            
            axs[1, 1].matshow(bottom_right, cmap='viridis')
            axs[1, 1].set_title(f'Esquina inferior derecha ({size_corner}x{size_corner})')
            
            plt.tight_layout()
            plt.savefig('visualizacion_matriz_dispersa.png')
            plt.show()
            
        else:
            # Si la matriz es pequeña, mostrarla completa
            plt.figure(figsize=(10, 8))
            plt.matshow(self.matriz.toarray(), cmap='viridis', fignum=1)
            plt.colorbar()
            plt.title(f'Matriz dispersa {self.size}x{self.size}')
            plt.savefig('matriz_dispersa_completa.png')
            plt.show()
        
        # También visualizar el espectro de densidad
        densidad = np.zeros(self.size)
        matriz_csr = self.matriz.tocsr()
        for i in range(self.size):
            densidad[i] = matriz_csr[i, :].nnz
        
        plt.figure(figsize=(10, 6))
        plt.plot(densidad, '.')
        plt.title('Densidad por fila')
        plt.xlabel('Índice de fila')
        plt.ylabel('Elementos no cero')
        plt.savefig('densidad_matriz_dispersa.png')
        plt.show()

def prueba_rendimiento(valores_n, densidad=0.0001):
    """
    Realiza pruebas de rendimiento para diferentes tamaños de matriz.
    
    Args:
        valores_n: Lista de valores de n a probar
        densidad: Densidad de la matriz
    
    Returns:
        Tupla de (tiempos_generacion, tiempos_multiplicacion, memorias)
    """
    tiempos_generacion = []
    tiempos_multiplicacion = []
    memorias = []
    
    for n in valores_n:
        tamanio = n**(2**n)
        print(f"\nProbando n={n}, tamaño de matriz={tamanio}x{tamanio}")
        
        try:
            # Inicializar matriz
            matriz = MatrizEspacioEstados(n)
            
            # Medir tiempo de generación
            start = timer()
            matriz.generar_matriz_aleatoria(densidad=densidad)
            end = timer()
            tiempo_gen = end - start
            tiempos_generacion.append(tiempo_gen)
            
            # Obtener información
            info = matriz.obtener_info()
            memorias.append(info['memoria_MB'])
            
            print(f"Elementos no cero: {info['elementos_no_cero']}")
            print(f"Densidad: {info['densidad']:.6f}%")
            print(f"Memoria utilizada: {info['memoria_MB']:.2f} MB")
            
            # Medir tiempo de multiplicación matriz-vector
            vector = np.random.rand(matriz.size)
            start = timer()
            resultado = matriz.aplicar_vector(vector)
            end = timer()
            tiempo_mult = end - start
            tiempos_multiplicacion.append(tiempo_mult)
            
            print(f"Tiempo de generación: {tiempo_gen:.4f} segundos")
            print(f"Tiempo de multiplicación: {tiempo_mult:.4f} segundos")
            
        except MemoryError:
            print(f"Memoria insuficiente para n={n}")
            tiempos_generacion.append(None)
            tiempos_multiplicacion.append(None)
            memorias.append(None)
            break
        
        except Exception as e:
            print(f"Error con n={n}: {e}")
            tiempos_generacion.append(None)
            tiempos_multiplicacion.append(None)
            memorias.append(None)
    
    return tiempos_generacion, tiempos_multiplicacion, memorias

def graficar_resultados(valores_n, tiempos_generacion, tiempos_multiplicacion, memorias):
    """
    Grafica los resultados de las pruebas de rendimiento.
    
    Args:
        valores_n: Lista de valores de n probados
        tiempos_generacion: Lista de tiempos de generación
        tiempos_multiplicacion: Lista de tiempos de multiplicación
        memorias: Lista de uso de memoria
    """
    # Filtrar valores válidos
    n_validos = []
    tg_validos = []
    tm_validos = []
    mem_validos = []
    
    for i, n in enumerate(valores_n):
        if tiempos_generacion[i] is not None:
            n_validos.append(n)
            tg_validos.append(tiempos_generacion[i])
            tm_validos.append(tiempos_multiplicacion[i])
            mem_validos.append(memorias[i])
    
    # Crear gráficas
    plt.figure(figsize=(15, 10))
    
    # Gráfica de tiempos
    plt.subplot(2, 1, 1)
    plt.plot(n_validos, tg_validos, 'o-', label='Tiempo de generación')
    plt.plot(n_validos, tm_validos, 's-', label='Tiempo de multiplicación')
    plt.xlabel('n')
    plt.ylabel('Tiempo (segundos)')
    plt.title('Tiempos de ejecución vs n')
    plt.grid(True)
    plt.legend()
    
    # Gráfica de memoria
    plt.subplot(2, 1, 2)
    plt.plot(n_validos, mem_validos, 'D-', color='green')
    plt.xlabel('n')
    plt.ylabel('Memoria (MB)')
    plt.title('Memoria utilizada vs n')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('resultados_matriz_dispersa.png')
    plt.show()

def demostrar_autovalores(n=2):
    """
    Demuestra el cálculo de autovalores para una matriz pequeña.
    
    Args:
        n: Tamaño de la dimensión base
    """
    print(f"\nDemostración de cálculo de autovalores para n={n}")
    matriz = MatrizEspacioEstados(n)
    matriz.generar_matriz_aleatoria(densidad=0.01)  # Mayor densidad para matrices pequeñas
    
    try:
        start = timer()
        autovalores = matriz.calcular_autovalores(k=5)
        end = timer()
        
        print(f"Autovalores más grandes: {autovalores}")
        print(f"Tiempo de cálculo: {end - start:.4f} segundos")
        return autovalores
    except Exception as e:
        print(f"Error al calcular autovalores: {e}")
        return None

def generar_matriz_csv(n, densidad, archivo_salida):
    """
    Genera una matriz dispersa y la guarda en formato CSV.
    
    Args:
        n: Tamaño de la dimensión base
        densidad: Densidad de la matriz
        archivo_salida: Ruta del archivo CSV de salida
    """
    print(f"\nGenerando matriz dispersa para n={n} (tamaño {n**(2**n)}x{n**(2**n)})...")
    matriz = MatrizEspacioEstados(n)
    matriz.generar_matriz_aleatoria(densidad=densidad)
    
    # Obtener información de la matriz
    info = matriz.obtener_info()
    print(f"Matriz generada con {info['elementos_no_cero']} elementos no cero")
    print(f"Densidad: {info['densidad']:.6f}%")
    print(f"Memoria utilizada: {info['memoria_MB']:.2f} MB")
    
    # Convertir a formato COO para exportar fácilmente
    coo_matriz = matriz.matriz.tocoo()
    
    print(f"Guardando matriz en {archivo_salida}...")
    
    # Crear el directorio si no existe
    os.makedirs(os.path.dirname(archivo_salida), exist_ok=True)
    
    # Guardar en formato COO (fila, columna, valor)
    with open(archivo_salida, 'w') as f:
        # Encabezado con información
        f.write(f"# Matriz dispersa {n**(2**n)}x{n**(2**n)}, {info['elementos_no_cero']} elementos no cero, densidad {info['densidad']:.6f}%\n")
        f.write("# formato: fila,columna,valor\n")
        
        # Escribir cada elemento no cero
        for i, j, v in zip(coo_matriz.row, coo_matriz.col, coo_matriz.data):
            f.write(f"{i},{j},{v}\n")
    
    print(f"Matriz guardada exitosamente en formato COO.")
    print(f"Tamaño del archivo: {os.path.getsize(archivo_salida)/1024/1024:.2f} MB")
    
    return matriz

def cargar_matriz_csv(archivo):
    """
    Carga una matriz dispersa desde un archivo CSV en formato COO.
    
    Args:
        archivo: Ruta del archivo CSV
        
    Returns:
        Matriz dispersa en formato CSR
    """
    print(f"Cargando matriz desde {archivo}...")
    
    filas = []
    columnas = []
    datos = []
    
    with open(archivo, 'r') as f:
        # Saltar las líneas de encabezado
        for linea in f:
            if linea.startswith('#'):
                continue
                
            # Procesar cada línea de datos
            fila, columna, valor = linea.strip().split(',')
            filas.append(int(fila))
            columnas.append(int(columna))
            datos.append(float(valor))
    
    # Obtener dimensiones
    if filas and columnas:
        n = max(max(filas), max(columnas)) + 1
    else:
        n = 0
    
    print(f"Matriz cargada: {n}x{n} con {len(datos)} elementos no cero")
    
    # Crear matriz en formato COO y convertir a CSR
    return sp.coo_matrix((datos, (filas, columnas)), shape=(n, n)).tocsr()

def ejecutar_pruebas_rendimiento():
    """
    Ejecuta pruebas de rendimiento para diferentes valores de n.
    """
    valores_n = [1, 2, 3]  # Para n=4, el tamaño sería 4^16 lo cual es enorme
    print("\nEjecutando pruebas de rendimiento...")
    tiempos_gen, tiempos_mult, memorias = prueba_rendimiento(valores_n)
    print("\nGenerando gráficas...")
    graficar_resultados(valores_n, tiempos_gen, tiempos_mult, memorias)

def generar_matriz_y_guardar():
    """
    Genera una matriz dispersa de tamaño n * 2^n y la guarda en formato CSV.
    
    Returns:
        Tupla (matriz, archivo_salida) con la matriz generada y la ruta del archivo CSV.
    """
    # Para n=20, el tamaño de la matriz es 20 * 2^20, lo cual es enorme 
    n = 8
    
    # Usamos una densidad del 20% para que sea interesante pero manejable
    densidad = 0.2  # 20% de elementos no cero
    
    print(f"Generando matriz dispersa para n={n} (tamaño {n * (2**n)}x{n * (2**n)})...")
    matriz = MatrizEspacioEstados(n)
    matriz.generar_matriz_aleatoria(densidad=densidad)
    
    # Obtener información de la matriz
    info = matriz.obtener_info()
    print(f"Matriz generada con {info['elementos_no_cero']} elementos no cero")
    print(f"Densidad: {info['densidad']:.6f}%")
    print(f"Memoria utilizada: {info['memoria_MB']:.2f} MB")
    
    # Crear el directorio si no existe
    os.makedirs("datos", exist_ok=True)
    
    # Guardamos la matriz completa en formato CSV
    archivo_salida = os.path.join("datos", f"matriz_n{n}_tamaño_{n * (2**n)}.csv")
    matriz.guardar_como_csv_completo(archivo_salida)
    
    return matriz, archivo_salida