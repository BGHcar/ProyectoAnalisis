import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def visualizar_matriz(matriz, titulo="Matriz"):
    """
    Visualiza una matriz como un mapa de calor
    
    Args:
        matriz: Matriz a visualizar
        titulo: Título para la visualización
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(matriz, cmap='Blues', interpolation='none')
    plt.colorbar(label='Valor')
    plt.title(titulo)
    
    # Añadir etiquetas de ejes
    ticks = np.arange(len(matriz))
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.xlabel('Nodo destino')
    plt.ylabel('Nodo origen')
    
    plt.tight_layout()
    plt.show()

def visualizar_grafo(matriz, titulo="Grafo de la matriz"):
    """
    Visualiza la matriz como un grafo dirigido
    
    Args:
        matriz: Matriz de adyacencia
        titulo: Título para la visualización
    """
    G = nx.DiGraph()
    
    # Añadir nodos
    n = len(matriz)
    G.add_nodes_from(range(n))
    
    # Añadir aristas
    for i in range(n):
        for j in range(n):
            if matriz[i, j] > 0:
                G.add_edge(i, j)
    
    # Dibujar el grafo
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', 
            node_size=700, arrowsize=20, font_size=15, font_weight='bold')
    
    plt.title(titulo)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def graficar_comparacion(x_valores, y_valores, etiquetas, titulo, xlabel, ylabel):
    """
    Genera un gráfico de comparación
    
    Args:
        x_valores: Lista de valores para el eje X
        y_valores: Lista de listas de valores para el eje Y
        etiquetas: Lista de etiquetas para la leyenda
        titulo: Título del gráfico
        xlabel: Etiqueta para el eje X
        ylabel: Etiqueta para el eje Y
    """
    plt.figure(figsize=(10, 6))
    
    for i, y in enumerate(y_valores):
        plt.plot(x_valores, y, 'o-', linewidth=2, label=etiquetas[i])
    
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.show()