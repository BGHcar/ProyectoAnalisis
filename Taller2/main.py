from src.matriz_dispersa import generar_matriz_y_guardar
from src.procesador_bloques import procesar_matriz_por_bloques
import os
import argparse
import multiprocessing

def main():
    """
    Punto de entrada principal que genera una matriz dispersa de buen tamaño
    y la guarda en formato CSV, luego ofrece la opción de procesarla por bloques.
    """
    parser = argparse.ArgumentParser(description='Generación y procesamiento de matrices de espacios de estados')
    parser.add_argument('--solo-generar', action='store_true', help='Solo genera la matriz sin procesar por bloques')
    parser.add_argument('--tamanio-bloque', type=int, default=16, help='Tamaño de los bloques para procesamiento (defecto: 16)')
    parser.add_argument('--todos-bloques', action='store_true', help='Procesa todos los bloques, incluso los que son todos ceros')
    parser.add_argument('--umbral-relevancia', type=float, default=0.0, help='Umbral mínimo de densidad para considerar un bloque relevante (defecto: 0.0)')
    parser.add_argument('--paralelo', action='store_true', help='Utiliza procesamiento paralelo con múltiples núcleos')
    parser.add_argument('--num-procesos', type=int, default=None, help='Número de procesos a utilizar en modo paralelo (defecto: automático)')
    args = parser.parse_args()
    
    print("======== GENERACIÓN DE MATRIZ DISPERSA PARA ESPACIO DE ESTADOS ========")
    print(f"Núcleos disponibles: {multiprocessing.cpu_count()}")
    
    # Generar matriz y guardarla en CSV
    matriz, archivo_csv = generar_matriz_y_guardar()
    print(f"Matriz guardada en: {archivo_csv}")
    print(f"Tamaño de la matriz: {matriz.size}x{matriz.size}")
    
    info = matriz.obtener_info()
    print(f"Elementos no cero: {info['elementos_no_cero']}")
    print(f"Densidad: {info['densidad']:.6f}%")
    print(f"Memoria utilizada: {info['memoria_MB']:.2f} MB")
    
    print("\n======== MATRIZ GENERADA EXITOSAMENTE ========")
    
    # Si no se especifica --solo-generar, procesamos por bloques
    if not args.solo_generar:
        # Determinamos si procesamos solo bloques relevantes o todos
        solo_relevantes = not args.todos_bloques
        
        # Mensaje del modo de procesamiento
        if args.paralelo:
            print("\n======== PROCESAMIENTO PARALELO POR BLOQUES ========")
            if solo_relevantes:
                print(f"Modo: Procesamiento paralelo selectivo (solo bloques con densidad > {args.umbral_relevancia}%)")
            else:
                print("Modo: Procesamiento paralelo completo (todos los bloques)")
        else:
            print("\n======== PROCESAMIENTO SECUENCIAL POR BLOQUES ========")
            if solo_relevantes:
                print(f"Modo: Procesamiento selectivo (solo bloques con densidad > {args.umbral_relevancia}%)")
            else:
                print("Modo: Procesamiento completo (todos los bloques)")
        
        # Procesamos la matriz con los parámetros especificados
        # Usamos procesador_bloques.py para ambos casos (paralelo y secuencial)
        resultados = procesar_matriz_por_bloques(
            archivo_csv, 
            tamanio_bloque=args.tamanio_bloque,
            solo_relevantes=solo_relevantes,
            umbral_relevancia=args.umbral_relevancia,
            paralelo=args.paralelo,
            num_procesos=args.num_procesos
        )
        
        # Determinamos el directorio de resultados según el modo de procesamiento
        if args.paralelo:
            directorio_resultados = os.path.join(os.path.dirname(archivo_csv), "resultados_bloques")
        else:
            directorio_resultados = os.path.join(os.path.dirname(archivo_csv), "resultados_bloques")
        
        # Mostramos estadísticas del procesamiento
        print("\nEstadísticas del procesamiento por bloques:")
        print(f"Tiempo total: {resultados['tiempo_total']:.2f} segundos")
        print(f"Bloques procesados: {resultados['bloques_procesados']} de {resultados['bloques_relevantes']} relevantes")
        
        if 'speedup' in resultados:
            print(f"Speedup: {resultados['speedup']:.2f}x")
            print(f"Eficiencia: {resultados['eficiencia']*100:.2f}%")
        
        if solo_relevantes and 'bloques_ignorados' in resultados:
            print(f"Bloques ignorados: {resultados['bloques_ignorados']}")
            total_bloques = resultados['bloques_procesados'] + resultados['bloques_ignorados']
            print(f"Eficiencia espacial: {resultados['bloques_procesados']/total_bloques*100:.2f}% (solo se procesó lo necesario)")
        
        print(f"\nLos resultados detallados están disponibles en: {directorio_resultados}")

if __name__ == "__main__":
    main()