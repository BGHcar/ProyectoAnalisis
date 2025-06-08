# src/main.py
from src.controllers.manager import Manager
from src.strategies.geometry import Geometric
from src.strategies.q_nodes import QNodes
from src.strategies.phi import Phi
import time

def calcular_metricas_patron(resultado_qnodes, resultado_geo, t_qnodes, t_geo):
    """Calcula métricas para UN patrón específico."""
    phi_qnodes = getattr(resultado_qnodes, 'perdida', 0.0)
    phi_geo = getattr(resultado_geo, 'perdida', 0.0)
    
    # Acierto exacto para este caso
    acierto_exacto = abs(phi_qnodes - phi_geo) < 1e-9
    
    # Error relativo
    if abs(phi_qnodes) > 1e-10:
        error_relativo = abs(phi_qnodes - phi_geo) / abs(phi_qnodes)
    else:
        error_relativo = 0.0 if abs(phi_geo) < 1e-10 else float('inf')
    
    # Speedup
    speedup = t_qnodes / t_geo if t_geo > 0 else float('inf')
    
    return acierto_exacto, error_relativo, speedup, phi_qnodes, phi_geo

def calcular_metricas_globales(todos_resultados):
    """Calcula métricas globales de todos los casos."""
    if not todos_resultados:
        return 0.0, 0.0, 0.0
    
    # Tasa de acierto exacto global
    aciertos = sum(1 for r in todos_resultados if r['acierto_exacto'])
    tasa_acierto_global = (aciertos / len(todos_resultados)) * 100
    
    # Error relativo máximo
    errores = [r['error_relativo'] for r in todos_resultados if r['error_relativo'] != float('inf')]
    error_max = max(errores) if errores else 0.0
    
    # Speedup promedio
    speedups = [r['speedup'] for r in todos_resultados if r['speedup'] != float('inf')]
    speedup_prom = sum(speedups) / len(speedups) if speedups else 0.0
    
    return tasa_acierto_global, error_max, speedup_prom

def generar_patrones(n: int) -> dict:
    """
    Genera todos los patrones estándar para un sistema de n variables.
    
    Args:
        n (int): Número de variables del sistema
        
    Returns:
        dict: Diccionario con nombre_patron: cadena_binaria
    """
    patrones = {}
    
    # Completo - todas las variables activas
    patrones["completo"] = "1" * n
    
    # Desaparece último - última variable inactiva
    patrones["desaparece_ultimo"] = "1" * (n - 1) + "0" if n > 0 else ""
    
    # Desaparece primer - primera variable inactiva  
    patrones["desaparece_primer"] = "0" + "1" * (n - 1) if n > 0 else ""
    
    # Desaparece primero y último - primera y última variables inactivas
    if n >= 2:
        patrones["desaparece_primero_ultimo"] = "0" + "1" * (n - 2) + "0"
    else:
        patrones["desaparece_primero_ultimo"] = "0" * n
    
    # Solo impares - variables en posiciones impares activas (índices 0, 2, 4...)
    patrones["solo_impares"] = "".join("1" if i % 2 == 0 else "0" for i in range(n))
    
    # Solo pares - variables en posiciones pares activas (índices 1, 3, 5...)
    patrones["solo_pares"] = "".join("1" if i % 2 == 1 else "0" for i in range(n))
    
    # Desaparece penúltimo - penúltima variable inactiva
    if n >= 2:
        patrones["desaparece_penultimo"] = "".join("0" if i == n - 2 else "1" for i in range(n))
    else:
        patrones["desaparece_penultimo"] = "1" * n
        
    return patrones

def convertir_sistema_candidato(sistema_candidato: str) -> str:
    """
    Convierte el sistema candidato a formato binario compatible con Manager.
    
    Args:
        sistema_candidato (str): Puede ser:
                               - Letras: "ABCDEFGHIJ" 
                               - Binario: "1000000000"
        
    Returns:
        str: Estado inicial binario válido
    """
    # Si ya es binario (solo contiene 0s y 1s), usar directamente
    if all(c in '01' for c in sistema_candidato):
        return sistema_candidato
    
    # Si son letras, generar estado inicial estándar (primer bit en 1, resto en 0)
    n = len(sistema_candidato)
    return "1" + "0" * (n - 1)

def iniciar(sistema_candidato: str):
    """
    Ejecuta análisis de estrategias para un sistema candidato específico.
    Los patrones se generan dinámicamente basados en la longitud del sistema.
    
    Args:
        sistema_candidato (str): Sistema candidato en cualquier formato válido
    """
    # Convertir a formato binario compatible
    estado_inicial = convertir_sistema_candidato(sistema_candidato)
    n = len(sistema_candidato)
    condiciones = "1" * n
    
    # Información del sistema
    print(f"🔬 Sistema original: {sistema_candidato}")
    if sistema_candidato != estado_inicial:
        print(f"📊 Estado inicial: {estado_inicial}")
    print(f"🎯 Variables: {n}")
    
    # Generar patrones dinámicamente
    patrones_dict = generar_patrones(n)
    
    # Orden de patrones para mantener compatibilidad
    nombres_patrones = [
        "completo",
        "desaparece_ultimo", 
        "desaparece_primer",
        "desaparece_primero_ultimo",
        "solo_impares",
        "solo_pares",
        "desaparece_penultimo"
    ]
    
    gestor_sistema = Manager(estado_inicial)
    contador = 0
    todos_resultados = []
    
    for patron_alcance_nombre in nombres_patrones:
        for patron_mecanismo_nombre in nombres_patrones:
            alcance = patrones_dict[patron_alcance_nombre]
            mecanismo = patrones_dict[patron_mecanismo_nombre]
            
            # Análisis con QNodes (referencia)
            analizador_qnodes = QNodes(gestor_sistema)
            t_qnodes_start = time.time()
            resultado_qnodes = analizador_qnodes.aplicar_estrategia(condiciones, alcance, mecanismo)
            t_qnodes = time.time() - t_qnodes_start
            print(resultado_qnodes)
            
            contador += 1

            # Análisis con Geometric
            analizador_geo = Geometric(gestor_sistema)
            t_geo_start = time.time()
            resultado_geo = analizador_geo.aplicar_estrategia(condiciones, alcance, mecanismo)
            t_geo = time.time() - t_geo_start
            print(resultado_geo, contador)
            
            # CALCULAR Y MOSTRAR MÉTRICAS PARA ESTE PATRÓN
            exacto, error, speedup, phi_qnodes, phi_geo = calcular_metricas_patron(
                resultado_qnodes, resultado_geo, t_qnodes, t_geo
            )
            
            exacto_str = "✓" if exacto else "✗"
            print(f"Patrón {patron_alcance_nombre}×{patron_mecanismo_nombre}: QNodes_Φ={phi_qnodes:.6f} | Geo_Φ={phi_geo:.6f} | Acierto={exacto_str} | Error={error:.6f} | Speedup={speedup:.2f}x")
            
            # Guardar para métricas globales
            todos_resultados.append({
                'acierto_exacto': exacto,
                'error_relativo': error,
                'speedup': speedup
            })
    
    # MOSTRAR MÉTRICAS GLOBALES AL FINAL
    tasa_global, error_max_global, speedup_prom_global = calcular_metricas_globales(todos_resultados)
    
    print("\n" + "="*50)
    print("MÉTRICAS GLOBALES (QNodes vs Geometric):")
    print(f"Tasa de acierto exacto: {tasa_global:.1f}%")
    print(f"Error relativo máximo: {error_max_global:.6f}")
    print(f"Speedup promedio: {speedup_prom_global:.2f}x")
    print("="*50)
