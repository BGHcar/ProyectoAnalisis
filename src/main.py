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
    
    # Error relativo
    if abs(phi_qnodes) > 1e-10:
        error_relativo = abs(phi_qnodes - phi_geo) / abs(phi_qnodes)
    elif abs(phi_geo) > 1e-10:
        error_relativo = abs(phi_qnodes - phi_geo) / abs(phi_geo)
    else:
        error_relativo = 0.0
    
    # Tasa de acierto para este patrón (1 o 0)
    acierto_exacto = error_relativo < 0.01
    
    # Speedup relativo según fórmula (5.3)
    if t_geo > 1e-6:
        speedup_relativo = t_qnodes / t_geo
    else:
        speedup_relativo = 1000.0
    
    # Distancia estructural (Jaccard) - simplificada para biparticiones
    # Asumiendo que las biparticiones son conjuntos binarios
    distancia_estructural = abs(phi_qnodes - phi_geo) / max(abs(phi_qnodes), abs(phi_geo), 1e-10)
    
    return acierto_exacto, error_relativo, speedup_relativo, distancia_estructural, phi_qnodes, phi_geo

def calcular_metricas_globales(todos_resultados):
    """Calcula métricas globales según el documento."""
    if not todos_resultados:
        return 0.0, 0.0, 0.0, 0.0, "Insuficiente", False
    
    # Tasa de acierto exacto global (%)
    aciertos = sum(1 for r in todos_resultados if r['acierto_exacto'])
    tasa_acierto_global = (aciertos / len(todos_resultados)) * 100
    
    # Error relativo máximo
    error_max = max(r['error_relativo'] for r in todos_resultados)
    
    # Distancia estructural máxima
    distancia_max = max(r['distancia_estructural'] for r in todos_resultados)
    
    # Speedup promedio
    speedup_prom = sum(r['speedup'] for r in todos_resultados) / len(todos_resultados)
    
    # Clasificación según TODAS las métricas del Cuadro 5.1
    if (tasa_acierto_global > 90 and error_max < 0.01 and distancia_max < 0.1):
        clasificacion = "Excelente"
    elif (tasa_acierto_global > 80 and error_max < 0.05 and distancia_max < 0.2):
        clasificacion = "Bueno"
    elif (tasa_acierto_global > 70 and error_max < 0.10 and distancia_max < 0.3):
        clasificacion = "Aceptable"
    else:
        clasificacion = "Insuficiente"
    
    # Validez
    es_valido = clasificacion in ["Excelente", "Bueno", "Aceptable"]
    
    return tasa_acierto_global, error_max, distancia_max, speedup_prom, clasificacion, es_valido

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
    
    """ for patron_alcance_nombre in nombres_patrones:
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
            
            # CALCULAR MÉTRICAS ACTUALIZADAS
            exacto, error, speedup, distancia, phi_qnodes, phi_geo = calcular_metricas_patron(
                resultado_qnodes, resultado_geo, t_qnodes, t_geo
            )
            
            exacto_str = "✓" if exacto else "✗"
            print(f"Patrón {patron_alcance_nombre}×{patron_mecanismo_nombre}: QNodes_Φ={phi_qnodes:.6f} | Geo_Φ={phi_geo:.6f} | Acierto={exacto_str} | Error={error:.6f} | Dist={distancia:.6f} | Speedup={speedup:.2f}x")
            
            # Guardar métricas completas
            todos_resultados.append({
                'acierto_exacto': exacto,
                'error_relativo': error,
                'distancia_estructural': distancia,
                'speedup': speedup
            }) """
    
    # PATRÓN PERSONALIZADO
    if n >= 20:
        alcance_personalizado = "10111111111111111111"
        mecanismo_personalizado = "10111111111111111111"
    elif n >= 15:
        alcance_personalizado = "011111100111111"
        mecanismo_personalizado = "111111111111111"
    else:
        alcance_personalizado = "0111111001"  
        mecanismo_personalizado = "1111111111"

    
    # Análisis con QNodes
    analizador_qnodes = QNodes(gestor_sistema)
    t_qnodes_start = time.time()
    resultado_qnodes = analizador_qnodes.aplicar_estrategia(condiciones, alcance_personalizado, mecanismo_personalizado)
    t_qnodes = time.time() - t_qnodes_start
    print(resultado_qnodes)
    
    contador += 1

    # Análisis con Geometric
    analizador_geo = Geometric(gestor_sistema)
    t_geo_start = time.time()
    resultado_geo = analizador_geo.aplicar_estrategia(condiciones, alcance_personalizado, mecanismo_personalizado)
    t_geo = time.time() - t_geo_start
    print(resultado_geo, contador)
    
    # CALCULAR MÉTRICAS PARA EL PATRÓN PERSONALIZADO
    exacto, error, speedup, distancia, phi_qnodes, phi_geo = calcular_metricas_patron(
        resultado_qnodes, resultado_geo, t_qnodes, t_geo
    )
    
    exacto_str = "✓" if exacto else "✗"
    print(f"Patrón PERSONALIZADO: QNodes_Φ={phi_qnodes:.6f} | Geo_Φ={phi_geo:.6f} | Acierto={exacto_str} | Error={error:.6f} | Dist={distancia:.6f} | Speedup={speedup:.2f}x")
    
    # Guardar métricas completas
    todos_resultados.append({
        'acierto_exacto': exacto,
        'error_relativo': error,
        'distancia_estructural': distancia,
        'speedup': speedup
    })
    
    # MOSTRAR MÉTRICAS GLOBALES COMPLETAS
    tasa_global, error_max_global, distancia_max_global, speedup_prom_global, clasificacion, es_valido = calcular_metricas_globales(todos_resultados)
    
    validez_str = "VÁLIDO ✓" if es_valido else "INVÁLIDO ✗"
    
    print("\n" + "="*60)
    print("MÉTRICAS GLOBALES (QNodes vs Geometric):")
    print(f"Tasa de acierto exacto: {tasa_global:.1f}%")
    print(f"Error relativo máximo: {error_max_global:.6f}")
    print(f"Distancia estructural máxima: {distancia_max_global:.6f}")
    print(f"Speedup promedio: {speedup_prom_global:.2f}x")
    print(f"Nivel de calidad: {clasificacion}")
    print(f"Resultado: {validez_str}")
    print("="*60)
