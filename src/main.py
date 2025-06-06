# src/main.py
from src.controllers.manager import Manager
from src.strategies.geometry import Geometric
from src.strategies.q_nodes import QNodes
import time

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
    
    for patron_alcance_nombre in nombres_patrones:
        for patron_mecanismo_nombre in nombres_patrones:
            alcance = patrones_dict[patron_alcance_nombre]
            mecanismo = patrones_dict[patron_mecanismo_nombre]
            
            # Análisis con QNodes
            """ analizador_qnodes = QNodes(gestor_sistema)
            resultado_qnodes = analizador_qnodes.aplicar_estrategia(condiciones, alcance, mecanismo)
            print(resultado_qnodes) """
            
            contador += 1

            # Análisis con Geometric
            analizador_geo = Geometric(gestor_sistema)
            resultado_geo = analizador_geo.aplicar_estrategia(condiciones, alcance, mecanismo)
            print(resultado_geo, contador)

if __name__ == "__main__":
    # Ejemplo de uso por defecto
    iniciar("ABC")  # Sistema de 24 variables
