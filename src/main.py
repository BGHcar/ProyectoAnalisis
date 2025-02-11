from src.controllers.manager import Manager
from src.controllers.strategies.q_nodes import QNodes
from src.controllers.strategies.phi import Phi


def generar_alcance(patron, sistema_candidato):
    """Genera el alcance (t+1) basado en el patrón y el sistema candidato."""
    n = len(sistema_candidato)
    if patron == "completo":
        return "1" * n
    elif patron == "desaparece_ultimo":
        return "1" * (n - 1) + "0"
    elif patron == "desaparece_primer":
        return "0" + "1" * (n - 1)
    elif patron == "desaparece_primero_ultimo":
        return "0" + "1" * (n - 2) + "0"
    elif patron == "solo_impares":
        return "".join(["1" if i % 2 == 0 else "0" for i in range(n)])
    elif patron == "solo_pares":
        return "".join(["1" if i % 2 == 1 else "0" for i in range(n)])
    elif patron == "desaparece_penultimo":
        return "1" * (n - 2) + "01"
    else:
        raise ValueError("Patrón no válido")

def iniciar(sistema_candidato):
    """Punto de entrada principal"""
    # Estado inicial del sistema (siempre el mismo)
    estado_inicio = "1" + "0" * (len(sistema_candidato) - 1)
    
    # Configuración del sistema
    config_sistema = Manager(estado_inicial=estado_inicio)
    
    # Condiciones (siempre todo 1)
    condiciones = "1" * len(sistema_candidato)
    
    # Patrones de subsistemas
    patrones = [
        "completo",
        "desaparece_ultimo",
        "desaparece_primer",
        "desaparece_primero_ultimo",
        "solo_impares",
        "solo_pares",
        "desaparece_penultimo"
    ]
    
    # Iterar sobre los patrones de t+1
    for patron_t1 in patrones:
        # Generar alcance (t+1) basado en el patrón
        alcance = generar_alcance(patron_t1, sistema_candidato)
        
        # Iterar sobre los patrones de t
        for patron_t in patrones:
            # Mecanismo (t) siempre es todo 1
            mecanismo = "1" * len(sistema_candidato)
            
            # Aplicar la estrategia QNodes
            analizador_fb = QNodes(config_sistema)
            resultado = analizador_fb.aplicar_estrategia(condiciones, alcance, mecanismo)
            
            # Mostrar resultados
            print(f"t+1: {patron_t1} | t: {patron_t}")
            print(f"Alcance: {alcance}")
            print(f"Resultado: {resultado}\n")