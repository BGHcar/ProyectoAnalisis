from pathlib import Path
import time
from src.controllers.manager import Manager

from src.controllers.strategies.force import BruteForce
from src.controllers.strategies.q_nodes import QNodes
# Asumiendo que Phi es una estrategia adicional que necesitas importar
from src.controllers.strategies.phi import Phi

class ResultadosAnalisis:
    def __init__(self):
        self.results_dir = Path('resultados')
        self.results_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.results_file = self.results_dir / f'Qnodes20A.txt'
        
        # Inicializar archivo
        with open(self.results_file, 'w', encoding='utf-8') as f:
            f.write("RESULTADOS DEL ANÁLISIS\n")
            f.write("=" * 50 + "\n\n")
    
    def save_to_file(self, patron_t1: str, patron_t: str, alcance: str, resultado_str: str):
        """Guarda los resultados de un análisis específico"""
        try:
            with open(self.results_file, 'a', encoding='utf-8') as f:
                f.write(f"\nPATRÓN t+1: {patron_t1}\n")
                f.write(f"PATRÓN t: {patron_t}\n")
                f.write(f"ALCANCE: {alcance}\n")
                f.write(f"RESULTADO: {resultado_str}\n")
                f.write("-" * 30 + "\n")
        except Exception as e:
            print(f"Error al guardar resultados: {e}")
    
    def save_summary(self, total: int):
        """Guarda el resumen final"""
        try:
            with open(self.results_file, 'a', encoding='utf-8') as f:
                f.write("\nRESUMEN FINAL\n")
                f.write("-" * 50 + "\n")
                f.write(f"Total de análisis realizados: {total}\n")
                f.write(f"Tiempo de finalización: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
        except Exception as e:
            print(f"Error al guardar resumen: {e}")

class SistemaAnalisis:
    def __init__(self, sistema_candidato: str):
        self.sistema_candidato = sistema_candidato
        self.resultados = ResultadosAnalisis()
        self.total = 0
        
        # Estado inicial del sistema
        self.estado_inicio = "1" + "0" * (len(sistema_candidato) - 1)
        
        # Configuración del sistema
        self.config_sistema = Manager(estado_inicial=self.estado_inicio)
        
        # Condiciones (siempre todo 1)
        self.condiciones = "1" * len(sistema_candidato)
        
        # Patrones de subsistemas
        self.patrones = [
            "completo",
            "desaparece_ultimo",
            "desaparece_primer",
            "desaparece_primero_ultimo",
            "solo_impares",
            "solo_pares",
            "desaparece_penultimo"
        ]

    def generar_alcance(self, patron: str) -> str:
        """Genera el alcance (t+1) basado en el patrón"""
        n = len(self.sistema_candidato)
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
            # Cada tercer elemento es 0, los demás son 1
            return "".join(["0" if i > 0 and (i + 1) % 3 == 0 else "1" for i in range(n)])
        else:
            raise ValueError("Patrón no válido")

    def ejecutar_analisis(self):
        """Ejecuta el análisis completo"""
        try:
            # Primero ejecuta los patrones existentes
            for patron_t1 in self.patrones:
                alcance = self.generar_alcance(patron_t1)
                
                for patron_t in self.patrones:
                    self.total += 1
                    print(f"\nAnálisis {self.total}: {patron_t1} - {patron_t}")
                    
                    mecanismo = "1" * len(self.sistema_candidato)
                    
                    analizador_fb = QNodes(self.config_sistema)
                    resultado = analizador_fb.aplicar_estrategia(
                        self.condiciones, 
                        alcance, 
                        mecanismo
                    )
                    
                    self.resultados.save_to_file(
                        patron_t1=patron_t1,
                        patron_t=patron_t,
                        alcance=alcance,
                        resultado_str=str(resultado)
                    )
                    
                    print(f"t+1: {patron_t1} | t: {patron_t}")
                    print(f"Alcance: {alcance}")
                    print(f"Resultado: {resultado}\n")

            # Añadir el patrón especial después de los bucles
            self.total += 1
            n = len(self.sistema_candidato)
            
            # Generar el patrón dinámicamente
            # Para t: elimina el primer elemento
            patron_t = "0" + "1" * (n-1) 
            #BCDEFGJ_{t+1}|BCDEFGHIJ_{t}
            
            # Construcción del patrón por partes:
            primer_parte = "0" + "1" * 6  # 0111111 (primeros 7 valores)
            segunda_parte = "00"          # Los dos ceros después del sexto valor
            tercera_parte = "1" * (n-9)   # Unos hasta completar la longitud total
            
            patron_t1 = primer_parte + segunda_parte + tercera_parte
            
            mecanismo = "1" * n
            
            analizador_fb = Phi(self.config_sistema)
            resultado = analizador_fb.aplicar_estrategia(
                self.condiciones,
                patron_t1,
                mecanismo
            )
            
            # Guardar el resultado del patrón especial
            self.resultados.save_to_file(
                patron_t1="patron_especial_t1",
                patron_t="patron_especial_t",
                alcance=patron_t1,
                resultado_str=str(resultado)
            )
            
            print(f"\nAnálisis del patrón especial")
            print(f"t+1: {patron_t1}")
            print(f"t: {patron_t}")
            print(f"Resultado: {resultado}\n")

        except Exception as e:
            print(f"Error durante la ejecución: {e}")
            with open(self.resultados.results_file, 'a', encoding='utf-8') as f:
                f.write(f"\nERROR EN LA EJECUCIÓN: {str(e)}\n")
        
        finally:
            self.resultados.save_summary(self.total)
            
            print(f"\nTotal de análisis realizados: {self.total}")
            print(f"Resultados guardados en: {self.resultados.results_file}")

def iniciar(sistema_candidato: str):
    """Función de entrada para iniciar el análisis"""
    sistema = SistemaAnalisis(sistema_candidato)
    sistema.ejecutar_analisis()
