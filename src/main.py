from src.controllers.manager import Manager
from src.controllers.strategies.q_nodes import QNodes
from src.controllers.strategies.phi import Phi
from pathlib import Path
import time
import re

class ResultadosAnalisis:
    def __init__(self):
        self.results_dir = Path('resultados')
        self.results_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.results_file = self.results_dir / f'resultados_{timestamp}.txt'
        
        # Inicializar archivo
        with open(self.results_file, 'w', encoding='utf-8') as f:
            f.write("RESULTADOS DEL ANÁLISIS\n")
            f.write("=" * 50 + "\n\n")

    def clean_ansi(self, text: str) -> str:
        """Elimina códigos ANSI y mantiene el texto limpio"""
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    def save_to_file(self, patron_t1: str, patron_t: str, alcance: str, resultado_str: str):
        """Guarda los resultados sin códigos ANSI"""
        try:
            with open(self.results_file, 'a', encoding='utf-8') as f:
                f.write(f"Fecha y hora: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Patrón t+1: {patron_t1}\n")
                f.write(f"Patrón t: {patron_t}\n")
                f.write(f"Alcance: {alcance}\n")
                f.write("-" * 50 + "\n")
                # Limpiar y escribir el resultado
                clean_result = self.clean_ansi(resultado_str)
                f.write(clean_result + "\n")
                f.write("=" * 50 + "\n\n")
                
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
            return "1" * (n - 2) + "01"
        else:
            raise ValueError("Patrón no válido")

    def ejecutar_analisis(self):
        """Ejecuta el análisis completo"""
        try:
            for patron_t1 in self.patrones:
                alcance = self.generar_alcance(patron_t1)
                
                for patron_t in self.patrones:
                    self.total += 1
                    print(f"\nAnálisis {self.total}: {patron_t1} - {patron_t}")
                    
                    mecanismo = "1" * len(self.sistema_candidato)
                    
                    analizador_fb = Phi(self.config_sistema)
                    resultado = analizador_fb.aplicar_estrategia(
                        self.condiciones, 
                        alcance, 
                        mecanismo
                    )
                    
                    # Guardar resultado
                    self.resultados.save_to_file(
                        patron_t1=patron_t1,
                        patron_t=patron_t,
                        alcance=alcance,
                        resultado_str=str(resultado)
                    )
                    
                    # Mostrar en consola
                    print(f"t+1: {patron_t1} | t: {patron_t}")
                    print(f"Alcance: {alcance}")
                    print(f"Resultado: {resultado}\n")
        
        except Exception as e:
            print(f"Error durante la ejecución: {e}")
            with open(self.resultados.results_file, 'a', encoding='utf-8') as f:
                f.write(f"\nERROR EN LA EJECUCIÓN: {str(e)}\n")
        
        finally:
            # Guardar resumen final
            self.resultados.save_summary(self.total)
            
            print(f"\nTotal de análisis realizados: {self.total}")
            print(f"Resultados guardados en: {self.resultados.results_file}")

def iniciar(sistema_candidato: str):
    """Función de entrada para iniciar el análisis"""
    sistema = SistemaAnalisis(sistema_candidato)
    sistema.ejecutar_analisis()
