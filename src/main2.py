# src/main.py
from pathlib import Path
import time
from src.controllers.manager import Manager

from src.strategies.force import BruteForce
from src.strategies.q_nodes import QNodes
from src.strategies.geometry import Geometry
from src.strategies.phi import Phi

class ResultadosAnalisis:
    def __init__(self, estrategia_principal: str = "Geometry"):
        self.results_dir = Path('resultados')
        self.results_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.results_file = self.results_dir / f'Analisis_{estrategia_principal}_{timestamp}.txt'
        
        # Inicializar archivo con mejor formato
        with open(self.results_file, 'w', encoding='utf-8') as f:
            f.write(f"ANÁLISIS DE BIPARTICIÓN ÓPTIMA - {estrategia_principal}\n")
            f.write("=" * 70 + "\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Estrategia principal: {estrategia_principal}\n\n")
    
    def save_to_file(self, patron_t1: str, patron_t: str, alcance: str, 
                     resultado_str: str, estrategia: str = "", tiempo: float = 0.0):
        """Guarda los resultados con información de rendimiento"""
        try:
            with open(self.results_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"ANÁLISIS #{self.get_analysis_count()}\n")
                f.write(f"Estrategia: {estrategia}\n")
                f.write(f"Patrón t+1: {patron_t1}\n")
                f.write(f"Patrón t: {patron_t}\n")
                f.write(f"Alcance: {alcance}\n")
                f.write(f"Tiempo ejecución: {tiempo:.4f}s\n")
                f.write(f"Resultado: {resultado_str}\n")
                f.write("-" * 60 + "\n")
        except Exception as e:
            print(f"⚠️ Error al guardar resultados: {e}")
    
    def save_comparison(self, comparaciones: dict):
        """Guarda comparación entre estrategias cuando están disponibles"""
        try:
            with open(self.results_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*70}\n")
                f.write("COMPARACIÓN DE ESTRATEGIAS\n")
                f.write(f"{'='*70}\n")
                
                for caso, resultados in comparaciones.items():
                    f.write(f"\nCaso: {caso}\n")
                    f.write("-" * 40 + "\n")
                    for estrategia, datos in resultados.items():
                        f.write(f"{estrategia:12} | EMD: {datos.get('emd', 'N/A'):>10} | ")
                        f.write(f"Tiempo: {datos.get('tiempo', 0):>6.3f}s\n")
        except Exception as e:
            print(f"⚠️ Error al guardar comparación: {e}")
    
    def save_summary(self, total: int, estadisticas: dict = None):
        """Guarda el resumen final"""
        try:
            with open(self.results_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*70}\n")
                f.write("RESUMEN FINAL\n")
                f.write(f"{'='*70}\n")
                f.write(f"Total de análisis: {total}\n")
                f.write(f"Finalizado: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                if estadisticas:
                    f.write(f"\nEstadísticas:\n")
                    for key, value in estadisticas.items():
                        f.write(f"  {key}: {value}\n")
                f.write("\n")
        except Exception as e:
            print(f"⚠️ Error al guardar resumen: {e}")
    
    def get_analysis_count(self):
        """Cuenta el número de análisis realizados"""
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                return f.read().count("ANÁLISIS #") + 1
        except:
            return 1

class SistemaAnalisis:
    def __init__(self, sistema_candidato: str):
        self.sistema_candidato = sistema_candidato
        self.resultados = ResultadosAnalisis("Geometry")
        self.total = 0
        
        # Estado inicial del sistema
        self.estado_inicio = "1" + "0" * (len(sistema_candidato) - 1)
        
        # Configuración del sistema
        self.config_sistema = Manager(estado_inicial=self.estado_inicio)
        
        # Condiciones (siempre todo 1)
        self.condiciones = "1" * len(sistema_candidato)
        
        # Patrones optimizados para eficiencia
        self.patrones = [
            "completo",
            "desaparece_ultimo",
            "desaparece_primer",
            "solo_impares",
            "solo_pares"
        ]
        
        # Estadísticas de rendimiento
        self.estadisticas = {
            'geometry_exitosos': 0,
            'geometry_errores': 0,
            'tiempo_total_geometry': 0.0,
            'otros_exitosos': 0,
            'otros_errores': 0
        }

    def generar_alcance(self, patron: str) -> str:
        """Genera el alcance (t+1) basado en el patrón"""
        n = len(self.sistema_candidato)
        patrones_map = {
            "completo": "1" * n,
            "desaparece_ultimo": "1" * (n - 1) + "0",
            "desaparece_primer": "0" + "1" * (n - 1),
            "desaparece_primero_ultimo": "0" + "1" * (n - 2) + "0",
            "solo_impares": "".join(["1" if i % 2 == 0 else "0" for i in range(n)]),
            "solo_pares": "".join(["1" if i % 2 == 1 else "0" for i in range(n)]),
            "desaparece_penultimo": "".join(["0" if i > 0 and (i + 1) % 3 == 0 else "1" for i in range(n)])
        }
        
        if patron not in patrones_map:
            raise ValueError(f"Patrón no válido: {patron}")
        
        return patrones_map[patron]

    def ejecutar_con_estrategia(self, estrategia_class, nombre: str, condiciones: str, 
                               alcance: str, mecanismo: str):
        """Ejecuta una estrategia específica con manejo de errores mejorado"""
        try:
            print(f"  🔄 Ejecutando {nombre}...")
            inicio = time.time()
            
            analizador = estrategia_class(self.config_sistema)
            resultado = analizador.aplicar_estrategia(condiciones, alcance, mecanismo)
            
            tiempo_total = time.time() - inicio
            
            # Actualizar estadísticas
            if nombre.lower() == 'geometry':
                self.estadisticas['geometry_exitosos'] += 1
                self.estadisticas['tiempo_total_geometry'] += tiempo_total
            else:
                self.estadisticas['otros_exitosos'] += 1
            
            print(f"    ✅ {nombre}: Completado en {tiempo_total:.3f}s")
            return resultado, tiempo_total, None
            
        except Exception as e:
            if nombre.lower() == 'geometry':
                self.estadisticas['geometry_errores'] += 1
            else:
                self.estadisticas['otros_errores'] += 1
                
            print(f"    ❌ {nombre}: Error - {str(e)[:100]}...")
            return None, 0.0, str(e)

    def ejecutar_analisis(self):
        """Ejecuta el análisis completo con enfoque en Geometry"""
        try:
            print(f"🚀 INICIANDO ANÁLISIS GEOMÉTRICO")
            print(f"Sistema: {self.sistema_candidato} ({len(self.sistema_candidato)} variables)")
            print(f"Estado inicial: {self.estado_inicio}")
            print("=" * 70)
            
            # Análisis principal con Geometry
            for patron_t1 in self.patrones:
                alcance = self.generar_alcance(patron_t1)
                
                for patron_t in self.patrones:
                    self.total += 1
                    print(f"\n📊 Análisis {self.total}: {patron_t1} vs {patron_t}")
                    
                    mecanismo = "1" * len(self.sistema_candidato)
                    
                    # Ejecutar con Geometry (estrategia principal)
                    resultado, tiempo, error = self.ejecutar_con_estrategia(
                        Geometry, "Geometry", self.condiciones, alcance, mecanismo
                    )
                    
                    # Guardar resultado
                    resultado_str = str(resultado) if not error else f"ERROR: {error}"
                    self.resultados.save_to_file(
                        patron_t1=patron_t1,
                        patron_t=patron_t,
                        alcance=alcance,
                        resultado_str=resultado_str,
                        estrategia="Geometry",
                        tiempo=tiempo
                    )
                    
                    # Mostrar resultado
                    if not error:
                        if hasattr(resultado, 'emd'):
                            print(f"    📈 EMD: {resultado.emd:.6f}")
                        if hasattr(resultado, 'biparticion'):
                            print(f"    🎯 Bipartición: {resultado.biparticion}")
                    
                    print(f"    ⏱️ Tiempo: {tiempo:.3f}s")

            # Ejecutar caso especial (manteniendo tu lógica original)
            self._ejecutar_caso_especial()

        except Exception as e:
            print(f"❌ Error durante la ejecución: {e}")
            with open(self.resultados.results_file, 'a', encoding='utf-8') as f:
                f.write(f"\nERROR CRÍTICO: {str(e)}\n")
        
        finally:
            self._generar_resumen_final()

    def _ejecutar_caso_especial(self):
        """Ejecuta el caso especial con QNodes"""
        try:
            print(f"\n🎯 EJECUTANDO CASO ESPECIAL")
            print("-" * 50)
            
            self.total += 1
            n = len(self.sistema_candidato)
            
            # Patrones especiales (ajustados según el tamaño del sistema)
            if n >= 3:
                patron_t = "101"
                patron_t1 = "101"
            else:
                patron_t = "1" * n
                patron_t1 = "1" * n
                
            mecanismo = "1" * n
            
            resultado, tiempo, error = self.ejecutar_con_estrategia(
                QNodes, "QNodes_Especial", self.condiciones, patron_t1, mecanismo
            )
            
            # Guardar resultado especial
            resultado_str = str(resultado) if not error else f"ERROR: {error}"
            self.resultados.save_to_file(
                patron_t1="patron_especial_t1",
                patron_t="patron_especial_t",
                alcance=patron_t1,
                resultado_str=resultado_str,
                estrategia="QNodes_Especial",
                tiempo=tiempo
            )
            
            if not error:
                print(f"✅ Caso especial completado exitosamente")
            else:
                print(f"❌ Error en caso especial: {error}")
                
        except Exception as e:
            print(f"❌ Error en caso especial: {e}")

    def _generar_resumen_final(self):
        """Genera el resumen final del análisis"""
        print(f"\n📋 RESUMEN FINAL")
        print("=" * 50)
        
        # Calcular estadísticas
        geometry_total = self.estadisticas['geometry_exitosos'] + self.estadisticas['geometry_errores']
        geometry_tasa_exito = (self.estadisticas['geometry_exitosos'] / geometry_total * 100) if geometry_total > 0 else 0
        
        tiempo_promedio = (self.estadisticas['tiempo_total_geometry'] / 
                          self.estadisticas['geometry_exitosos']) if self.estadisticas['geometry_exitosos'] > 0 else 0
        
        estadisticas_finales = {
            'Sistema analizado': self.sistema_candidato,
            'Total de análisis': self.total,
            'Geometry exitosos': self.estadisticas['geometry_exitosos'],
            'Geometry errores': self.estadisticas['geometry_errores'],
            'Tasa de éxito Geometry': f"{geometry_tasa_exito:.1f}%",
            'Tiempo promedio Geometry': f"{tiempo_promedio:.3f}s",
            'Tiempo total Geometry': f"{self.estadisticas['tiempo_total_geometry']:.3f}s"
        }
        
        # Mostrar estadísticas
        for key, value in estadisticas_finales.items():
            print(f"  {key}: {value}")
        
        # Guardar resumen
        self.resultados.save_summary(self.total, estadisticas_finales)
        
        print(f"\n✅ Análisis completado!")
        print(f"📁 Resultados en: {self.resultados.results_file}")

def iniciar(sistema_candidato: str):
    """Función principal de entrada para el análisis"""
    print(f"🎯 INICIANDO ANÁLISIS PARA: {sistema_candidato}")
    
    # Validación básica
    if not sistema_candidato or len(sistema_candidato) < 2:
        raise ValueError("Sistema candidato debe tener al menos 2 variables")
    
    if len(sistema_candidato) > 10:
        print(f"⚠️ Sistema grande ({len(sistema_candidato)} variables), el análisis puede tardar...")
    
    # Crear y ejecutar análisis
    sistema = SistemaAnalisis(sistema_candidato)
    sistema.ejecutar_analisis()
    
    return sistema
