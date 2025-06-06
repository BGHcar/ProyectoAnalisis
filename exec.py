# exec.py
import sys
from src.models.base.application import aplicacion
from src.main import iniciar

def main():
    
    iniciar()
    """Inicializar el aplicativo con análisis geométrico."""
    
"""    # Configuración de la aplicación
    aplicacion.profiler_habilitado = True
    aplicacion.pagina_sample_network = "A"
    
    print("🚀 SISTEMA DE ANÁLISIS GEOMÉTRICO DE BIPARTICIONES")
    print("=" * 60)
    
    # Determinar sistema candidato
    if len(sys.argv) > 1:
        sistema_candidato = sys.argv[1]
        print(f"📋 Sistema desde argumentos: {sistema_candidato}")
    else:
        # Sistema por defecto (puedes cambiarlo aquí)
        sistema_candidato = "ABC"  # Cambia esto por el sistema que necesites
        print(f"📋 Sistema por defecto: {sistema_candidato}")
    
    print(f"🔬 Variables: {len(sistema_candidato)}")
    print(f"⏱️ Iniciando análisis...")
    
    try:
        # Ejecutar análisis
        resultado = iniciar(sistema_candidato)
        
        # Mostrar resumen rápido
        print(f"\n🎉 ¡ANÁLISIS COMPLETADO EXITOSAMENTE!")
        print(f"📊 Total de análisis realizados: {resultado.total}")
        print(f"📁 Resultados guardados en: {resultado.resultados.results_file}")
        
        # Estadísticas rápidas
        geometry_exitosos = resultado.estadisticas['geometry_exitosos']
        geometry_errores = resultado.estadisticas['geometry_errores']
        
        if geometry_exitosos > 0:
            tiempo_promedio = resultado.estadisticas['tiempo_total_geometry'] / geometry_exitosos
            print(f"⚡ Rendimiento Geometry: {geometry_exitosos} exitosos, {geometry_errores} errores")
            print(f"⏱️ Tiempo promedio: {tiempo_promedio:.3f}s por análisis")
        
    except Exception as e:
        print(f"❌ ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) """

if __name__ == "__main__":
    main()
