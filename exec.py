# exec.py
import sys
from src.models.base.application import aplicacion
from src.main import iniciar

def main():
    # Configuración de la aplicación
    aplicacion.profiler_habilitado = True
    aplicacion.pagina_sample_network = "A"
    
    print("🚀 SISTEMA DE ANÁLISIS GEOMÉTRICO DE BIPARTICIONES")
    print("=" * 60)
    
    # Determinar sistema candidato
    if len(sys.argv) > 1:
        sistema_candidato = sys.argv[1]
        print(f"📋 Sistema desde argumentos: {sistema_candidato}")
    else:
        # Tu sistema por defecto preferido
        sistema_candidato = "ABCDEFGHIJKLMNOPQRST"  
        print(f"📋 Sistema por defecto: {sistema_candidato}")
    
    print(f"🔬 Variables: {len(sistema_candidato)}")
    print(f"⏱️ Iniciando análisis...")
    
    try:
        # Ejecutar análisis
        iniciar(sistema_candidato)
        
        print(f"\n🎉 ¡ANÁLISIS COMPLETADO EXITOSAMENTE!")
        print(f"📊 Se analizaron {len(sistema_candidato)} variables")
        print(f"🎯 Total de combinaciones: {7 * 7 * 2} análisis")  # 7 patrones × 7 patrones × 2 estrategias
        
    except Exception as e:
        print(f"❌ ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
