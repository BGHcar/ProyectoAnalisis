from src.middlewares.profile import profiler_manager
from src.models.base.application import aplicacion
from src.main import iniciar
from src.controllers.manager import Manager


def main():
    """Inicializar el aplicativo."""
    profiler_manager.enabled = True

    aplicacion.pagina_sample_network = "A"

    print(f"{aplicacion.pagina_sample_network=}")  

    sistema_candidato = "ABCDEFGHIJKLMNOPQRST"  # Puedes cambiar el sistema candidato aquí  ABCDEFGHIJKLMNOPQRST
    iniciar(sistema_candidato)
    

if __name__ == "__main__":
    main()
