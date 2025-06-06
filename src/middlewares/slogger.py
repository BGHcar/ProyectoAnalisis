import sys
import logging
from pathlib import Path
from datetime import datetime
from functools import wraps
from typing import Any, Callable

# Importar colorama de forma segura
try:
    from colorama import init, Fore, Style
    _COLORAMA_AVAILABLE = True
except ImportError:
    _COLORAMA_AVAILABLE = False
    class DummyColor:
        def __getattr__(self, name):
            return ''
    Fore = DummyColor()
    Style = DummyColor()

from src.constants.base import LOGS_PATH

# ¡CLAVE! - Conjunto global para evitar reconfiguración repetitiva
_configured_loggers = set()

class ColorFormatter(logging.Formatter):
    """Formatter personalizado para consola con colores usando colorama."""

    COLORS = {
        logging.DEBUG: Fore.LIGHTBLACK_EX,
        logging.INFO: Fore.BLUE,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
        logging.FATAL: Fore.RED + Style.BRIGHT,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if _COLORAMA_AVAILABLE:
            init(autoreset=True)

    def format(self, record: logging.LogRecord) -> str:
        if _COLORAMA_AVAILABLE:
            color = self.COLORS.get(record.levelno, "")
            original_levelname = record.levelname
            record.levelname = f"{color}{original_levelname}{Style.RESET_ALL}"

        formatted = super().format(record)
        
        if _COLORAMA_AVAILABLE:
            record.levelname = original_levelname
        return formatted

class SafeLogger:
    """Logger optimizado que evita reconfiguración repetitiva."""

    def __init__(self, name: str):
        # Obtener logger singleton por nombre
        self._logger = logging.getLogger(name)
        
        # ¡OPTIMIZACIÓN CLAVE! - Solo configurar si no se ha hecho antes
        if name not in _configured_loggers:
            self.__setup_handlers_once(self._logger, name)
            _configured_loggers.add(name)
            print(f"Logger '{name}' configurado por primera vez")
        else:
            print(f"Logger '{name}' reutilizado (sin reconfiguración)")

    def _safe_str(self, obj: Any) -> str:
        """Convierte cualquier objeto a string de forma segura."""
        try:
            if isinstance(obj, (list, tuple, set, dict)):
                return str(obj)
            return str(obj).encode("utf-8", errors="replace").decode("utf-8")
        except Exception:
            return "[Objeto no representable]"

    def _safe_format(self, *args, **kwargs) -> str:
        """Formatea los argumentos de forma segura."""
        args_str = " ".join(self._safe_str(arg) for arg in args)
        if kwargs:
            kwargs_str = " ".join(f"{k}={self._safe_str(v)}" for k, v in kwargs.items())
            return f"{args_str} {kwargs_str}"
        return args_str

    def __setup_handlers_once(self, logger_instance: logging.Logger, name: str) -> None:
        """Configura handlers UNA SOLA VEZ por nombre de logger."""
        
        # Limpiar handlers existentes y cerrar recursos
        if logger_instance.handlers:
            for handler in logger_instance.handlers:
                handler.close()  # ¡IMPORTANTE! - Liberar recursos
            logger_instance.handlers.clear()

        logger_instance.setLevel(logging.ERROR)  # Reducir nivel para mejor rendimiento
        logger_instance.propagate = False

        # Crear estructura de directorios
        base_log_dir = Path(LOGS_PATH)
        base_log_dir.mkdir(exist_ok=True)

        current_time = datetime.now()
        date_dir = base_log_dir / current_time.strftime("%d_%m_%Y")
        date_dir.mkdir(exist_ok=True)

        hour_dir = date_dir / f"{current_time.strftime('%H')}hrs"
        hour_dir.mkdir(exist_ok=True)

        # Archivos de log
        detailed_log_file = hour_dir / f"{name}.log"
        last_log_file = base_log_dir / f"last_{name}.log"

        # Formatters
        plain_formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        colored_formatter = ColorFormatter(
            "%(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )

        # Handlers optimizados
        # Usar 'a' en lugar de 'w' para evitar sobrescritura constante
        detailed_file_handler = logging.FileHandler(
            detailed_log_file, mode="a", encoding="utf-8"
        )
        detailed_file_handler.setLevel(logging.DEBUG)
        detailed_file_handler.setFormatter(plain_formatter)

        last_file_handler = logging.FileHandler(
            last_log_file, mode="w", encoding="utf-8"
        )
        last_file_handler.setLevel(logging.DEBUG)
        last_file_handler.setFormatter(plain_formatter)

        # Console handler con nivel más alto para reducir output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)  # Solo warnings y errores
        console_handler.setFormatter(colored_formatter)

        logger_instance.addHandler(detailed_file_handler)
        logger_instance.addHandler(last_file_handler)
        logger_instance.addHandler(console_handler)

    def set_log(self, level: int, *args, **kwargs) -> None:
        """Método genérico de logging."""
        message = self._safe_format(*args, **kwargs)
        self._logger.log(level, message)

    def debug(self, *args, **kwargs) -> None:
        self.set_log(logging.DEBUG, *args, **kwargs)

    def info(self, *args, **kwargs) -> None:
        self.set_log(logging.INFO, *args, **kwargs)

    def warn(self, *args, **kwargs) -> None:
        self.set_log(logging.WARNING, *args, **kwargs)

    def error(self, *args, **kwargs) -> None:
        self.set_log(logging.ERROR, *args, **kwargs)

    def critic(self, *args, **kwargs) -> None:
        self.set_log(logging.CRITICAL, *args, **kwargs)

    def fatal(self, *args, **kwargs) -> None:
        self.set_log(logging.FATAL, *args, **kwargs)

def get_logger(name: str) -> SafeLogger:
    """Función de conveniencia para obtener una instancia del logger."""
    return SafeLogger(name)

# Decorador opcional para logging automático
def log_execution(logger: SafeLogger):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                logger.debug(f"Iniciando {func.__name__}")
                result = func(*args, **kwargs)
                logger.debug(f"Completado {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"Error en {func.__name__}: {e}")
                raise
        return wrapper
    return decorator
