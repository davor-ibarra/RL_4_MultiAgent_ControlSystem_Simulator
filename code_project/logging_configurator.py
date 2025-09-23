import os
import logging
from typing import Dict, List, Set # Añadir Set para type hint

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class LevelFilter(logging.Filter):
    """
    Filtro para permitir sólo registros cuyos levelnames estén en allowed.
    """
    def __init__(self, allowed_levels: List[str]):
        super().__init__()
        # Convertir nombres a mayúsculas para comparación insensible a mayúsculas/minúsculas
        self.allowed: Set[str] = set(level.upper() for level in allowed_levels)
        # Validar que los niveles sean reconocidos por logging
        valid_logging_levels = set(logging._nameToLevel.keys()) # type: ignore # Acceso a atributo "privado"
        if not self.allowed.issubset(valid_logging_levels):
             invalid_levels = self.allowed - valid_logging_levels
             logger.warning(f"Filtro de nivel contiene niveles no reconocidos por logging: {invalid_levels}. Estos serán ignorados.")
             # Opcional: eliminar niveles inválidos
             # self.allowed = self.allowed.intersection(valid_logging_levels)

    def filter(self, record: logging.LogRecord) -> bool:
        """Devuelve True si el levelname del registro está permitido."""
        return record.levelname.upper() in self.allowed


def configure_file_logger(logging_config: Dict, results_folder: str):
    """
    Configura un FileHandler para logging hacia un fichero dentro de results_folder.
    Obtiene el logger raíz y le añade el handler configurado.

    Args:
        logging_config (Dict): La sección 'logging' del archivo de configuración.
        results_folder (str): La ruta absoluta a la carpeta de resultados para esta ejecución.
    """
    log_to_file = logging_config.get('log_to_file', False)
    if not log_to_file:
        logger.info("Logging a fichero deshabilitado en la configuración.")
        return

    # Nombre del fichero y niveles permitidos
    filename = logging_config.get('filename', 'simulation_run.log')
    if not filename:
        logger.warning("Nombre de archivo de log no especificado o vacío en config. Usando 'simulation_run.log'.")
        filename = 'simulation_run.log'

    # Obtener niveles permitidos, por defecto a ['INFO', 'WARNING', 'ERROR', 'CRITICAL']
    levels_to_log: List[str] = logging_config.get('levels', ['INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    if not isinstance(levels_to_log, list) or not all(isinstance(lvl, str) for lvl in levels_to_log):
        logger.warning(f"'levels' en config de logging no es una lista de strings válida ({levels_to_log}). Usando default.")
        levels_to_log = ['INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if not levels_to_log: # Si la lista está vacía
         logger.warning("'levels' en config de logging está vacía. Ningún mensaje se guardará en el archivo.")
         # Podríamos decidir no añadir el handler si no hay niveles
         # return

    # Mapear levels a valores numéricos y tomar el mínimo para el handler
    # Usar logging._nameToLevel para mapeo robusto
    level_map = logging._nameToLevel # type: ignore
    numeric_levels = [level_map.get(l.upper(), logging.INFO) for l in levels_to_log] # Default a INFO si no se reconoce
    min_level = min(numeric_levels) if numeric_levels else logging.INFO # Default a INFO si la lista está vacía

    # Ruta completa
    # Asegurar que results_folder existe (aunque main.py debería haberlo creado)
    if not os.path.isdir(results_folder):
         logger.error(f"La carpeta de resultados '{results_folder}' no existe. No se puede crear el archivo de log.")
         # Intentar crearla? O simplemente abortar configuración de fichero? Abortar es más seguro.
         return

    filepath = os.path.join(results_folder, filename)
    logger.info(f"Configurando logging a fichero: {filepath}")

    try:
        # Crear el FileHandler
        # Usar encoding='utf-8' es buena práctica
        file_handler = logging.FileHandler(filepath, mode='w', encoding='utf-8') # 'w' para sobrescribir en cada ejecución
        file_handler.setLevel(min_level)

        # Añadir el filtro de niveles específicos
        level_filter = LevelFilter(levels_to_log)
        file_handler.addFilter(level_filter)

        # Usar un formato consistente (igual que el de consola en main.py)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Añadir el handler configurado al logger RAÍZ
        root_logger = logging.getLogger()
        # Evitar añadir múltiples handlers si esta función se llama varias veces (poco probable aquí)
        # Comprobar si ya existe un FileHandler con la misma ruta
        handler_exists = any(isinstance(h, logging.FileHandler) and h.baseFilename == filepath for h in root_logger.handlers)
        if not handler_exists:
             root_logger.addHandler(file_handler)
             logger.info(f"FileHandler añadido al logger raíz para niveles {levels_to_log} (min={logging.getLevelName(min_level)}).")
        else:
             logger.warning(f"Ya existe un FileHandler para '{filepath}'. No se añadió de nuevo.")

    except OSError as e:
        logger.error(f"Error de OS al crear o acceder al archivo de log '{filepath}': {e}")
    except Exception as e:
        logger.error(f"Error inesperado configurando el FileHandler para '{filepath}': {e}", exc_info=True)