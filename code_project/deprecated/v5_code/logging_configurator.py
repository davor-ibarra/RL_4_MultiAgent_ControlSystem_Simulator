import os
import logging
from typing import Dict, List, Set # Añadir Set para type hint

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class LevelFilter(logging.Filter):
    """
    Filtro para permitir sólo registros cuyos levelnames estén en allowed.
    Comparación insensible a mayúsculas/minúsculas.
    """
    def __init__(self, allowed_levels: List[str]):
        super().__init__()
        # Convertir nombres a mayúsculas para comparación insensible
        self.allowed: Set[str] = set(level.upper() for level in allowed_levels)
        # Validar que los niveles sean reconocidos por logging
        valid_logging_levels = set(logging._nameToLevel.keys()) # type: ignore # Acceso a atributo "privado"
        if not self.allowed.issubset(valid_logging_levels):
             invalid_levels = self.allowed - valid_logging_levels
             logger.warning(f"Filtro de nivel contiene niveles no reconocidos por logging: {invalid_levels}. Estos serán ignorados.")
             # Opcional: eliminar niveles inválidos del set
             # self.allowed = self.allowed.intersection(valid_logging_levels)
             # logger.debug(f"Niveles válidos efectivos en filtro: {self.allowed}")

    def filter(self, record: logging.LogRecord) -> bool:
        """Devuelve True si el levelname del registro está permitido."""
        return record.levelname.upper() in self.allowed


def configure_file_logger(logging_config: Dict, results_folder: str):
    """
    Configura un FileHandler para logging hacia un fichero dentro de results_folder,
    basado en la configuración proporcionada. Añade el handler al logger raíz.

    Args:
        logging_config (Dict): La sección 'logging' del archivo de configuración.
                               Debe ser un diccionario.
        results_folder (str): La ruta absoluta a la carpeta de resultados para esta ejecución.
    """
    if not isinstance(logging_config, dict):
         logger.error("configure_file_logger recibió logging_config inválido (no es dict). No se configurará log a fichero.")
         return

    log_to_file = logging_config.get('log_to_file', False)
    if not log_to_file:
        logger.info("Logging a fichero deshabilitado en la configuración ('log_to_file': false).")
        return

    # Nombre del fichero
    filename = logging_config.get('filename', 'simulation_run.log')
    if not filename or not isinstance(filename, str):
        logger.warning(f"Nombre de archivo de log inválido o ausente ('{filename}') en config. Usando 'simulation_run.log'.")
        filename = 'simulation_run.log'

    # Niveles permitidos
    levels_to_log: List[str] = logging_config.get('levels', ['INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    if not isinstance(levels_to_log, list) or not all(isinstance(lvl, str) for lvl in levels_to_log):
        logger.warning(f"'levels' en config de logging no es una lista de strings válida ({levels_to_log}). Usando default ['INFO', 'WARNING', 'ERROR', 'CRITICAL'].")
        levels_to_log = ['INFO', 'WARNING', 'ERROR', 'CRITICAL']

    if not levels_to_log:
         logger.warning("'levels' en config de logging está vacía. Ningún mensaje se guardará en el archivo.")
         # No añadir el handler si no hay niveles que loguear
         return

    # Mapear levels a valores numéricos y tomar el mínimo para el handler
    # Usar logging._nameToLevel para mapeo robusto
    level_map = logging._nameToLevel # type: ignore
    numeric_levels = [level_map.get(l.upper()) for l in levels_to_log if level_map.get(l.upper()) is not None]
    if not numeric_levels: # Si ninguno de los niveles especificados es válido
         logger.error(f"Ninguno de los niveles especificados en 'levels' ({levels_to_log}) es válido. No se puede configurar FileHandler.")
         return
    min_level = min(numeric_levels)

    # Ruta completa al archivo de log
    # Asegurar que results_folder existe (main.py debería haberla creado)
    if not os.path.isdir(results_folder):
         logger.error(f"La carpeta de resultados '{results_folder}' no existe. No se puede crear el archivo de log.")
         # Abortar configuración de fichero es lo más seguro.
         return

    filepath = os.path.join(results_folder, filename)
    logger.info(f"Configurando logging a fichero: {filepath}")
    logger.info(f"Niveles a loguear en fichero: {levels_to_log} (Mínimo efectivo: {logging.getLevelName(min_level)})")

    try:
        # Crear el FileHandler
        # Usar encoding='utf-8' es buena práctica
        # Usar mode='w' para sobrescribir en cada ejecución nueva
        file_handler = logging.FileHandler(filepath, mode='w', encoding='utf-8')
        file_handler.setLevel(min_level) # Establecer nivel mínimo en el handler

        # Añadir el filtro de niveles específicos
        level_filter = LevelFilter(levels_to_log)
        file_handler.addFilter(level_filter)

        # Usar un formato consistente (igual que el de consola propuesto en main.py)
        formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - %(name)-25s - %(message)s') # Ajustar formato si es necesario
        file_handler.setFormatter(formatter)

        # Añadir el handler configurado al logger RAÍZ
        root_logger = logging.getLogger()

        # Evitar añadir múltiples handlers idénticos si esta función se llama varias veces
        handler_exists = any(
            isinstance(h, logging.FileHandler) and h.baseFilename == filepath
            for h in root_logger.handlers
        )

        if not handler_exists:
             root_logger.addHandler(file_handler)
             logger.info(f"FileHandler añadido al logger raíz.")
             # Verificar el nivel del logger raíz también
             if root_logger.getEffectiveLevel() > min_level:
                  logger.warning(f"El nivel del logger raíz ({logging.getLevelName(root_logger.getEffectiveLevel())}) "
                                 f"es más alto que el nivel mínimo del FileHandler ({logging.getLevelName(min_level)}). "
                                 f"Algunos mensajes podrían no llegar al archivo. Considere ajustar el nivel raíz en main.py.")
        else:
             logger.warning(f"Ya existe un FileHandler para '{filepath}'. No se añadió de nuevo.")

    except OSError as e:
        logger.error(f"Error de OS al crear o acceder al archivo de log '{filepath}': {e}")
    except Exception as e:
        logger.error(f"Error inesperado configurando el FileHandler para '{filepath}': {e}", exc_info=True)