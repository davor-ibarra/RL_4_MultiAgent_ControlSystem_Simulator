# utils/config/logging_configurator.py
import os
import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__) # Logger específico del módulo

class LevelFilter(logging.Filter):
    """
    Filtro para permitir sólo registros cuyos levelnames estén en allowed.
    Comparación insensible a mayúsculas/minúsculas.
    """
    def __init__(self, allowed_levels: List[str]):
        super().__init__()
        self.allowed: Set[str] = set(level.upper() for level in allowed_levels)
        valid_logging_levels = set(logging._nameToLevel.keys()) # type: ignore[attr-defined]
        if not self.allowed.issubset(valid_logging_levels):
            invalid_levels = self.allowed - valid_logging_levels
            logger.warning(f"[LevelFilter] Filter contains unrecognized levels: {invalid_levels}. They will be ignored.")
            self.allowed = self.allowed.intersection(valid_logging_levels)
            logger.debug(f"[LevelFilter] Effective valid levels in filter: {self.allowed}")

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelname.upper() in self.allowed

def configure_file_logger(logging_config: Dict, results_folder: str):
    """
    Configura un FileHandler para logging hacia un fichero dentro de results_folder,
    basado en la configuración proporcionada. Añade el handler al logger raíz.

    Args:
        logging_config (Dict): La sección 'logging' del archivo de configuración.
        results_folder (str): La ruta absoluta a la carpeta de resultados.
    """
    if not isinstance(logging_config, dict):
        logger.error("[LoggingConfig] Invalid logging_config (not a dict). File logging skipped.")
        return

    log_to_file = logging_config.get('log_to_file', False)
    if not log_to_file:
        logger.info("[LoggingConfig] File logging disabled in config ('log_to_file': false).")
        return

    filename = logging_config.get('filename', 'simulation_run.log')
    if not filename or not isinstance(filename, str):
        logger.warning(f"[LoggingConfig] Invalid log filename ('{filename}'). Using 'simulation_run.log'.")
        filename = 'simulation_run.log'

    levels_to_log: List[str] = logging_config.get('levels', ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    if not isinstance(levels_to_log, list) or not all(isinstance(lvl, str) for lvl in levels_to_log):
        logger.warning(f"[LoggingConfig] 'levels' in logging config not a valid list of strings. Using default.")
        levels_to_log = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if not levels_to_log:
        logger.warning("[LoggingConfig] 'levels' in logging config is empty. No file logging will occur.")
        return

    level_map = logging._nameToLevel # type: ignore[attr-defined]
    numeric_levels = [level_map.get(l.upper()) for l in levels_to_log if level_map.get(l.upper()) is not None]
    if not numeric_levels:
        logger.error(f"[LoggingConfig] None of the levels specified ({levels_to_log}) are valid. FileHandler not configured.")
        return
    min_level_for_handler = min(numeric_levels)

    if not os.path.isdir(results_folder): # results_folder es validado por main antes de llamar
        logger.error(f"[LoggingConfig] Results folder '{results_folder}' does not exist. Cannot create log file.")
        return

    filepath = os.path.join(results_folder, filename)
    logger.info(f"[LoggingConfig] Configuring file logging to: {filepath}")
    logger.info(f"[LoggingConfig] File log levels: {levels_to_log} (Handler Min Level: {logging.getLevelName(min_level_for_handler)})")

    try:
        file_handler = logging.FileHandler(filepath, mode='w', encoding='utf-8')
        file_handler.setLevel(min_level_for_handler)

        level_filter = LevelFilter(levels_to_log)
        file_handler.addFilter(level_filter)

        # Formato estándar (consistente con consola si se desea)
        formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - %(name)-25s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        handler_exists = any(
            isinstance(h, logging.FileHandler) and h.baseFilename == filepath
            for h in root_logger.handlers
        )

        if not handler_exists:
            root_logger.addHandler(file_handler)
            logger.info(f"[LoggingConfig] FileHandler added to root logger for '{filepath}'.")
            root_level = root_logger.getEffectiveLevel()
            if root_level > min_level_for_handler:
                logger.warning(f"[LoggingConfig] Root logger level ({logging.getLevelName(root_level)}) "
                               f"is higher than FileHandler level ({logging.getLevelName(min_level_for_handler)}). "
                               f"Some messages might not reach the file. Root level set in main.py.")
        else:
            logger.warning(f"[LoggingConfig] FileHandler for '{filepath}' already exists. Not added again.")

    except OSError as e:
        logger.error(f"[LoggingConfig] OS error creating/accessing log file '{filepath}': {e}")
    except Exception as e:
        logger.error(f"[LoggingConfig] Unexpected error configuring FileHandler for '{filepath}': {e}", exc_info=True)