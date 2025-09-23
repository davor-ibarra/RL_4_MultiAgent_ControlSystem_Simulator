# utils/config/logging_configurator.py
import os
import logging
from typing import Dict, List, Set, Any

logger_cfg_mod = logging.getLogger(__name__) # Logger específico del módulo

class LevelFilter(logging.Filter):
    def __init__(self, allowed_levels: List[str]):
        super().__init__()
        self.allowed_levelname_set: Set[str] = set(level.upper() for level in allowed_levels)
        # Validar contra niveles de logging reales
        valid_logging_levels_set = set(logging._nameToLevel.keys()) # type: ignore[attr-defined]
        unrecognized = self.allowed_levelname_set - valid_logging_levels_set
        if unrecognized:
            logger_cfg_mod.warning(f"[LevelFilter] Filter contains unrecognized levels: {unrecognized}. They will be ignored.")
            self.allowed_levelname_set.intersection_update(valid_logging_levels_set)
        # logger_cfg_mod.debug(f"[LevelFilter] Effective valid levels in filter: {self.allowed_levelname_set}")

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelname.upper() in self.allowed_levelname_set

def configure_file_logger(logging_config: Dict[str, Any], output_dir: str):
    """Configura FileHandler para logging basado en config."""
    if not isinstance(logging_config, dict):
        logger_cfg_mod.error("[LoggingConfig] Invalid logging_config (not a dict). File logging skipped.")
        return

    if not logging_config.get('log_to_file', False):
        logger_cfg_mod.info("[LoggingConfig] File logging disabled in config ('log_to_file': false).")
        return

    log_filename_cfg = logging_config.get('filename', 'simulation_run.log')
    if not log_filename_cfg or not isinstance(log_filename_cfg, str):
        log_filename_cfg = 'simulation_run.log'
        logger_cfg_mod.warning(f"[LoggingConfig] Invalid log filename. Using default: '{log_filename_cfg}'.")

    levels_to_log_file = logging_config.get('levels', ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    if not isinstance(levels_to_log_file, list) or not all(isinstance(lvl, str) for lvl in levels_to_log_file):
        levels_to_log_file = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        logger_cfg_mod.warning("[LoggingConfig] 'levels' in logging config invalid. Using defaults.")
    if not levels_to_log_file:
        logger_cfg_mod.warning("[LoggingConfig] 'levels' in logging config is empty. No file logging will occur.")
        return

    # Determinar el nivel mínimo para el handler a partir de los niveles permitidos
    level_map = logging._nameToLevel # type: ignore[attr-defined]
    numeric_levels = [level_map.get(l.upper()) for l in levels_to_log_file if level_map.get(l.upper()) is not None]
    if not numeric_levels:
        logger_cfg_mod.error(f"[LoggingConfig] None of the specified levels ({levels_to_log_file}) are valid. FileHandler not configured.")
        return
    min_handler_level = min(numeric_levels)

    log_file_full_path = os.path.join(output_dir, log_filename_cfg)
    logger_cfg_mod.info(f"[LoggingConfig] Configuring file logging to: {log_file_full_path}")
    logger_cfg_mod.info(f"[LoggingConfig] File log levels (filter): {levels_to_log_file} (Handler Min Level: {logging.getLevelName(min_handler_level)})")

    # Crear y configurar el FileHandler
    file_handler = logging.FileHandler(log_file_full_path, mode='w', encoding='utf-8')
    file_handler.setLevel(min_handler_level) # Nivel mínimo que el handler procesará

    level_filter_for_file = LevelFilter(levels_to_log_file) # Filtro para seleccionar niveles específicos
    file_handler.addFilter(level_filter_for_file)

    file_formatter = logging.Formatter('%(asctime)s - %(levelname)-4s - %(name)-12s - %(message)s',
                                       datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    # Añadir handler al logger raíz (o a un logger específico si se prefiere)
    root_log = logging.getLogger()
    # Evitar añadir múltiples handlers idénticos si esta función se llama varias veces (poco probable)
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file_full_path for h in root_log.handlers):
        root_log.addHandler(file_handler)
        logger_cfg_mod.info(f"[LoggingConfig] FileHandler added to root logger for '{log_file_full_path}'.")
        # Advertir si el nivel del root logger es más restrictivo que el del handler
        if root_log.getEffectiveLevel() > min_handler_level:
            logger_cfg_mod.warning(f"[LoggingConfig] Root logger level ({logging.getLevelName(root_log.getEffectiveLevel())}) "
                                   f"is higher than FileHandler min level ({logging.getLevelName(min_handler_level)}). "
                                   f"Some messages might not reach the file.")
    # else:
        # logger_cfg_mod.warning(f"[LoggingConfig] FileHandler for '{log_file_full_path}' already exists. Not added again.")