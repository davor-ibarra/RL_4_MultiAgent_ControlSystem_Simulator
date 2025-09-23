import os
import logging
from typing import Dict, List, Set # Añadir Set para type hint

# 3.1: Usar logger específico del módulo
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
        valid_logging_levels = set(logging._nameToLevel.keys()) # type: ignore[attr-defined] # Acceso a atributo "privado"
        if not self.allowed.issubset(valid_logging_levels):
            invalid_levels = self.allowed - valid_logging_levels
            # 3.2: Usar logger del módulo para advertencias internas
            logger.warning(f"Filtro de nivel contiene niveles no reconocidos: {invalid_levels}. Serán ignorados.")
            # Opcional: eliminar niveles inválidos
            self.allowed = self.allowed.intersection(valid_logging_levels)
            logger.debug(f"Niveles válidos efectivos en filtro: {self.allowed}")

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
    # 3.3: Validar que logging_config sea un diccionario
    if not isinstance(logging_config, dict):
        logger.error("configure_file_logger recibió logging_config inválido (no es dict). No se configurará log a fichero.")
        return

    log_to_file = logging_config.get('log_to_file', False)
    if not log_to_file:
        logger.info("Logging a fichero deshabilitado en config ('log_to_file': false).")
        return

    filename = logging_config.get('filename', 'simulation_run.log')
    if not filename or not isinstance(filename, str):
        logger.warning(f"Nombre de archivo log inválido ('{filename}'). Usando 'simulation_run.log'.")
        filename = 'simulation_run.log'

    levels_to_log: List[str] = logging_config.get('levels', ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']) # Default incluye DEBUG
    if not isinstance(levels_to_log, list) or not all(isinstance(lvl, str) for lvl in levels_to_log):
        logger.warning(f"'levels' en config logging no es lista de strings válida. Usando default.")
        levels_to_log = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if not levels_to_log:
        logger.warning("'levels' en config logging está vacía. No se guardará nada en el archivo.")
        return

    # Mapear levels a valores numéricos y tomar el mínimo para el handler
    level_map = logging._nameToLevel # type: ignore[attr-defined]
    numeric_levels = [level_map.get(l.upper()) for l in levels_to_log if level_map.get(l.upper()) is not None]
    if not numeric_levels:
        logger.error(f"Ninguno de los niveles especificados en 'levels' ({levels_to_log}) es válido. No se configura FileHandler.")
        return
    min_level_for_handler = min(numeric_levels) # Nivel MÍNIMO que el handler aceptará

    # 3.4: Usar results_folder pasado como argumento
    if not os.path.isdir(results_folder):
        logger.error(f"La carpeta de resultados '{results_folder}' no existe. No se puede crear archivo de log.")
        return # Fail-Fast si la carpeta no existe

    filepath = os.path.join(results_folder, filename)
    logger.info(f"Configurando logging a fichero: {filepath}")
    logger.info(f"Niveles a filtrar en fichero: {levels_to_log} (Handler Level: {logging.getLevelName(min_level_for_handler)})")

    try:
        # Crear FileHandler (modo 'w' sobrescribe)
        file_handler = logging.FileHandler(filepath, mode='w', encoding='utf-8')
        # 3.5: Establecer nivel mínimo en el handler
        file_handler.setLevel(min_level_for_handler)

        # Añadir filtro de niveles específicos (LevelFilter)
        level_filter = LevelFilter(levels_to_log)
        file_handler.addFilter(level_filter)

        # Usar formato estándar (el mismo que la consola)
        formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - %(name)-25s - %(message)s')
        file_handler.setFormatter(formatter)

        # 3.6: Obtener logger raíz de forma estándar
        root_logger = logging.getLogger()

        # Evitar añadir múltiples handlers idénticos (aunque 'w' sobrescribe, mejor evitar)
        handler_exists = any(
            isinstance(h, logging.FileHandler) and h.baseFilename == filepath
            for h in root_logger.handlers
        )

        if not handler_exists:
            root_logger.addHandler(file_handler)
            logger.info(f"FileHandler añadido al logger raíz.")
            # Verificar nivel raíz (importante)
            root_level = root_logger.getEffectiveLevel()
            if root_level > min_level_for_handler:
                logger.warning(f"Nivel del logger raíz ({logging.getLevelName(root_level)}) "
                               f"es más alto que el nivel del FileHandler ({logging.getLevelName(min_level_for_handler)}). "
                               f"Algunos mensajes (e.g., DEBUG) podrían no llegar al archivo. Ajustar nivel raíz en main.py si es necesario (actualmente es DEBUG).")
        else:
            logger.warning(f"Ya existe un FileHandler para '{filepath}'. No se añadió de nuevo.")

    except OSError as e:
        logger.error(f"Error OS creando/accediendo a archivo log '{filepath}': {e}")
    except Exception as e:
        logger.error(f"Error inesperado configurando FileHandler para '{filepath}': {e}", exc_info=True)