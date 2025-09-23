import os
import yaml
import logging
from typing import Dict, Any, Optional, Tuple

# Configuración básica de logging si este módulo se usa standalone
# En la ejecución normal, main.py configura el logging principal.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__) # Usar logger específico del módulo

def load_and_validate_config(
    config_filename: str = 'config.yaml',
    vis_config_filename_rel: Optional[str] = None # Relativo al directorio del script - Argumento obsoleto ahora
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Carga la configuración principal, la de visualización y la de logging desde archivos YAML.
    Realiza validaciones básicas de estructura y contenido.

    Args:
        config_filename (str): Nombre del archivo de configuración principal (relativo al script).

    Returns:
        Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
          (main_config, vis_config, logging_config).
          Devuelve (None, None, {}) si hay un error crítico cargando main_config.
          vis_config será None si la visualización está desactivada o el archivo no existe/es inválido.
          logging_config será un diccionario (vacío si no se encuentra o es inválido).
    """
    # Obtener directorio base del script actual (__file__)
    # Esto es más robusto que depender del directorio de trabajo actual
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_config_path = os.path.join(script_dir, config_filename)
    main_config, vis_config = None, None
    # Inicializar logging_config como diccionario vacío por defecto
    logging_config = {}

    # --- [1] Carga Main Config ---
    logger.info(f"Intentando cargar configuración principal desde: {main_config_path}")
    if not os.path.exists(main_config_path):
         logger.error(f"Error crítico: Archivo de configuración principal no encontrado en {main_config_path}")
         return None, None, {} # Retornar tupla indicando fallo

    try:
        with open(main_config_path, 'r', encoding='utf-8') as file: # Especificar encoding
            main_config = yaml.safe_load(file)
            if not isinstance(main_config, dict):
                 # Si el archivo está vacío o no es un dict, yaml.safe_load puede devolver None u otro tipo
                 logger.error(f"Error crítico: El contenido de {main_config_path} no es un diccionario YAML válido.")
                 raise TypeError("El contenido del archivo de configuración principal no es un diccionario válido.")
            logger.info(f"Configuración principal cargada desde: {main_config_path}")

        # --- [2] Validación Básica de Main Config ---
        # (Validaciones existentes mantenidas)
        required_sections = ['environment', 'simulation', 'pid_adaptation', 'initial_conditions']
        for section in required_sections:
            if section not in main_config:
                logger.warning(f"Sección requerida '{section}' no encontrada en {config_filename}.")
                # Podría ser un error crítico dependiendo de la sección
                # Considerar lanzar ValueError aquí si alguna es absolutamente esencial
            elif not isinstance(main_config.get(section), dict):
                 logger.warning(f"Sección '{section}' en {config_filename} no es un diccionario.")
                 # Considerar lanzar TypeError

        # Validación específica de Q-tables (mantenida)
        if 'environment' in main_config and isinstance(main_config.get('environment'), dict) and \
           'agent' in main_config['environment'] and isinstance(main_config['environment'].get('agent'), dict) and \
           'params' in main_config['environment']['agent'] and isinstance(main_config['environment']['agent'].get('params'), dict):

            agent_params = main_config['environment']['agent']['params']
            state_cfg = agent_params.get('state_config', {})
            if isinstance(state_cfg, dict):
                 enabled_bins = []
                 for var_name, var_cfg in state_cfg.items():
                      if isinstance(var_cfg, dict) and var_cfg.get('enabled'):
                           bins = var_cfg.get('bins')
                           if isinstance(bins, int) and bins > 0:
                                enabled_bins.append(bins)
                           else:
                                logger.warning(f"Configuración de estado para '{var_name}': 'bins' inválido o ausente ({bins}). No se incluirá en la estimación de tamaño.")

                 if enabled_bins: # Solo calcular si hay bins habilitados válidos
                    total_states = 1
                    for b in enabled_bins:
                        total_states *= b
                    num_actions = agent_params.get('num_actions', 3) # Default a 3 si no está
                    if not isinstance(num_actions, int) or num_actions <= 0:
                        logger.warning(f"'num_actions' ({num_actions}) inválido en config. Usando 3 para estimación.")
                        num_actions = 3

                    estimated_entries = total_states * num_actions
                    # Threshold: 1 million entries per gain
                    threshold = 1_000_000
                    if estimated_entries > threshold:
                        logger.warning(
                            f"Estimación Q-table: Discretizando {len(enabled_bins)} variables con {enabled_bins} bins "
                            f"y {num_actions} acciones resulta en ≈{estimated_entries:,.0f} entradas por tabla de ganancia. "
                            f"Esto excede el umbral ({threshold:,.0f}) y podría consumir mucha memoria/tiempo. "
                            "Considerar reducir bins o deshabilitar variables de estado."
                        )
                    else:
                         logger.info(f"Estimación Q-table: ≈{estimated_entries:,.0f} entradas por tabla de ganancia (OK).")
                 else:
                      logger.info("Estimación Q-table: No hay variables de estado habilitadas con bins válidos.")
            else:
                 logger.warning("'state_config' no es un diccionario válido. No se puede estimar tamaño de Q-table.")

    except yaml.YAMLError as e:
        logger.error(f"Error crítico parseando YAML en {main_config_path}: {e}")
        return None, None, {} # Retornar tupla indicando fallo
    except FileNotFoundError: # Ya cubierto arriba, pero por si acaso
        logger.error(f"Error crítico: Archivo de configuración principal no encontrado en {main_config_path}")
        return None, None, {}
    except TypeError as e: # Captura la excepción de tipo lanzada arriba
         logger.error(f"Error crítico cargando {main_config_path}: {e}")
         return None, None, {}
    except Exception as e:
        logger.error(f"Error crítico inesperado cargando {main_config_path}: {e}", exc_info=True)
        return None, None, {}

    # --- [3] Extraer logging_config (asegurarse que es un dict) ---
    # main_config debe existir si llegamos aquí
    logging_config = main_config.get('logging', {})
    if not isinstance(logging_config, dict):
         logger.warning("'logging' section in config is not a dictionary. Using default logging config {}")
         logging_config = {}

    # --- [4] Carga Visualización Config si aplica ---
    # main_config debe existir si llegamos aquí
    vis_settings = main_config.get('visualization', {})
    if not isinstance(vis_settings, dict):
        logger.warning("'visualization' section in config is not a dictionary. Disabling visualization.")
        vis_settings = {'enabled': False} # Forzar deshabilitado si la sección es inválida

    if vis_settings.get('enabled', False):
        vis_file_rel = vis_settings.get('config_file') # Obtener nombre relativo desde config

        if vis_file_rel and isinstance(vis_file_rel, str):
            # Construir ruta absoluta basada en el directorio del *script*
            vis_path_abs = os.path.join(script_dir, vis_file_rel)
            logger.info(f"Intentando cargar configuración de visualización desde: {vis_path_abs}")
            if not os.path.exists(vis_path_abs):
                 logger.warning(f"Archivo de configuración de visualización '{vis_path_abs}' no encontrado. Visualización deshabilitada.")
                 vis_config = None # Visualización no posible sin config
            else:
                 try:
                     with open(vis_path_abs, 'r', encoding='utf-8') as vf:
                         vis_config_loaded = yaml.safe_load(vf)
                         if not isinstance(vis_config_loaded, dict):
                              logger.warning(f"Contenido de {vis_path_abs} no es un diccionario. Visualización deshabilitada.")
                              vis_config = None # Resetear a None si el contenido es inválido
                         else:
                              vis_config = vis_config_loaded # Asignar config cargada
                              # Añadir 'enabled' flag dentro de vis_config para consistencia? Opcional.
                              # vis_config['enabled'] = True
                              logger.info(f"Configuración de visualización cargada desde: {vis_path_abs}")
                 except yaml.YAMLError as e:
                      logger.warning(f"Error parseando YAML en {vis_path_abs}: {e}. Visualización deshabilitada.")
                      vis_config = None
                 except Exception as e:
                      logger.warning(f"No se pudo cargar o leer vis config desde {vis_path_abs}: {e}", exc_info=True)
                      vis_config = None
        else:
            logger.warning("Visualización habilitada pero 'config_file' no especificado o inválido en la sección 'visualization'. Visualización deshabilitada.")
            vis_config = None # No hay config si no se especifica archivo
    else:
        logger.info("Visualización deshabilitada en la configuración principal.")
        vis_config = None # Asegurar que es None si está deshabilitado


    # Devolver la tupla, incluyendo logging_config (siempre es un dict)
    return main_config, vis_config, logging_config