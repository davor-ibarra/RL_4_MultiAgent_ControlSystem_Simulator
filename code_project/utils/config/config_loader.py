# utils/config/config_loader.py
import os
import sys
import numpy as np
import yaml
import logging
from typing import Dict, Any, Optional, Tuple, List

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)-4s - %(name)-4s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)

def _get_data_save_directives(data_handling_content: Dict[str, Any], sub_config_filename: str) -> Dict[str, Any]:
    """
    Extrae el diccionario 'data_save' del contenido del sub-archivo de configuración.
    Ya no se valida la estructura interna; esa responsabilidad se delega al consumidor.
    Devuelve un diccionario vacío si no se encuentra.
    """
    err_prefix = f"[ConfigLoader:_get_data_save_directives] en '{sub_config_filename}':"

    if not isinstance(data_handling_content, dict):
        logger.warning(f"{err_prefix} El contenido no es un diccionario. Se devuelven directivas vacías.")
        return {}

    data_save_cfg = data_handling_content.get('data_save')
    if not isinstance(data_save_cfg, dict):
        logger.warning(f"{err_prefix} La sección 'data_save' está ausente o no es un diccionario. Se devuelven directivas vacías.")
        return {}

    logger.debug(f"[ConfigLoader] Se extrajeron las directivas 'data_save' en crudo de '{sub_config_filename}'.")
    return data_save_cfg

def load_and_validate_config(
    config_filename: str = 'super_config.yaml'
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    Carga y valida configuraciones.
    Devuelve: (main_config, vis_config, logging_config, processed_data_directives).
    processed_data_directives ahora contiene el diccionario 'data_save' completo del sub-archivo.
    """
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    super_config_path = os.path.join(script_dir, config_filename)
    main_config: Optional[Dict[str, Any]] = None
    vis_config: Optional[Dict[str, Any]] = None
    logging_config_data: Dict[str, Any] = {} # Renombrado para evitar colisión
    data_directives_processed: Dict[str, Any] # Siempre se inicializa

    if not os.path.exists(super_config_path):
        logger.critical(f"[ConfigLoader] CRITICAL: 'super_config.yaml' not found in the root directory.")
        return None, None, {}, _get_data_save_directives({}, "dummy_for_defaults.yaml")
    
    with open(super_config_path, 'r', encoding='utf-8') as f_super:
        super_config = yaml.safe_load(f_super)
    
    active_config_filename = super_config.get('active_config_file')
    if not active_config_filename:
        logger.critical(f"[ConfigLoader] CRITICAL: 'active_config_file' key not found or is empty in 'super_config.yaml'.")
        return None, None, {}, _get_data_save_directives({}, "dummy_for_defaults.yaml")
        
    config_dir = os.path.join(script_dir, 'config')
    main_config_path = os.path.join(config_dir, active_config_filename)

    logger.info(f"[ConfigLoader] Attempting to load main config from: {main_config_path}")
    if not os.path.exists(main_config_path):
        logger.critical(f"[ConfigLoader] CRITICAL: Main config file not found: {main_config_path}")
        # Devolver defaults para data_directives_processed para que main no falle si SimMan lo necesita
        return None, None, {}, _get_data_save_directives({}, "dummy_for_defaults.yaml")

    try:
        with open(main_config_path, 'r', encoding='utf-8') as file:
            main_config = yaml.safe_load(file)
        if not isinstance(main_config, dict):
            raise TypeError(f"Content of '{active_config_filename}' is not a valid YAML dictionary.")
        logger.info(f"[ConfigLoader] Main configuration loaded from: {main_config_path}")

        # --- Validación Estructural Mínima de main_config ---
        # (Esta es una validación de alto nivel; los componentes validarán sus propias secciones más a fondo)
        required_sections = ['environment', 'data_handling', 'visualization', 'logging']
        for section in required_sections:
            if section not in main_config or not isinstance(main_config[section], dict):
                raise KeyError(f"Required top-level section '{section}' missing or not a dictionary in '{active_config_filename}'.")
        
        # Validaciones más específicas para environment (ejemplos)
        env_cfg = main_config['environment']
        if 'type' not in env_cfg or not isinstance(env_cfg['type'], str):
            raise ValueError("main_config: 'environment.type' is missing or invalid.")
        if 'simulation' not in env_cfg or not isinstance(env_cfg['simulation'], dict):
            raise ValueError("main_config: 'environment.simulation' section is missing or invalid.")
        if 'dt_sec' not in env_cfg['simulation'] or not isinstance(env_cfg['simulation']['dt_sec'], (int, float)):
            raise ValueError("main_config: 'environment.simulation.dt_sec' is missing or invalid.")
        # ... (añadir más validaciones cruciales para system, controller, agent, reward_setup si es necesario) ...
        # Reward setup validation
        reward_setup_cfg = env_cfg.get('reward_setup')
        if not isinstance(reward_setup_cfg, dict): raise ValueError("Config 'environment.reward_setup' missing or not dict.")
        if 'reward_strategy' not in reward_setup_cfg or not isinstance(reward_setup_cfg['reward_strategy'], dict):
            raise ValueError("Config 'environment.reward_setup.reward_strategy' missing or not dict.")
        if 'type' not in reward_setup_cfg['reward_strategy']: raise ValueError("Config 'reward_strategy.type' missing.")
        if 'calculation' not in reward_setup_cfg or not isinstance(reward_setup_cfg['calculation'], dict):
            raise ValueError("Config 'environment.reward_setup.calculation' missing or not dict.")
        if 'method' not in reward_setup_cfg['calculation']: raise ValueError("Config 'calculation.method' missing.")
        # 'stability_measure' es opcional en 'calculation', si no está, RewardFactory usará NullStabilityCalculator.

        logger.debug("[ConfigLoader] Basic structure of main_config validated.")

    except (yaml.YAMLError, FileNotFoundError, TypeError, ValueError, KeyError) as e_cfg_load:
        logger.critical(f"[ConfigLoader] CRITICAL error loading/validating '{active_config_filename}': {e_cfg_load}", exc_info=True)
        return None, None, {}, _get_data_save_directives({}, "dummy_for_defaults_on_error.yaml")


    # --- Logging Config ---
    logging_config_data = main_config.get('logging', {}) # Ya se validó que existe y es dict
    
    # --- Visualization Config ---
    vis_settings = main_config.get('visualization', {}) # Ya se validó
    if vis_settings.get('enabled', False):
        vis_file_path = vis_settings.get('config_file')
        if vis_file_path and isinstance(vis_file_path, str):
            abs_vis_path = os.path.join(config_dir, vis_file_path)
            if os.path.exists(abs_vis_path):
                try:
                    with open(abs_vis_path, 'r', encoding='utf-8') as vf: vis_config = yaml.safe_load(vf)
                    if not isinstance(vis_config, dict) or 'plots' not in vis_config or not isinstance(vis_config['plots'], list):
                        logger.error(f"Vis config '{abs_vis_path}' invalid. Disabling vis.")
                        vis_config = None
                    else: logger.info(f"Visualization config loaded from: {abs_vis_path}")
                except Exception as e_vis: 
                    logger.error(f"Error loading vis_config '{abs_vis_path}': {e_vis}. Disabling vis.")
                    vis_config = None
            else: logger.warning(f"Vis config file '{abs_vis_path}' not found. Disabling vis.")
        else: logger.warning("Vis enabled but 'config_file' missing/invalid. Disabling vis.")
    else: logger.info("Visualization disabled in main config.")


    # --- Data Handling Directives ---
    data_handling_cfg_main = main_config.get('data_handling', {})
    sub_config_file_path = data_handling_cfg_main.get('config_file')
    
    if not sub_config_file_path or not isinstance(sub_config_file_path, str):
        # Esto no debería ocurrir si la validación de main_config fue estricta.
        logger.error("[ConfigLoader] 'data_handling.config_file' missing or invalid in main_config. Using default (disabled) data directives.")
        data_directives_processed = _get_data_save_directives({}, sub_config_file_path if sub_config_file_path else "N/A")
    else:
        abs_sub_config_path = os.path.join(config_dir, sub_config_file_path)
        if not os.path.exists(abs_sub_config_path):
            logger.error(f"[ConfigLoader] Data handling sub-config '{abs_sub_config_path}' NOT found. Using default (disabled) data directives.")
            data_directives_processed = _get_data_save_directives({}, sub_config_file_path)
        else:
            try:
                with open(abs_sub_config_path, 'r', encoding='utf-8') as dh_f:
                    dh_content = yaml.safe_load(dh_f)
                if not isinstance(dh_content, dict) : # El archivo debe ser un dict en la raíz
                    raise TypeError(f"Content of '{sub_config_file_path}' is not a valid YAML dictionary at its root.")
                data_directives_processed = _get_data_save_directives(dh_content, sub_config_file_path)
                logger.info(f"Data handling directives loaded and validated from: {abs_sub_config_path}")
            except Exception as e_dh_sub_load: # Captura YAML, TypeError, ValueError de _validate
                logger.critical(f"[ConfigLoader] CRITICAL error loading/validating data handling sub-config '{sub_config_file_path}': {e_dh_sub_load}. Aborting.", exc_info=True)
                # Si falla la carga/validación del sub-archivo de datos, es crítico.
                return None, None, {}, _get_data_save_directives({}, "dummy_on_sub_config_fail.yaml")

    return main_config, vis_config, logging_config_data, data_directives_processed