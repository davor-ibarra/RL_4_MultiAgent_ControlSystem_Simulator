# utils/config/config_loader.py
import os
import sys
import numpy as np
import yaml
import logging
from typing import Dict, Any, Optional, Tuple

# Configuración básica de logging si este módulo se usa standalone
# (Se mantiene por si acaso, pero main.py configura el logging principal)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)-4s - %(name)-4s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M')
logger = logging.getLogger(__name__)

def load_and_validate_config(
    config_filename: str = 'config.yaml'
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Carga la configuración principal, la de visualización y la de logging desde archivos YAML.
    Realiza validaciones de estructura y contenido esenciales.

    Args:
        config_filename (str): Nombre del archivo de configuración principal.

    Returns:
        Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
          (main_config, vis_config, logging_config).
          Devuelve (None, None, {}) si hay un error crítico cargando/validando main_config.
          vis_config será None si la visualización está desactivada o hay error.
          logging_config será un diccionario (vacío por defecto).
    """
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    main_config_path = os.path.join(script_dir, config_filename)
    main_config: Optional[Dict[str, Any]] = None
    vis_config: Optional[Dict[str, Any]] = None
    logging_config: Dict[str, Any] = {}

    logger.info(f"[ConfigLoader] Attempting to load main config from: {main_config_path}")
    if not os.path.exists(main_config_path):
        logger.error(f"[ConfigLoader] CRITICAL: Main config file not found: {main_config_path}")
        return None, None, {}

    try:
        with open(main_config_path, 'r', encoding='utf-8') as file:
            main_config = yaml.safe_load(file)
            if not isinstance(main_config, dict):
                raise TypeError(f"Content of '{config_filename}' is not a valid YAML dictionary.")
            logger.info(f"[ConfigLoader] Main configuration loaded from: {main_config_path}")

        # --- [2] Validación Estructura Esencial ---
        # Nivel Superior
        required_top_level_sections = ['environment', 'visualization', 'logging']
        for section in required_top_level_sections:
            if section not in main_config:
                raise KeyError(f"Required top-level section '{section}' absent in '{config_filename}'.")
            if section == 'environment' and not isinstance(main_config[section], dict):
                raise TypeError(f"Top-level section '{section}' must be a dictionary.")
            elif section in ['visualization', 'logging'] and \
                 main_config[section] is not None and not isinstance(main_config[section], dict):
                if isinstance(main_config[section], bool) and section == 'visualization' and not main_config[section]:
                    pass # Permitir visualization: false
                else:
                    raise TypeError(f"Top-level section '{section}' must be a dictionary or null/false (for visualization).")
        logger.debug("[ConfigLoader] Top-level sections validated.")

        # Nivel 'environment'
        env_cfg = main_config.get('environment', {})
        required_env_sections = [
            'results_folder', 'simulation', 'system',
            'controller', 'agent', 'reward_setup', 'initial_conditions'
        ]
        for section in required_env_sections:
            if section not in env_cfg:
                raise KeyError(f"Required section 'environment.{section}' absent.")
            if not isinstance(env_cfg[section], dict) and section not in ['results_folder']: # results_folder es string
                 raise TypeError(f"Section 'environment.{section}' must be a dictionary (except results_folder).")
        logger.debug("[ConfigLoader] 'environment' sub-sections validated.")

        # Nivel 'environment.controller'
        controller_cfg = env_cfg.get('controller', {})
        if 'pid_adaptation' not in controller_cfg or not isinstance(controller_cfg['pid_adaptation'], dict):
            raise KeyError("Required section 'environment.controller.pid_adaptation' absent or not a dictionary.")
        logger.debug("[ConfigLoader] 'environment.controller.pid_adaptation' validated.")

        # Nivel 'environment.reward_setup'
        reward_setup = env_cfg.get('reward_setup', {})
        if 'reward_strategy' not in reward_setup or not isinstance(reward_setup['reward_strategy'], dict):
            raise KeyError("Section 'environment.reward_setup.reward_strategy' absent or not a dictionary.")
        if 'calculation' not in reward_setup or not isinstance(reward_setup['calculation'], dict):
            raise KeyError("Section 'environment.reward_setup.calculation' absent or not a dictionary.")
        logger.debug("[ConfigLoader] 'environment.reward_setup' sub-sections validated.")

        # Validación 'reward_setup.calculation.stability_calculator'
        calc_cfg = reward_setup.get('calculation', {})
        stab_cfg = calc_cfg.get('stability_calculator', {}) # Puede ser None o no existir
        stability_calculator_type_present = False
        if stab_cfg is not None and isinstance(stab_cfg, dict):
            stab_calc_type = stab_cfg.get('type')
            if stab_calc_type and isinstance(stab_calc_type, str):
                stability_calculator_type_present = True
                # Validar que si 'type' está, los params correspondientes existan como dict
                # Ejemplo: si type: 'ira_instantaneous', debe existir 'ira_instantaneous_params: {}'
                expected_params_key = f"{stab_calc_type}_params"
                if expected_params_key not in stab_cfg or not isinstance(stab_cfg[expected_params_key], dict):
                    raise ValueError(f"Stability calculator type '{stab_calc_type}' is defined, but its parameters section 'stability_calculator.{expected_params_key}' is missing or not a dictionary.")
            elif stab_calc_type is not None: # Existe 'type' pero no es string
                 raise TypeError(f"'stability_calculator.type' must be a string, found: {type(stab_calc_type).__name__}")
            # Si 'type' no está, se considera no configurado, lo cual es válido si no se usa.
        elif stab_cfg is not None: # Existe 'stability_calculator' pero no es un dict
            raise TypeError("'environment.reward_setup.calculation.stability_calculator' must be a dictionary or null.")
        logger.debug(f"[ConfigLoader] 'stability_calculator' basic validation done. Type present: {stability_calculator_type_present}")

        # Validación cruzada: 'shadow_baseline' requiere 'stability_calculator.type'
        learn_strat_cfg = reward_setup.get('reward_strategy', {})
        learn_type = learn_strat_cfg.get('type')
        if learn_type == 'shadow_baseline' and not stability_calculator_type_present:
            raise ValueError("Learning strategy 'shadow_baseline' requires 'stability_calculator.type' to be defined.")
        allowed_strategies = ['global', 'shadow_baseline', 'echo_baseline']
        if learn_type not in allowed_strategies:
             raise ValueError(f"Unknown 'reward_strategy.type': '{learn_type}'. Allowed: {allowed_strategies}")
        if 'strategy_params' not in learn_strat_cfg or not isinstance(learn_strat_cfg['strategy_params'], dict):
            raise TypeError("'reward_strategy.strategy_params' must be a dictionary.")
        if learn_type in learn_strat_cfg['strategy_params'] and not isinstance(learn_strat_cfg['strategy_params'][learn_type], dict):
            raise TypeError(f"'reward_strategy.strategy_params.{learn_type}' must be a dictionary.")
        logger.debug("[ConfigLoader] 'reward_strategy' and cross-validation with stability_calculator done.")

        # Estimación Q-Table (usando nuevas rutas)
        agent_cfg_est = env_cfg.get('agent', {})
        if agent_cfg_est.get('type') == 'pid_qlearning':
            agent_params_est = agent_cfg_est.get('params', {})
            state_cfg_q_est = agent_params_est.get('state_config', {})
            if isinstance(state_cfg_q_est, dict):
                enabled_bins = []
                for var_name, var_cfg_est in state_cfg_q_est.items():
                    if isinstance(var_cfg_est, dict) and var_cfg_est.get('enabled'):
                        bins = var_cfg_est.get('bins')
                        if isinstance(bins, int) and bins > 0:
                            enabled_bins.append(bins)
                if enabled_bins:
                    total_states = np.prod(enabled_bins) if enabled_bins else 1
                    num_actions = agent_params_est.get('num_actions', 3)
                    if not isinstance(num_actions, int) or num_actions <= 0: num_actions = 3
                    estimated_entries = total_states * num_actions
                    threshold = 1_000_000 # Mismo umbral
                    msg_q_table = f"[ConfigLoader] Q-table estimate: ~{estimated_entries:,.0f} entries/table ({len(enabled_bins)} vars: {enabled_bins} bins, {num_actions} acts)."
                    if estimated_entries > threshold: logger.warning(f"{msg_q_table} Exceeds threshold ({threshold:,.0f}).")
                    else: logger.info(f"{msg_q_table} (OK).")
                else: logger.info("[ConfigLoader] Q-table estimate: No state variables enabled/valid for estimation.")
            else: logger.warning("[ConfigLoader] Q-table estimate: 'agent.params.state_config' is not a dictionary.")

    except (yaml.YAMLError, FileNotFoundError, TypeError, ValueError, KeyError) as e:
        logger.error(f"[ConfigLoader] CRITICAL error loading/validating '{config_filename}': {e}", exc_info=True)
        return None, None, {}
    except Exception as e:
        logger.error(f"[ConfigLoader] UNEXPECTED error loading/validating '{config_filename}': {e}", exc_info=True)
        return None, None, {}

    # --- [3] Extraer logging_config ---
    logging_config = main_config.get('logging', {})
    if not isinstance(logging_config, dict):
        logger.warning("[ConfigLoader] 'logging' section not a dictionary or null. Using default {}.")
        logging_config = {}

    # --- [4] Carga Visualización Config ---
    vis_settings = main_config.get('visualization', {})
    if vis_settings is None: vis_settings = {} # Tratar null como dict vacío
    if not isinstance(vis_settings, dict):
        logger.warning("[ConfigLoader] 'visualization' section not a dictionary or null. Disabling visualization.")
        vis_settings = {'enabled': False}

    vis_enabled = vis_settings.get('enabled', False)
    if vis_enabled:
        vis_file_rel = vis_settings.get('config_file')
        if not vis_file_rel or not isinstance(vis_file_rel, str):
            logger.warning("[ConfigLoader] Visualization enabled but 'config_file' missing/invalid. Vis_config will be None.")
        else:
            vis_path_abs = os.path.join(script_dir, vis_file_rel)
            logger.info(f"[ConfigLoader] Attempting to load visualization config from: {vis_path_abs}")
            if not os.path.exists(vis_path_abs):
                logger.error(f"[ConfigLoader] Visualization config file '{vis_path_abs}' NOT found. Vis_config will be None.")
            else:
                try:
                    with open(vis_path_abs, 'r', encoding='utf-8') as vf:
                        vis_config_loaded = yaml.safe_load(vf)
                        if not isinstance(vis_config_loaded, dict):
                            logger.error(f"[ConfigLoader] Content of '{vis_path_abs}' is not a YAML dictionary. Vis_config set to None.")
                        elif 'plots' not in vis_config_loaded or not isinstance(vis_config_loaded['plots'], list):
                            logger.error(f"[ConfigLoader] Vis config '{vis_path_abs}' loaded, but 'plots' key missing or not a list. Vis_config set to None.")
                        else:
                            vis_config = vis_config_loaded # ¡Éxito!
                            logger.info(f"[ConfigLoader] Visualization config loaded and validated from: {vis_path_abs}")
                except (yaml.YAMLError, OSError) as e:
                    logger.error(f"[ConfigLoader] Error loading/parsing vis_config '{vis_path_abs}': {e}. Vis_config set to None.")
                except Exception as e_vis: # Captura genérica para errores inesperados
                    logger.error(f"[ConfigLoader] UNEXPECTED error loading vis_config '{vis_path_abs}': {e_vis}", exc_info=True)
    else:
        logger.info("[ConfigLoader] Visualization explicitly disabled in main config.")

    return main_config, vis_config, logging_config