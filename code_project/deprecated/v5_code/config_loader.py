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
    config_filename: str = 'config.yaml'
    # vis_config_filename_rel: Optional[str] = None # Argumento obsoleto
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Carga la configuración principal, la de visualización y la de logging desde archivos YAML.
    Realiza validaciones básicas de estructura y contenido, adaptadas a la nueva estructura de reward_setup.

    Args:
        config_filename (str): Nombre del archivo de configuración principal (relativo al script).

    Returns:
        Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
          (main_config, vis_config, logging_config).
          Devuelve (None, None, {}) si hay un error crítico cargando main_config.
          vis_config será None si la visualización está desactivada o el archivo no existe/es inválido.
          logging_config será un diccionario (vacío si no se encuentra o es inválido).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_config_path = os.path.join(script_dir, config_filename)
    main_config, vis_config = None, None
    logging_config = {} # Inicializar por defecto

    # --- [1] Carga Main Config ---
    logger.info(f"Intentando cargar configuración principal desde: {main_config_path}")
    if not os.path.exists(main_config_path):
         logger.error(f"Error crítico: Archivo de configuración principal no encontrado en {main_config_path}")
         return None, None, {}

    try:
        with open(main_config_path, 'r', encoding='utf-8') as file:
            main_config = yaml.safe_load(file)
            if not isinstance(main_config, dict):
                 raise TypeError("El contenido del archivo de configuración principal no es un diccionario válido.")
            logger.info(f"Configuración principal cargada desde: {main_config_path}")

        # --- [2] Validación Estructura Básica y Reward Setup ---
        required_sections = ['environment', 'simulation', 'pid_adaptation', 'initial_conditions']
        for section in required_sections:
            if section not in main_config:
                logger.error(f"Sección requerida '{section}' no encontrada en {config_filename}. Abortando.")
                raise KeyError(f"Sección requerida '{section}' ausente.")
            elif not isinstance(main_config.get(section), dict):
                 logger.error(f"Sección '{section}' en {config_filename} no es un diccionario. Abortando.")
                 raise TypeError(f"Sección '{section}' debe ser un diccionario.")

        # Validar nueva estructura 'reward_setup'
        if 'reward_setup' not in main_config.get('environment', {}):
            raise KeyError("'environment' debe contener la sección 'reward_setup'.")

        reward_setup = main_config['environment']['reward_setup']
        if not isinstance(reward_setup, dict):
            raise TypeError("'environment.reward_setup' debe ser un diccionario.")

        # 2.a Validación 'calculation'
        calc_cfg = reward_setup.get('calculation')
        if not isinstance(calc_cfg, dict): raise TypeError("'reward_setup.calculation' debe ser un diccionario.")
        calc_method = calc_cfg.get('method')
        if not calc_method or calc_method not in ['gaussian', 'stability_calculator']:
             raise ValueError(f"Valor inválido o ausente para 'reward_setup.calculation.method': {calc_method}")
        if calc_method == 'gaussian' and not isinstance(calc_cfg.get('gaussian_params'), dict):
             raise TypeError("Si method='gaussian', 'gaussian_params' debe ser un diccionario.")

        # 2.b Validación 'stability_calculator'
        stab_cfg = reward_setup.get('stability_calculator')
        if not isinstance(stab_cfg, dict): raise TypeError("'reward_setup.stability_calculator' debe ser un diccionario.")
        stab_enabled = stab_cfg.get('enabled')
        if not isinstance(stab_enabled, bool): raise TypeError("'stability_calculator.enabled' debe ser booleano.")
        if stab_enabled:
             stab_type = stab_cfg.get('type')
             if not stab_type or stab_type not in ['ira_instantaneous', 'simple_exponential']:
                  raise ValueError(f"Si stability_calculator está habilitado, 'type' debe ser especificado y válido: {stab_type}")
             # Validar parámetros específicos del tipo habilitado
             if stab_type == 'ira_instantaneous' and not isinstance(stab_cfg.get('ira_params'), dict):
                 raise TypeError("Si type='ira_instantaneous', 'ira_params' debe ser un diccionario.")
             if stab_type == 'simple_exponential' and not isinstance(stab_cfg.get('simple_exponential_params'), dict):
                 raise TypeError("Si type='simple_exponential', 'simple_exponential_params' debe ser un diccionario.")

        # 2.c Validación 'learning_strategy'
        learn_strat_cfg = reward_setup.get('learning_strategy')
        if not isinstance(learn_strat_cfg, dict): raise TypeError("'reward_setup.learning_strategy' debe ser un diccionario.")
        learn_strat_type = learn_strat_cfg.get('type')
        if not learn_strat_type or learn_strat_type not in ['global', 'shadow_baseline', 'echo_baseline']:
             raise ValueError(f"Valor inválido o ausente para 'reward_setup.learning_strategy.type': {learn_strat_type}")
        if not isinstance(learn_strat_cfg.get('strategy_params'), dict):
             raise TypeError("'learning_strategy.strategy_params' debe ser un diccionario.")
        # Validar params específicos si aplica
        if learn_strat_type == 'shadow_baseline' and not isinstance(learn_strat_cfg.get('strategy_params', {}).get('shadow_baseline'), dict):
             raise TypeError("Si type='shadow_baseline', 'strategy_params.shadow_baseline' debe ser un diccionario.")
        # Validar params para Echo (actualmente ninguno, pero estructura debe estar)
        if learn_strat_type == 'echo_baseline' and 'echo_baseline' not in learn_strat_cfg.get('strategy_params', {}):
              raise TypeError("Si type='echo_baseline', 'strategy_params.echo_baseline' debe existir (puede ser vacío: {} o null).")

        # 2.d Validación Cruzada Shadow <-> Stability
        if learn_strat_type == 'shadow_baseline' and not stab_enabled:
             raise ValueError("La estrategia 'shadow_baseline' requiere que 'reward_setup.stability_calculator.enabled' sea true.")

        logger.info("Validación de estructura 'reward_setup' completada.")

        # --- [3] Validación Específica de Q-tables (mantenida) ---
        env_cfg = main_config.get('environment', {})
        agent_cfg = env_cfg.get('agent', {})
        if agent_cfg.get('type') == 'pid_qlearning':
            agent_params = agent_cfg.get('params', {})
            state_cfg_q = agent_params.get('state_config', {})
            if isinstance(state_cfg_q, dict):
                 enabled_bins = []
                 for var_name, var_cfg in state_cfg_q.items():
                      if isinstance(var_cfg, dict) and var_cfg.get('enabled'):
                           bins = var_cfg.get('bins')
                           if isinstance(bins, int) and bins > 0:
                                enabled_bins.append(bins)
                           else:
                                logger.warning(f"Q-Table Estimation: Config estado para '{var_name}': 'bins' inválido ({bins}).")

                 if enabled_bins:
                    total_states = 1
                    for b in enabled_bins: total_states *= b
                    num_actions = agent_params.get('num_actions', 3)
                    if not isinstance(num_actions, int) or num_actions <= 0: num_actions = 3
                    estimated_entries = total_states * num_actions
                    threshold = 1_000_000
                    if estimated_entries > threshold:
                        logger.warning(
                            f"Estimación Q-table: ≈{estimated_entries:,.0f} entradas/tabla ({len(enabled_bins)} vars: {enabled_bins} bins, {num_actions} acts). "
                            f"Excede umbral ({threshold:,.0f}). Considerar reducir bins/vars."
                        )
                    else:
                         logger.info(f"Estimación Q-table: ≈{estimated_entries:,.0f} entradas/tabla (OK).")
                 else:
                      logger.info("Estimación Q-table: No hay variables de estado habilitadas con bins válidos.")
            else:
                 logger.warning("'agent.params.state_config' no es diccionario. No se estima tamaño Q-table.")


    except yaml.YAMLError as e:
        logger.error(f"Error crítico parseando YAML en {main_config_path}: {e}")
        return None, None, {}
    except (FileNotFoundError, TypeError, ValueError, KeyError) as e: # Capturar errores de validación/carga
         logger.error(f"Error crítico cargando/validando {main_config_path}: {e}")
         return None, None, {}
    except Exception as e:
        logger.error(f"Error crítico inesperado cargando {main_config_path}: {e}", exc_info=True)
        return None, None, {}

    # --- [4] Extraer logging_config (asegurarse que es un dict) ---
    logging_config = main_config.get('logging', {})
    if not isinstance(logging_config, dict):
         logger.warning("'logging' section in config is not a dictionary. Using default logging config {}")
         logging_config = {}

    # --- [5] Carga Visualización Config si aplica ---
    vis_settings = main_config.get('visualization', {})
    if not isinstance(vis_settings, dict):
        logger.warning("'visualization' section in config is not a dictionary. Disabling visualization.")
        vis_settings = {'enabled': False}

    if vis_settings.get('enabled', False):
        vis_file_rel = vis_settings.get('config_file')
        if vis_file_rel and isinstance(vis_file_rel, str):
            vis_path_abs = os.path.join(script_dir, vis_file_rel)
            logger.info(f"Intentando cargar config visualización desde: {vis_path_abs}")
            if not os.path.exists(vis_path_abs):
                 logger.warning(f"Archivo config visualización '{vis_path_abs}' no encontrado. Vis deshabilitada.")
                 vis_config = None
            else:
                 try:
                     with open(vis_path_abs, 'r', encoding='utf-8') as vf:
                         vis_config_loaded = yaml.safe_load(vf)
                         if not isinstance(vis_config_loaded, dict):
                              logger.warning(f"Contenido de {vis_path_abs} no es dict. Vis deshabilitada.")
                              vis_config = None
                         else:
                              vis_config = vis_config_loaded
                              logger.info(f"Config visualización cargada desde: {vis_path_abs}")
                 except yaml.YAMLError as e:
                      logger.warning(f"Error parseando YAML en {vis_path_abs}: {e}. Vis deshabilitada.")
                      vis_config = None
                 except Exception as e:
                      logger.warning(f"No se pudo cargar/leer vis config desde {vis_path_abs}: {e}")
                      vis_config = None
        else:
            logger.warning("Visualización habilitada pero 'config_file' no especificado o inválido. Vis deshabilitada.")
            vis_config = None
    else:
        logger.info("Visualización deshabilitada en la configuración principal.")
        vis_config = None

    # Devolver la tupla
    return main_config, vis_config, logging_config