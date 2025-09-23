import os
import sys
import numpy as np
import yaml
import logging
from typing import Dict, Any, Optional, Tuple

# Configuración básica de logging si este módulo se usa standalone
# (Se mantiene por si acaso, pero main.py configura el logging principal)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
# 2.1: Usar logger específico del módulo
logger = logging.getLogger(__name__)

def load_and_validate_config(
    config_filename: str = 'config.yaml'
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Carga la configuración principal, la de visualización y la de logging desde archivos YAML.
    Realiza validaciones de estructura y contenido esenciales.
    Las validaciones detalladas de parámetros de componentes se delegan a las factorías/componentes.

    Args:
        config_filename (str): Nombre del archivo de configuración principal (relativo a este script).

    Returns:
        Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
          (main_config, vis_config, logging_config).
          Devuelve (None, None, {}) si hay un error crítico cargando/validando main_config.
          vis_config será None si la visualización está desactivada o hay error.
          logging_config será un diccionario (vacío por defecto).
    """
    # 2.2: Obtener directorio del script actual para rutas relativas
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    main_config_path = os.path.join(script_dir, config_filename)
    main_config, vis_config = None, None
    logging_config = {} # Default

    # --- [1] Carga Main Config ---
    logger.info(f"Intentando cargar config principal desde: {main_config_path}")
    if not os.path.exists(main_config_path):
        logger.error(f"Error crítico: Archivo config principal no encontrado: {main_config_path}")
        return None, None, {} # Fail-Fast

    try:
        with open(main_config_path, 'r', encoding='utf-8') as file:
            main_config = yaml.safe_load(file)
            if not isinstance(main_config, dict):
                # Lanzar error si el contenido no es un diccionario
                raise TypeError(f"Contenido de '{config_filename}' no es un diccionario YAML válido.")
            logger.info(f"Configuración principal cargada desde: {main_config_path}")

        # --- [2] Validación Estructura Esencial ---
        # 2.3: Validar solo presencia y tipo de secciones clave.
        required_sections = [
            'environment', 'simulation', 'pid_adaptation',
            'initial_conditions', 'logging', 'visualization' # Añadir logging/vis para asegurar extracción
        ]
        for section in required_sections:
            if section not in main_config:
                raise KeyError(f"Sección requerida '{section}' ausente en '{config_filename}'.")
            # Permitir que logging/visualization sean None o False, pero deben existir o ser dict
            if section not in ['logging', 'visualization'] and not isinstance(main_config[section], dict):
                 raise TypeError(f"Sección '{section}' debe ser un diccionario en '{config_filename}'.")
            elif section in ['logging', 'visualization'] and main_config[section] is not None and not isinstance(main_config[section], dict):
                 # Si existen pero no son dict (y no son None), es error
                 if isinstance(main_config[section], bool) and section == 'visualization' and not main_config[section]:
                      pass # Permitir visualization: false
                 else:
                      raise TypeError(f"Sección '{section}' debe ser un diccionario o null/false(vis) en '{config_filename}'.")


        # 2.4: Validar estructura interna de 'reward_setup' (se mantiene por complejidad)
        env_cfg = main_config.get('environment', {})
        if 'reward_setup' not in env_cfg:
            raise KeyError("Sección 'environment' debe contener 'reward_setup'.")
        reward_setup = env_cfg['reward_setup']
        if not isinstance(reward_setup, dict):
            raise TypeError("'environment.reward_setup' debe ser un diccionario.")

        # 2.4.1: Validar 'calculation'
        calc_cfg = reward_setup.get('calculation')
        if not isinstance(calc_cfg, dict): raise TypeError("'reward_setup.calculation' debe ser dict.")
        if 'method' not in calc_cfg: raise KeyError("'reward_setup.calculation' falta 'method'.")
        # No validar params internos de 'gaussian', se hará en RewardFactory/Componente

        # 2.4.2: Validar 'stability_calculator' (estructura básica)
        stab_cfg = reward_setup.get('calculation').get('stability_calculator')
        if not isinstance(stab_cfg, dict): raise TypeError("'reward_setup.stability_calculator' debe ser dict.")
        if 'enabled' not in stab_cfg or not isinstance(stab_cfg['enabled'], bool):
            raise TypeError("'stability_calculator' falta 'enabled' o no es booleano.")
        if stab_cfg['enabled'] and ('type' not in stab_cfg or not stab_cfg['type']):
            raise ValueError("Si stability_calculator está habilitado, debe especificar un 'type'.")
        # No validar params internos (ira_params, etc.), se hará en RewardFactory

        # 2.4.3: Validar 'learning_strategy' (estructura básica)
        learn_strat_cfg = reward_setup.get('learning_strategy')
        if not isinstance(learn_strat_cfg, dict): raise TypeError("'reward_setup.learning_strategy' debe ser dict.")
        if 'type' not in learn_strat_cfg: raise KeyError("'learning_strategy' falta 'type'.")
        learn_type = learn_strat_cfg['type']
        allowed_strategies = ['global', 'shadow_baseline', 'echo_baseline']
        if learn_type not in allowed_strategies:
             raise ValueError(f"Tipo de 'learning_strategy' desconocido: '{learn_type}'. Permitidos: {allowed_strategies}")
        if 'strategy_params' not in learn_strat_cfg or not isinstance(learn_strat_cfg['strategy_params'], dict):
            raise TypeError("'learning_strategy' debe tener 'strategy_params' como diccionario.")
        # No validar contenido de strategy_params (beta, etc.), se hará en DI/_create_reward_strategy

        # 2.4.4: Validación Cruzada Shadow <-> Stability (se mantiene)
        if learn_type == 'shadow_baseline' and not stab_cfg.get('enabled', False):
            raise ValueError("Estrategia 'shadow_baseline' requiere 'stability_calculator.enabled' = true.")

        logger.info("Validación de estructura 'reward_setup' completada.")

        # 2.5: Estimación Q-Table (se mantiene como informativo, no bloqueante)
        agent_cfg = env_cfg.get('agent', {})
        if agent_cfg.get('type') == 'pid_qlearning':
             params = agent_cfg.get('params', {})
             state_cfg_q = params.get('state_config', {})
             # ... (lógica de estimación sin cambios, solo loguea WARNING/INFO) ...
             if isinstance(state_cfg_q, dict):
                 enabled_bins = []
                 for var_name, var_cfg in state_cfg_q.items():
                     if isinstance(var_cfg, dict) and var_cfg.get('enabled'):
                         bins = var_cfg.get('bins')
                         if isinstance(bins, int) and bins > 0:
                             enabled_bins.append(bins)
                 if enabled_bins:
                     total_states = np.prod(enabled_bins) if enabled_bins else 1 # Usar np.prod
                     num_actions = params.get('num_actions', 3)
                     if not isinstance(num_actions, int) or num_actions <= 0: num_actions = 3
                     estimated_entries = total_states * num_actions
                     threshold = 1_000_000
                     if estimated_entries > threshold:
                         logger.warning(
                             f"Estimación Q-table: ≈{estimated_entries:,.0f} entradas/tabla "
                             f"({len(enabled_bins)} vars: {enabled_bins} bins, {num_actions} acts). "
                             f"Excede umbral ({threshold:,.0f}). Considerar reducir bins/vars."
                         )
                     else:
                         logger.info(f"Estimación Q-table: ≈{estimated_entries:,.0f} entradas/tabla (OK).")
                 else:
                     logger.info("Estimación Q-table: No hay variables de estado habilitadas/válidas.")
             else:
                 logger.warning("Estimación Q-table: 'agent.params.state_config' no es diccionario.")


    # 2.6: Simplificar captura de errores
    except (yaml.YAMLError, FileNotFoundError, TypeError, ValueError, KeyError) as e:
        logger.error(f"Error crítico cargando/validando '{config_filename}': {e}", exc_info=True)
        return None, None, {} # Fail-Fast
    except Exception as e: # Captura genérica para errores inesperados
        logger.error(f"Error inesperado cargando/validando '{config_filename}': {e}", exc_info=True)
        return None, None, {} # Fail-Fast

    # --- [3] Extraer logging_config (asegurarse que es un dict o {}) ---
    # 3.1: Extraer sección logging, default a {} si no existe o no es dict
    logging_config = main_config.get('logging', {})
    if not isinstance(logging_config, dict):
        logger.warning("'logging' section in config is not a dictionary or null. Using default {}")
        logging_config = {}

    # --- [4] Carga Visualización Config si aplica --- (Revisado) ---
    # 4.1: Extraer sección visualization de main_config
    vis_settings = main_config.get('visualization', {}) # Default a dict vacío si falta la sección
    if vis_settings is None: # Permitir 'visualization: null' en YAML
        vis_settings = {} # Tratar null como vacío
    if not isinstance(vis_settings, dict):
        logger.warning("'visualization' section in config is not a dictionary or null. Disabling visualization.")
        vis_settings = {'enabled': False} # Forzar deshabilitado si no es dict

    vis_enabled = vis_settings.get('enabled', False)
    vis_config = None # Inicializar vis_config como None

    if vis_enabled:
        vis_file_rel = vis_settings.get('config_file')
        if not vis_file_rel or not isinstance(vis_file_rel, str):
            # Loguear advertencia pero NO deshabilitar aquí, permitir fallback
            logger.warning("Visualización habilitada pero 'config_file' falta o es inválido en config principal.")
            # Podríamos intentar cargar un archivo por defecto? O simplemente fallará más adelante.
            # Por ahora, vis_config sigue None.
        else:
            # Construir ruta absoluta al archivo de visualización
            vis_path_abs = os.path.join(script_dir, vis_file_rel)
            logger.info(f"Intentando cargar config visualización desde: {vis_path_abs}")

            if not os.path.exists(vis_path_abs):
                logger.error(f"Archivo config visualización '{vis_path_abs}' NO encontrado. Visualización no funcionará.")
                # vis_config sigue None
            else:
                try:
                    with open(vis_path_abs, 'r', encoding='utf-8') as vf:
                        vis_config_loaded = yaml.safe_load(vf)
                        # --- Validación Clave: Asegurar que carga un dict y TIENE 'plots' ---
                        if not isinstance(vis_config_loaded, dict):
                            logger.error(f"Contenido de '{vis_path_abs}' no es un diccionario YAML. Visualización deshabilitada.")
                            # vis_config sigue None
                        elif 'plots' not in vis_config_loaded or not isinstance(vis_config_loaded['plots'], list):
                            logger.error(f"Archivo config visualización '{vis_path_abs}' cargado, pero falta la clave 'plots' o no es una lista. Visualización deshabilitada.")
                            # vis_config sigue None
                        else:
                            # ¡Éxito! Asignar el diccionario cargado
                            vis_config = vis_config_loaded
                            logger.info(f"Config visualización cargada y validada (contiene 'plots') desde: {vis_path_abs}")

                except (yaml.YAMLError, OSError) as e:
                    logger.error(f"Error cargando/parseando '{vis_path_abs}': {e}. Visualización deshabilitada.")
                    # vis_config sigue None
                except Exception as e:
                    logger.error(f"Error inesperado cargando vis config '{vis_path_abs}': {e}", exc_info=True)
                    # vis_config sigue None
    else:
        logger.info("Visualización deshabilitada explícitamente ('enabled: false') en la config principal.")
        # vis_config sigue None

    # Devolver la tupla (main_config, vis_config (puede ser None), logging_config)
    return main_config, vis_config, logging_config