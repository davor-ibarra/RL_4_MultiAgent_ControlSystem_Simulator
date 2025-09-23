import logging
import time
from typing import Dict, Any, List, Tuple, Optional, Union, TYPE_CHECKING
import numpy as np
import pandas as pd # Needed for isnan checks
import os
import copy # Potentially needed if virtual sim copies controller

# Interfaces (Type Hinting)
from interfaces.environment import Environment
from interfaces.rl_agent import RLAgent
from interfaces.controller import Controller
from interfaces.virtual_simulator import VirtualSimulator
from interfaces.metrics_collector import MetricsCollector
from interfaces.reward_strategy import RewardStrategy

# Importaciones concretas para isinstance (si es necesario)
# from components.reward_strategies.shadow_baseline_reward_strategy import ShadowBaselineRewardStrategy
# from components.reward_strategies.echo_baseline_reward_strategy import EchoBaselineRewardStrategy

# Servicios Auxiliares
from result_handler import ResultHandler
from utils.data_processing import summarize_episode

# Romper ciclo de importación para type hints
if TYPE_CHECKING:
    from di_container import Container


class SimulationManager:
    """ Orquesta la ejecución de la simulación usando DI. """

    def __init__(self,
                 logger: logging.Logger,
                 result_handler: ResultHandler,
                 container: 'Container' # Usar string literal
                ):
        """ Inicializa el SimulationManager. """
        self.logger = logger
        self.result_handler = result_handler
        self.container = container
        self.logger.info("SimulationManager instance created.")
        if container is None:
             msg = "SimulationManager requiere una instancia Container válida."
             self.logger.critical(msg); raise ValueError(msg)

    def _get_dependencies(self) -> Tuple[Environment, RLAgent, Controller, MetricsCollector, RewardStrategy, Optional[VirtualSimulator], Dict[str, Any], str]:
        """ Resuelve dependencias desde el contenedor. """
        # Código de resolución de dependencias mantenido.
        self.logger.debug("Resolviendo dependencias para SimulationManager.run...")
        try:
            environment = self.container.resolve(Environment); agent = self.container.resolve(RLAgent)
            controller = self.container.resolve(Controller); metrics_collector = self.container.resolve(MetricsCollector)
            reward_strategy = self.container.resolve(RewardStrategy); virtual_simulator = self.container.resolve(Optional[VirtualSimulator])
            config = self.container.resolve(dict); results_folder = self.container.resolve(str)
            self.logger.debug("Dependencias resueltas.")
            if None in [environment, agent, controller, metrics_collector, reward_strategy, config, results_folder]:
                 missing = [name for name, var in locals().items() if var is None and name != 'virtual_simulator']
                 raise ValueError(f"Fallo al resolver dependencias clave: {missing}")
            return environment, agent, controller, metrics_collector, reward_strategy, virtual_simulator, config, results_folder
        except ValueError as e: self.logger.critical(f"Error fatal resolviendo dependencias DI: {e}", exc_info=True); raise
        except Exception as e: self.logger.critical(f"Error fatal resolviendo dependencias DI: {e}", exc_info=True); raise

    def run(self) -> Tuple[List[Dict], List[Dict]]:
        """ Ejecuta el bucle principal de simulación. """
        self.logger.info("--- Iniciando Bucle de Simulación ---")
        all_episodes_detailed_data: List[Dict] = []
        summary_data: List[Dict] = []
        # Variables que necesitan estar definidas fuera del try para el finally
        results_folder: Optional[str] = None
        episode: int = -1 # Initialize episode counter
        current_episode_batch: List[Dict] = [] # Batch actual para guardar
        file_handlers: List[logging.Handler] = [] # Para flush de logs

        try:
            # --- 1. Resolver Dependencias y Extraer Config ---
            environment, agent, controller, metrics_collector, reward_strategy, \
                virtual_simulator, config, results_folder = self._get_dependencies()

            # --- Extraer Parámetros de Configuración (con defaults y validación segura) ---
            sim_cfg = config.get('simulation', {}); env_cfg = config.get('environment', {})
            agent_cfg_full = env_cfg.get('agent', {}); agent_params_cfg = agent_cfg_full.get('params', {})
            pid_adapt_cfg = config.get('pid_adaptation', {}); init_cond_cfg = config.get('initial_conditions', {})
            reward_setup_cfg = env_cfg.get('reward_setup', {}); logging_cfg = config.get('logging', {})

            max_episodes = env_cfg.get('max_episodes', 1); total_sim_time_per_episode = env_cfg.get('total_time', 5.0)
            decision_interval = env_cfg.get('decision_interval', 0.01); dt = env_cfg.get('dt', 0.001)
            if not isinstance(dt, (float, int)) or dt <= 0: raise ValueError(f"dt ({dt}) inválido")
            if not isinstance(decision_interval, (float, int)) or decision_interval < dt:
                 self.logger.warning(f"decision_interval ({decision_interval}) < dt ({dt}). Usando dt como intervalo."); decision_interval = dt
            max_steps = int(round(total_sim_time_per_episode / dt)) if dt > 0 else 0

            episodes_per_file = sim_cfg.get('episodes_per_file', 100)
            agent_state_save_freq = sim_cfg.get('agent_state_save_frequency', 1000) if sim_cfg.get('save_agent_state', False) else 0
            log_flush_frequency = logging_cfg.get('log_save_frequency', 0)

            initial_state_vector = init_cond_cfg.get('x0', [0.0, 0.0, 0.0, 0.0])
            if initial_state_vector is None: raise KeyError("'initial_conditions: x0' is missing")

            # Determinar tipo de estrategia para lógica condicional
            is_echo_baseline = "EchoBaseline" in type(reward_strategy).__name__
            is_shadow_baseline = "ShadowBaseline" in type(reward_strategy).__name__
            if is_echo_baseline and virtual_simulator is None:
                 self.logger.error("Echo Baseline activado pero VirtualSimulator no está disponible/resuelto. Abortando."); return [], []

            gain_step_config = pid_adapt_cfg.get('gain_step', 5.0)
            variable_step = pid_adapt_cfg.get('variable_step', False)
            # Configuración de límites de ganancias desde el agente (para clipping)
            gain_limits_cfg = agent_cfg_full.get('params', {}).get('state_config', {})

            # Preparar handlers para flush de logs
            if log_flush_frequency > 0:
                 file_handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler) and hasattr(h, 'flush')]
                 if not file_handlers: self.logger.warning("log_flush_frequency > 0 pero no se encontraron FileHandlers."); log_flush_frequency = 0

            total_virtual_sim_time = 0.0 # Para Echo Baseline

            # --- 2. Bucle Principal de Episodios ---
            self.logger.info(f"Iniciando simulación para {max_episodes} episodios...")
            for episode in range(max_episodes):
                episode_start_time = time.time()
                self.logger.info(f"--- [ Episodio {episode}/{max_episodes-1} ] ---")

                # --- Resetear Entorno y Colector ---
                try:
                    current_state_vector = environment.reset(initial_state_vector)
                    metrics_collector.reset(episode_id=episode)
                except Exception as e:
                    self.logger.error(f"Error reseteando entorno/colector ep {episode}: {e}", exc_info=True); continue # Saltar episodio

                # --- Inicializar Variables del Episodio ---
                cumulative_reward, interval_reward = 0.0, 0.0
                interval_stability_scores: List[float] = [] # Para Shadow Baseline avg_w_stab
                next_decision_time = decision_interval
                agent_decision_count = 0 # **Contador de decisiones REQUERIDO**
                done = False; termination_reason = "unknown"
                current_time = 0.0
                last_learn_time = np.nan # Para log learn_select_duration_ms
                last_td_errors_logged = False # Flag para evitar log doble de NaN

                # Diccionario para almacenar datos entre intervalos para `agent.learn`
                last_interval_data_for_learn: Optional[Dict[str, Any]] = None

                # --- Log Estado Inicial (t=0) ---
                # (Similar al código antiguo, usando metrics_collector)
                metrics_collector.log('time', 0.0)
                metrics_collector.log('cart_position', current_state_vector[0])
                metrics_collector.log('cart_velocity', current_state_vector[1])
                metrics_collector.log('pendulum_angle', current_state_vector[2])
                metrics_collector.log('pendulum_velocity', current_state_vector[3])
                try:
                    # Usar interfaz Controller para acceder a params y estado
                    ctrl_setpoint = getattr(controller, 'setpoint', 0.0)
                    metrics_collector.log('error', current_state_vector[2] - ctrl_setpoint)
                    metrics_collector.log('kp', controller.get_params().get('kp', np.nan))
                    metrics_collector.log('ki', controller.get_params().get('ki', np.nan))
                    metrics_collector.log('kd', controller.get_params().get('kd', np.nan))
                    metrics_collector.log('integral_error', getattr(controller, 'integral_error', np.nan))
                    metrics_collector.log('derivative_error', getattr(controller, 'derivative_error', np.nan))
                    # Usar interfaz RLAgent para acceder a params
                    metrics_collector.log('epsilon', getattr(agent, 'epsilon', np.nan))
                    metrics_collector.log('learning_rate', getattr(agent, 'learning_rate', np.nan))
                except AttributeError as ae: self.logger.warning(f"No se pudo loguear atributo inicial: {ae}")
                metrics_collector.log('reward', 0.0); metrics_collector.log('cumulative_reward', 0.0)
                metrics_collector.log('force', 0.0); metrics_collector.log('stability_score', 1.0) # Asumir estabilidad inicial
                metrics_collector.log('action_kp', np.nan); metrics_collector.log('action_ki', np.nan); metrics_collector.log('action_kd', np.nan)
                metrics_collector.log('learn_select_duration_ms', np.nan)
                # Log Q-values/Visits/Baselines iniciales (serán mayormente NaN/defaults si agente no entrenado)
                initial_agent_state_dict = agent.build_agent_state(current_state_vector, controller, agent_params_cfg.get('state_config', {}))
                if hasattr(metrics_collector, 'log_q_values'): metrics_collector.log_q_values(agent, initial_agent_state_dict) # type: ignore
                if hasattr(metrics_collector, 'log_q_visit_counts'): metrics_collector.log_q_visit_counts(agent, initial_agent_state_dict) # type: ignore
                if hasattr(metrics_collector, 'log_baselines'): metrics_collector.log_baselines(agent, initial_agent_state_dict) # type: ignore
                # Log otros valores iniciales como NaN
                metrics_collector.log('td_error_kp', np.nan); metrics_collector.log('td_error_ki', np.nan); metrics_collector.log('td_error_kd', np.nan)
                metrics_collector.log('virtual_reward_kp', np.nan); metrics_collector.log('virtual_reward_ki', np.nan); metrics_collector.log('virtual_reward_kd', np.nan)
                metrics_collector.log('id_agent_decision', 0) # Primera decisión (implícita en t=0)
                # Log gain step inicial
                step_kp, step_ki, step_kd = np.nan, np.nan, np.nan
                if variable_step and isinstance(gain_step_config, dict):
                    step_kp = float(gain_step_config.get('kp', np.nan))
                    step_ki = float(gain_step_config.get('ki', np.nan))
                    step_kd = float(gain_step_config.get('kd', np.nan))
                    metrics_collector.log('gain_step_kp', step_kp); metrics_collector.log('gain_step_ki', step_ki); metrics_collector.log('gain_step_kd', step_kd)
                else:
                     step_kp = step_ki = step_kd = float(gain_step_config) if isinstance(gain_step_config, (int,float)) else np.nan
                     metrics_collector.log('gain_step', step_kp) # Log genérico
                # Adaptive stats iniciales (serán NaN)
                if hasattr(metrics_collector, 'log_adaptive_stats'): metrics_collector.log_adaptive_stats({}) # type: ignore


                # --- Selección de Acción Inicial (A') ---
                # Seleccionar la acción que se aplicará durante el *primer* intervalo
                learn_block_start_time = time.time()
                try:
                    current_agent_state_dict = agent.build_agent_state(current_state_vector, controller, agent_params_cfg.get('state_config', {}))
                    actions_prime = agent.select_action(current_agent_state_dict) # Acción A' para el primer intervalo
                except Exception as e:
                    self.logger.error(f"Error selección acción inicial ep {episode}: {e}. Usando acción neutral.", exc_info=True)
                    actions_prime = {'kp': 1, 'ki': 1, 'kd': 1}

                # Aplicar acción A' para obtener las ganancias del *primer* intervalo
                try:
                    kp, ki, kd = controller.kp, controller.ki, controller.kd
                    if variable_step and isinstance(gain_step_config, dict):
                        if actions_prime['kp'] == 0: kp -= step_kp
                        elif actions_prime['kp'] == 2: kp += step_kp
                        if actions_prime['ki'] == 0: ki -= step_ki
                        elif actions_prime['ki'] == 2: ki += step_ki
                        if actions_prime['kd'] == 0: kd -= step_kd
                        elif actions_prime['kd'] == 2: kd += step_kd
                    else: # Paso fijo
                        step = float(gain_step_config) if isinstance(gain_step_config, (int,float)) else 0.0
                        if actions_prime['kp'] == 0: kp -= step
                        elif actions_prime['kp'] == 2: kp += step
                        if actions_prime['ki'] == 0: ki -= step
                        elif actions_prime['ki'] == 2: ki += step
                        if actions_prime['kd'] == 0: kd -= step
                        elif actions_prime['kd'] == 2: kd += step

                    # Clip gains usando límites desde config
                    kp_cfg = gain_limits_cfg.get('kp', {}); ki_cfg = gain_limits_cfg.get('ki', {}); kd_cfg = gain_limits_cfg.get('kd', {})
                    kp = np.clip(kp, kp_cfg.get('min', -np.inf), kp_cfg.get('max', np.inf))
                    ki = np.clip(ki, ki_cfg.get('min', -np.inf), ki_cfg.get('max', np.inf))
                    kd = np.clip(kd, kd_cfg.get('min', -np.inf), kd_cfg.get('max', np.inf))

                    controller.update_params(kp, ki, kd)
                    self.logger.debug(f"SM Acción A' Inicial Aplicada Ep {episode}. New Gains: kp={kp:.4f}, ki={ki:.4f}, kd={kd:.4f}")

                except KeyError as e: self.logger.error(f"SM Error aplicando acción inicial: Clave {e}. Actions: {actions_prime}")
                except Exception as e: self.logger.error(f"SM Error aplicando acción inicial: {e}", exc_info=True)

                # Guardar datos para el primer bloque de learn (que ocurrirá al final del primer intervalo)
                last_interval_data_for_learn = {
                     'start_state_dict': current_agent_state_dict, # S al inicio (t=0)
                     'start_raw_state_vector': np.copy(current_state_vector),
                     'actions_dict': actions_prime.copy(),         # A' (acciones a aplicar durante el 1er int)
                     'reward_sum': 0.0,
                     'w_stab_sum': 0.0,
                     'steps': 0,
                     'end_state_dict': None, # Se calculará al final del intervalo
                     'done': False,
                     'reward_dict_for_learn': None # Se calculará si es Echo
                 }
                last_learn_time = (time.time() - learn_block_start_time) * 1000 # Duración selección/aplicación inicial

                # --- 3. Bucle de Pasos Internos (dt) ---
                for t_step in range(max_steps):
                    current_time = round((t_step + 1) * dt, 6) # Tiempo al FINAL del paso dt actual

                    # --- A. Ejecutar Paso del Entorno ---
                    try:
                        next_state_vector, (reward, stability_score), force = environment.step()
                    except Exception as e:
                        self.logger.error(f"CRITICAL: Error env.step t={current_time:.4f} ep {episode}: {e}. Terminando episodio.", exc_info=True)
                        done = True; termination_reason = "env_step_error"
                        # Loguear estado final ANTES de romper
                        metrics_collector.log('time', current_time); metrics_collector.log('reward', np.nan)
                        metrics_collector.log('cumulative_reward', cumulative_reward); metrics_collector.log('stability_score', np.nan)
                        break # Salir del bucle de pasos

                    # --- B. Acumular Métricas del Intervalo ---
                    # Asegurar que sean flotantes y finitos
                    reward_f = float(reward) if np.isfinite(reward) else 0.0
                    stability_score_f = float(stability_score) if np.isfinite(stability_score) else 0.0 # Usar 0 si es NaN/inf? O 1? Usar 0 para ser conservador.
                    force_f = float(force) if np.isfinite(force) else np.nan

                    cumulative_reward += reward_f
                    interval_reward += reward_f
                    interval_stability_scores.append(stability_score_f)
                    # Actualizar datos para el learn del *próximo* intervalo
                    if last_interval_data_for_learn:
                        last_interval_data_for_learn['reward_sum'] += reward_f
                        last_interval_data_for_learn['w_stab_sum'] += stability_score_f
                        last_interval_data_for_learn['steps'] += 1


                    # --- C. Loguear Métricas de ESTE Paso (dt) ---
                    metrics_collector.log('time', current_time)
                    metrics_collector.log('cart_position', next_state_vector[0])
                    metrics_collector.log('cart_velocity', next_state_vector[1])
                    metrics_collector.log('pendulum_angle', next_state_vector[2])
                    metrics_collector.log('pendulum_velocity', next_state_vector[3])
                    try:
                        ctrl_setpoint = getattr(controller, 'setpoint', 0.0)
                        metrics_collector.log('error', next_state_vector[2] - ctrl_setpoint)
                        metrics_collector.log('kp', controller.get_params().get('kp', np.nan)) # Gains *actuales*
                        metrics_collector.log('ki', controller.get_params().get('ki', np.nan))
                        metrics_collector.log('kd', controller.get_params().get('kd', np.nan))
                        metrics_collector.log('integral_error', getattr(controller, 'integral_error', np.nan))
                        metrics_collector.log('derivative_error', getattr(controller, 'derivative_error', np.nan))
                        # Acciones A' que se están aplicando *durante* este intervalo
                        metrics_collector.log('action_kp', actions_prime.get('kp', np.nan))
                        metrics_collector.log('action_ki', actions_prime.get('ki', np.nan))
                        metrics_collector.log('action_kd', actions_prime.get('kd', np.nan))
                        # Parámetros actuales del agente
                        metrics_collector.log('epsilon', getattr(agent, 'epsilon', np.nan))
                        metrics_collector.log('learning_rate', getattr(agent, 'learning_rate', np.nan))
                        # Log gain step usado
                        if variable_step:
                             metrics_collector.log('gain_step_kp', step_kp); metrics_collector.log('gain_step_ki', step_ki); metrics_collector.log('gain_step_kd', step_kd)
                        else: metrics_collector.log('gain_step', step_kp)
                    except AttributeError as ae: self.logger.warning(f"No se pudo loguear atributo en step: {ae}")
                    metrics_collector.log('reward', reward_f); metrics_collector.log('cumulative_reward', cumulative_reward)
                    metrics_collector.log('force', force_f); metrics_collector.log('stability_score', stability_score_f)

                    # Loguear NaNs para métricas de decisión si no estamos en boundary
                    # (se sobreescribirán si estamos en boundary)
                    metrics_collector.log('learn_select_duration_ms', np.nan)
                    metrics_collector.log('id_agent_decision', np.nan)
                    if hasattr(metrics_collector, 'log_q_values'): metrics_collector.log_q_values(agent, {}) # Log NaNs
                    if hasattr(metrics_collector, 'log_q_visit_counts'): metrics_collector.log_q_visit_counts(agent, {})
                    if hasattr(metrics_collector, 'log_baselines'): metrics_collector.log_baselines(agent, {})
                    metrics_collector.log('td_error_kp', np.nan); metrics_collector.log('td_error_ki', np.nan); metrics_collector.log('td_error_kd', np.nan)
                    metrics_collector.log('virtual_reward_kp', np.nan); metrics_collector.log('virtual_reward_ki', np.nan); metrics_collector.log('virtual_reward_kd', np.nan)
                    last_td_errors_logged = False # Resetear flag

                    # --- D. Verificar Terminación ---
                    try:
                        angle_exc, cart_exc, stab = environment.check_termination(config)
                        time_limit_reached = (current_time >= total_sim_time_per_episode - dt / 2)
                        if angle_exc or cart_exc or stab or time_limit_reached:
                            done = True
                            if termination_reason == "unknown": # Asignar solo la primera vez
                                if angle_exc: termination_reason = "angle_limit"
                                elif cart_exc: termination_reason = "cart_limit"
                                elif stab: termination_reason = "stabilized"
                                elif time_limit_reached: termination_reason = "time_limit"
                                else: termination_reason = "unknown_done"
                            self.logger.info(f"Ep {episode} terminando: {termination_reason} at t={current_time:.3f}")
                    except Exception as e:
                        self.logger.error(f"Error check_termination t={current_time:.3f} ep {episode}: {e}", exc_info=True)
                        done = True; termination_reason = "termination_check_error"

                    # --- E. Bloque de Decisión y Aprendizaje (si corresponde) ---
                    # Usar current_time (tiempo al final del paso dt actual)
                    is_decision_boundary = current_time >= next_decision_time - dt / 2 # Añadir tolerancia

                    if is_decision_boundary or done:
                        learn_block_start_time = time.time() # Iniciar cronómetro para este bloque
                        agent_decision_count += 1 # **INCREMENTAR CONTADOR DE DECISIÓN**
                        metrics_collector.log('id_agent_decision', agent_decision_count) # Log ID

                        # --- 1. Estado S' al final del intervalo ---
                        # Usar next_state_vector que es el estado al final de este paso dt
                        next_agent_state_dict = agent.build_agent_state(next_state_vector, controller, agent_params_cfg.get('state_config', {}))

                        # --- 2. Completar datos del intervalo ANTERIOR para learn ---
                        if last_interval_data_for_learn:
                            last_interval_data_for_learn['end_state_dict'] = next_agent_state_dict # S'
                            last_interval_data_for_learn['done'] = done # Estado de terminación al final

                            # --- 3. Calcular R_learn para el intervalo ANTERIOR ---
                            # Media de estabilidad del intervalo que acaba de terminar
                            avg_interval_stability = np.mean(interval_stability_scores) if interval_stability_scores else 1.0
                            if not isinstance(avg_interval_stability, (float, int, np.number)) or pd.isna(avg_interval_stability):
                                avg_interval_stability = 1.0 # Fallback

                            # Recompensa acumulada en el intervalo que acaba de terminar
                            interval_reward_ended = last_interval_data_for_learn['reward_sum']

                            # Calcular R_diff si es Echo Baseline
                            reward_dict_echo: Optional[Dict[str, float]] = None
                            if is_echo_baseline and virtual_simulator is not None:
                                 echo_sim_start = time.time()
                                 reward_dict_echo = {}
                                 gains_before = controller.get_params() # Gains al inicio de ESTE intervalo (resultado de A' anterior)
                                 # Pre-calcular gains si se aplicara la *nueva* acción A'' (que aún no se selecciona)
                                 # Necesitamos seleccionar A'' primero
                                 temp_actions_prime_prime = agent.select_action(next_agent_state_dict) if not done else {'kp': 1, 'ki': 1, 'kd': 1}
                                 kp_after, ki_after, kd_after = gains_before['kp'], gains_before['ki'], gains_before['kd']
                                 # Aplicar temp_actions_prime_prime para calcular gains_after hipotéticos
                                 if variable_step and isinstance(gain_step_config, dict):
                                      if temp_actions_prime_prime['kp'] == 0: kp_after -= step_kp
                                      elif temp_actions_prime_prime['kp'] == 2: kp_after += step_kp
                                      if temp_actions_prime_prime['ki'] == 0: ki_after -= step_ki
                                      elif temp_actions_prime_prime['ki'] == 2: ki_after += step_ki
                                      if temp_actions_prime_prime['kd'] == 0: kd_after -= step_kd
                                      elif temp_actions_prime_prime['kd'] == 2: kd_after += step_kd
                                 else:
                                      step = float(gain_step_config) if isinstance(gain_step_config, (int,float)) else 0.0
                                      if temp_actions_prime_prime['kp'] == 0: kp_after -= step
                                      elif temp_actions_prime_prime['kp'] == 2: kp_after += step
                                      if temp_actions_prime_prime['ki'] == 0: ki_after -= step
                                      elif temp_actions_prime_prime['ki'] == 2: ki_after += step
                                      if temp_actions_prime_prime['kd'] == 0: kd_after -= step
                                      elif temp_actions_prime_prime['kd'] == 2: kd_after += step
                                 # Clip gains_after hipotéticos
                                 kp_cfg = gain_limits_cfg.get('kp', {}); ki_cfg = gain_limits_cfg.get('ki', {}); kd_cfg = gain_limits_cfg.get('kd', {})
                                 kp_after = np.clip(kp_after, kp_cfg.get('min', -np.inf), kp_cfg.get('max', np.inf))
                                 ki_after = np.clip(ki_after, ki_cfg.get('min', -np.inf), ki_cfg.get('max', np.inf))
                                 kd_after = np.clip(kd_after, kd_cfg.get('min', -np.inf), kd_cfg.get('max', np.inf))
                                 gains_after_hypothetical = {'kp': kp_after, 'ki': ki_after, 'kd': kd_after}

                                 # Definir estados contrafactuales
                                 interval_start_state = last_interval_data_for_learn['start_raw_state_vector'] # Necesitamos estado crudo
                                 interval_start_t = current_time - decision_interval # Tiempo aprox inicio intervalo
                                 gains_p_cf = {'kp': gains_before['kp'], 'ki': gains_after_hypothetical['ki'], 'kd': gains_after_hypothetical['kd']}
                                 gains_i_cf = {'kp': gains_after_hypothetical['kp'], 'ki': gains_before['ki'], 'kd': gains_after_hypothetical['kd']}
                                 gains_d_cf = {'kp': gains_after_hypothetical['kp'], 'ki': gains_after_hypothetical['ki'], 'kd': gains_before['kd']}
                                 # Correr simulaciones virtuales
                                 R_p_cf = virtual_simulator.run_interval_simulation(interval_start_state, interval_start_t, decision_interval, gains_p_cf)
                                 R_i_cf = virtual_simulator.run_interval_simulation(interval_start_state, interval_start_t, decision_interval, gains_i_cf)
                                 R_d_cf = virtual_simulator.run_interval_simulation(interval_start_state, interval_start_t, decision_interval, gains_d_cf)
                                 # Calcular recompensas diferenciales
                                 reward_dict_echo = {'kp': interval_reward_ended - R_p_cf, 'ki': interval_reward_ended - R_i_cf, 'kd': interval_reward_ended - R_d_cf}
                                 total_virtual_sim_time += (time.time() - echo_sim_start)
                                 if hasattr(metrics_collector, 'log_virtual_rewards'): metrics_collector.log_virtual_rewards(reward_dict_echo) # type: ignore
                            else:
                                 if hasattr(metrics_collector, 'log_virtual_rewards'): metrics_collector.log_virtual_rewards(None) # Log NaNs

                            # Determinar qué pasar a agent.learn basado en estrategia
                            reward_info_for_agent_learn: Union[float, Tuple[float, float], Dict[str, float]]
                            if is_echo_baseline: reward_info_for_agent_learn = reward_dict_echo if reward_dict_echo is not None else {}
                            elif is_shadow_baseline: reward_info_for_agent_learn = (interval_reward_ended, avg_interval_stability)
                            else: reward_info_for_agent_learn = interval_reward_ended # Global

                            # --- 4. Llamar a Agent Learn ---
                            # Usa S (start_state_dict), A (actions_dict), R_learn, S' (end_state_dict), Done
                            try:
                                agent.learn(
                                    current_agent_state_dict=last_interval_data_for_learn['start_state_dict'],
                                    actions_dict=last_interval_data_for_learn['actions_dict'],
                                    reward_info=reward_info_for_agent_learn,
                                    next_agent_state_dict=last_interval_data_for_learn['end_state_dict'],
                                    done=last_interval_data_for_learn['done']
                                )
                                # Log TD Errors DESPUÉS de learn
                                if hasattr(metrics_collector, 'log_td_errors'):
                                     metrics_collector.log_td_errors(agent.get_last_td_errors()) # type: ignore
                                     last_td_errors_logged = True # Marcar que se logueó
                            except Exception as learn_e:
                                 self.logger.error(f"Error en agent.learn ep {episode}: {learn_e}", exc_info=True)
                                 if hasattr(metrics_collector, 'log_td_errors'): metrics_collector.log_td_errors({}) # Log NaN
                                 last_td_errors_logged = True

                        # --- 5. Seleccionar Siguiente Acción (A') ---
                        if not done:
                            actions_prime = agent.select_action(next_agent_state_dict) # Nueva A'
                        else:
                            actions_prime = {'kp': 1, 'ki': 1, 'kd': 1} # Acción neutral si termina

                        # --- 6. Aplicar Nueva Acción A' (actualizar controller para *próximo* intervalo) ---
                        if not done:
                            try:
                                kp, ki, kd = controller.kp, controller.ki, controller.kd
                                if variable_step and isinstance(gain_step_config, dict):
                                     if actions_prime['kp'] == 0: kp -= step_kp
                                     elif actions_prime['kp'] == 2: kp += step_kp
                                     if actions_prime['ki'] == 0: ki -= step_ki
                                     elif actions_prime['ki'] == 2: ki += step_ki
                                     if actions_prime['kd'] == 0: kd -= step_kd
                                     elif actions_prime['kd'] == 2: kd += step_kd
                                else:
                                     step = float(gain_step_config) if isinstance(gain_step_config, (int,float)) else 0.0
                                     if actions_prime['kp'] == 0: kp -= step
                                     elif actions_prime['kp'] == 2: kp += step
                                     if actions_prime['ki'] == 0: ki -= step
                                     elif actions_prime['ki'] == 2: ki += step
                                     if actions_prime['kd'] == 0: kd -= step
                                     elif actions_prime['kd'] == 2: kd += step

                                # Clip gains
                                kp_cfg = gain_limits_cfg.get('kp', {}); ki_cfg = gain_limits_cfg.get('ki', {}); kd_cfg = gain_limits_cfg.get('kd', {})
                                kp = np.clip(kp, kp_cfg.get('min', -np.inf), kp_cfg.get('max', np.inf))
                                ki = np.clip(ki, ki_cfg.get('min', -np.inf), ki_cfg.get('max', np.inf))
                                kd = np.clip(kd, kd_cfg.get('min', -np.inf), kd_cfg.get('max', np.inf))

                                controller.update_params(kp, ki, kd)
                                self.logger.debug(f"SM Acción A' Aplicada Ep {episode} Dec {agent_decision_count}. New Gains: kp={kp:.4f}, ki={ki:.4f}, kd={kd:.4f}")
                            except KeyError as e: self.logger.error(f"SM Error aplicando nueva acción A': Clave {e}. Actions: {actions_prime}")
                            except Exception as e: self.logger.error(f"SM Error aplicando nueva acción A': {e}", exc_info=True)

                        # --- 7. Loguear Métricas del Bloque de Decisión ---
                        # Log Q-values, Visits, Baselines para S' (estado actual)
                        if hasattr(metrics_collector, 'log_q_values'): metrics_collector.log_q_values(agent, next_agent_state_dict) # type: ignore
                        if hasattr(metrics_collector, 'log_q_visit_counts'): metrics_collector.log_q_visit_counts(agent, next_agent_state_dict) # type: ignore
                        if hasattr(metrics_collector, 'log_baselines'): metrics_collector.log_baselines(agent, next_agent_state_dict) # type: ignore
                        # Log TD errors si no se loguearon ya tras learn()
                        if not last_td_errors_logged:
                             if hasattr(metrics_collector, 'log_td_errors'): metrics_collector.log_td_errors(agent.get_last_td_errors()) # type: ignore
                        # Logear tiempo de ejecución del bloque
                        last_learn_time = (time.time() - learn_block_start_time) * 1000
                        metrics_collector.log('learn_select_duration_ms', last_learn_time)

                        # --- 8. Preparar datos para el *siguiente* ciclo de learn ---
                        if not done:
                            last_interval_data_for_learn = {
                                'start_state_dict': next_agent_state_dict, # S será S'
                                'start_raw_state_vector': np.copy(current_state_vector), # Estado crudo actual (que será el inicio del siguiente intervalo)
                                'actions_dict': actions_prime.copy(),      # A serán A'
                                'reward_sum': 0.0, 'w_stab_sum': 0.0, 'steps': 0, # Resetear contadores
                                'end_state_dict': None, 'done': False, 'reward_dict_for_learn': None
                             }
                        else:
                            last_interval_data_for_learn = None # No más learning

                        # --- 9. Resetear acumuladores del intervalo y actualizar next_decision_time ---
                        interval_reward = 0.0
                        interval_stability_scores = []
                        next_decision_time = round((np.floor(current_time / decision_interval) + 1) * decision_interval, 6)
                        # Asegurarse que next_decision_time avance
                        if next_decision_time <= current_time: next_decision_time = round(current_time + decision_interval, 6)

                    # --- F. Actualizar Estado y Salir si 'done' ---
                    current_state_vector = next_state_vector # Avanzar estado
                    if done: break # Salir del bucle de pasos

                # --- 4. Fin Bucle de Pasos (dt) ---

                # --- Procesamiento Post-Episodio ---
                if not done: # Si terminó por max_steps
                     current_time = total_sim_time_per_episode
                     termination_reason = "time_limit"
                     self.logger.info(f"Episodio {episode} finalizado: Límite de tiempo alcanzado t={current_time:.3f}")
                     # Asegurarse que el último learn se ejecute si faltó
                     if last_interval_data_for_learn and last_interval_data_for_learn['steps'] > 0:
                         # El estado final (S') es el último current_state_vector
                         last_interval_data_for_learn['end_state_dict'] = agent.build_agent_state(current_state_vector, controller, agent_params_cfg.get('state_config', {}))
                         last_interval_data_for_learn['done'] = True
                         avg_interval_stability = np.mean(interval_stability_scores) if interval_stability_scores else 1.0
                         if not isinstance(avg_interval_stability, (float, int, np.number)) or pd.isna(avg_interval_stability): avg_interval_stability = 1.0
                         interval_reward_ended = last_interval_data_for_learn['reward_sum']
                         reward_dict_echo = None # No calcular Echo en el último paso forzado
                         if is_echo_baseline: reward_info_for_agent_learn = {}
                         elif is_shadow_baseline: reward_info_for_agent_learn = (interval_reward_ended, avg_interval_stability)
                         else: reward_info_for_agent_learn = interval_reward_ended
                         try:
                             agent.learn(**last_interval_data_for_learn) # type: ignore # unpack dict
                         except Exception as learn_e: self.logger.error(f"Error en learn final ep {episode}: {learn_e}", exc_info=True)

                episode_duration_s = time.time() - episode_start_time
                episode_metrics = metrics_collector.get_metrics()

                # Añadir métricas finales calculadas
                episode_metrics['termination_reason'] = [termination_reason] * len(next(iter(episode_metrics.values()), []))
                episode_metrics['episode_duration_s'] = [episode_duration_s] * len(next(iter(episode_metrics.values()), []))
                final_gains = controller.get_params()
                episode_metrics['final_kp'] = [final_gains.get('kp', np.nan)] * len(next(iter(episode_metrics.values()), []))
                episode_metrics['final_ki'] = [final_gains.get('ki', np.nan)] * len(next(iter(episode_metrics.values()), []))
                episode_metrics['final_kd'] = [final_gains.get('kd', np.nan)] * len(next(iter(episode_metrics.values()), []))
                episode_metrics['total_agent_decisions'] = [agent_decision_count] * len(next(iter(episode_metrics.values()), [])) # Usar contador
                # Calcular avg_stability_score y total_reward (se añadirán en summarize_episode)
                episode_metrics['avg_stability_score'] = [np.nanmean(episode_metrics.get('stability_score', [np.nan]))] * len(next(iter(episode_metrics.values()), []))
                episode_metrics['total_reward'] = [np.nansum(episode_metrics.get('reward', [np.nan]))] * len(next(iter(episode_metrics.values()), []))

                # Llamar a summarize_episode (devuelve un dict con agregados)
                summary = summarize_episode(episode_metrics)
                summary['episode'] = episode # Añadir ID de episodio al resumen
                summary_data.append(summary)

                # Añadir datos detallados al batch actual
                current_episode_batch.append(episode_metrics)

                # Log Resumen
                self.logger.info(f"Ep {episode} Resumen: Kp_final={summary.get('final_kp', np.nan):.2f}, "
                                 f"Ki_final={summary.get('final_ki', np.nan):.2f}, "
                                 f"Kd_final={summary.get('final_kd', np.nan):.3f}, "
                                 f"N_decisions={summary.get('total_agent_decisions', 0)}")
                self.logger.info(f"Ep {episode} Resumen: T_Reward={summary.get('total_reward', np.nan):.2f}, "
                                 f"Perf={summary.get('performance', np.nan):.2f}, "
                                 f"AvgStab={summary.get('avg_stability_score', np.nan):.3f}, "
                                 f"epsilon={summary.get('final_epsilon', np.nan):.3f}, "
                                 f"alpha={summary.get('final_learning_rate', np.nan):.4f}, "
                                 f"dur={summary.get('episode_duration_s', np.nan):.2f}s")

                # Actualizar stats adaptativas (si aplica)
                try: environment.update_reward_calculator_stats(episode_metrics, episode)
                except Exception as e: self.logger.error(f"Error update reward stats ep {episode}: {e}", exc_info=True)

                # Guardado Periódico de Datos
                if episodes_per_file > 0 and ((episode + 1) % episodes_per_file == 0 or episode == max_episodes - 1):
                    if current_episode_batch:
                        self.result_handler.save_episode_batch(current_episode_batch, results_folder, episode)
                        current_episode_batch = [] # Limpiar batch después de guardar
                if agent_state_save_freq > 0 and (episode + 1) % agent_state_save_freq == 0:
                    self.result_handler.save_agent_state(agent, episode, results_folder)
                if log_flush_frequency > 0 and (episode + 1) % log_flush_frequency == 0:
                    self.logger.debug(f"Flushing logs file after ep {episode}...")
                    for h in file_handlers:
                        try: h.flush() # type: ignore
                        except Exception as e_flush: self.logger.warning(f"Error flush {h}: {e_flush}")

            # --- 5. Fin Bucle de Episodios ---

        except Exception as e:
            self.logger.error(f"Error inesperado en bucle simulación ep {episode}: {e}", exc_info=True)
            # Intentar guardar batch parcial si hay datos y carpeta válida
            if current_episode_batch and results_folder and os.path.isdir(results_folder):
                 self.logger.warning("Intentando guardar batch parcial de episodios tras error inesperado...")
                 self.result_handler.save_episode_batch(current_episode_batch, results_folder, episode if episode >= 0 else -1)

        finally:
            # Asegurar flush final de logs si aplica
            if file_handlers:
                 self.logger.info("Realizando flush final de logs...")
                 for h in file_handlers:
                      try: h.flush() # type: ignore
                      except Exception as e_flush_final: self.logger.warning(f"Error flush final {h}: {e_flush_final}")
            self.logger.info("--- Simulación Principal Finalizada ---")

        # Devolver los datos resumen. all_episodes_detailed_data estará vacío si se guardó por batch.
        return all_episodes_detailed_data, summary_data