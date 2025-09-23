# simulation_manager.py
import logging
import time
import numpy as np
import pandas as pd
import os
import copy # Necesario para deepcopy en simulación virtual (aunque ya no se haga aquí directamente)
from typing import Dict, Any, List, Tuple, Optional, Union, TYPE_CHECKING

# Interfaces (Type Hinting)
from interfaces.environment import Environment
from interfaces.rl_agent import RLAgent
from interfaces.controller import Controller
from interfaces.virtual_simulator import VirtualSimulator
from interfaces.metrics_collector import MetricsCollector
from interfaces.reward_strategy import RewardStrategy

# Importar estrategias concretas para check de tipo (EchoBaseline)
from components.reward_strategies.echo_baseline_reward_strategy import EchoBaselineRewardStrategy

# Servicios Auxiliares
from result_handler import ResultHandler
from utils.data_processing import summarize_episode

# Romper ciclo de importación para type hints
if TYPE_CHECKING:
    from di_container import Container

# Obtener logger
logger = logging.getLogger(__name__)

class SimulationManager:
    """
    Orquesta la ejecución de la simulación completa, episodio por episodio.
    Utiliza componentes resueltos vía DI y gestiona el flujo principal,
    incluyendo la ejecución de intervalos normales o con simulación virtual (Echo).
    """

    def __init__(self,
                 logger: logging.Logger, # Inyectar logger específico si se prefiere
                 result_handler: ResultHandler,
                 container: 'Container'
                 ):
        """ Inicializa el SimulationManager. """
        self.logger = logger # Usar logger inyectado
        self.result_handler = result_handler
        self.container = container
        self.logger.info("SimulationManager instance created.")
        if container is None:
             msg = "SimulationManager requiere una instancia Container válida."
             self.logger.critical(msg); raise ValueError(msg)

    # --- Métodos Privados de Orquestación ---

    def _resolve_dependencies(self) -> Tuple[Environment, RLAgent, Controller, MetricsCollector, RewardStrategy, Optional[VirtualSimulator], Dict[str, Any], str]:
        """ Resuelve las dependencias principales desde el contenedor DI. """
        self.logger.debug("Resolviendo dependencias para la simulación...")
        try:
            environment = self.container.resolve(Environment)
            agent = self.container.resolve(RLAgent)
            controller = self.container.resolve(Controller)
            metrics_collector = self.container.resolve(MetricsCollector)
            reward_strategy = self.container.resolve(RewardStrategy)
            virtual_simulator = self.container.resolve(Optional[VirtualSimulator]) # Puede ser None
            config = self.container.resolve(dict)
            results_folder = self.container.resolve(str) # Resuelto desde DI
            self.logger.debug("Dependencias resueltas.")

            # Validar dependencias no opcionales
            if None in [environment, agent, controller, metrics_collector, reward_strategy, config, results_folder]:
                 missing = [name for name, var in locals().items() if var is None and name not in ['virtual_simulator']]
                 raise ValueError(f"Fallo al resolver dependencias clave: {missing}")

            return environment, agent, controller, metrics_collector, reward_strategy, virtual_simulator, config, results_folder
        except ValueError as e: self.logger.critical(f"Error fatal resolviendo dependencias DI: {e}", exc_info=True); raise
        except Exception as e: self.logger.critical(f"Error inesperado resolviendo dependencias DI: {e}", exc_info=True); raise

    def _initialize_episode(self, episode_id: int, environment: Environment, metrics_collector: MetricsCollector, agent: RLAgent, controller: Controller, config: Dict) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Inicializa el entorno, colector, agente y estado para un nuevo episodio."""
        self.logger.info(f"--- [ Episodio {episode_id}/{config.get('environment', {}).get('max_episodes', 1)-1} ] ---")
        try:
            # Obtener condiciones iniciales
            initial_state_vector = config.get('initial_conditions', {}).get('x0', [0.0, 0.0, 0.0, 0.0])
            if initial_state_vector is None: raise KeyError("'initial_conditions: x0' no encontrado.")

            # Resetear componentes
            current_state_vector = environment.reset(initial_state_vector)
            metrics_collector.reset(episode_id=episode_id) # Agent reset ya ocurre dentro de environment.reset

            # Log estado inicial (t=0)
            self._log_initial_metrics(metrics_collector, current_state_vector, controller, agent, config)
            
            # Seleccionar acción inicial A' para el primer intervalo
            # Verificar el state_config que se va a pasar
            state_config_to_pass = config.get('environment', {}).get('agent', {}).get('params', {}).get('state_config', None) # Usar None como default para detectar si falta
            # Llamada original
            agent_state_dict = agent.build_agent_state(current_state_vector, controller, state_config_to_pass)
            initial_actions_prime = agent.select_action(agent_state_dict)

            # Aplicar acción A' inicial al controlador
            self._apply_actions_to_controller(controller, initial_actions_prime, config)

            # Preparar datos para el primer bloque de aprendizaje
            initial_interval_data = {
                'start_state_dict': agent_state_dict,
                'start_raw_state_vector': np.copy(current_state_vector),
                'actions_dict': initial_actions_prime.copy(),
                'reward_sum': 0.0,
                'w_stab_sum': 0.0,
                'steps': 0,
                'end_state_dict': None, # Se llenará al final del intervalo
                'done': False,
                'reward_dict_echo': None # Se llenará si es Echo
            }
            return current_state_vector, initial_interval_data

        except KeyError as e: self.logger.error(f"Error config inicializando episodio {episode_id}: Falta clave {e}", exc_info=True); raise
        except Exception as e: self.logger.error(f"Error inicializando episodio {episode_id}: {e}", exc_info=True); raise

    def _log_initial_metrics(self, metrics_collector: MetricsCollector, state: np.ndarray, controller: Controller, agent: RLAgent, config: Dict):
        """Registra las métricas iniciales (t=0) del episodio."""
        agent_params_cfg = config.get('environment', {}).get('agent', {}).get('params', {})
        pid_adapt_cfg = config.get('pid_adaptation', {})
        gain_step_config = pid_adapt_cfg.get('gain_step', 5.0)
        variable_step = pid_adapt_cfg.get('variable_step', False)

        metrics_collector.log('time', 0.0)
        metrics_collector.log('cart_position', state[0]); metrics_collector.log('cart_velocity', state[1])
        metrics_collector.log('pendulum_angle', state[2]); metrics_collector.log('pendulum_velocity', state[3])
        try:
            ctrl_setpoint = getattr(controller, 'setpoint', 0.0); metrics_collector.log('error', state[2] - ctrl_setpoint)
            gains = controller.get_params(); metrics_collector.log('kp', gains.get('kp', np.nan)); metrics_collector.log('ki', gains.get('ki', np.nan)); metrics_collector.log('kd', gains.get('kd', np.nan))
            metrics_collector.log('integral_error', getattr(controller, 'integral_error', np.nan)); metrics_collector.log('derivative_error', getattr(controller, 'derivative_error', np.nan))
            metrics_collector.log('epsilon', agent.epsilon); metrics_collector.log('learning_rate', agent.learning_rate)
            if variable_step and isinstance(gain_step_config, dict):
                 metrics_collector.log('gain_step_kp', float(gain_step_config.get('kp', np.nan)))
                 metrics_collector.log('gain_step_ki', float(gain_step_config.get('ki', np.nan)))
                 metrics_collector.log('gain_step_kd', float(gain_step_config.get('kd', np.nan)))
            else: metrics_collector.log('gain_step', float(gain_step_config) if isinstance(gain_step_config, (int,float)) else np.nan)
        except AttributeError as ae: self.logger.warning(f"No se pudo loguear atributo inicial: {ae}")

        metrics_collector.log('reward', 0.0); metrics_collector.log('cumulative_reward', 0.0); metrics_collector.log('force', 0.0); metrics_collector.log('stability_score', 1.0)
        # Log NaNs para valores que aún no existen
        nan_metrics = ['action_kp', 'action_ki', 'action_kd', 'learn_select_duration_ms', 'td_error_kp', 'td_error_ki', 'td_error_kd', 'virtual_reward_kp', 'virtual_reward_ki', 'virtual_reward_kd', 'id_agent_decision']
        for m in nan_metrics: metrics_collector.log(m, np.nan)
        # Pasar diccionario vacío a los loggers de Q/Visit/Baseline iniciales
        empty_state_dict = {}
        if hasattr(metrics_collector, 'log_q_values'): metrics_collector.log_q_values(agent, empty_state_dict) # type: ignore
        if hasattr(metrics_collector, 'log_q_visit_counts'): metrics_collector.log_q_visit_counts(agent, empty_state_dict) # type: ignore
        if hasattr(metrics_collector, 'log_baselines'): metrics_collector.log_baselines(agent, empty_state_dict) # type: ignore
        if hasattr(metrics_collector, 'log_adaptive_stats'): metrics_collector.log_adaptive_stats({}) # type: ignore

    def _apply_actions_to_controller(self, controller: Controller, actions_dict: Dict[str, int], config: Dict):
        """Aplica las acciones seleccionadas (0, 1, 2) para actualizar las ganancias del controlador."""
        pid_adapt_cfg = config.get('pid_adaptation', {})
        gain_step_config = pid_adapt_cfg.get('gain_step', 5.0)
        variable_step = pid_adapt_cfg.get('variable_step', False)
        gain_limits_cfg = config.get('environment', {}).get('agent', {}).get('params', {}).get('state_config', {})

        try:
            kp, ki, kd = controller.kp, controller.ki, controller.kd # type: ignore # Asumiendo PIDController por ahora
            step_kp, step_ki, step_kd = 0.0, 0.0, 0.0

            if variable_step and isinstance(gain_step_config, dict):
                step_kp = float(gain_step_config.get('kp', 0.0))
                step_ki = float(gain_step_config.get('ki', 0.0))
                step_kd = float(gain_step_config.get('kd', 0.0))
            elif isinstance(gain_step_config, (int, float)):
                step_kp = step_ki = step_kd = float(gain_step_config)

            # Aplicar cambios basados en acción
            if actions_dict.get('kp') == 0: kp -= step_kp
            elif actions_dict.get('kp') == 2: kp += step_kp
            if actions_dict.get('ki') == 0: ki -= step_ki
            elif actions_dict.get('ki') == 2: ki += step_ki
            if actions_dict.get('kd') == 0: kd -= step_kd
            elif actions_dict.get('kd') == 2: kd += step_kd

            # Clip gains
            kp_cfg = gain_limits_cfg.get('kp', {}); ki_cfg = gain_limits_cfg.get('ki', {}); kd_cfg = gain_limits_cfg.get('kd', {})
            kp = np.clip(kp, kp_cfg.get('min', -np.inf), kp_cfg.get('max', np.inf))
            ki = np.clip(ki, ki_cfg.get('min', -np.inf), ki_cfg.get('max', np.inf))
            kd = np.clip(kd, kd_cfg.get('min', -np.inf), kd_cfg.get('max', np.inf))

            controller.update_params(kp, ki, kd)

        except KeyError as e: self.logger.error(f"Error aplicando acciones al controlador: Clave {e} faltante en actions_dict {actions_dict}")
        except Exception as e: self.logger.error(f"Error inesperado aplicando acciones al controlador: {e}", exc_info=True)

    def _run_standard_interval_steps(self, start_time: float, duration: float, current_state: np.ndarray, environment: Environment, controller: Controller, agent: RLAgent, metrics_collector: MetricsCollector, config: Dict, actions_to_apply: Dict) -> Tuple[float, float, np.ndarray, bool, str]:
        """ Ejecuta los pasos dt para un intervalo estándar, acumulando recompensas y métricas. """
        interval_reward_sum = 0.0
        interval_stability_scores: List[float] = []
        done = False
        termination_reason = "unknown"
        final_state_in_interval = current_state # Estado al final del intervalo

        num_steps = max(1, int(round(duration / environment.dt))) # type: ignore # Asume dt en env
        dt = environment.dt # type: ignore

        for step in range(num_steps):
            current_step_time = start_time + (step + 1) * dt

            # A. Ejecutar paso del entorno
            try:
                next_state_vector, (reward, stability_score), force = environment.step()
                final_state_in_interval = next_state_vector # Actualizar estado final

            except Exception as e:
                self.logger.error(f"CRITICAL: Error env.step t={current_step_time:.4f}: {e}. Terminando episodio.", exc_info=True)
                done = True; termination_reason = "env_step_error"
                # Loguear estado mínimo antes de romper
                metrics_collector.log('time', current_step_time); metrics_collector.log('reward', np.nan); metrics_collector.log('cumulative_reward', np.nansum(metrics_collector.metrics.get('reward',[]))); metrics_collector.log('stability_score', np.nan)
                break # Salir del bucle de pasos del intervalo

            # B. Acumular métricas del intervalo
            reward_f = float(reward) if np.isfinite(reward) else 0.0
            stability_score_f = float(stability_score) if np.isfinite(stability_score) else 0.0
            force_f = float(force) if np.isfinite(force) else np.nan

            interval_reward_sum += reward_f
            interval_stability_scores.append(stability_score_f)

            # C. Loguear métricas del paso dt
            metrics_collector.log('time', current_step_time)
            metrics_collector.log('cart_position', next_state_vector[0]); metrics_collector.log('cart_velocity', next_state_vector[1])
            metrics_collector.log('pendulum_angle', next_state_vector[2]); metrics_collector.log('pendulum_velocity', next_state_vector[3])
            try:
                ctrl_setpoint = getattr(controller, 'setpoint', 0.0); metrics_collector.log('error', next_state_vector[2] - ctrl_setpoint)
                gains = controller.get_params(); metrics_collector.log('kp', gains.get('kp', np.nan)); metrics_collector.log('ki', gains.get('ki', np.nan)); metrics_collector.log('kd', gains.get('kd', np.nan))
                metrics_collector.log('integral_error', getattr(controller, 'integral_error', np.nan)); metrics_collector.log('derivative_error', getattr(controller, 'derivative_error', np.nan))
                # Log actions_to_apply (A' que se aplicó durante este intervalo)
                metrics_collector.log('action_kp', actions_to_apply.get('kp', np.nan)); metrics_collector.log('action_ki', actions_to_apply.get('ki', np.nan)); metrics_collector.log('action_kd', actions_to_apply.get('kd', np.nan))
                metrics_collector.log('epsilon', agent.epsilon); metrics_collector.log('learning_rate', agent.learning_rate)
                # Log gain step (asumiendo que no cambia intra-intervalo)
                pid_adapt_cfg = config.get('pid_adaptation', {}); gain_step_config = pid_adapt_cfg.get('gain_step', 5.0); variable_step = pid_adapt_cfg.get('variable_step', False)
                if variable_step: metrics_collector.log('gain_step_kp', float(gain_step_config.get('kp', np.nan))); metrics_collector.log('gain_step_ki', float(gain_step_config.get('ki', np.nan))); metrics_collector.log('gain_step_kd', float(gain_step_config.get('kd', np.nan)))
                else: metrics_collector.log('gain_step', float(gain_step_config) if isinstance(gain_step_config, (int,float)) else np.nan)
            except AttributeError as ae: self.logger.warning(f"No se pudo loguear atributo en step: {ae}")
            metrics_collector.log('reward', reward_f); metrics_collector.log('cumulative_reward', np.nansum(metrics_collector.metrics.get('reward',[]))) # Calcular cumulativo on-the-fly
            metrics_collector.log('force', force_f); metrics_collector.log('stability_score', stability_score_f)
            # Log NaNs para métricas de decisión si no estamos en boundary
            metrics_collector.log('learn_select_duration_ms', np.nan); metrics_collector.log('id_agent_decision', np.nan)
            if hasattr(metrics_collector, 'log_q_values'): metrics_collector.log_q_values(agent, {}) # Log NaNs
            if hasattr(metrics_collector, 'log_q_visit_counts'): metrics_collector.log_q_visit_counts(agent, {})
            if hasattr(metrics_collector, 'log_baselines'): metrics_collector.log_baselines(agent, {})
            metrics_collector.log('td_error_kp', np.nan); metrics_collector.log('td_error_ki', np.nan); metrics_collector.log('td_error_kd', np.nan)
            metrics_collector.log('virtual_reward_kp', np.nan); metrics_collector.log('virtual_reward_ki', np.nan); metrics_collector.log('virtual_reward_kd', np.nan)
            if hasattr(metrics_collector, 'log_adaptive_stats'): metrics_collector.log_adaptive_stats(getattr(environment.reward_function.stability_calculator, 'get_current_adaptive_stats', lambda: {})()) # type: ignore

            # D. Verificar Terminación
            if not done: # Solo chequear si no ha terminado ya por error
                try:
                    angle_exc, cart_exc, stab = environment.check_termination(config)
                    # Chequear límite de tiempo total del episodio también aquí
                    total_sim_time_per_episode = config.get('environment', {}).get('total_time', 5.0)
                    time_limit_reached = (current_step_time >= total_sim_time_per_episode - dt / 2)

                    if angle_exc or cart_exc or stab or time_limit_reached:
                        done = True
                        if termination_reason == "unknown": # Asignar solo la primera vez
                            if angle_exc: termination_reason = "angle_limit"
                            elif cart_exc: termination_reason = "cart_limit"
                            elif stab: termination_reason = "stabilized"
                            elif time_limit_reached: termination_reason = "time_limit"
                            else: termination_reason = "unknown_done"
                        self.logger.info(f"Episodio terminando: {termination_reason} at t={current_step_time:.3f}")
                except Exception as e:
                    self.logger.error(f"Error check_termination t={current_step_time:.3f}: {e}", exc_info=True)
                    done = True; termination_reason = "termination_check_error"

            # Salir del bucle de pasos si termina
            if done:
                break

        # Calcular media de estabilidad
        avg_interval_stability = np.mean(interval_stability_scores) if interval_stability_scores else 1.0
        if not np.isfinite(avg_interval_stability): avg_interval_stability = 1.0

        return interval_reward_sum, avg_interval_stability, final_state_in_interval, done, termination_reason

    def _run_echo_baseline_interval_steps(self, start_time: float, duration: float, current_state: np.ndarray, environment: Environment, controller: Controller, agent: RLAgent, metrics_collector: MetricsCollector, virtual_simulator: VirtualSimulator, config: Dict, actions_applied_in_interval: Dict) -> Tuple[float, float, np.ndarray, bool, str, Dict[str, float]]:
        """ Ejecuta intervalo real y 3 simulaciones virtuales para Echo Baseline. """

        # 1. Ejecutar intervalo real
        interval_reward_real, avg_w_stab_real, final_state_real, done_real, termination_reason_real = \
            self._run_standard_interval_steps(start_time, duration, current_state, environment, controller, agent, metrics_collector, config, actions_applied_in_interval)

        if done_real and termination_reason_real == "env_step_error":
             # Si falló el entorno real, no tiene sentido hacer simulaciones virtuales
             self.logger.error("EchoBaseline: Fallo en intervalo real, saltando simulaciones virtuales.")
             return interval_reward_real, avg_w_stab_real, final_state_real, done_real, termination_reason_real, {}

        # 2. Preparar para simulaciones virtuales
        reward_dict_echo: Dict[str, float] = {}
        interval_start_state = current_state # Estado al INICIO del intervalo real
        gains_applied = controller.get_params() # Ganancias que se aplicaron durante el intervalo real

        # 3. Ejecutar simulaciones virtuales (solo si el intervalo real no falló críticamente)
        virtual_sim_start_time = time.time()
        try:
            # Simulación contrafactual para Kp (manteniendo Kp anterior, usando Ki/Kd aplicados)
            gains_p_cf = {'kp': getattr(controller,'prev_kp',gains_applied['kp']), 'ki': gains_applied['ki'], 'kd': gains_applied['kd']}
            R_p_cf = virtual_simulator.run_interval_simulation(interval_start_state, start_time, duration, gains_p_cf)

            # Simulación contrafactual para Ki
            gains_i_cf = {'kp': gains_applied['kp'], 'ki': getattr(controller,'prev_ki',gains_applied['ki']), 'kd': gains_applied['kd']}
            R_i_cf = virtual_simulator.run_interval_simulation(interval_start_state, start_time, duration, gains_i_cf)

            # Simulación contrafactual para Ki
            gains_d_cf = {'kp': gains_applied['kp'], 'ki': gains_applied['ki'], 'kd': getattr(controller,'prev_kd',gains_applied['kd'])}
            R_d_cf = virtual_simulator.run_interval_simulation(interval_start_state, start_time, duration, gains_d_cf)

            # Calcular recompensas diferenciales
            reward_dict_echo = {
                'kp': interval_reward_real - R_p_cf,
                'ki': interval_reward_real - R_i_cf,
                'kd': interval_reward_real - R_d_cf
            }
            virtual_sim_duration = (time.time() - virtual_sim_start_time) * 1000
            self.logger.debug(f"EchoBaseline: Duración de simulaciones virtuales completadas ({virtual_sim_duration:.1f} ms)")
            self.logger.debug(f"SimulationManager -> _run_echo_baseline_interval_steps -> Recompensas contrafactuales -> R_x_cf: Kp={R_p_cf:.3f}, Ki={R_i_cf:.3f}, Kd={R_d_cf:.5f}")
            self.logger.debug(f"SimulationManager -> _run_echo_baseline_interval_steps -> Recompensas finales -> R_diff: Kp={reward_dict_echo['kp']:.3f}, Ki={reward_dict_echo['ki']:.3f}, Kd={reward_dict_echo['kd']:.3f}")
            # Log R_diffs virtuales
            if hasattr(metrics_collector, 'log_virtual_rewards'): 
                metrics_collector.log_virtual_rewards(reward_dict_echo) # type: ignore

        except Exception as e:
            self.logger.error(f"EchoBaseline: Error durante simulaciones virtuales: {e}", exc_info=True)
            # Devolver dict vacío si fallan las simulaciones
            reward_dict_echo = {}
            if hasattr(metrics_collector, 'log_virtual_rewards'): metrics_collector.log_virtual_rewards({}) # Log NaNs

        return interval_reward_real, avg_w_stab_real, final_state_real, done_real, termination_reason_real, reward_dict_echo


    def _handle_decision_boundary(self, current_time: float, current_state: np.ndarray, last_interval_data: Dict, interval_run_results: Dict, agent: RLAgent, controller: Controller, metrics_collector: MetricsCollector, config: Dict) -> Tuple[Dict[str, int], Optional[Dict]]:
        """ Maneja el aprendizaje, selección y aplicación de la nueva acción en el límite de decisión. """
        decision_start_time = time.time()
        agent_decision_count = last_interval_data.get('decision_count', 0) + 1
        metrics_collector.log('id_agent_decision', agent_decision_count)

        self.logger.debug(f"SimulationManager -> _handle_decision_boundary -> agent_decision_count={agent_decision_count}")

        # 1. Estado S' al final del intervalo
        next_raw_state_vector = interval_run_results['final_state']
        
        next_agent_state_dict = agent.build_agent_state(next_raw_state_vector, controller, config.get('environment', {}).get('agent', {}).get('params', {}).get('state_config', {}))
        episode_done = interval_run_results['done']

        # 2. Preparar datos para Agent Learn (intervalo ANTERIOR)
        learn_data = {
            'current_agent_state_dict': last_interval_data['start_state_dict'], # S
            'actions_dict': last_interval_data['actions_dict'],             # A' (aplicado en intervalo anterior)
            'next_agent_state_dict': next_agent_state_dict,                 # S'
            'done': episode_done,
            'controller': controller
        }

        self.logger.debug(f"SimulationManager -> _handle_decision_boundary -> last_interval_data={last_interval_data}")
        self.logger.debug(f"SimulationManager -> _handle_decision_boundary -> learn_data={learn_data}")

        # Determinar reward_info basado en estrategia (R_real, (R_real, w_stab), R_diff_dict)
        reward_info: Union[float, Tuple[float, float], Dict[str, float]]
        if 'reward_dict_echo' in interval_run_results: # Modo Echo
            reward_info = interval_run_results['reward_dict_echo']
        else: # Modo Global o Shadow
            reward_info = (interval_run_results['interval_reward'], interval_run_results['avg_w_stab'])
            # La estrategia Global/Shadow extraerá lo que necesite de la tupla

        learn_data['reward_info'] = reward_info

        # 3. Llamar a Agent Learn
        last_td_errors_logged = False
        try:
            agent.learn(**learn_data)
            if hasattr(metrics_collector, 'log_td_errors'):
                 metrics_collector.log_td_errors(agent.get_last_td_errors()) # type: ignore
                 last_td_errors_logged = True
        except Exception as learn_e:
             self.logger.error(f"Error en agent.learn(): {learn_e}", exc_info=True)
             if hasattr(metrics_collector, 'log_td_errors'): metrics_collector.log_td_errors({}) # Log NaN
             last_td_errors_logged = True

        # 4. Seleccionar Siguiente Acción (A'')
        if not episode_done:
            next_actions_prime = agent.select_action(next_agent_state_dict) # Nueva A''
        else:
            next_actions_prime = {'kp': 1, 'ki': 1, 'kd': 1} # Acción neutral si termina

        # 5. Aplicar Nueva Acción A'' al controlador (para el *próximo* intervalo)
        if not episode_done:
            self._apply_actions_to_controller(controller, next_actions_prime, config)

        # 6. Loguear Métricas del Bloque de Decisión (para estado S')
        if hasattr(metrics_collector, 'log_q_values'): metrics_collector.log_q_values(agent, next_agent_state_dict) # type: ignore
        if hasattr(metrics_collector, 'log_q_visit_counts'): metrics_collector.log_q_visit_counts(agent, next_agent_state_dict) # type: ignore
        if hasattr(metrics_collector, 'log_baselines'): metrics_collector.log_baselines(agent, next_agent_state_dict) # type: ignore
        if not last_td_errors_logged and hasattr(metrics_collector, 'log_td_errors'): metrics_collector.log_td_errors(agent.get_last_td_errors()) # type: ignore
        decision_duration_ms = (time.time() - decision_start_time) * 1000
        metrics_collector.log('learn_select_duration_ms', decision_duration_ms)

        # 7. Preparar datos para el *siguiente* intervalo (si no 'done')
        next_interval_data = None
        if not episode_done:
            next_interval_data = {
                'start_state_dict': next_agent_state_dict, # S será S'
                'start_raw_state_vector': np.copy(next_raw_state_vector),
                'actions_dict': next_actions_prime.copy(),      # A serán A''
                'reward_sum': 0.0, 'w_stab_sum': 0.0, 'steps': 0, # Resetear contadores
                'end_state_dict': None, 'done': False, 'reward_dict_echo': None,
                'decision_count': agent_decision_count # Pasar contador
             }

        return next_actions_prime, next_interval_data # Devolver acciones aplicadas y datos para próximo intervalo

    def _finalize_episode(self, episode: int, episode_metrics_dict: Dict, termination_reason: str, episode_start_time: float, controller: Controller, environment: Environment, agent: RLAgent, results_folder: str, config: Dict, summary_data_list: List, current_episode_batch: List, agent_state_save_freq: int):
        """ Realiza el resumen, loggeo y guardado parcial al final de un episodio. """
        episode_duration_s = time.time() - episode_start_time
        final_decision_count = np.nanmax(episode_metrics_dict.get('id_agent_decision', [0])) # Último ID de decisión

        # Añadir métricas finales calculadas al dict detallado
        len_first_metric = len(next(iter(episode_metrics_dict.values()), []))
        episode_metrics_dict['termination_reason'] = [termination_reason] * len_first_metric
        episode_metrics_dict['episode_duration_s'] = [episode_duration_s] * len_first_metric
        final_gains = controller.get_params()
        episode_metrics_dict['final_kp'] = [final_gains.get('kp', np.nan)] * len_first_metric
        episode_metrics_dict['final_ki'] = [final_gains.get('ki', np.nan)] * len_first_metric
        episode_metrics_dict['final_kd'] = [final_gains.get('kd', np.nan)] * len_first_metric
        episode_metrics_dict['total_agent_decisions'] = [final_decision_count] * len_first_metric
        # SimManager calcula y añade avg_stability_score y total_reward antes de llamar a summarize
        episode_metrics_dict['avg_stability_score'] = [np.nanmean(episode_metrics_dict.get('stability_score', [np.nan]))] * len_first_metric
        episode_metrics_dict['total_reward'] = [np.nansum(episode_metrics_dict.get('reward', [np.nan]))] * len_first_metric

        # Llamar a summarize_episode
        summary = summarize_episode(episode_metrics_dict)
        summary['episode'] = episode # Añadir ID episodio
        summary_data_list.append(summary)

        # Añadir datos detallados al batch actual
        current_episode_batch.append(episode_metrics_dict)

        # Log Resumen
        self.logger.info(f"Ep {episode} Resumen: Kp={summary.get('final_kp', np.nan):.2f}, Ki={summary.get('final_ki', np.nan):.2f}, Kd={summary.get('final_kd', np.nan):.3f}, n_decisions={summary.get('total_agent_decisions', 0)}")
        self.logger.info(f"Ep {episode} Resumen: total_reward={summary.get('total_reward', np.nan):.2f}, performance={summary.get('performance', np.nan):.2f}, stab={summary.get('avg_stability_score', np.nan):.3f}, epsilon={summary.get('final_epsilon', np.nan):.3f}, LR={summary.get('final_learning_rate', np.nan):.4f}, Dur={summary.get('episode_duration_s', np.nan):.2f}s")

        # Actualizar stats adaptativas (si aplica)
        try: environment.update_reward_calculator_stats(episode_metrics_dict, episode)
        except Exception as e: self.logger.error(f"Error update reward stats ep {episode}: {e}", exc_info=True)

        # Guardado Periódico de Estado del Agente
        if agent_state_save_freq > 0 and (episode + 1) % agent_state_save_freq == 0:
            self.result_handler.save_agent_state(agent, episode, results_folder)

    # --- Método Público Principal ---

    def run(self) -> Tuple[List[Dict], List[Dict]]:
        """ Ejecuta el bucle principal de simulación, orquestando episodios e intervalos. """
        self.logger.info("--- Iniciando Bucle de Simulación Principal (Refactorizado) ---")
        all_episodes_detailed_data: List[Dict] = [] # Rara vez usado si guardamos por batch
        summary_data: List[Dict] = []
        results_folder: Optional[str] = None; episode: int = -1
        current_episode_batch: List[Dict] = []
        file_handlers: List[logging.Handler] = []

        try:
            # --- 1. Resolver Dependencias y Extraer Config ---
            environment, agent, controller, metrics_collector, reward_strategy, \
                virtual_simulator, config, results_folder = self._resolve_dependencies()

            # --- Extraer Parámetros Clave ---
            sim_cfg = config.get('simulation', {}); env_cfg = config.get('environment', {})
            logging_cfg = config.get('logging', {})

            max_episodes = env_cfg.get('max_episodes', 1)
            decision_interval = env_cfg.get('decision_interval', 0.01)
            dt = env_cfg.get('dt', 0.001)
            if not isinstance(decision_interval, (float, int)) or decision_interval < dt: decision_interval = dt
            episodes_per_file = sim_cfg.get('episodes_per_file', 100)
            agent_state_save_freq = sim_cfg.get('agent_state_save_frequency', 1000) if sim_cfg.get('save_agent_state', False) else 0
            log_flush_frequency = logging_cfg.get('log_save_frequency', 0)

            # Determinar si estamos en modo Echo Baseline
            is_echo_baseline = isinstance(reward_strategy, EchoBaselineRewardStrategy)
            if is_echo_baseline and virtual_simulator is None:
                 self.logger.error("Echo Baseline activado pero VirtualSimulator no resuelto. Abortando."); return [], []

            # Preparar handlers para flush
            if log_flush_frequency > 0:
                 file_handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler) and hasattr(h, 'flush')]
                 if not file_handlers: self.logger.warning("log_flush_frequency > 0 pero no se encontraron FileHandlers."); log_flush_frequency = 0

            # --- 2. Bucle Principal de Episodios ---
            self.logger.info(f"Iniciando simulación para {max_episodes} episodios...")
            for episode in range(max_episodes):
                episode_start_time = time.time()
                # --- 2.a Inicializar Episodio ---
                try:
                    current_state, last_interval_data = self._initialize_episode(
                        episode, environment, metrics_collector, agent, controller, config
                    )
                    # Asegurar que last_interval_data no es None (manejar error en _initialize_episode)
                    if last_interval_data is None: raise RuntimeError("Fallo al inicializar datos del intervalo.")
                    last_interval_data['decision_count'] = 0 # Inicializar contador
                except Exception as init_e:
                    self.logger.error(f"Fallo crítico inicializando episodio {episode}: {init_e}. Saltando episodio.")
                    continue # Saltar al siguiente episodio

                # --- 2.b Bucle de Intervalos de Decisión ---
                current_time = 0.0
                next_decision_time = decision_interval
                episode_done = False
                termination_reason = "unknown"
                total_sim_time_per_episode = env_cfg.get('total_time', 5.0)

                while not episode_done:
                    # Calcular duración del intervalo actual
                    interval_duration = min(decision_interval, total_sim_time_per_episode - current_time)
                    if interval_duration <= 0: break # Evitar intervalos de duración cero o negativa

                    actions_applied = last_interval_data['actions_dict'].copy() # Acciones A' aplicadas

                    # --- Ejecutar Pasos del Intervalo (Real o con Echo) ---
                    interval_run_results: Dict[str, Any] = {} # Para almacenar resultados del run
                    if is_echo_baseline:
                        interval_reward, avg_w_stab, final_state, interval_done, term_reason, reward_dict_echo = \
                            self._run_echo_baseline_interval_steps(
                                current_time, interval_duration, current_state, environment, controller, agent,
                                metrics_collector, virtual_simulator, config, actions_applied # type: ignore
                            )
                        interval_run_results = {
                            'interval_reward': interval_reward, 'avg_w_stab': avg_w_stab,
                            'final_state': final_state, 'done': interval_done,
                            'termination_reason': term_reason, 'reward_dict_echo': reward_dict_echo
                        }
                    else: # Modo Global o Shadow
                        interval_reward, avg_w_stab, final_state, interval_done, term_reason = \
                            self._run_standard_interval_steps(
                                current_time, interval_duration, current_state, environment, controller, agent,
                                metrics_collector, config, actions_applied
                            )
                        interval_run_results = {
                            'interval_reward': interval_reward, 'avg_w_stab': avg_w_stab,
                            'final_state': final_state, 'done': interval_done,
                            'termination_reason': term_reason
                        }

                    # Actualizar estado y tiempo
                    current_state = interval_run_results['final_state']
                    current_time = round(current_time + interval_duration, 6) # Actualizar tiempo basado en duración real
                    episode_done = interval_run_results['done']
                    if episode_done and termination_reason == "unknown":
                         termination_reason = interval_run_results['termination_reason']

                    # --- Manejar Límite de Decisión ---
                    # Siempre se llama al final del intervalo o si 'done'
                    _, next_interval_data = self._handle_decision_boundary(
                         current_time, current_state, last_interval_data, interval_run_results,
                         agent, controller, metrics_collector, config
                    )
                    # Preparar para el siguiente intervalo
                    if episode_done:
                         last_interval_data = None # No más intervalos
                         if termination_reason == "unknown": termination_reason = "time_limit" # Si termina por tiempo exacto
                    elif next_interval_data is not None:
                         last_interval_data = next_interval_data
                    else: # Caso raro: no 'done' pero no hay datos para el siguiente
                         self.logger.error("Error lógico: Episodio no terminado pero no hay datos para el siguiente intervalo. Terminando.")
                         episode_done = True
                         termination_reason = "interval_logic_error"

                    # Actualizar próximo tiempo de decisión (innecesario con cálculo de duration)
                    # next_decision_time += decision_interval

                # --- 2.c Finalizar Episodio ---
                # Recopilar métricas finales
                episode_metrics = metrics_collector.get_metrics()
                self._finalize_episode(
                     episode, episode_metrics, termination_reason, episode_start_time, controller,
                     environment, agent, results_folder, config, summary_data, current_episode_batch,
                     agent_state_save_freq
                )

                # --- Guardado de Batch de Episodios ---
                if episodes_per_file > 0 and ((episode + 1) % episodes_per_file == 0 or episode == max_episodes - 1):
                    if current_episode_batch:
                        self.result_handler.save_episode_batch(current_episode_batch, results_folder, episode)
                        current_episode_batch = [] # Limpiar batch

                # --- Flush Periódico de Logs ---
                if log_flush_frequency > 0 and (episode + 1) % log_flush_frequency == 0:
                    self.logger.debug(f"Flushing logs file after ep {episode}...")
                    for h in file_handlers:
                        try: h.flush() # type: ignore
                        except Exception as e_flush: self.logger.warning(f"Error flush {h}: {e_flush}")

            # --- Fin Bucle de Episodios ---

        except Exception as e:
            self.logger.error(f"Error INESPERADO en bucle simulación ep {episode}: {e}", exc_info=True)
            if current_episode_batch and results_folder and os.path.isdir(results_folder):
                 self.logger.warning("Intentando guardar batch parcial de episodios tras error...")
                 self.result_handler.save_episode_batch(current_episode_batch, results_folder, episode if episode >= 0 else -1)

        finally:
            if file_handlers:
                 self.logger.info("Realizando flush final de logs...")
                 for h in file_handlers:
                      try: h.flush() # type: ignore
                      except Exception as e_flush_final: self.logger.warning(f"Error flush final {h}: {e_flush_final}")
            self.logger.info("--- Simulación Principal Finalizada ---")

        return all_episodes_detailed_data, summary_data