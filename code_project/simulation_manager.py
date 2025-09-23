# simulation_manager.py
import logging
import time
import numpy as np
import pandas as pd
import os
import gc
from typing import Dict, Any, List, Tuple, Optional, Union, TYPE_CHECKING

from interfaces.environment import Environment
from interfaces.rl_agent import RLAgent
from interfaces.controller import Controller
from interfaces.virtual_simulator import VirtualSimulator
from interfaces.metrics_collector import MetricsCollector
from interfaces.reward_strategy import RewardStrategy

# Importar estrategias concretas solo para atributos/type hints si es estrictamente necesario
# (Idealmente, esto se evitaría si la interfaz RewardStrategy define las propiedades necesarias)
from components.reward_strategies.echo_baseline_reward_strategy import EchoBaselineRewardStrategy

from utils.data.result_handler import ResultHandler
from utils.data.data_processing import summarize_episode

if TYPE_CHECKING:
    from di_container import Container

class SimulationManager:
    def __init__(self,
                 logger: logging.Logger,
                 result_handler: ResultHandler,
                 container: 'Container'
                 ):
        self.logger = logger
        self.result_handler = result_handler
        self.container = container
        self.logger.info("[SimMan] SimulationManager instance created.")
        if container is None:
             msg = "[SimMan] CRITICAL: SimulationManager requires a valid Container instance."
             self.logger.critical(msg); raise ValueError(msg)
        if logger is None or result_handler is None:
             msg = "[SimMan] CRITICAL: SimulationManager requires valid Logger and ResultHandler instances."
             self.logger.critical(msg); raise ValueError(msg)

    def _resolve_dependencies(self) -> Tuple[Environment, RLAgent, Controller, MetricsCollector, RewardStrategy, Optional[VirtualSimulator], Dict[str, Any], str]:
        self.logger.debug("[SimMan:_resolve_dependencies] Resolving dependencies for simulation...")
        try:
            environment = self.container.resolve(Environment)
            agent = self.container.resolve(RLAgent)
            controller = self.container.resolve(Controller)
            # MetricsCollector es transient, se resolverá por episodio dentro del bucle run()
            # No obstante, podemos resolver uno aquí para validar que el proveedor existe
            _ = self.container.resolve(MetricsCollector) # Valida que se puede resolver
            reward_strategy = self.container.resolve(RewardStrategy)
            virtual_simulator = self.container.resolve(Optional[VirtualSimulator]) # Resuelve Optional
            config = self.container.resolve(dict)
            results_folder = self.container.resolve(str)

            required_components = {
                "Environment": environment, "RLAgent": agent, "Controller": controller,
                "RewardStrategy": reward_strategy, "dict (config)": config,
                "str (results_folder)": results_folder
            } # MetricsCollector y VirtualSimulator son opcionales aquí
            missing = [name for name, var in required_components.items() if var is None]
            if missing:
                raise ValueError(f"Failed to resolve key DI dependencies: {missing}")

            self.logger.info(f"[SimMan:_resolve_dependencies] Core dependencies resolved. RewardStrategy: {type(reward_strategy).__name__}")
            # MetricsCollector se resuelve en el bucle de episodios.
            return environment, agent, controller, None, reward_strategy, virtual_simulator, config, results_folder # type: ignore

        except (ValueError, RecursionError) as e:
            self.logger.critical(f"[SimMan:_resolve_dependencies] CRITICAL DI Error: {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.critical(f"[SimMan:_resolve_dependencies] UNEXPECTED DI Error: {e}", exc_info=True)
            raise

    def _initialize_episode(self,
                            episode_id: int,
                            environment: Environment,
                            agent: RLAgent,
                            controller: Controller,
                            metrics_collector: MetricsCollector, # Instancia Transient
                            config: Dict
                           ) -> Tuple[np.ndarray, Dict[str, Any]]:
        max_episodes_cfg = config.get('environment', {}).get('simulation', {}).get('max_episodes', 1)
        self.logger.info(f"--- [ Episode {episode_id}/{max_episodes_cfg-1} Initializing ] ---")
        try:
            # Leer ruta de initial_conditions desde la nueva estructura
            initial_state_vector_cfg = config.get('environment', {}).get('initial_conditions', {}).get('x0')
            if initial_state_vector_cfg is None:
                raise KeyError("'environment.initial_conditions.x0' not found in config.")

            current_state_vector = environment.reset(initial_state_vector_cfg)
            metrics_collector.reset(episode_id=episode_id) # Resetear collector transient
            self._log_initial_metrics(metrics_collector, current_state_vector, controller, agent, config)

            # Leer state_config para el agente desde la nueva estructura
            state_config_for_agent = config.get('environment', {}).get('agent', {}).get('params', {}).get('state_config', {})
            agent_state_dict = agent.build_agent_state(current_state_vector, controller, state_config_for_agent)
            initial_actions_prime = agent.select_action(agent_state_dict)
            self._apply_actions_to_controller(controller, initial_actions_prime, config)

            initial_interval_data = {
                'start_state_dict': agent_state_dict,
                'start_raw_state_vector': np.copy(current_state_vector),
                'actions_dict': initial_actions_prime.copy(),
                'reward_sum': 0.0, 'w_stab_sum': 0.0, 'steps_in_interval': 0,
                'end_state_dict': None, 'done': False, 'reward_dict_echo': None,
                'decision_count': 0
            }
            self.logger.debug(f"[SimMan:_initialize_episode] Ep {episode_id} initialized. Initial state: {np.round(current_state_vector[:4],3)}, Action A': {initial_actions_prime}")
            return current_state_vector, initial_interval_data
        except (KeyError, ValueError, AttributeError, TypeError, RuntimeError) as e:
            self.logger.error(f"[SimMan:_initialize_episode] Error initializing episode {episode_id}: {e}", exc_info=True)
            raise

    def _log_initial_metrics(self, metrics_collector: MetricsCollector, state: np.ndarray, controller: Controller, agent: RLAgent, config: Dict):
        self.logger.debug(f"[SimMan:_log_initial_metrics] Logging initial metrics for state: {np.round(state[:4],3)}")
        metrics_collector.log('time', 0.0)
        try:
            metrics_collector.log('cart_position', state[0]); metrics_collector.log('cart_velocity', state[1])
            metrics_collector.log('pendulum_angle', state[2]); metrics_collector.log('pendulum_velocity', state[3])

            ctrl_params = controller.get_params()
            metrics_collector.log('kp', ctrl_params.get('kp', np.nan)); metrics_collector.log('ki', ctrl_params.get('ki', np.nan)); metrics_collector.log('kd', ctrl_params.get('kd', np.nan))
            # Setpoint ahora está en environment.controller.params.setpoint
            setpoint = config.get('environment',{}).get('controller',{}).get('params',{}).get('setpoint', 0.0)
            metrics_collector.log('error', state[2] - setpoint)
            metrics_collector.log('integral_error', getattr(controller, 'integral_error', np.nan))
            metrics_collector.log('derivative_error', getattr(controller, 'derivative_error', np.nan))

            metrics_collector.log('epsilon', agent.epsilon); metrics_collector.log('learning_rate', agent.learning_rate)

            # Leer pid_adaptation desde la nueva estructura environment.controller.pid_adaptation
            pid_adapt_cfg = config.get('environment', {}).get('controller', {}).get('pid_adaptation', {})
            gain_step_config = pid_adapt_cfg.get('gain_step', 5.0) # Default si falta
            variable_step = pid_adapt_cfg.get('variable_step', False)
            if variable_step and isinstance(gain_step_config, dict):
                 metrics_collector.log('gain_step_kp', float(gain_step_config.get('kp', np.nan)))
                 metrics_collector.log('gain_step_ki', float(gain_step_config.get('ki', np.nan)))
                 metrics_collector.log('gain_step_kd', float(gain_step_config.get('kd', np.nan)))
            elif isinstance(gain_step_config, (int,float)):
                 metrics_collector.log('gain_step', float(gain_step_config))
            else: metrics_collector.log('gain_step', np.nan)

            metrics_collector.log('reward', 0.0); metrics_collector.log('cumulative_reward', 0.0)
            metrics_collector.log('force', 0.0); metrics_collector.log('stability_score', 1.0)
            nan_metrics = ['action_kp', 'action_ki', 'action_kd', 'learn_select_duration_ms',
                           'td_error_kp', 'td_error_ki', 'td_error_kd',
                           'virtual_reward_kp', 'virtual_reward_ki', 'virtual_reward_kd',
                           'id_agent_decision', 'q_value_max_kp', 'q_value_max_ki', 'q_value_max_kd',
                           'q_visit_count_state_kp', 'q_visit_count_state_ki', 'q_visit_count_state_kd',
                           'baseline_value_kp', 'baseline_value_ki', 'baseline_value_kd',
                           'virtual_w_stab_kp_cf', 'virtual_w_stab_ki_cf', 'virtual_w_stab_kd_cf']
            adaptive_vars = ['angle', 'angular_velocity', 'cart_position', 'cart_velocity']
            for var in adaptive_vars: nan_metrics.extend([f'adaptive_mu_{var}', f'adaptive_sigma_{var}'])
            for m in nan_metrics: metrics_collector.log(m, np.nan)

            # Construir el estado discreto del agente para inicialización
            state_config_for_agent = config.get('environment', {}) .get('agent', {}).get('params', {}).get('state_config', {})
            agent_state_dict = agent.build_agent_state(state, controller, state_config_for_agent)

            if hasattr(metrics_collector, 'log_early_termination_metrics'): metrics_collector.log_early_termination_metrics(agent) # type: ignore
            if hasattr(metrics_collector, 'log_q_values'): metrics_collector.log_q_values(agent, agent_state_dict) # type: ignore[operator]
            if hasattr(metrics_collector, 'log_q_visit_counts'): metrics_collector.log_q_visit_counts(agent, agent_state_dict) # type: ignore[operator]
            if hasattr(metrics_collector, 'log_baselines'): metrics_collector.log_baselines(agent, agent_state_dict) # type: ignore[operator]
            if hasattr(metrics_collector, 'log_adaptive_stats'): metrics_collector.log_adaptive_stats({}) # type: ignore[operator]
        except IndexError: self.logger.warning("[SimMan:_log_initial_metrics] State vector unexpected length.")
        except Exception as e: self.logger.error(f"[SimMan:_log_initial_metrics] Error logging: {e}", exc_info=True)


    def _apply_actions_to_controller(self, controller: Controller, actions_dict: Dict[str, int], config: Dict):
        try:
            current_gains = controller.get_params()
            kp, ki, kd = current_gains.get('kp', 0.0), current_gains.get('ki', 0.0), current_gains.get('kd', 0.0)

            # Leer pid_adaptation desde environment.controller.pid_adaptation
            pid_adapt_cfg = config.get('environment', {}).get('controller', {}).get('pid_adaptation', {})
            gain_step_config = pid_adapt_cfg.get('gain_step', 1.0)
            variable_step = pid_adapt_cfg.get('variable_step', False)
            step_kp, step_ki, step_kd = 0.0, 0.0, 0.0

            if variable_step and isinstance(gain_step_config, dict):
                step_kp = float(gain_step_config.get('kp', 0.0))
                step_ki = float(gain_step_config.get('ki', 0.0))
                step_kd = float(gain_step_config.get('kd', 0.0))
            elif isinstance(gain_step_config, (int, float)):
                step_kp = step_ki = step_kd = float(gain_step_config)
            else:
                 self.logger.warning(f"[SimMan:_apply_actions] Invalid 'gain_step' config ({gain_step_config}). Using steps 0.0.")

            new_kp = kp + (actions_dict.get('kp', 1) - 1) * step_kp
            new_ki = ki + (actions_dict.get('ki', 1) - 1) * step_ki
            new_kd = kd + (actions_dict.get('kd', 1) - 1) * step_kd

            # Leer límites de ganancia desde environment.agent.params.state_config
            gain_limits_cfg = config.get('environment', {}).get('agent', {}).get('params', {}).get('state_config', {})
            kp_cfg = gain_limits_cfg.get('kp', {}); ki_cfg = gain_limits_cfg.get('ki', {}); kd_cfg = gain_limits_cfg.get('kd', {})
            kp_min, kp_max = kp_cfg.get('min', -np.inf), kp_cfg.get('max', np.inf)
            ki_min, ki_max = ki_cfg.get('min', -np.inf), ki_cfg.get('max', np.inf)
            kd_min, kd_max = kd_cfg.get('min', -np.inf), kd_cfg.get('max', np.inf)

            new_kp = np.clip(new_kp, kp_min, kp_max)
            new_ki = np.clip(new_ki, ki_min, ki_max)
            new_kd = np.clip(new_kd, kd_min, kd_max)
            controller.update_params(new_kp, new_ki, new_kd)
            
            self.logger.debug(f"[SimMan:_apply_actions] Controller gains updated to: Kp={new_kp:.2f}, Ki={new_ki:.2f}, Kd={new_kd:.2f}")
        except KeyError as e:
            self.logger.error(f"[SimMan:_apply_actions] Error: Missing key {e} in config or actions_dict {actions_dict}")
        except Exception as e:
            self.logger.error(f"[SimMan:_apply_actions] Unexpected error: {e}", exc_info=True)


    def _run_standard_interval_steps(self, start_time: float, duration: float, current_state: np.ndarray,
                                      environment: Environment, controller: Controller, agent: RLAgent,
                                      metrics_collector: MetricsCollector, config: Dict,
                                      actions_applied_in_interval: Dict
                                     ) -> Tuple[float, float, np.ndarray, bool, str]:
        interval_reward_sum = 0.0
        interval_stability_scores: List[float] = []
        done = False
        termination_reason = "unknown"
        final_state_in_interval = current_state
        try:
            dt = environment.dt # type: ignore[attr-defined]
            if not isinstance(dt, (float, int)) or dt <= 0: raise ValueError("Invalid dt from environment")
        except (AttributeError, ValueError) as e:
             self.logger.error(f"[SimMan:_run_std_interval] Invalid dt from env: {e}. Using 0.001 (fallback).")
             dt = 0.001
        num_steps = max(1, int(round(duration / dt)))

        for step in range(num_steps):
            current_step_time = round(start_time + (step + 1) * dt, 6)
            try:
                next_state_vector, (reward_step, stability_score_step), force = environment.step()
                final_state_in_interval = next_state_vector
            except RuntimeError as e:
                self.logger.error(f"[SimMan:_run_std_interval] CRITICAL env.step() error at t={current_step_time:.4f}: {e}. Terminating episode.", exc_info=True)
                done = True; termination_reason = "env_step_error"
                metrics_collector.log('time', current_step_time)
                nan_metrics = ['reward', 'stability_score', 'force', 'cart_position', 'cart_velocity', 'pendulum_angle', 'pendulum_velocity', 'error', 'kp', 'ki', 'kd', 'integral_error', 'derivative_error', 'action_kp', 'action_ki', 'action_kd', 'epsilon', 'learning_rate', 'gain_step', 'gain_step_kp', 'gain_step_ki', 'gain_step_kd', 'cumulative_reward']
                for m in nan_metrics: metrics_collector.log(m, np.nan)
                break

            reward_f = float(reward_step) if np.isfinite(reward_step) else 0.0
            stability_score_f = float(stability_score_step) if np.isfinite(stability_score_step) else 0.0
            force_f = float(force) if np.isfinite(force) else np.nan
            interval_reward_sum += reward_f
            interval_stability_scores.append(stability_score_f)

            metrics_collector.log('time', current_step_time)
            metrics_collector.log('cart_position', next_state_vector[0]); metrics_collector.log('cart_velocity', next_state_vector[1])
            metrics_collector.log('pendulum_angle', next_state_vector[2]); metrics_collector.log('pendulum_velocity', next_state_vector[3])
            try:
                # Setpoint desde config
                ctrl_setpoint = config.get('environment',{}).get('controller',{}).get('params',{}).get('setpoint', 0.0)
                metrics_collector.log('error', next_state_vector[2] - ctrl_setpoint)
                gains = controller.get_params(); metrics_collector.log('kp', gains.get('kp', np.nan)); metrics_collector.log('ki', gains.get('ki', np.nan)); metrics_collector.log('kd', gains.get('kd', np.nan))
                metrics_collector.log('integral_error', getattr(controller, 'integral_error', np.nan)); metrics_collector.log('derivative_error', getattr(controller, 'derivative_error', np.nan))
                metrics_collector.log('action_kp', actions_applied_in_interval.get('kp', np.nan)); metrics_collector.log('action_ki', actions_applied_in_interval.get('ki', np.nan)); metrics_collector.log('action_kd', actions_applied_in_interval.get('kd', np.nan))
                metrics_collector.log('epsilon', agent.epsilon); metrics_collector.log('learning_rate', agent.learning_rate)
                # Gain step desde config environment.controller.pid_adaptation
                pid_adapt_cfg = config.get('environment', {}).get('controller', {}).get('pid_adaptation', {})
                gain_step_config = pid_adapt_cfg.get('gain_step', 5.0); variable_step = pid_adapt_cfg.get('variable_step', False)
                if variable_step and isinstance(gain_step_config, dict):
                     metrics_collector.log('gain_step_kp', float(gain_step_config.get('kp', np.nan)))
                     metrics_collector.log('gain_step_ki', float(gain_step_config.get('ki', np.nan)))
                     metrics_collector.log('gain_step_kd', float(gain_step_config.get('kd', np.nan)))
                elif isinstance(gain_step_config, (int,float)): metrics_collector.log('gain_step', float(gain_step_config))
                else: metrics_collector.log('gain_step', np.nan)
            except AttributeError as ae: self.logger.warning(f"[SimMan:_run_std_interval] Attr error logging step metrics: {ae}")
            except IndexError: self.logger.warning("[SimMan:_run_std_interval] State vector unexpected length in step log.")
            metrics_collector.log('reward', reward_f)
            cumulative_rewards_list = metrics_collector.get_metrics().get('reward', [])
            metrics_collector.log('cumulative_reward', np.nansum(np.array(cumulative_rewards_list, dtype=float)))
            metrics_collector.log('force', force_f); metrics_collector.log('stability_score', stability_score_f)
            nan_boundary_metrics = ['learn_select_duration_ms', 'id_agent_decision', 'td_error_kp', 'td_error_ki', 'td_error_kd', 'virtual_reward_kp', 'virtual_reward_ki', 'virtual_reward_kd', 'q_value_max_kp', 'q_value_max_ki', 'q_value_max_kd', 'q_visit_count_state_kp', 'q_visit_count_state_ki', 'q_visit_count_state_kd', 'baseline_value_kp', 'baseline_value_ki', 'baseline_value_kd', 'virtual_w_stab_kp_cf', 'virtual_w_stab_ki_cf', 'virtual_w_stab_kd_cf']
            for m in nan_boundary_metrics: metrics_collector.log(m, np.nan)
            if hasattr(metrics_collector, 'log_adaptive_stats'):
                adaptive_stats = {}
                try:
                    calculator = getattr(environment.reward_function, 'stability_calculator', None) # type: ignore[attr-defined]
                    if calculator and hasattr(calculator, 'get_current_adaptive_stats'):
                         adaptive_stats = calculator.get_current_adaptive_stats()
                except Exception as e_stats: self.logger.debug(f"[SimMan:_run_std_interval] Could not get adaptive stats: {e_stats}")
                metrics_collector.log_adaptive_stats(adaptive_stats) # type: ignore[operator]

            if not done:
                try:
                    # Pasar config completa a check_termination
                    limit_exceeded, goal_reached, _ = environment.check_termination(config)
                    # total_time desde environment.simulation.total_time
                    total_sim_time_per_episode = config.get('environment', {}).get('simulation', {}).get('total_time', 5.0)
                    time_limit_reached = (current_step_time >= total_sim_time_per_episode - (dt / 2.0))

                    if limit_exceeded or goal_reached or time_limit_reached:
                        done = True
                        if termination_reason == "unknown": # Asignar solo la primera vez
                            if limit_exceeded: termination_reason = "limit_exceeded" # Simplificar
                            elif goal_reached: termination_reason = "goal_reached"
                            elif time_limit_reached: termination_reason = "time_limit"
                            else: termination_reason = "unknown_termination"
                        self.logger.info(f"[SimMan:_run_std_interval] Episode ending: {termination_reason} at t={current_step_time:.3f}s")
                except Exception as e:
                    self.logger.error(f"[SimMan:_run_std_interval] Error in env.check_termination at t={current_step_time:.3f}s: {e}", exc_info=True)
                    done = True; termination_reason = "termination_check_error"
            if done: break
        avg_interval_stability = np.nanmean(interval_stability_scores) if interval_stability_scores else 1.0
        if not np.isfinite(avg_interval_stability): avg_interval_stability = 1.0
        return interval_reward_sum, avg_interval_stability, final_state_in_interval, done, termination_reason

    def _run_echo_baseline_interval_steps(self, start_time: float, duration: float, current_state: np.ndarray,
                                           environment: Environment, controller: Controller, agent: RLAgent,
                                           metrics_collector: MetricsCollector, virtual_simulator: VirtualSimulator,
                                           config: Dict, actions_applied_in_interval: Dict
                                          ) -> Tuple[float, float, np.ndarray, bool, str, Dict[str, float]]:
        interval_reward_real, avg_w_stab_real, final_state_real, done_real, termination_reason_real = \
            self._run_standard_interval_steps(
                start_time, duration, current_state, environment, controller, agent,
                metrics_collector, config, actions_applied_in_interval
            )
        if done_real and termination_reason_real == "env_step_error":
             self.logger.error("[SimMan:_run_echo_interval] Critical failure in real interval, skipping virtual simulations.")
             return interval_reward_real, avg_w_stab_real, final_state_real, done_real, termination_reason_real, {}

        reward_dict_echo: Dict[str, float] = {}
        interval_start_state = current_state
        gains_applied = controller.get_params()
        prev_kp = getattr(controller, 'prev_kp', gains_applied.get('kp', np.nan))
        prev_ki = getattr(controller, 'prev_ki', gains_applied.get('ki', np.nan))
        prev_kd = getattr(controller, 'prev_kd', gains_applied.get('kd', np.nan))
        if any(pd.isna(k) or not np.isfinite(k) for k in [prev_kp, prev_ki, prev_kd]):
             self.logger.warning(f"[SimMan:_run_echo_interval] Invalid previous gains (Kp:{prev_kp}, Ki:{prev_ki}, Kd:{prev_kd}). Skipping counterfactuals.")
             return interval_reward_real, avg_w_stab_real, final_state_real, done_real, termination_reason_real, {}

        try:
            gains_p_cf = {'kp': prev_kp, 'ki': gains_applied['ki'], 'kd': gains_applied['kd']}
            R_p_cf, Ws_p_cf = virtual_simulator.run_interval_simulation(interval_start_state, start_time, duration, gains_p_cf)
            gains_i_cf = {'kp': gains_applied['kp'], 'ki': prev_ki, 'kd': gains_applied['kd']}
            R_i_cf, Ws_i_cf = virtual_simulator.run_interval_simulation(interval_start_state, start_time, duration, gains_i_cf)
            gains_d_cf = {'kp': gains_applied['kp'], 'ki': gains_applied['ki'], 'kd': prev_kd}
            R_d_cf, Ws_d_cf = virtual_simulator.run_interval_simulation(interval_start_state, start_time, duration, gains_d_cf)
            reward_dict_echo = {
                'kp': interval_reward_real - R_p_cf, 'ki': interval_reward_real - R_i_cf, 'kd': interval_reward_real - R_d_cf
            }
            if hasattr(metrics_collector, 'log_virtual_rewards'): metrics_collector.log_virtual_rewards(reward_dict_echo) # type: ignore[operator]
            else: metrics_collector.log('virtual_reward_kp', reward_dict_echo.get('kp', np.nan)); metrics_collector.log('virtual_reward_ki', reward_dict_echo.get('ki', np.nan)); metrics_collector.log('virtual_reward_kd', reward_dict_echo.get('kd', np.nan))
            metrics_collector.log('virtual_w_stab_kp_cf', Ws_p_cf if np.isfinite(Ws_p_cf) else np.nan)
            metrics_collector.log('virtual_w_stab_ki_cf', Ws_i_cf if np.isfinite(Ws_i_cf) else np.nan)
            metrics_collector.log('virtual_w_stab_kd_cf', Ws_d_cf if np.isfinite(Ws_d_cf) else np.nan)
        except Exception as e:
            self.logger.error(f"[SimMan:_run_echo_interval] Error during virtual simulations: {e}", exc_info=True)
            reward_dict_echo = {}
            if hasattr(metrics_collector, 'log_virtual_rewards'): metrics_collector.log_virtual_rewards({}) # type: ignore[operator]
            else: metrics_collector.log('virtual_reward_kp', np.nan); metrics_collector.log('virtual_reward_ki', np.nan); metrics_collector.log('virtual_reward_kd', np.nan)
            metrics_collector.log('virtual_w_stab_kp_cf', np.nan); metrics_collector.log('virtual_w_stab_ki_cf', np.nan); metrics_collector.log('virtual_w_stab_kd_cf', np.nan)
        return interval_reward_real, avg_w_stab_real, final_state_real, done_real, termination_reason_real, reward_dict_echo

    def _handle_decision_boundary(self,
                                 current_time: float, current_state: np.ndarray,
                                 last_interval_data: Dict, interval_run_results: Dict,
                                 agent: RLAgent, controller: Controller,
                                 metrics_collector: MetricsCollector, config: Dict
                                 ) -> Tuple[Dict[str, int], Optional[Dict]]:
        decision_start_time = time.time()
        agent_decision_count = last_interval_data.get('decision_count', 0) + 1
        metrics_collector.log('id_agent_decision', agent_decision_count)
        self.logger.debug(f"[SimMan:_handle_decision_boundary] Start Decision #{agent_decision_count} @ t={current_time:.3f}s")

        next_raw_state_vector = current_state
        # state_config para agente desde config
        state_config_for_agent = config.get('environment', {}).get('agent', {}).get('params', {}).get('state_config', {})
        next_agent_state_dict = agent.build_agent_state(next_raw_state_vector, controller, state_config_for_agent)
        episode_done = interval_run_results['done']

        # Preparar reward_info para agent.learn()
        # Esta es la "info cruda" del intervalo. El agente pasará esto a su RewardStrategy.
        reward_info_for_agent: Union[float, Tuple[float, float], Dict[str, float]]
        if 'reward_dict_echo' in interval_run_results and interval_run_results['reward_dict_echo']:
            reward_info_for_agent = interval_run_results['reward_dict_echo']
        else:
            interval_reward_val = interval_run_results.get('interval_reward', 0.0)
            avg_w_stab_val = interval_run_results.get('avg_w_stab', 1.0)
            reward_info_for_agent = (float(interval_reward_val), float(avg_w_stab_val))

        learn_data = {
            'current_agent_state_dict': last_interval_data['start_state_dict'],
            'actions_dict': last_interval_data['actions_dict'],
            'reward_info': reward_info_for_agent, # Pasar la info cruda
            'next_agent_state_dict': next_agent_state_dict,
            'controller': controller,
            'done': episode_done
        }

        last_td_errors_logged = False
        try:
            agent.learn(**learn_data) # El agente usará su RewardStrategy internamente
            td_errors = agent.get_last_td_errors()
            if hasattr(metrics_collector, 'log_td_errors'): metrics_collector.log_td_errors(td_errors) # type: ignore[operator]
            else: metrics_collector.log('td_error_kp', td_errors.get('kp', np.nan)); metrics_collector.log('td_error_ki', td_errors.get('ki', np.nan)); metrics_collector.log('td_error_kd', td_errors.get('kd', np.nan))
            last_td_errors_logged = True
        except Exception as learn_e:
            self.logger.error(f"[SimMan:_handle_decision_boundary] Error in agent.learn(): {learn_e}", exc_info=True)
            if not last_td_errors_logged:
                 if hasattr(metrics_collector, 'log_td_errors'): metrics_collector.log_td_errors({}) # type: ignore[operator]
                 else: metrics_collector.log('td_error_kp', np.nan); metrics_collector.log('td_error_ki', np.nan); metrics_collector.log('td_error_kd', np.nan)
                 last_td_errors_logged = True

        next_actions_prime = {g: 1 for g in ['kp', 'ki', 'kd']} # Default neutral
        if not episode_done:
            next_actions_prime = agent.select_action(next_agent_state_dict)
            self._apply_actions_to_controller(controller, next_actions_prime, config)

        if hasattr(metrics_collector, 'log_early_termination_metrics'): metrics_collector.log_early_termination_metrics(agent) # type: ignore
        if hasattr(metrics_collector, 'log_q_values'): metrics_collector.log_q_values(agent, next_agent_state_dict) # type: ignore[operator]
        if hasattr(metrics_collector, 'log_q_visit_counts'): metrics_collector.log_q_visit_counts(agent, next_agent_state_dict) # type: ignore[operator]
        if hasattr(metrics_collector, 'log_baselines'): metrics_collector.log_baselines(agent, next_agent_state_dict) # type: ignore[operator]
        if not last_td_errors_logged: # Log TD errors si learn falló antes
             if hasattr(metrics_collector, 'log_td_errors'): metrics_collector.log_td_errors(agent.get_last_td_errors()) # type: ignore[operator]
             else: td_errors = agent.get_last_td_errors(); metrics_collector.log('td_error_kp', td_errors.get('kp', np.nan)); metrics_collector.log('td_error_ki', td_errors.get('ki', np.nan)); metrics_collector.log('td_error_kd', td_errors.get('kd', np.nan))

        decision_duration_ms = (time.time() - decision_start_time) * 1000
        metrics_collector.log('learn_select_duration_ms', decision_duration_ms)

        next_interval_data = None
        if not episode_done:
            next_interval_data = {
                'start_state_dict': next_agent_state_dict,
                'start_raw_state_vector': np.copy(next_raw_state_vector),
                'actions_dict': next_actions_prime.copy(),
                'reward_sum': 0.0, 'w_stab_sum': 0.0, 'steps_in_interval': 0,
                'end_state_dict': None, 'done': False, 'reward_dict_echo': None,
                'decision_count': agent_decision_count
             }
        self.logger.debug(f"[SimMan:_handle_decision_boundary] End Decision #{agent_decision_count}. Duration: {decision_duration_ms:.2f}ms. Next Action A'': {next_actions_prime if not episode_done else 'N/A'}")
        return next_actions_prime, next_interval_data

    def _finalize_episode(self, episode: int, episode_metrics_dict: Dict, termination_reason: str,
                          episode_start_time: float, controller: Controller, environment: Environment,
                          agent: RLAgent, results_folder: str, config: Dict,
                          summary_data_list: List, current_episode_batch: List,
                          agent_state_save_freq: int):
        episode_duration_s = time.time() - episode_start_time
        last_decision_id_list = episode_metrics_dict.get('id_agent_decision', [])
        final_decision_count = np.nanmax(last_decision_id_list) if last_decision_id_list and np.any(pd.notna(last_decision_id_list)) else 0

        rewards = np.array(episode_metrics_dict.get('reward', []), dtype=float)
        stability_scores = np.array(episode_metrics_dict.get('stability_score', []), dtype=float)
        total_reward = np.nansum(rewards)
        avg_stability = np.nanmean(stability_scores) if stability_scores.size > 0 else np.nan
        final_time_list = episode_metrics_dict.get('time', [])
        final_time = np.nanmax(final_time_list) if final_time_list and np.any(pd.notna(final_time_list)) else np.nan

        len_first_metric = len(next(iter(episode_metrics_dict.values()), []))
        if len_first_metric > 0:
             final_gains = controller.get_params()
             perf = total_reward / final_time if pd.notna(total_reward) and pd.notna(final_time) and final_time > 1e-9 else np.nan
             # Añadir métricas precalculadas al dict detallado para que summarize_episode las use
             for k, v_list in [('termination_reason', [termination_reason]), ('episode_duration_s', [episode_duration_s]),
                               ('final_kp', [final_gains.get('kp', np.nan)]), ('final_ki', [final_gains.get('ki', np.nan)]),
                               ('final_kd', [final_gains.get('kd', np.nan)]), ('total_agent_decisions', [final_decision_count]),
                               ('avg_stability_score', [avg_stability]), ('total_reward', [total_reward]),
                               ('episode_time', [final_time]), ('performance', [perf])]:
                 episode_metrics_dict[k] = v_list * len_first_metric

        # Agregar agentes
        if hasattr(agent, 'get_agent_defining_vars'): episode_metrics_dict['_agent_defining_vars'] = agent.get_agent_defining_vars()
        
        summary = summarize_episode(episode_metrics_dict)
        summary['episode'] = episode # Asegurar que 'episode' esté en el resumen
        summary_data_list.append(summary)
        current_episode_batch.append(episode_metrics_dict)

        self.logger.info(f"[SimMan:_finalize_episode] Ep {episode} Summary: Term='{summary.get('termination_reason', '?')}', "
                         f"R={summary.get('total_reward', np.nan):.2f}, Perf={summary.get('performance', np.nan):.2f}, "
                         f"Stab={summary.get('avg_stability_score', np.nan):.3f}, T={summary.get('episode_time', np.nan):.2f}s, "
                         f"Decisions={summary.get('total_agent_decisions', 0)}, Gains(Kp={summary.get('final_kp', np.nan):.2f}, "
                         f"Ki={summary.get('final_ki', np.nan):.2f}, Kd={summary.get('final_kd', np.nan):.3f}), "
                         f"Dur={summary.get('episode_duration_s', np.nan):.2f}s")
        try:
            environment.update_reward_calculator_stats(episode_metrics_dict, episode)
        except Exception as e:
            self.logger.error(f"[SimMan:_finalize_episode] Error update_reward_calculator_stats ep {episode}: {e}", exc_info=True)

        if agent_state_save_freq > 0 and (episode + 1) % agent_state_save_freq == 0:
            self.result_handler.save_agent_state(agent, episode, results_folder)

    def run(self) -> Tuple[List[Dict], List[Dict]]:
        self.logger.info("[SimMan:run] --- Starting Main Simulation Loop ---")
        all_episodes_detailed_data: List[Dict] = []
        summary_data: List[Dict] = []
        results_folder: Optional[str] = None
        current_episode_batch: List[Dict] = []
        file_handlers: List[logging.Handler] = []
        environment: Optional[Environment] = None
        agent: Optional[RLAgent] = None
        controller: Optional[Controller] = None
        reward_strategy: Optional[RewardStrategy] = None
        virtual_simulator: Optional[VirtualSimulator] = None
        config: Optional[Dict] = None
        episode_idx = -1 # Renombrar para evitar confusión con la variable de episodio de summarize_episode

        try:
            # Resolver dependencias una vez
            environment, agent, controller, _, reward_strategy, \
                virtual_simulator, config, results_folder = self._resolve_dependencies()
            # MetricsCollector se resuelve dentro del bucle

            # Leer config desde nuevas rutas
            env_cfg_section = config.get('environment', {})
            sim_cfg_params = env_cfg_section.get('simulation', {})
            logging_cfg_params = config.get('logging', {})

            max_episodes = sim_cfg_params.get('max_episodes', 1)
            decision_interval = sim_cfg_params.get('decision_interval', 0.01)
            dt = environment.dt if hasattr(environment, 'dt') else sim_cfg_params.get('dt', 0.001) # type: ignore[attr-defined]

            if not isinstance(decision_interval, (float, int)) or decision_interval < dt:
                 self.logger.warning(f"[SimMan:run] Decision interval ({decision_interval}) < dt ({dt}). Using dt as interval.")
                 decision_interval = dt

            episodes_per_file = env_cfg_section.get('episodes_per_file', 100)
            agent_state_save_freq = env_cfg_section.get('agent_state_save_frequency', 0)
            if env_cfg_section.get('save_agent_state', False) and agent_state_save_freq <= 0:
                 agent_state_save_freq = max_episodes
            elif not env_cfg_section.get('save_agent_state', False):
                 agent_state_save_freq = 0

            log_flush_frequency = logging_cfg_params.get('log_save_frequency', 0)

            # Determinar si se necesita simulador virtual basado en la estrategia
            # Asumir que la estrategia tiene un atributo 'needs_virtual_simulation'
            # Este atributo debe ser añadido a la interfaz RewardStrategy y sus implementaciones.
            # Por ahora, haremos un isinstance check, pero esto se debe cambiar.
            # needs_virtual_sim = getattr(reward_strategy, 'needs_virtual_simulation', False) # Ideal
            needs_virtual_sim = isinstance(reward_strategy, EchoBaselineRewardStrategy) # Temporal
            if needs_virtual_sim and virtual_simulator is None:
                 msg = f"[SimMan:run] RewardStrategy '{type(reward_strategy).__name__}' requires VirtualSimulator, but none was resolved."
                 self.logger.critical(msg); raise ValueError(msg)
            self.logger.info(f"[SimMan:run] Simulation Config: MaxEp={max_episodes}, DecisionInt={decision_interval:.4f}s, dt={dt:.4f}s, BatchSize={episodes_per_file}, AgentSaveFreq={agent_state_save_freq}, VirtualSimNeeded={needs_virtual_sim}")

            if log_flush_frequency > 0:
                 file_handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)]
                 if not file_handlers: log_flush_frequency = 0; self.logger.warning("[SimMan:run] Log flush freq > 0 but no FileHandlers found.")

            self.logger.info(f"[SimMan:run] Starting simulation for {max_episodes} episodes...")
            for episode_idx in range(max_episodes):
                episode_start_time = time.time()
                metrics_collector = self.container.resolve(MetricsCollector) # Resolver transient
                if metrics_collector is None:
                     self.logger.critical(f"[SimMan:run] CRITICAL: Failed to resolve MetricsCollector for ep {episode_idx}. Aborting."); break

                try:
                    current_state, last_interval_data = self._initialize_episode(
                        episode_idx, environment, agent, controller, metrics_collector, config # type: ignore[arg-type]
                    )
                except Exception as init_e:
                    self.logger.error(f"[SimMan:run] Failed initializing ep {episode_idx}: {init_e}. Skipping.", exc_info=True)
                    continue

                current_time = 0.0
                episode_done = False
                termination_reason = "unknown"
                total_sim_time_per_episode = sim_cfg_params.get('total_time', 5.0)

                while not episode_done and current_time < total_sim_time_per_episode:
                    interval_duration = min(decision_interval, total_sim_time_per_episode - current_time)
                    if interval_duration <= dt / 2.0: break

                    actions_applied = last_interval_data['actions_dict'].copy()
                    interval_run_results: Dict[str, Any] = {}
                    try:
                        if needs_virtual_sim and virtual_simulator: # Chequear virtual_simulator de nuevo
                            interval_reward, avg_w_stab, final_state, interval_done, term_reason, r_dict_echo = \
                                self._run_echo_baseline_interval_steps(
                                    current_time, interval_duration, current_state, environment, controller, agent, # type: ignore[arg-type]
                                    metrics_collector, virtual_simulator, config, actions_applied
                                )
                            interval_run_results = {'interval_reward': interval_reward, 'avg_w_stab': avg_w_stab, 'final_state': final_state, 'done': interval_done, 'termination_reason': term_reason, 'reward_dict_echo': r_dict_echo}
                        else:
                            interval_reward, avg_w_stab, final_state, interval_done, term_reason = \
                                self._run_standard_interval_steps(
                                    current_time, interval_duration, current_state, environment, controller, agent, # type: ignore[arg-type]
                                    metrics_collector, config, actions_applied
                                )
                            interval_run_results = {'interval_reward': interval_reward, 'avg_w_stab': avg_w_stab, 'final_state': final_state, 'done': interval_done, 'termination_reason': term_reason}
                    except RuntimeError as step_e:
                         self.logger.error(f"[SimMan:run] Runtime error during interval ep {episode_idx} t={current_time:.3f}s: {step_e}. Terminating episode.")
                         episode_done = True; termination_reason = "interval_runtime_error"
                         current_state = last_interval_data.get('start_raw_state_vector', current_state)
                         break

                    current_state = interval_run_results['final_state']
                    last_logged_time_list = metrics_collector.get_metrics().get('time', [current_time])
                    current_time = last_logged_time_list[-1] if last_logged_time_list else current_time
                    episode_done = interval_run_results['done']
                    if episode_done and termination_reason == "unknown":
                         termination_reason = interval_run_results.get('termination_reason', 'interval_ended_done')

                    try:
                        _, next_interval_data = self._handle_decision_boundary(
                            current_time, current_state, last_interval_data, interval_run_results,
                            agent, controller, metrics_collector, config # type: ignore[arg-type]
                        )
                    except Exception as decision_e:
                        self.logger.error(f"[SimMan:run] Error in _handle_decision_boundary ep {episode_idx} t={current_time:.3f}s: {decision_e}. Terminating.", exc_info=True)
                        episode_done = True; termination_reason = "decision_boundary_error"; next_interval_data = None

                    # --- EVALUAR SI episode_done POR EARLY TERMINATION
                    if next_interval_data is None: # Esto significa que episode_done se activó en _handle_decision_boundary | OJO podría ser por ERROR
                        episode_done = True 
                        # termination_reason ya debería estar seteada si _handle_decision_boundary falló
                        if termination_reason == "unknown": 
                            termination_reason = "decision_boundary_no_next_data"
                        last_interval_data = None # Se propaga valor por consistencia
                        break # Salir del bucle while

                    last_interval_data = next_interval_data # Asignar para el siguiente ciclo del while

                    if agent.early_termination_enabled and hasattr(agent, 'should_episode_terminate_early') and agent.should_episode_terminate_early():
                        self.logger.info(f"[SimMan:run Ep {episode_idx}] Agent requested early termination at t={current_time:.3f}s.")
                        episode_done = True
                        if termination_reason == "unknown": # Solo si no hay otra razón más específica
                            termination_reason = "agent_early_termination"
                        # No hay 'break' inmediato aquí; la condición de `episode_done` se evaluará al inicio del siguiente `while`
                    
                    if episode_done: last_interval_data = None
                    elif next_interval_data is not None: last_interval_data = next_interval_data
                    else:
                        self.logger.error(f"[SimMan:run] Logic error: Ep {episode_idx} not done but no next_interval_data. Terminating.")
                        episode_done = True; termination_reason = "interval_logic_error"; last_interval_data = None
                    if episode_done: break

                try:
                    episode_metrics = metrics_collector.get_metrics()
                    self._finalize_episode(
                        episode_idx, episode_metrics, termination_reason, episode_start_time, controller, # type: ignore[arg-type]
                        environment, agent, results_folder, config, summary_data, current_episode_batch, # type: ignore[arg-type]
                        agent_state_save_freq
                    )
                except Exception as finalize_e:
                     self.logger.error(f"[SimMan:run] Error finalizing ep {episode_idx}: {finalize_e}", exc_info=True)

                if episodes_per_file > 0 and ((episode_idx + 1) % episodes_per_file == 0 or episode_idx == max_episodes - 1):
                    if current_episode_batch and results_folder:
                        self.result_handler.save_episode_batch(current_episode_batch, results_folder, episode_idx)
                        current_episode_batch = []

                if log_flush_frequency > 0 and (episode_idx + 1) % log_flush_frequency == 0:
                    for h in file_handlers:
                        try: h.flush()
                        except Exception as e_flush: self.logger.warning(f"[SimMan:run] Error flushing handler {h}: {e_flush}")
            self.logger.info(f"[SimMan:run] --- All {max_episodes} Episodes Processed ---")

        except (ValueError, RuntimeError, AttributeError, TypeError, KeyError) as e:
            self.logger.critical(f"[SimMan:run] CRITICAL Error in simulation loop (Ep ~{episode_idx}): {e}", exc_info=True)
            if current_episode_batch and results_folder and os.path.isdir(results_folder):
                 self.logger.warning("[SimMan:run] Attempting to save partial batch after critical error...")
                 try: self.result_handler.save_episode_batch(current_episode_batch, results_folder, episode_idx)
                 except Exception as save_e: self.logger.error(f"[SimMan:run] Failed saving partial batch: {save_e}")
        except Exception as e:
            self.logger.critical(f"[SimMan:run] UNEXPECTED Error in simulation loop (Ep ~{episode_idx}): {e}", exc_info=True)
            if current_episode_batch and results_folder and os.path.isdir(results_folder):
                 self.logger.warning("[SimMan:run] Attempting to save partial batch after unexpected error...")
                 try: self.result_handler.save_episode_batch(current_episode_batch, results_folder, episode_idx)
                 except Exception as save_e: self.logger.error(f"[SimMan:run] Failed saving partial batch: {save_e}")
        finally:
            if file_handlers:
             self.logger.info("[SimMan:run] Performing final log flush...")
             for h in file_handlers:
                  try: h.flush()
                  except Exception as e_flush_final: self.logger.warning(f"[SimMan:run] Error final flushing handler {h}: {e_flush_final}")
            # Limpieza explícita
            #del environment, agent, controller, reward_strategy, virtual_simulator, config, metrics_collector
            #del all_episodes_detailed_data, summary_data, current_episode_batch # Aunque algunos ya deberían estar vacíos/liberados
            #gc.collect()
            self.logger.info("[SimMan:run] --- Main Simulation Loop Finished & Cleanup Attempted ---")
        return all_episodes_detailed_data, summary_data # all_episodes_detailed_data debería estar vacío si se usa batch