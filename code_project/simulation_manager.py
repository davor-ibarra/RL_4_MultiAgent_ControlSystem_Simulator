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
from interfaces.reward_function import RewardFunction
from interfaces.reward_strategy import RewardStrategy
from interfaces.stability_calculator import BaseStabilityCalculator

from utils.data.result_handler import ResultHandler
from utils.data.data_processing import summarize_episode

if TYPE_CHECKING:
    from di_container import Container


# Definir los tokens string que SimulationManager necesita resolver
# (deben coincidir con los usados en build_container)
_PROCESSED_DATA_DIRECTIVES_TOKEN_STR_ = "processed_data_directives_dict_token"
_OUTPUT_DIR_TOKEN_STR_ = "output_dir_path_token"


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

    # --- SECCIÓN 1: Resolución de Dependencias y Configuración ---
    def _resolve_dependencies(self) -> Tuple[
        Environment, RLAgent, Controller, RewardStrategy, Optional[VirtualSimulator],
        BaseStabilityCalculator, Dict[str, Any], Dict[str, Any], str  # main_config, data_directives, output_dir
    ]:
        """Resuelve todas las dependencias necesarias una sola vez."""
        self.logger.info("[SimMan:_resolve_dependencies] Resolving core simulation components...")
        env_instance = self.container.resolve(Environment)
        agent_instance = self.container.resolve(RLAgent)
        controllers_dict_token = "controllers_dict_token"
        controllers_dict_instance = self.container.resolve(controllers_dict_token)
        reward_strategy_instance = self.container.resolve(RewardStrategy)
        virtual_simulator_instance = self.container.resolve(Optional[VirtualSimulator]) # Es opcional
        stability_calculator_instance = self.container.resolve(BaseStabilityCalculator)
        main_config_dict = self.container.resolve(dict)
        output_dir_path = self.container.resolve(_OUTPUT_DIR_TOKEN_STR_)
        processed_data_directives_dict = self.container.resolve(_PROCESSED_DATA_DIRECTIVES_TOKEN_STR_)

        critical_deps = { # type: ignore
            "Environment": env_instance, "RLAgent": agent_instance, "Controller": controllers_dict_instance,
            "RewardStrategy": reward_strategy_instance, "BaseStabilityCalculator": stability_calculator_instance, 
            "main_config": main_config_dict,"output_dir": output_dir_path, "processed_data_directives": processed_data_directives_dict
        }
        missing = [name for name, dep in critical_deps.items() if dep is None]
        if missing:
            raise ValueError(f"[SimMan:_resolve_dependencies] Failed to resolve critical DI dependencies: {missing}")

        self.logger.info("[SimMan:_resolve_dependencies] Core dependencies resolved.")
        #self.logger.info(f"[SimMan:_resolve_dependencies] Core dependencies resolved. StabilityCalculator: {type(stability_calculator_instance).__name__ if stability_calculator_instance else 'None'}.")
        return (env_instance, agent_instance, controllers_dict_instance, reward_strategy_instance, virtual_simulator_instance, 
                stability_calculator_instance, main_config_dict, processed_data_directives_dict, output_dir_path)

    def _extract_simulation_parameters(self, config: Dict[str, Any], env: Environment) -> Dict[str, Any]:
        """Extrae y valida parámetros de simulación de la configuración."""
        sim_params_cfg = config.get('environment', {}).get('simulation', {})
        data_handling_cfg = config.get('data_handling', {})
        log_cfg = config.get('logging', {})

        params = {
            'max_episodes': sim_params_cfg.get('max_episodes', 1),
            'episode_duration_sec': sim_params_cfg.get('episode_duration_sec', 5.0),
            'decision_interval_sec': sim_params_cfg.get('agent_decision_period_sec', 0.01),
            'sim_dt_sec': env.dt, # Obtener dt del entorno resuelto
            'episodes_per_chunk': data_handling_cfg.get('episodes_per_dataset_chunk', 100),
            'agent_state_save_freq': data_handling_cfg.get('agent_state_save_frequency', 0),
            'save_agent_state_enabled': data_handling_cfg.get('save_agent_state', False),
            'log_flush_frequency_episodes': log_cfg.get('log_save_frequency', 0),
            'state_definition_config': config.get('environment', {}).get('agent', {}).get('params', {}).get('state_config', {})
        }

        if not isinstance(params['decision_interval_sec'], (float, int)) or params['decision_interval_sec'] < params['sim_dt_sec']:
            self.logger.warning(f"[SimMan:_extract_params] Agent decision period ({params['decision_interval_sec']}) invalid or < dt ({params['sim_dt_sec']}). Using dt as interval.")
            params['decision_interval_sec'] = params['sim_dt_sec']

        if params['save_agent_state_enabled'] and params['agent_state_save_freq'] <= 0:
            params['agent_state_save_freq'] = params['max_episodes']
        elif not params['save_agent_state_enabled']:
            params['agent_state_save_freq'] = 0
        
        self.logger.info(f"[SimMan:_extract_params] Simulation parameters extracted. MaxEp={params['max_episodes']}, DecisionInterval={params['decision_interval_sec']:.4f}s, SimDT={params['sim_dt_sec']:.4f}s")
        return params

    # --- SECCIÓN 2: Orquestación Principal (`run`) ---
    def run(self) -> Tuple[List[Dict], List[Dict]]:
        self.logger.info("[SimMan:run] --- Simulation Run Starting ---")
        summary_data_all_eps: List[Dict] = []
        detailed_data_batch_buffer: List[Dict] = []

        try:
            (env, agent, ctrls, reward_strategy, virtual_sim, stability_calc,
             config, data_directives, output_dir) = self._resolve_dependencies()

            sim_run_params = self._extract_simulation_parameters(config, env)
            
            if reward_strategy.needs_virtual_simulation and virtual_sim is None:
                raise ValueError(f"RewardStrategy '{type(reward_strategy).__name__}' needs VirtualSimulator, but none was resolved.")

            self.logger.info(f"[SimMan:run] Starting simulation for {sim_run_params['max_episodes']} episodes...")

            for ep_idx in range(sim_run_params['max_episodes']):
                metrics_collector = self.container.resolve(MetricsCollector) # Transient
                if metrics_collector is None:
                    raise RuntimeError(f"Failed to resolve MetricsCollector for episode {ep_idx}")

                episode_summary_dict, episode_detailed_metrics = self._run_episode(
                    ep_idx, env, agent, ctrls, reward_strategy, virtual_sim, 
                    stability_calc, metrics_collector, config, sim_run_params, data_directives
                )
                summary_data_all_eps.append(episode_summary_dict)

                if data_directives.get('config', {}).get('enabled_json_history', False) and episode_detailed_metrics:
                    detailed_data_batch_buffer.append(episode_detailed_metrics)

                # Guardar datos detallados en lotes (delegado a ResultHandler)
                is_last_episode = (ep_idx == sim_run_params['max_episodes'] - 1)
                if detailed_data_batch_buffer and (len(detailed_data_batch_buffer) >= sim_run_params['episodes_per_chunk'] or is_last_episode):
                    self.result_handler.save_episode_batch(detailed_data_batch_buffer, output_dir, ep_idx)
                    detailed_data_batch_buffer.clear()

                # Guardar estado del agente periódicamente
                if sim_run_params['agent_state_save_freq'] > 0 and (ep_idx + 1) % sim_run_params['agent_state_save_freq'] == 0:
                    self.result_handler.save_agent_state(agent, ep_idx, output_dir)

                # Flush logs periódicamente
                if sim_run_params['log_flush_frequency_episodes'] > 0 and (ep_idx + 1) % sim_run_params['log_flush_frequency_episodes'] == 0:
                    for handler in logging.getLogger().handlers:
                        if isinstance(handler, logging.FileHandler): handler.flush()
                
                gc.collect()

            self.logger.info(f"[SimMan:run] All {sim_run_params['max_episodes']} episodes processed.")
            # Los datos detallados se guardan en lotes, por lo que devolvemos lista vacía aquí.
            return [], summary_data_all_eps

        except Exception as e:
            self.logger.critical(f"[SimMan:run] UNHANDLED EXCEPTION during simulation: {e}", exc_info=True)
            raise
        finally:
            self.logger.info("[SimMan:run] --- Simulation Run Finished ---")

    # --- SECCIÓN 3: Orquestación de Episodio Individual ---
    def _run_episode(self,
                     episode_idx: int,
                     env: Environment, agent: RLAgent, ctrls: Dict[str, Controller],
                     reward_strategy: RewardStrategy, virtual_sim: Optional[VirtualSimulator],
                     stability_calc: BaseStabilityCalculator,
                     metrics_collector: MetricsCollector, config: Dict[str, Any],
                     sim_run_params: Dict[str, Any],
                     data_directives: Dict[str, Any]
                    ) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
        """Orquesta la ejecución de un solo episodio, delegando el logging y el resumen."""
        self.logger.info(f"--- [ Episode {episode_idx}/{sim_run_params['max_episodes']-1} Starting ] ---")

        # 1. Inicialización del Episodio
        ep_wall_start_time = time.time()
        initial_conditions_cfg = config.get('environment', {}).get('initial_conditions', {}).get('x0')
        current_raw_state_s = env.reset(initial_conditions_cfg)
        initial_state_dict = env._create_state_dict(current_raw_state_s) if hasattr(env, '_create_state_dict') else {}
        
        metrics_collector.log_on_episode_start(context={
            'episode_id': episode_idx,
            'id_agent_decision': np.nan,
            'state': initial_state_dict,
            'controllers': ctrls, 'agent': agent, 'stability_calculator': stability_calc
        })
        
        current_sim_time_sec = 0.0
        episode_done_flag = False
        episode_term_reason = "unknown"
        decision_counter = 0

        # Bucle de Intervalos de Decisión
        while not episode_done_flag and current_sim_time_sec < sim_run_params['episode_duration_sec']:
            interval_remaining_time = sim_run_params['episode_duration_sec'] - current_sim_time_sec
            current_interval_duration_actual = min(sim_run_params['decision_interval_sec'], interval_remaining_time)
            id_agent_decision = decision_counter
            if current_interval_duration_actual < sim_run_params['sim_dt_sec'] / 2.0: break

            learn_select_boundary_start_time = time.time()

            # 2. Selección y Aplicación de Acción
            current_state_dict = env._create_state_dict(current_raw_state_s) if hasattr(env, '_create_state_dict') else {}
            current_agent_s_dict = agent.build_agent_state(current_state_dict, ctrls)
            actions_a_prime_dict = agent.select_action(current_agent_s_dict)
            self._apply_actions_to_controller(ctrls, actions_a_prime_dict, config)

            # 3. Ejecución del Intervalo de Simulación (delega el logging a los métodos internos)
            if reward_strategy.needs_virtual_simulation and virtual_sim:
                (interval_reward, avg_stability_score_interval, final_env_state_s_prime_raw,
                 interval_done_flag, term_reason_from_interval, diff_rewards) = \
                    self._run_echo_baseline_interval_steps(
                        id_agent_decision, current_sim_time_sec, current_interval_duration_actual, current_raw_state_s,
                        env, ctrls, agent, env.reward_function, metrics_collector, virtual_sim, stability_calc, config
                    )
                reward_info_for_learn = {'interval_reward': interval_reward, 
                                         'avg_stability_score_interval': avg_stability_score_interval, 
                                         'differential_rewards': diff_rewards}
            else:
                (interval_reward, avg_stability_score_interval, final_env_state_s_prime_raw,
                 interval_done_flag, term_reason_from_interval) = \
                    self._run_standard_interval_steps(
                        id_agent_decision, current_sim_time_sec, current_interval_duration_actual, current_raw_state_s,
                        env, ctrls, agent, env.reward_function, metrics_collector, stability_calc, config
                    )
                reward_info_for_learn = {'interval_reward': interval_reward, 
                                         'avg_stability_score_interval': avg_stability_score_interval}

            # 4. Aprendizaje del Agente
            next_state_dict = env._create_state_dict(final_env_state_s_prime_raw) if hasattr(env, '_create_state_dict') else {}
            next_agent_s_prime_dict = agent.build_agent_state(next_state_dict, ctrls)
            learning_metrics = agent.learn(
                current_agent_s_dict, actions_a_prime_dict,
                reward_info_for_learn,
                next_agent_s_prime_dict, ctrls, episode_done_flag, episode_idx
            )
            
            # 5. Logging en el Límite de Decisión
            metrics_collector.log_on_decision_boundary(context={
                'actions_map': actions_a_prime_dict,
                'reward_info_for_learn': reward_info_for_learn,
                'learn_metrics': learning_metrics,
                'controllers': ctrls,
                'agent': agent,
                'stability_calculator': stability_calc
                })

            decision_counter += 1 # Incrementar para el siguiente intervalo
            
            # 6. Comprobar terminación temprana
            episode_done_flag = interval_done_flag
            if episode_done_flag and episode_term_reason == "unknown":
                episode_term_reason = term_reason_from_interval
            if not episode_done_flag and agent.early_termination_enabled and agent.should_episode_terminate_early():
                episode_done_flag = True
                if episode_term_reason == "unknown": 
                    episode_term_reason = "agent_early_termination"
            if episode_done_flag: 
                break
        
            # 7. Actualizar estado y métricas globales
            current_raw_state_s = final_env_state_s_prime_raw
            current_sim_time_sec += current_interval_duration_actual

        # --- Finalización del Episodio ---
        metrics_collector.log_on_episode_end(context={
            'total_agent_decisions': decision_counter,
            'termination_reason': episode_term_reason,
            'episode_wall_time_sec': time.time() - ep_wall_start_time,
            'controllers': ctrls, # Para obtener ganancias finales
            'agent': agent, # Para obtener epsilon/lr finales
        })
        final_metrics = metrics_collector.get_metrics()
        env.update_reward_and_stability_calculator_stats(final_metrics, episode_idx)

        # 8. Pasar las directivas y la config global a la función de resumen

        # Obtener directivas del colector
        summary_directives = metrics_collector.get_summary_directives()
        # Obtener la config global para el resumen (para 'summary_first_cols', etc.)
        global_summary_config = data_directives.get('config', {})

        episode_summary = summarize_episode(final_metrics, summary_directives, global_summary_config)
        episode_summary['episode'] = episode_idx
        
        self.logger.info(f"--- [ Episode {episode_idx} Finished: {episode_summary.get('termination_reason','?')} ] --- "
                         f"angle_mean={episode_summary.get('pendulum_angle_mean', np.nan):.3f} | "
                         f"cart_mean={episode_summary.get('cart_position_mean', np.nan):.3f} | "
                         #f"R_base_theta_mean={episode_summary.get('reward_base_pendulum_angle_mean', np.nan):.3f} | "
                         f"R={episode_summary.get('total_reward', np.nan):.2f} | "
                         f"Stab={episode_summary.get('avg_stability_score', np.nan):.3f} ] ---"
                         )
        
        self.logger.info(f"t_max={episode_summary.get('time_duration', np.nan):.2f} | "
                         f"angle_kp={episode_summary.get('final_kp_pendulum_angle', np.nan):.2f} | "
                         f"angle_ki={episode_summary.get('final_ki_pendulum_angle', np.nan):.3f} | "
                         f"angle_kd={episode_summary.get('final_kd_pendulum_angle', np.nan):.3f} | "
                         f"cart_kp={episode_summary.get('final_kp_cart_position', np.nan):.2f} | "
                         f"cart_ki={episode_summary.get('final_ki_cart_position', np.nan):.3f} | "
                         f"cart_kd={episode_summary.get('final_kd_cart_position', np.nan):.3f} | "
                         )
        
        return episode_summary, final_metrics

    # --- SECCIÓN 4: Ejecución de Intervalos ---
    def _run_standard_interval_steps(self,
                                     id_agent_decision: int,
                                     interval_start_sim_time: float,
                                     interval_duration_to_run: float,
                                     current_raw_state_at_interval_start: np.ndarray,
                                     env: Environment, ctrls: Dict[str, Controller], agent: RLAgent, reward_function: RewardFunction,
                                     metrics_collector: MetricsCollector,
                                     stability_calc: BaseStabilityCalculator,
                                     config: Dict[str, Any]
                                     ) -> Tuple[float, float, np.ndarray, bool, str]:
        """Ejecuta los pasos de simulación para un intervalo estándar, delegando el logging."""
        accumulated_interval_reward = 0.0
        stability_scores_in_interval: List[float] = []
        is_interval_terminal = False
        termination_reason_interval = "unknown"
        last_state_in_interval = np.copy(current_raw_state_at_interval_start)
        sim_dt = env.dt

        num_steps_this_interval = max(1, int(round(interval_duration_to_run / sim_dt)))

        for step_idx in range(num_steps_this_interval):
            current_step_sim_time = round(interval_start_sim_time + (step_idx + 1) * sim_dt, 6)
            
            next_state_vec, reward, stability_score, _ = env.step()
            
            last_state_in_interval = next_state_vec
            accumulated_interval_reward += float(reward) if pd.notna(reward) else 0.0
            stability_scores_in_interval.append(float(stability_score) if pd.notna(stability_score) else 1.0)

            next_state_dict = env._create_state_dict(next_state_vec) if hasattr(env, '_create_state_dict') else {}            
            metrics_collector.log_on_step(context={
                'id_agent_decision': id_agent_decision, # only for save param
                'time': current_step_sim_time,
                'state': next_state_dict,
                'reward': reward,
                'stability_score': stability_score,
                'env': env,
                'controllers': ctrls,
                'agent': agent,
                'reward_calc' : reward_function,
                'stability_calculator': stability_calc
            })

            limit_exceeded, goal_reached, _ = env.check_termination()
            max_ep_duration = config.get('environment', {}).get('simulation', {}).get('episode_duration_sec', 5.0)
            time_limit_reached = (current_step_sim_time >= max_ep_duration - (sim_dt / 2.0))

            if limit_exceeded or goal_reached or time_limit_reached:
                is_interval_terminal = True
                if termination_reason_interval == "unknown":
                    termination_reason_interval = "limit_exceeded" if limit_exceeded else "goal_reached" if goal_reached else "time_limit"
                break

        avg_stability = np.nanmean(stability_scores_in_interval) if stability_scores_in_interval else 1.0
        return accumulated_interval_reward, float(avg_stability), last_state_in_interval, is_interval_terminal, termination_reason_interval

    def _run_echo_baseline_interval_steps(self,
                                          id_agent_decision: int,
                                          interval_start_sim_time: float,
                                          interval_duration_to_run: float,
                                          current_raw_state_at_interval_start: np.ndarray,
                                          env: Environment, ctrls: Dict[str, Controller], agent: RLAgent, reward_function: RewardFunction,
                                          metrics_collector: MetricsCollector, virtual_sim: VirtualSimulator,
                                          stability_calc: BaseStabilityCalculator,
                                          config: Dict[str, Any]
                                          ) -> Tuple[float, float, np.ndarray, bool, str, Dict[str, float]]:
        """Ejecuta un intervalo estándar y luego las simulaciones virtuales para Echo Baseline."""
        (real_reward, real_avg_stability, final_real_state,
         real_interval_done, real_term_reason) = self._run_standard_interval_steps(
            id_agent_decision, interval_start_sim_time, interval_duration_to_run, current_raw_state_at_interval_start,
            env, ctrls, agent, reward_function, metrics_collector, stability_calc, config
        )

        ### --- La lógica de Echo Baseline se complica con múltiples controladores. ---
        ### --- Se necesitaría una refactorización mayor aquí si se quiere usar Echo. ---
        '''
        echo_differential_rewards: Dict[str, float] = {}
        
        initial_state_for_virtual = current_raw_state_at_interval_start
        current_gains_real = ctrl.get_params()
        
        prev_kp = getattr(ctrl, 'previous_kp', current_gains_real.get('kp', np.nan))
        prev_ki = getattr(ctrl, 'previous_ki', current_gains_real.get('ki', np.nan))
        prev_kd = getattr(ctrl, 'previous_kd', current_gains_real.get('kd', np.nan))

        if any(pd.isna(g) for g in [prev_kp, prev_ki, prev_kd]):
            self.logger.warning(f"[SimMan:_run_echo_interval] Previous gains not available for Echo. Skipping virtual runs.")
        else:
            try:
                gains_cf_kp = {'kp': prev_kp, 'ki': current_gains_real['ki'], 'kd': current_gains_real['kd']}
                reward_kp_cf, stability_kp_cf  = virtual_sim.run_interval_simulation(initial_state_for_virtual, interval_start_sim_time, interval_duration_to_run, gains_cf_kp)
                echo_differential_rewards['kp'] = real_reward - reward_kp_cf
                
                gains_cf_ki = {'kp': current_gains_real['kp'], 'ki': prev_ki, 'kd': current_gains_real['kd']}
                reward_ki_cf, stability_ki_cf  = virtual_sim.run_interval_simulation(initial_state_for_virtual, interval_start_sim_time, interval_duration_to_run, gains_cf_ki)
                echo_differential_rewards['ki'] = real_reward - reward_ki_cf

                gains_cf_kd = {'kp': current_gains_real['kp'], 'ki': current_gains_real['ki'], 'kd': prev_kd}
                reward_kd_cf, stability_kd_cf = virtual_sim.run_interval_simulation(initial_state_for_virtual, interval_start_sim_time, interval_duration_to_run, gains_cf_kd)
                echo_differential_rewards['kd'] = real_reward - reward_kd_cf
            except Exception as e_virt:
                self.logger.error(f"[SimMan:_run_echo_interval] Error during virtual simulations: {e_virt}", exc_info=True)
        
        return real_reward, real_avg_stability, final_real_state, real_interval_done, real_term_reason, echo_differential_rewards
        '''
    # --- SECCIÓN 5: Métodos Auxiliares ---
    def _apply_actions_to_controller(self,
                                     controllers_dict: Dict[str, Controller],
                                     actions_to_apply_map: Dict[str, int],
                                     config_for_apply: Dict[str, Any]):
        """Aplica las acciones (cambios de ganancia) a todos los controladores."""
        agent_state_cfg = config_for_apply.get('environment', {}).get('agent', {}).get('params', {}).get('state_config', {})
        
        # Mapear name_objective_var a la configuración de su controlador
        controller_configs = config_for_apply.get('environment', {}).get('controller', {})
        objective_to_config_map = {
            cfg.get('params', {}).get('name_objective_var'): cfg
            for cfg in controller_configs.values() if isinstance(cfg, dict)
        }

        all_new_gains = {}
        # Iterar sobre las ganancias gestionadas por el agente
        for gain_name, gain_cfg in agent_state_cfg.items():
            if not gain_cfg.get('enabled_agent', False):
                continue

            # Extraer el tipo de ganancia y el objetivo (ej. 'kp', 'pendulum_angle')
            parts = gain_name.split('_')
            gain_type = parts[0]
            objective_var = "_".join(parts[1:])

            # Encontrar la configuración de pid_adaptation para este controlador
            ctrl_cfg = objective_to_config_map.get(objective_var, {})
            pid_adapt_cfg = ctrl_cfg.get('pid_adaptation', {})
            per_gain_delta = pid_adapt_cfg.get('per_gain_delta', False)
            gain_delta_config = pid_adapt_cfg.get('gain_delta', 1.0)
            
            gain_delta = float(gain_delta_config.get(gain_type, 1.0)) if per_gain_delta and isinstance(gain_delta_config, dict) else float(gain_delta_config)

            # Obtener el valor actual de la ganancia
            current_val = 0.0
            for ctrl in controllers_dict.values():
                if gain_name in ctrl.get_params():
                    current_val = ctrl.get_params()[gain_name]
                    break
            
            action = actions_to_apply_map.get(f'action_{gain_name}', 1)
            change = action - 1
            new_val = current_val + change * gain_delta
            
            all_new_gains[gain_name] = np.clip(new_val, gain_cfg.get('min', -np.inf), gain_cfg.get('max', np.inf))

        # Pasar el diccionario completo de nuevas ganancias a CADA controlador
        for ctrl in controllers_dict.values():
            ctrl.update_params(all_new_gains)
