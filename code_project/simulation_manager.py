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
        # Las dependencias logger y result_handler son críticas.
        # El contenedor DI debería asegurar que no sean None.

    # --- SECCIÓN 1: Resolución de Dependencias y Configuración ---
    def _resolve_dependencies(self) -> Tuple[
        Environment, RLAgent, Controller, RewardStrategy, Optional[VirtualSimulator],
        BaseStabilityCalculator, Dict[str, Any], Dict[str, Any], str  # main_config, data_directives, output_dir
    ]:
        """Resuelve todas las dependencias necesarias una sola vez."""
        self.logger.info("[SimMan:_resolve_dependencies] Resolving core simulation components...")
        env_instance = self.container.resolve(Environment)
        agent_instance = self.container.resolve(RLAgent)
        controller_instance = self.container.resolve(Controller)
        reward_strategy_instance = self.container.resolve(RewardStrategy)
        virtual_simulator_instance = self.container.resolve(Optional[VirtualSimulator]) # Es opcional
        stability_calculator_instance = self.container.resolve(BaseStabilityCalculator)
        main_config_dict = self.container.resolve(dict)
        output_dir_path = self.container.resolve(_OUTPUT_DIR_TOKEN_STR_)
        processed_data_directives_dict = self.container.resolve(_PROCESSED_DATA_DIRECTIVES_TOKEN_STR_)

        critical_deps = { # type: ignore
            "Environment": env_instance, "RLAgent": agent_instance, "Controller": controller_instance,
            "RewardStrategy": reward_strategy_instance, "BaseStabilityCalculator": stability_calculator_instance, 
            "main_config": main_config_dict,"output_dir": output_dir_path, "processed_data_directives": processed_data_directives_dict
        }
        missing = [name for name, dep in critical_deps.items() if dep is None]
        if missing:
            raise ValueError(f"[SimMan:_resolve_dependencies] Failed to resolve critical DI dependencies: {missing}")

        self.logger.info("[SimMan:_resolve_dependencies] Core dependencies resolved.")
        #self.logger.info(f"[SimMan:_resolve_dependencies] Core dependencies resolved. StabilityCalculator: {type(stability_calculator_instance).__name__ if stability_calculator_instance else 'None'}.")
        return (env_instance, agent_instance, controller_instance, reward_strategy_instance, virtual_simulator_instance, 
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
        aggregated_summary_data_global: List[Dict] = []
        current_episode_batch_data_buffer: List[Dict] = []
        
        env: Optional[Environment] = None
        agent: Optional[RLAgent] = None
        output_dir: Optional[str] = None
        data_directives: Optional[Dict[str, Any]] = None
        current_episode_idx_global = 0 # Para el finally

        try:
            (env, agent, ctrl, reward_strategy, virtual_sim, stability_calc,
             config, data_directives, output_dir) = self._resolve_dependencies()

            sim_run_params = self._extract_simulation_parameters(config, env)

            active_file_handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)] if sim_run_params['log_flush_frequency_episodes'] > 0 else []
            
            if reward_strategy.needs_virtual_simulation and virtual_sim is None:
                raise ValueError(f"RewardStrategy '{type(reward_strategy).__name__}' needs VirtualSimulator, but none was resolved.")

            self.logger.info(f"[SimMan:run] Starting simulation for {sim_run_params['max_episodes']} episodes...")

            for ep_idx in range(sim_run_params['max_episodes']):
                current_episode_idx_global = ep_idx # Actualizar para el finally
                ep_wall_start_time = time.time()
                
                metrics_collector = self.container.resolve(MetricsCollector) # Transient
                if metrics_collector is None:
                    # Este caso es improbable si el DI está bien configurado y ExtendedMetricsCollector está registrado.
                    # Si ocurre, es un error de configuración del DI.
                    self.logger.critical(f"[SimMan:run] CRITICAL: Failed to resolve MetricsCollector for ep {ep_idx}. Aborting episode run.")
                    # Podríamos decidir abortar la simulación completa aquí o intentar continuar sin métricas.
                    # Por robustez, podríamos usar un dummy, pero es mejor que falle si DI no funciona.
                    raise RuntimeError(f"Failed to resolve MetricsCollector for episode {ep_idx}")

                episode_summary_dict, episode_detailed_metrics = self._run_episode(
                    ep_idx, env, agent, ctrl, reward_strategy, virtual_sim, 
                    stability_calc, metrics_collector, config, 
                    sim_run_params, data_directives, ep_wall_start_time
                )
                aggregated_summary_data_global.append(episode_summary_dict)

                if data_directives.get('json_history_enabled', False) and episode_detailed_metrics:
                    current_episode_batch_data_buffer.append(episode_detailed_metrics)

                # Guardar datos detallados en lotes
                if data_directives.get('json_history_enabled', False) and sim_run_params['episodes_per_chunk'] > 0 and \
                   ((ep_idx + 1) % sim_run_params['episodes_per_chunk'] == 0 or ep_idx == sim_run_params['max_episodes'] - 1):
                    if current_episode_batch_data_buffer:
                        self.result_handler.save_episode_batch(current_episode_batch_data_buffer, output_dir, ep_idx)
                        current_episode_batch_data_buffer = []

                # Guardar estado del agente periódicamente
                if sim_run_params['agent_state_save_freq'] > 0 and (ep_idx + 1) % sim_run_params['agent_state_save_freq'] == 0:
                    self.result_handler.save_agent_state(agent, ep_idx, output_dir)

                # Flush logs periódicamente
                if active_file_handlers and (ep_idx + 1) % sim_run_params['log_flush_frequency_episodes'] == 0:
                    for handler in active_file_handlers: handler.flush()
                
                gc.collect()

            self.logger.info(f"[SimMan:run] All {sim_run_params['max_episodes']} episodes processed.")
            # Los datos detallados se guardan en lotes, por lo que devolvemos lista vacía aquí.
            return [], aggregated_summary_data_global

        except Exception as e_global_run:
            self.logger.critical(f"[SimMan:run] UNEXPECTED GLOBAL EXCEPTION during simulation (around ep ~{current_episode_idx_global}): {e_global_run}", exc_info=True)
            if data_directives and data_directives.get('json_history_enabled', False) and \
               current_episode_batch_data_buffer and output_dir and os.path.isdir(output_dir):
                self.logger.warning("[SimMan:run] Attempting to save partial batch data after critical error...")
                try:
                    self.result_handler.save_episode_batch(current_episode_batch_data_buffer, output_dir, current_episode_idx_global)
                except Exception as save_err:
                    self.logger.error(f"[SimMan:run] Failed to save partial batch data: {save_err}")
            raise
        finally:
            if active_file_handlers: # type: ignore
                self.logger.info("[SimMan:run] Performing final log flush...")
                for handler_final_flush in active_file_handlers: handler_final_flush.flush() # type: ignore
            self.logger.info("[SimMan:run] --- Simulation Run Finished ---")

    # --- SECCIÓN 3: Orquestación de Episodio Individual ---
    def _run_episode(self,
                     episode_idx: int,
                     env: Environment, agent: RLAgent, ctrl: Controller,
                     reward_strategy: RewardStrategy, virtual_sim: Optional[VirtualSimulator],
                     stability_calc: BaseStabilityCalculator,
                     metrics_collector: MetricsCollector, config: Dict[str, Any],
                     sim_run_params: Dict[str, Any], # Parámetros de simulación ya extraídos
                     data_directives: Dict[str, Any],
                     ep_wall_start_time: float
                    ) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
        """Orquesta la ejecución de un solo episodio, devolviendo su resumen y métricas detalladas."""
        max_eps_cfg = sim_run_params['max_episodes']
        self.logger.info(f"--- [ Episode {episode_idx}/{max_eps_cfg-1} Starting ] ---")

        # 1. Inicialización del Episodio (Reset y primer estado/acción)
        initial_conditions_cfg = config.get('environment', {}).get('initial_conditions', {}).get('x0')
        current_raw_state_s = env.reset(initial_conditions_cfg) # Resetea env, system, controller, agent
        metrics_collector.reset(episode_idx)
        self._log_initial_metrics(metrics_collector, current_raw_state_s, ctrl, agent, config, stability_calc)
        
        current_sim_time_sec = 0.0
        episode_done_flag = False
        episode_term_reason = "unknown"
        decision_counter = 0

        # Bucle de Intervalos de Decisión
        while not episode_done_flag and current_sim_time_sec < sim_run_params['episode_duration_sec']:
            # Calcular duración real de este intervalo (puede ser menor al final del episodio)
            interval_remaining_time = sim_run_params['episode_duration_sec'] - current_sim_time_sec
            current_interval_duration_actual = min(sim_run_params['decision_interval_sec'], interval_remaining_time)
            if current_interval_duration_actual < sim_run_params['sim_dt_sec'] / 2.0: # Evitar intervalos demasiado pequeños
                break

            decision_counter += 1
            metrics_collector.log('id_agent_decision', decision_counter)
            learn_select_boundary_start_time = time.time()

            # 2. Obtener Estado S del Agente y Seleccionar Acción A'
            current_agent_s_dict = agent.build_agent_state(current_raw_state_s, ctrl, sim_run_params['state_definition_config'])
            actions_a_prime_dict = agent.select_action(current_agent_s_dict)
            
            # 3. Aplicar Acción A' al Controlador (las ganancias se mantendrán durante el intervalo)
            self._apply_actions_to_controller(ctrl, actions_a_prime_dict, config)

            # 4. Ejecutar Pasos de Simulación del Intervalo
            interval_reward_val: float; avg_stability_score_interval: float
            final_env_state_s_prime_raw: np.ndarray; interval_done_flag: bool; term_reason_from_interval: str
            echo_rewards_map: Optional[Dict[str, float]] = None # Para EchoBaseline

            if reward_strategy.needs_virtual_simulation and virtual_sim:
                (interval_reward_val, avg_stability_score_interval, final_env_state_s_prime_raw,
                 interval_done_flag, term_reason_from_interval, echo_rewards_map) = \
                    self._run_echo_baseline_interval_steps(
                        current_sim_time_sec, current_interval_duration_actual, current_raw_state_s,
                        env, ctrl, agent, metrics_collector, virtual_sim, stability_calc, config, actions_a_prime_dict
                    )
            else:
                (interval_reward_val, avg_stability_score_interval, final_env_state_s_prime_raw,
                 interval_done_flag, term_reason_from_interval) = \
                    self._run_standard_interval_steps(
                        current_sim_time_sec, current_interval_duration_actual, current_raw_state_s,
                        env, ctrl, agent, metrics_collector, stability_calc, config, actions_a_prime_dict
                    )
            
            # Actualizar estado y tiempo globales del episodio
            current_raw_state_s = final_env_state_s_prime_raw
            logged_times = metrics_collector.get_metrics().get('time', [current_sim_time_sec])
            current_sim_time_sec = logged_times[-1] if logged_times else current_sim_time_sec # Tiempo real simulado
            
            episode_done_flag = interval_done_flag # Si el intervalo marcó fin (límite, meta, tiempo)
            if episode_done_flag and episode_term_reason == "unknown":
                episode_term_reason = term_reason_from_interval

            # 5. Obtener Estado S' del Agente y Aprender
            next_agent_s_prime_dict = agent.build_agent_state(current_raw_state_s, ctrl, sim_run_params['state_definition_config'])

            agent.learn(
                current_agent_s_dict = current_agent_s_dict,        # S (estado al inicio del intervalo)
                taken_actions_map = actions_a_prime_dict,                # A' (acción que llevó a S')
                interval_reward_information = interval_reward_val,
                interval_stability_information = avg_stability_score_interval,
                next_agent_s_prime_dict = next_agent_s_prime_dict,  # S'
                current_controller_instance = ctrl,
                is_episode_done = episode_done_flag # True si el intervalo marcó el fin
            )
            
            # Loguear métricas post-aprendizaje
            self._log_decision_boundary_metrics(metrics_collector, agent, next_agent_s_prime_dict, learn_select_boundary_start_time)
            
            # 6. Comprobar terminación temprana solicitada por el agente (después de aprender)
            if not episode_done_flag and agent.early_termination_enabled and agent.should_episode_terminate_early():
                self.logger.info(f"[SimMan:_run_episode Ep {episode_idx}] Agent requested early termination at t={current_sim_time_sec:.3f}s.")
                episode_done_flag = True
                if episode_term_reason == "unknown": episode_term_reason = "agent_early_termination"
            
            if episode_done_flag: 
                # Si el episodio terminó, current_sim_time_sec tiene el tiempo final
                # Asegurarse de que 'episode_time_duration_sec' refleje esto para el resumen
                # Esto es crucial si el episodio termina antes del episode_duration_sec configurado
                
                # Obtener la lista actual de tiempos del metrics_collector
                time_list_from_metrics = metrics_collector.get_metrics().get('time', [])
                actual_final_sim_time = current_sim_time_sec # El último tiempo del bucle
                if time_list_from_metrics: # Si hay tiempos logueados, el último es el más preciso
                    actual_final_sim_time = time_list_from_metrics[-1]                
                pass # Salir del bucle de intervalos

        # --- Finalización del Episodio ---
        final_metrics_for_episode = metrics_collector.get_metrics()
        try:
            env.update_reward_and_stability_calculator_stats(final_metrics_for_episode, episode_idx)
        except Exception as e_update_rew_stats:
            self.logger.error(f"[SimMan:_run_episode Ep {episode_idx}] Error updating reward calculator stats: {e_update_rew_stats}", exc_info=True)

        # Loguear 'episode_duration_sec' al final, usando el tiempo real simulado.
        if 'episode_duration_sec' not in final_metrics_for_episode or not final_metrics_for_episode['episode_duration_sec']:
             len_any_metric = len(next(iter(final_metrics_for_episode.values()), []))
             final_metrics_for_episode['episode_duration_sec'] = [current_sim_time_sec] * len_any_metric if len_any_metric > 0 else [current_sim_time_sec]
        
        episode_summary_dict = self._finalize_episode_metrics_and_summary(
            episode_idx, final_metrics_for_episode, episode_term_reason,
            ep_wall_start_time, ctrl, agent, data_directives, current_sim_time_sec
        )
        
        # El log que usa episode_summary_dict
        self.logger.info(f"[SimMan:_finalize_episode] Ep {episode_idx} Summary: Term='{episode_summary_dict.get('termination_reason', '?')}', "
                        f"R={episode_summary_dict.get('total_reward', np.nan):.2f}, Perf={episode_summary_dict.get('performance', np.nan):.2f}, "
                        f"Stab={episode_summary_dict.get('avg_stability_score', np.nan):.3f}, T={episode_summary_dict.get('episode_time_duration_sec', np.nan):.2f}s, "
                        f"Decisions={episode_summary_dict.get('total_agent_decisions', 0)}")
        
        self.logger.info(f"[SimMan:_finalize_episode] Ep {episode_idx} Summary: Term='{episode_summary_dict.get('termination_reason', '?')}', "
                        f"Gains(Kp={episode_summary_dict.get('final_kp', np.nan):.2f}, "
                        f"Ki={episode_summary_dict.get('final_ki', np.nan):.2f}, Kd={episode_summary_dict.get('final_kd', np.nan):.3f}), "
                        f"Dur={episode_summary_dict.get('episode_wall_time_sec', np.nan):.2f}s")
        self.logger.info(f"--- [ Episode {episode_idx} Finished: {episode_term_reason} ] ---")
        return episode_summary_dict, final_metrics_for_episode

    # --- SECCIÓN 4: Ejecución de Intervalos de Simulación (Real y Virtual) ---
    def _run_standard_interval_steps(self,
                                     interval_start_sim_time: float,
                                     interval_duration_to_run: float,
                                     current_raw_state_at_interval_start: np.ndarray,
                                     env: Environment, ctrl: Controller, agent: RLAgent,
                                     metrics_collector: MetricsCollector,
                                     stability_calc: BaseStabilityCalculator,
                                     config: Dict[str, Any],
                                     actions_applied_during_interval: Dict[str, int]
                                     ) -> Tuple[float, float, np.ndarray, bool, str]:
        """Ejecuta los pasos de simulación para un intervalo estándar, devuelve (reward_total, avg_stability, final_state, done, reason)."""
        accumulated_interval_reward = 0.0
        stability_scores_in_interval: List[float] = []
        is_interval_terminal = False
        termination_reason_interval = "unknown" # Razón de terminación *dentro* de este intervalo
        last_state_in_interval = np.copy(current_raw_state_at_interval_start)
        sim_dt = env.dt # type: ignore

        num_steps_this_interval = max(1, int(round(interval_duration_to_run / sim_dt)))

        for step_idx in range(num_steps_this_interval):
            current_step_sim_time = round(interval_start_sim_time + (step_idx + 1) * sim_dt, 6)
            
            # Environment.step() devuelve (next_state, reward, stability_score, info_dict_force)
            next_state_from_env, reward_at_step, stability_score_at_step, force_applied_in_step = env.step() # type: ignore
            
            last_state_in_interval = next_state_from_env
            accumulated_interval_reward += float(reward_at_step) if pd.notna(reward_at_step) and np.isfinite(reward_at_step) else 0.0
            stability_scores_in_interval.append(float(stability_score_at_step) if pd.notna(stability_score_at_step) and np.isfinite(stability_score_at_step) else 0.0)

            self._log_step_metrics(metrics_collector, current_step_sim_time, next_state_from_env,
                                   reward_at_step, stability_score_at_step, force_applied_in_step,
                                   ctrl, agent, config, actions_applied_during_interval, stability_calc) # type: ignore

            # Comprobar terminación del episodio (por Environment)
            limit_exceeded, goal_reached, _ = env.check_termination()
            
            # Comprobar si se alcanzó el tiempo máximo del episodio (manejado por SimulationManager)
            max_ep_duration_from_config = config.get('environment', {}).get('simulation', {}).get('episode_duration_sec', 5.0)
            time_limit_for_episode_reached = (current_step_sim_time >= max_ep_duration_from_config - (sim_dt / 2.0))

            if limit_exceeded or goal_reached or time_limit_for_episode_reached:
                is_interval_terminal = True # Marcar que el intervalo debe terminar.
                if termination_reason_interval == "unknown": # Asignar la primera razón que cause terminación.
                    if limit_exceeded: termination_reason_interval = "limit_exceeded"
                    elif goal_reached: termination_reason_interval = "goal_reached"
                    elif time_limit_for_episode_reached: termination_reason_interval = "time_limit"
                self.logger.debug(f"[SimMan:_run_std_interval] Interval ending: {termination_reason_interval} at step_time={current_step_sim_time:.4f}s")
                break # Salir del bucle de pasos del intervalo

        avg_stability_score_for_interval = np.nanmean(stability_scores_in_interval) if stability_scores_in_interval else 1.0
        if not np.isfinite(avg_stability_score_for_interval): avg_stability_score_for_interval = 1.0 # Default a neutro
        
        return accumulated_interval_reward, avg_stability_score_for_interval, last_state_in_interval, is_interval_terminal, termination_reason_interval

    def _run_echo_baseline_interval_steps(self,
                                          interval_start_sim_time: float,
                                          interval_duration_to_run: float,
                                          current_raw_state_at_interval_start: np.ndarray,
                                          env: Environment, ctrl: Controller, agent: RLAgent,
                                          metrics_collector: MetricsCollector, virtual_sim: VirtualSimulator,
                                          stability_calc: BaseStabilityCalculator,
                                          config: Dict[str, Any],
                                          actions_applied_during_interval: Dict[str, int]
                                          ) -> Tuple[float, float, np.ndarray, bool, str, Dict[str, float]]:
        """Ejecuta un intervalo estándar y luego las simulaciones virtuales para Echo Baseline."""
        (real_reward, real_avg_stability, final_real_state,
         real_interval_done, real_term_reason) = self._run_standard_interval_steps(
            interval_start_sim_time, interval_duration_to_run, current_raw_state_at_interval_start,
            env, ctrl, agent, metrics_collector, stability_calc, config, actions_applied_during_interval
        )

        echo_differential_rewards: Dict[str, float] = {}
        
        initial_state_for_virtual = current_raw_state_at_interval_start # Usar S (estado al inicio del intervalo real)
        current_gains_real = ctrl.get_params()
        # Asumimos que el controlador tiene 'previous_kp', etc. como en PIDController.
        # Esto es un acoplamiento implícito. Una mejor forma sería que el agente
        # o la estrategia de recompensa manejen el estado "anterior" de las ganancias si es necesario.
        # Por ahora, se mantiene como en el código original para Echo.
        prev_kp = getattr(ctrl, 'previous_kp', current_gains_real.get('kp', np.nan))
        prev_ki = getattr(ctrl, 'previous_ki', current_gains_real.get('ki', np.nan))
        prev_kd = getattr(ctrl, 'previous_kd', current_gains_real.get('kd', np.nan))

        if any(pd.isna(g) for g in [prev_kp, prev_ki, prev_kd]):
            self.logger.warning(f"[SimMan:_run_echo_interval] Previous gains not available for Echo. Skipping virtual runs. Prev: kp={prev_kp}, ki={prev_ki}, kd={prev_kd}")
        else:
            try:
                gains_cf_kp = {'kp': prev_kp, 'ki': current_gains_real['ki'], 'kd': current_gains_real['kd']}
                reward_kp_cf, stability_kp_cf  = virtual_sim.run_interval_simulation(initial_state_for_virtual, interval_start_sim_time, interval_duration_to_run, gains_cf_kp)
                echo_differential_rewards['kp'] = real_reward - reward_kp_cf
                metrics_collector.log('virtual_stability_score_kp_cf', stability_kp_cf  if pd.notna(stability_kp_cf ) and np.isfinite(stability_kp_cf ) else np.nan)

                gains_cf_ki = {'kp': current_gains_real['kp'], 'ki': prev_ki, 'kd': current_gains_real['kd']}
                reward_ki_cf, stability_ki_cf  = virtual_sim.run_interval_simulation(initial_state_for_virtual, interval_start_sim_time, interval_duration_to_run, gains_cf_ki)
                echo_differential_rewards['ki'] = real_reward - reward_ki_cf
                metrics_collector.log('virtual_stability_score_ki_cf', stability_ki_cf if pd.notna(stability_ki_cf) and np.isfinite(stability_ki_cf) else np.nan)

                gains_cf_kd = {'kp': current_gains_real['kp'], 'ki': current_gains_real['ki'], 'kd': prev_kd}
                reward_kd_cf, stability_kd_cf = virtual_sim.run_interval_simulation(initial_state_for_virtual, interval_start_sim_time, interval_duration_to_run, gains_cf_kd)
                echo_differential_rewards['kd'] = real_reward - reward_kd_cf
                metrics_collector.log('virtual_stability_score_kd_cf', stability_kd_cf if pd.notna(stability_kd_cf) and np.isfinite(stability_kd_cf) else np.nan)
                
                # Loguear R_diff
                if hasattr(metrics_collector, 'log_virtual_rewards'): metrics_collector.log_virtual_rewards(echo_differential_rewards) # type: ignore
                
            except Exception as e_virt:
                self.logger.error(f"[SimMan:_run_echo_interval] Error during virtual simulations: {e_virt}", exc_info=True)
                if hasattr(metrics_collector, 'log_virtual_rewards'): metrics_collector.log_virtual_rewards({}) # type: ignore
                metrics_collector.log('virtual_stability_score_kp_cf', np.nan); metrics_collector.log('virtual_stability_score_ki_cf', np.nan); metrics_collector.log('virtual_stability_score_kd_cf', np.nan)
        
        return real_reward, real_avg_stability, final_real_state, real_interval_done, real_term_reason, echo_differential_rewards

    # --- SECCIÓN 5: Métodos Auxiliares de Logging y Aplicación de Acciones ---
    def _log_initial_metrics(self,
                             metrics_collector: MetricsCollector, initial_raw_state: np.ndarray,
                             ctrl: Controller, agent: RLAgent, config: Dict[str, Any], 
                             stability_calc: BaseStabilityCalculator):
        """Loguea las métricas iniciales al comienzo de un episodio."""
        self.logger.debug(f"[SimMan:_log_initial_metrics] Logging initial metrics. State: {np.round(initial_raw_state[:4],3)}")
        metrics_collector.log('time', 0.0) # Tiempo inicial es 0
        # Llamar a _log_step_metrics con valores iniciales/default para recompensa, fuerza, etc.
        self._log_step_metrics(metrics_collector, 0.0, initial_raw_state, 0.0, 1.0, 0.0, ctrl, agent, config, {}, stability_calc) # type: ignore
        # Las métricas de decisión/aprendizaje se loguearán después de la primera decisión
        boundary_metrics_to_init_nan = [
            'learn_select_duration_ms', 'id_agent_decision', 'td_error_kp', 'td_error_ki', 'td_error_kd',
            'q_value_max_kp', 'q_value_max_ki', 'q_value_max_kd',
            'q_visit_count_state_kp', 'q_visit_count_state_ki', 'q_visit_count_state_kd',
            'virtual_reward_kp', 'virtual_reward_ki', 'virtual_reward_kd',
            'virtual_stability_score_kp_cf', 'virtual_stability_score_ki_cf', 'virtual_stability_score_kd_cf',
            'baseline_value_kp', 'baseline_value_ki', 'baseline_value_kd'
        ]
        for metric_key_nan in boundary_metrics_to_init_nan: metrics_collector.log(metric_key_nan, np.nan)
        
        # Loguear estado inicial de ET y Q-tables/Baselines
        agent_state_def_cfg_log = config.get('environment', {}).get('agent', {}).get('params', {}).get('state_config', {})
        current_agent_s_for_log = agent.build_agent_state(initial_raw_state, ctrl, agent_state_def_cfg_log)
        if hasattr(metrics_collector, 'log_early_termination_metrics'): metrics_collector.log_early_termination_metrics(agent) # type: ignore
        if hasattr(metrics_collector, 'log_q_values'): metrics_collector.log_q_values(agent, current_agent_s_for_log) # type: ignore
        if hasattr(metrics_collector, 'log_q_visit_counts'): metrics_collector.log_q_visit_counts(agent, current_agent_s_for_log) # type: ignore
        if hasattr(metrics_collector, 'log_baselines'): metrics_collector.log_baselines(agent, current_agent_s_for_log) # type: ignore
        if hasattr(metrics_collector, 'log_adaptive_stats'): metrics_collector.log_adaptive_stats({}) # type: ignore # Stats adaptativas se loguean por paso si RewardFunc las expone


    def _log_step_metrics(self,
                          metrics_collector: MetricsCollector, current_sim_time: float,
                          current_raw_state: np.ndarray, reward: float, stability_score: float, force: float,
                          ctrl: Controller, agent: RLAgent, config: Dict[str, Any],
                          actions_in_interval: Dict[str, int], # Acciones A' que están activas
                          stability_calc_instance: Optional[BaseStabilityCalculator] = None): # Para stats adaptativas
        """Loguea las métricas de un paso de simulación individual."""
        metrics_collector.log('time', current_sim_time)
        metrics_collector.log('cart_position', current_raw_state[0]); metrics_collector.log('cart_velocity', current_raw_state[1])
        metrics_collector.log('pendulum_angle', current_raw_state[2]); metrics_collector.log('pendulum_velocity', current_raw_state[3])

        gains = ctrl.get_params()
        metrics_collector.log('kp', gains.get('kp', np.nan)); metrics_collector.log('ki', gains.get('ki', np.nan)); metrics_collector.log('kd', gains.get('kd', np.nan))
        setpoint = config.get('environment',{}).get('controller',{}).get('params',{}).get('setpoint', 0.0)
        metrics_collector.log('error', current_raw_state[2] - setpoint)
        # *Asume PIDController
        metrics_collector.log('integral_error', getattr(ctrl, 'accumulated_integral_error', np.nan))
        metrics_collector.log('derivative_error', getattr(ctrl, 'current_derivative_error', np.nan))
        metrics_collector.log('proportional_term', getattr(ctrl, 'last_proportional_term', np.nan))
        metrics_collector.log('integral_term', getattr(ctrl, 'last_integral_term', np.nan))
        metrics_collector.log('derivative_term', getattr(ctrl, 'last_derivative_term', np.nan))

        metrics_collector.log('action_kp', actions_in_interval.get('kp', np.nan)) # Estas son las A'
        metrics_collector.log('action_ki', actions_in_interval.get('ki', np.nan))
        metrics_collector.log('action_kd', actions_in_interval.get('kd', np.nan))

        metrics_collector.log('epsilon', agent.epsilon); metrics_collector.log('learning_rate', agent.learning_rate)
        pid_adapt_cfg = config.get('environment', {}).get('controller', {}).get('pid_adaptation', {})
        gain_delta_cfg = pid_adapt_cfg.get('gain_delta', 5.0)
        per_gain_delta_cfg = pid_adapt_cfg.get('per_gain_delta', False)
        if per_gain_delta_cfg and isinstance(gain_delta_cfg, dict):
            metrics_collector.log('gain_delta_kp', float(gain_delta_cfg.get('kp', np.nan)))
            # ... para ki, kd
        elif isinstance(gain_delta_cfg, (int,float)): metrics_collector.log('gain_delta', float(gain_delta_cfg))
        else: metrics_collector.log('gain_delta', np.nan)

        metrics_collector.log('reward', float(reward) if pd.notna(reward) and np.isfinite(reward) else 0.0)
        all_rewards = metrics_collector.get_metrics().get('reward', [])
        metrics_collector.log('cumulative_reward', np.nansum(np.array(all_rewards, dtype=float)))
        metrics_collector.log('force', float(force) if pd.notna(force) and np.isfinite(force) else 0.0)
        metrics_collector.log('stability_score', float(stability_score) if pd.notna(stability_score) and np.isfinite(stability_score) else 0.0)

        if hasattr(metrics_collector, 'log_adaptive_stats') and stability_calc_instance:
            adaptive_stats = {}
            try: # El stability_calculator está dentro de RewardFunction
                stability_calc = getattr(stability_calc_instance, 'stability_measure_instance', None)
                if stability_calc and hasattr(stability_calc, 'get_current_adaptive_stats'):
                    adaptive_stats = stability_calc.get_current_adaptive_stats() # type: ignore
            except Exception: pass
            metrics_collector.log_adaptive_stats(adaptive_stats) # type: ignore

    def _log_decision_boundary_metrics(self, metrics_collector: MetricsCollector, agent: RLAgent,
                                       next_agent_s_prime_dict: Dict[str, Any], boundary_start_time: float):
        """Loguea métricas específicas del límite de decisión (post-learn, post-select)."""
        if hasattr(metrics_collector, 'log_td_errors'): metrics_collector.log_td_errors(agent.get_last_td_errors()) # type: ignore
        if hasattr(metrics_collector, 'log_q_values'): metrics_collector.log_q_values(agent, next_agent_s_prime_dict) # type: ignore
        if hasattr(metrics_collector, 'log_baselines'): metrics_collector.log_baselines(agent, next_agent_s_prime_dict) # type: ignore
        if hasattr(metrics_collector, 'log_q_visit_counts'): metrics_collector.log_q_visit_counts(agent, next_agent_s_prime_dict) # type: ignore
        if hasattr(metrics_collector, 'log_early_termination_metrics'): metrics_collector.log_early_termination_metrics(agent) # type: ignore
        metrics_collector.log('learn_select_duration_ms', (time.time() - boundary_start_time) * 1000)

    def _apply_actions_to_controller(self,
                                     ctrl_instance: Controller,
                                     actions_to_apply_map: Dict[str, int], # ej: {'kp': 0, 'ki': 1, 'kd': 2}
                                     config_for_apply: Dict[str, Any]):
        """Aplica las acciones (cambios de ganancia) al controlador."""
        current_gains = ctrl_instance.get_params()
        
        pid_adapt_config = config_for_apply.get('environment', {}).get('controller', {}).get('pid_adaptation', {})
        gain_delta_config_val = pid_adapt_config.get('gain_delta', 1.0) # 'gain_delta' es el nuevo nombre
        per_gain_delta_is_active = pid_adapt_config.get('per_gain_delta', False) # 'per_gain_delta' es el nuevo nombre

        # Determinar el delta para cada ganancia
        delta_values = {'kp': 0.0, 'ki': 0.0, 'kd': 0.0}
        if per_gain_delta_is_active and isinstance(gain_delta_config_val, dict):
            for gain in ['kp', 'ki', 'kd']: delta_values[gain] = float(gain_delta_config_val.get(gain, 0.0))
        elif isinstance(gain_delta_config_val, (int, float)):
            common_delta_val = float(gain_delta_config_val)
            for gain in ['kp', 'ki', 'kd']: delta_values[gain] = common_delta_val
        
        new_gains = {}
        for gain_name_key, current_gain_val in current_gains.items():
            action_for_gain = actions_to_apply_map.get(gain_name_key, 1) # Default: maintain (índice 1)
            # action 0 (decrease), 1 (maintain), 2 (increase) => (action - 1) da -1, 0, 1
            change_multiplier = action_for_gain - 1
            new_gains[gain_name_key] = current_gain_val + change_multiplier * delta_values.get(gain_name_key, 0.0)

        # Aplicar límites de ganancia (de state_config del agente)
        agent_state_cfg_limits = config_for_apply.get('environment', {}).get('agent', {}).get('params', {}).get('state_config', {})
        for gain_name_limit, gain_val_new in new_gains.items():
            min_lim = agent_state_cfg_limits.get(gain_name_limit, {}).get('min', -np.inf)
            max_lim = agent_state_cfg_limits.get(gain_name_limit, {}).get('max', np.inf)
            new_gains[gain_name_limit] = np.clip(gain_val_new, min_lim, max_lim)
            
        ctrl_instance.update_params(new_gains.get('kp',0.0), new_gains.get('ki',0.0), new_gains.get('kd',0.0))
        # self.logger.debug(f"[SimMan:_apply_actions] Controller gains updated to: Kp={new_gains.get('kp'):.3f}, Ki={new_gains.get('ki'):.3f}, Kd={new_gains.get('kd'):.3f}")

    def _finalize_episode_metrics_and_summary(self,
                                              episode_idx_finalize: int,
                                              final_episode_metrics_dict: Dict[str, List[Any]],
                                              term_reason_finalize: str,
                                              ep_wall_start_time_finalize: float,
                                              ctrl_finalize: Controller, agent_finalize: RLAgent,
                                              data_directives_finalize: Dict[str, Any],
                                              actual_episode_sim_duration: float 
                                             ) -> Dict[str, Any]:
        """Prepara el diccionario de resumen del episodio y añade métricas finales."""
        # Añadir información final al diccionario de métricas detalladas ANTES de resumir
        len_any_metric_list = len(next(iter(final_episode_metrics_dict.values()), []))
        if len_any_metric_list == 0: len_any_metric_list = 1 # Para que la replicación funcione si está vacío

        final_gains_dict = ctrl_finalize.get_params()

        # Calcular total_reward y performance
        total_reward_ep_val = np.nansum(np.array(final_episode_metrics_dict.get('reward', [np.nan]), dtype=float))
        
        # Usar el actual_episode_sim_duration pasado
        performance_ep_val = total_reward_ep_val / actual_episode_sim_duration if pd.notna(actual_episode_sim_duration) and actual_episode_sim_duration > 1e-9 else np.nan
        
        # Calcular avg_stability_score si no está
        avg_stability_score_val = np.nan # Default
        if 'avg_stability_score' in final_episode_metrics_dict and final_episode_metrics_dict['avg_stability_score']:
            avg_stability_score_val = final_episode_metrics_dict['avg_stability_score'][-1] # Tomar el valor ya calculado
        elif 'stability_score' in final_episode_metrics_dict: # Calcular si no está pre-calculado
            stability_scores_list = final_episode_metrics_dict.get('stability_score', [])
            if stability_scores_list and not all(pd.isna(s) for s in stability_scores_list if pd.notna(s)): # Evitar mean of empty slice
                 avg_stability_score_val = np.nanmean(np.array(stability_scores_list, dtype=float))
            final_episode_metrics_dict['avg_stability_score'] = [avg_stability_score_val] * len_any_metric_list

        summary_extra_data = {
            'termination_reason': term_reason_finalize,
            'episode_wall_time_sec': time.time() - ep_wall_start_time_finalize,
            'final_kp': final_gains_dict.get('kp', np.nan),
            'final_ki': final_gains_dict.get('ki', np.nan),
            'final_kd': final_gains_dict.get('kd', np.nan),
            'total_agent_decisions': final_episode_metrics_dict.get('id_agent_decision', [0])[-1] if final_episode_metrics_dict.get('id_agent_decision') else 0,
            'final_epsilon': agent_finalize.epsilon,
            'final_learning_rate': agent_finalize.learning_rate,
            'total_reward': total_reward_ep_val,
            'performance': performance_ep_val,
            'episode_time_duration_sec': actual_episode_sim_duration, # <<< USAR EL VALOR PASADO
            'avg_stability_score': avg_stability_score_val if pd.notna(avg_stability_score_val) else np.nan, # Asegurar que esté
            '_agent_defining_vars': agent_finalize.get_agent_defining_vars()
        }
        for key, val in summary_extra_data.items():
            # Replicar para que tenga la misma longitud que otras listas de métricas
            if key not in final_episode_metrics_dict or not isinstance(final_episode_metrics_dict[key], list) or not final_episode_metrics_dict[key] :
                final_episode_metrics_dict[key] = [val] * len_any_metric_list
            elif isinstance(final_episode_metrics_dict[key], list) and len(final_episode_metrics_dict[key]) == len_any_metric_list:
                final_episode_metrics_dict[key][-1] = val # Actualizar solo el último valor para que get_last_valid_value lo tome

        # Crear el resumen usando las directivas
        episode_summary_output_dict = summarize_episode(final_episode_metrics_dict, data_directives_finalize)
        episode_summary_output_dict['episode'] = episode_idx_finalize # Asegurar que el ID esté

        # Asegurar que las claves que el log espera estén en el summary_output_dict
        # Si summarize_episode no las incluyó (porque no estaban en direct_columns), las añadimos aquí
        # con los valores de summary_extra_data o los calculados.
        for key_to_ensure in ['termination_reason', 'total_reward', 'performance', 'avg_stability_score', 'episode_time_duration_sec', 'total_agent_decisions', 'final_kp', 'final_ki', 'final_kd', 'episode_wall_time_sec']:
            if key_to_ensure not in episode_summary_output_dict:
                if key_to_ensure in summary_extra_data:
                    episode_summary_output_dict[key_to_ensure] = summary_extra_data[key_to_ensure]
                elif key_to_ensure == 'avg_stability_score' and 'avg_stability_score' in final_episode_metrics_dict: # Tomar el calculado antes
                     episode_summary_output_dict[key_to_ensure] = final_episode_metrics_dict['avg_stability_score'][-1]

        return episode_summary_output_dict