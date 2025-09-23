import time
import numpy as np
import logging
import pandas as pd # Needed for isnan checks
from typing import Dict, Any, Optional, List, Tuple, Union

# Import components for type hinting
from interfaces.environment import Environment
from interfaces.rl_agent import RLAgent
from interfaces.controller import Controller
from interfaces.metrics_collector import MetricsCollector
from interfaces.virtual_simulator import VirtualSimulator
from result_handler import save_episode_batch, save_agent_state

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

def run_simulation(components: Dict[str, Any], config: Dict[str, Any], results_folder: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Runs the main simulation loop over episodes and steps, performing detailed logging.

    Args:
        components: Dictionary containing initialized components.
        config: The main configuration dictionary.
        results_folder: Path to the folder where results will be saved.

    Returns:
        Tuple containing:
            - all_episodes_data (List[Dict]): Detailed data collected for each episode.
            - summary_data (List[Dict]): Summary statistics for each episode.
    """
    logging.info("--- Starting Simulation Run ---")
    start_time_sim = time.time()

    # --- Extract Components ---
    env: Environment = components.get('env')
    agent: RLAgent = components.get('agent')
    controller: Controller = components.get('controller')
    metrics_collector: MetricsCollector = components.get('metrics_collector')
    virtual_simulator: Optional[VirtualSimulator] = components.get('virtual_simulator')

    if not all([env, agent, metrics_collector, controller]):
        logging.error("CRITICAL: Missing essential components for simulation run.")
        return [], []

    # --- Extract Simulation Parameters ---
    try:
        simulation_cfg = config.get('simulation', {})
        env_cfg = config.get('environment', {})
        agent_cfg_full = env_cfg.get('agent', {})
        agent_params_cfg = agent_cfg_full.get('params', {})
        init_cond_cfg = config.get('initial_conditions', {})
        reward_setup_cfg = env_cfg.get('reward_setup', {})
        pid_adapt_cfg = config.get('pid_adaptation', {}) # Get PID adaptation config

        episodes_per_file = simulation_cfg.get('episodes_per_file', 200)
        decision_interval = env_cfg.get('decision_interval', 0.01)
        dt = env_cfg.get('dt', 0.001)
        total_time = env_cfg.get('total_time', 5.0)
        max_steps = int(round(total_time / dt)) if dt > 0 else 0
        max_episodes = env_cfg.get('max_episodes', 1000)
        initial_state_vector = init_cond_cfg.get('x0')
        if initial_state_vector is None: raise KeyError("'initial_conditions: x0' is missing")

        current_reward_mode = reward_setup_cfg.get('learning_strategy', 'global')
        gain_step_config = pid_adapt_cfg.get('gain_step', 1.0) # Default or dict
        variable_step = pid_adapt_cfg.get('variable_step', False)
        # Configuración de límites de ganancias desde el agente
        gain_limits_cfg = agent_cfg_full.get('params', {}).get('state_config', {})

    except KeyError as e:
        logging.error(f"CRITICAL: Missing essential configuration key: {e}. Aborting.", exc_info=True)
        return [], []
    except Exception as e:
        logging.error(f"CRITICAL: Error reading simulation parameters: {e}. Aborting.", exc_info=True)
        return [], []

    # --- Main Simulation Loop ---
    episode_batch, summary_data, all_episodes_data = [], [], []
    total_virtual_sim_time = 0.0

    logging.info(f"Starting simulation for {max_episodes} episodes...")
    for episode in range(max_episodes):
        episode_start_time = time.time()
        logging.info(f"--- Starting Episode: {episode}/{max_episodes-1} ---")

        # --- Reset Environment and Agent ---
        try:
            state_vector = env.reset(initial_conditions=initial_state_vector)
        except Exception as e:
            logging.error(f"Error during env reset ep {episode}: {e}. Skipping.", exc_info=True)
            continue

        # --- Initialize Episode Variables ---
        metrics_collector.reset(episode_id=episode)
        cumulative_reward, interval_reward = 0.0, 0.0
        interval_stability_scores = []
        next_decision_time = decision_interval if decision_interval > dt else dt
        agent_decision_count = 0 # Counter for decisions in this episode

        # --- Log Initial State (t=0) ---
        metrics_collector.log('time', 0.0)
        metrics_collector.log('cart_position', state_vector[0])
        metrics_collector.log('cart_velocity', state_vector[1])
        metrics_collector.log('pendulum_angle', state_vector[2])
        metrics_collector.log('pendulum_velocity', state_vector[3])
        try:
            controller_setpoint = getattr(controller, 'setpoint', 0.0)
            metrics_collector.log('error', state_vector[2] - controller_setpoint)
            metrics_collector.log('kp', getattr(controller, 'kp', np.nan))
            metrics_collector.log('ki', getattr(controller, 'ki', np.nan))
            metrics_collector.log('kd', getattr(controller, 'kd', np.nan))
            metrics_collector.log('integral_error', getattr(controller, 'integral_error', np.nan))
            metrics_collector.log('derivative_error', getattr(controller, 'derivative_error', np.nan)) # Init is 0
            metrics_collector.log('epsilon', getattr(agent, 'epsilon', np.nan))
            metrics_collector.log('learning_rate', getattr(agent, 'learning_rate', np.nan))
        except AttributeError as ae:
            logging.warning(f"Could not log initial state attribute: {ae}")
        metrics_collector.log('reward', 0.0); metrics_collector.log('cumulative_reward', 0.0)
        metrics_collector.log('force', 0.0); metrics_collector.log('stability_score', 1.0)
        metrics_collector.log('action_kp', np.nan); metrics_collector.log('action_ki', np.nan); metrics_collector.log('action_kd', np.nan)
        metrics_collector.log('learn_select_duration_ms', np.nan)
        # Log initial agent state metrics (will be mostly NaNs or default values)
        if hasattr(agent, 'get_q_values_for_state'): # Check if agent has the helper methods
             initial_agent_state_dict_for_log = agent.build_agent_state(state_vector, controller, agent_params_cfg.get('state_config', {}))
             metrics_collector.log_q_values(agent, initial_agent_state_dict_for_log)
             metrics_collector.log_q_visit_counts(agent, initial_agent_state_dict_for_log)
             metrics_collector.log_baselines(agent, initial_agent_state_dict_for_log)
        metrics_collector.log('td_error_kp', np.nan); metrics_collector.log('td_error_ki', np.nan); metrics_collector.log('td_error_kd', np.nan)
        metrics_collector.log('virtual_reward_kp', np.nan); metrics_collector.log('virtual_reward_ki', np.nan); metrics_collector.log('virtual_reward_kd', np.nan)
        metrics_collector.log('id_agent_decision', 0) # First decision ID
        # Log gain step (initial value, may change if variable)
        if variable_step and isinstance(gain_step_config, dict):
             metrics_collector.log('gain_step_kp', gain_step_config.get('kp', np.nan))
             metrics_collector.log('gain_step_ki', gain_step_config.get('ki', np.nan))
             metrics_collector.log('gain_step_kd', gain_step_config.get('kd', np.nan))
        else:
             metrics_collector.log('gain_step', float(gain_step_config) if isinstance(gain_step_config, (int,float)) else np.nan )

        # --- Initial Action Selection ---
        try:
            current_agent_state_dict = agent.build_agent_state(state_vector, controller, agent_params_cfg.get('state_config', {}))
            actions = agent.select_action(current_agent_state_dict) # Acción A para el primer intervalo
            # Log the selected action (overwrites initial NaN)
            metrics_collector.log('action_kp', actions.get('kp', np.nan))
            metrics_collector.log('action_ki', actions.get('ki', np.nan))
            metrics_collector.log('action_kd', actions.get('kd', np.nan))
        except Exception as e:
            logging.error(f"Error during initial action selection ep {episode}: {e}. Using default.", exc_info=True)
            actions = {'kp': 1, 'ki': 1, 'kd': 1}
            metrics_collector.log('action_kp', actions['kp'])
            metrics_collector.log('action_ki', actions['ki'])
            metrics_collector.log('action_kd', actions['kd'])

        # ===========================================================
        # LOGICA DE ACTUALIZACIÓN DE GANANCIAS (se repite abajo)
        # ===========================================================
        try:
            kp, ki, kd = controller.kp, controller.ki, controller.kd
            if not variable_step:
                step = float(gain_step_config) # Asegurar float
                if actions['kp'] == 0: kp -= step
                elif actions['kp'] == 2: kp += step
                if actions['ki'] == 0: ki -= step
                elif actions['ki'] == 2: ki += step
                if actions['kd'] == 0: kd -= step
                elif actions['kd'] == 2: kd += step
            else:
                # Variable step logic (asegurar floats también)
                if not isinstance(gain_step_config, dict):
                    step_kp = step_ki = step_kd = float(gain_step_config)
                else:
                    step_kp = float(gain_step_config.get('kp', 0))
                    step_ki = float(gain_step_config.get('ki', 0))
                    step_kd = float(gain_step_config.get('kd', 0))
                if actions['kp'] == 0: kp -= step_kp
                elif actions['kp'] == 2: kp += step_kp
                if actions['ki'] == 0: ki -= step_ki
                elif actions['ki'] == 2: ki += step_ki
                if actions['kd'] == 0: kd -= step_kd
                elif actions['kd'] == 2: kd += step_kd

            # Clip gains using boundaries from agent's state config
            # Asegurar que los límites existen antes de usarlos
            kp_cfg = gain_limits_cfg.get('kp', {})
            ki_cfg = gain_limits_cfg.get('ki', {})
            kd_cfg = gain_limits_cfg.get('kd', {})
            kp = np.clip(kp, kp_cfg.get('min', -np.inf), kp_cfg.get('max', np.inf))
            ki = np.clip(ki, ki_cfg.get('min', -np.inf), ki_cfg.get('max', np.inf))
            kd = np.clip(kd, kd_cfg.get('min', -np.inf), kd_cfg.get('max', np.inf))

            controller.update_params(kp, ki, kd)
            logging.debug(f"SM Initial Action Applied Ep {episode}. New Gains: kp={kp:.4f}, ki={ki:.4f}, kd={kd:.4f}")

        except KeyError as e:
            logging.error(f"SM Error applying initial action to gains: Invalid key {e}. Actions: {actions}")
        except Exception as e:
            logging.error(f"SM Error applying initial action to gains: {e}", exc_info=True)
        # ===========================================================
        
        interval_start_state_vector = np.array(state_vector)
        interval_start_time = 0.0
        done = False
        termination_reason = "unknown"
        current_time = 0.0

        # --- Inner Step Loop ---
        for t_step in range(max_steps):
            current_time = round((t_step + 1) * dt, 6)

            # --- Store state S if nearing decision boundary ---
            is_decision_boundary = current_time >= next_decision_time - dt / 2
            if is_decision_boundary:
                interval_start_state_vector = np.array(state_vector)
                interval_start_time = round(next_decision_time - decision_interval, 6)

            # --- Environment Step ---
            try:
                next_state_vector, reward_stability_tuple, force = env.step()
                reward, stability_score = reward_stability_tuple
            except Exception as e:
                logging.error(f"CRITICAL: Env step error {t_step} ep {episode}: {e}. Terminating.", exc_info=True)
                done = True; termination_reason = "step_error"
                metrics_collector.log('time', current_time)
                metrics_collector.log('reward', np.nan)
                metrics_collector.log('cumulative_reward', cumulative_reward)
                metrics_collector.log('stability_score', np.nan)
                break # Exit step loop

            cumulative_reward += reward
            interval_reward += reward
            interval_stability_scores.append(stability_score)

            # --- Logging Step Metrics (EVERY TIMESTEP) ---
            metrics_collector.log('time', current_time)
            metrics_collector.log('cart_position', next_state_vector[0])
            metrics_collector.log('cart_velocity', next_state_vector[1])
            metrics_collector.log('pendulum_angle', next_state_vector[2])
            metrics_collector.log('pendulum_velocity', next_state_vector[3])
            try:
                ctrl_setpoint = getattr(controller, 'setpoint', 0.0)
                metrics_collector.log('error', next_state_vector[2] - ctrl_setpoint)
                metrics_collector.log('kp', getattr(controller, 'kp', np.nan))
                metrics_collector.log('ki', getattr(controller, 'ki', np.nan))
                metrics_collector.log('kd', getattr(controller, 'kd', np.nan))
                metrics_collector.log('integral_error', getattr(controller, 'integral_error', np.nan))
                metrics_collector.log('derivative_error', getattr(controller, 'derivative_error', np.nan))
                metrics_collector.log('action_kp', actions.get('kp', np.nan))
                metrics_collector.log('action_ki', actions.get('ki', np.nan))
                metrics_collector.log('action_kd', actions.get('kd', np.nan))
                metrics_collector.log('epsilon', getattr(agent, 'epsilon', np.nan))
                metrics_collector.log('learning_rate', getattr(agent, 'learning_rate', np.nan))
                # Log gain step value used
                if variable_step and isinstance(gain_step_config, dict):
                     metrics_collector.log('gain_step_kp', gain_step_config.get('kp', np.nan))
                     metrics_collector.log('gain_step_ki', gain_step_config.get('ki', np.nan))
                     metrics_collector.log('gain_step_kd', gain_step_config.get('kd', np.nan))
                else:
                     metrics_collector.log('gain_step', float(gain_step_config) if isinstance(gain_step_config, (int,float)) else np.nan )
            except AttributeError: pass
            metrics_collector.log('reward', reward)
            metrics_collector.log('cumulative_reward', cumulative_reward)
            metrics_collector.log('force', force)
            metrics_collector.log('stability_score', stability_score)
            # Log actions leading to this state AFTER potential decision block

            # --- Check Termination ---
            try:
                angle_exc, cart_exc, stab = env.check_termination(config)
                time_limit_reached = (current_time >= total_time - dt / 2)
                if angle_exc or cart_exc or stab or time_limit_reached:
                    done = True
                    if termination_reason == "unknown":
                        if angle_exc: termination_reason = "angle_limit"
                        elif cart_exc: termination_reason = "cart_limit"
                        elif stab: termination_reason = "stabilized"
                        elif time_limit_reached: termination_reason = "time_limit"
                        else: termination_reason = "unknown_done"
            except Exception as e:
                logging.error(f"Error checking termination {t_step} ep {episode}: {e}", exc_info=True)
                done = True; termination_reason = "termination_check_error"

            # --- Agent Learning & Next Action Selection (at Decision Interval or Done) ---
            time_for_decision_check = current_time
            decision_block_executed = False
            if time_for_decision_check >= next_decision_time - dt / 2 or done:
                decision_block_executed = True
                agent_decision_count += 1 # Increment decision counter
                learn_start_time = time.time()
                metrics_collector.log('id_agent_decision', agent_decision_count) # Log decision ID

                # --- Log Gains BEFORE potential update ---
                gains_before_dict = controller.get_params() # Gains al final del intervalo (antes de aplicar A')
                #logging.info(f"Ep {episode} Decision {agent_decision_count}: Gains_before = {gains_before_dict}")

                try:
                    # --- 1. Determine S' ---
                    next_agent_state_dict = agent.build_agent_state(next_state_vector, controller, agent_params_cfg.get('state_config', {})) # S'

                    # --- 2. Select Action A'---
                    if not done:
                        actions_prime = agent.select_action(next_agent_state_dict) # A'
                    else:
                        actions_prime = {'kp': 1, 'ki': 1, 'kd': 1} # Neutral action if done
                    #logging.info(f"Ep {episode} Decision {agent_decision_count}: actions' = {actions_prime}")

                    # --- 3. Calculate resulting gains if A' is applied (Always needed) ---
                    kp_after, ki_after, kd_after = gains_before_dict['kp'], gains_before_dict['ki'], gains_before_dict['kd'] # Start from current gains
                    # === Apply action A' to calculate potential next gains ===
                    if not variable_step:
                        step = float(gain_step_config)
                        if actions_prime['kp'] == 0: kp_after -= step
                        elif actions_prime['kp'] == 2: kp_after += step
                        if actions_prime['ki'] == 0: ki_after -= step
                        elif actions_prime['ki'] == 2: ki_after += step
                        if actions_prime['kd'] == 0: kd_after -= step
                        elif actions_prime['kd'] == 2: kd_after += step
                    else:
                        # Variable step logic for kp_after, ki_after, kd_after...
                        if not isinstance(gain_step_config, dict):
                            step_kp = step_ki = step_kd = float(gain_step_config)
                        else:
                            step_kp = float(gain_step_config.get('kp', 0))
                            step_ki = float(gain_step_config.get('ki', 0))
                            step_kd = float(gain_step_config.get('kd', 0))
                        if actions_prime['kp'] == 0: kp_after -= step_kp
                        elif actions_prime['kp'] == 2: kp_after += step_kp
                        if actions_prime['ki'] == 0: ki_after -= step_ki
                        elif actions_prime['ki'] == 2: ki_after += step_ki
                        if actions_prime['kd'] == 0: kd_after -= step_kd
                        elif actions_prime['kd'] == 2: kd_after += step_kd

                        # Clip calculated gains_after
                        kp_cfg = gain_limits_cfg.get('kp', {})
                        ki_cfg = gain_limits_cfg.get('ki', {})
                        kd_cfg = gain_limits_cfg.get('kd', {})
                        kp_after = np.clip(kp_after, kp_cfg.get('min', -np.inf), kp_cfg.get('max', np.inf))
                        ki_after = np.clip(ki_after, ki_cfg.get('min', -np.inf), ki_cfg.get('max', np.inf))
                        kd_after = np.clip(kd_after, kd_cfg.get('min', -np.inf), kd_cfg.get('max', np.inf))
                        gains_after_applying_actions_prime = {'kp': kp_after, 'ki': ki_after, 'kd': kd_after}
                        #logging.info(f"Ep {episode} Decision {agent_decision_count}: Gains_after (predicted) = {gains_after_applying_actions_prime}")
                        # === FIN Cálculo estado post-acción ===

                        # --- 4. Calculate reward_for_agent_info based on mode ---
                        reward_for_agent_info: Union[float, Tuple[float, float], Dict[str, float]] = 0.0 # Default

                        if current_reward_mode == 'global':
                            reward_for_agent_info = interval_reward
                        elif current_reward_mode == 'shadow_baseline':
                            avg_interval_stability = np.mean(interval_stability_scores) if interval_stability_scores else 1.0
                            # Ensure avg_w_stab is valid
                            if not isinstance(avg_interval_stability, (float, int, np.number)) or np.isnan(avg_interval_stability):
                                avg_interval_stability = 1.0
                            reward_for_agent_info = (interval_reward, avg_interval_stability)
                        elif current_reward_mode == 'echo_baseline':
                            if virtual_simulator:
                                echo_sim_start = time.time()
                                # Define CONTRAFACTUAL gains using gains_before_dict and predicted gains_after_applying_actions_prime
                                gains_p_cf = {'kp': gains_before_dict['kp'], 'ki': gains_after_applying_actions_prime['ki'], 'kd': gains_after_applying_actions_prime['kd']}
                                gains_i_cf = {'kp': gains_after_applying_actions_prime['kp'], 'ki': gains_before_dict['ki'], 'kd': gains_after_applying_actions_prime['kd']}
                                gains_d_cf = {'kp': gains_after_applying_actions_prime['kp'], 'ki': gains_after_applying_actions_prime['ki'], 'kd': gains_before_dict['kd']}

                                # Run virtual simulations
                                R_p_cf = virtual_simulator.run_interval_simulation(interval_start_state_vector, interval_start_time, decision_interval, gains_p_cf)
                                R_i_cf = virtual_simulator.run_interval_simulation(interval_start_state_vector, interval_start_time, decision_interval, gains_i_cf)
                                R_d_cf = virtual_simulator.run_interval_simulation(interval_start_state_vector, interval_start_time, decision_interval, gains_d_cf)

                                #logging.info(f"Ep {episode} Decision {agent_decision_count}: Interval Reward={interval_reward:.4f}, R_p_cf={R_p_cf:.4f}, R_i_cf={R_i_cf:.4f}, R_d_cf={R_d_cf:.4f}")
                                reward_for_agent_info = {'kp': interval_reward - R_p_cf, 'ki': interval_reward - R_i_cf, 'kd': interval_reward - R_d_cf}
                                #logging.info(f"Ep {episode} Decision {agent_decision_count}: Calculated reward_for_agent_info = {reward_for_agent_info}")
                                total_virtual_sim_time += (time.time() - echo_sim_start)
                                metrics_collector.log_virtual_rewards(reward_for_agent_info) # Log virtual rewards
                            else:
                                logging.error("Echo mode but no VirtualSimulator! Using 0.")
                                reward_for_agent_info = {'kp': 0.0, 'ki': 0.0, 'kd': 0.0}
                                metrics_collector.log_virtual_rewards(reward_for_agent_info)

                        # --- Log agent state BEFORE learning ---
                        metrics_collector.log_q_values(agent, current_agent_state_dict)
                        metrics_collector.log_q_visit_counts(agent, current_agent_state_dict)
                        metrics_collector.log_baselines(agent, current_agent_state_dict)

                        # --- 5. Agent Learn Step (Uses S, A, R, S') ---
                        # 'actions' aquí son las acciones A tomadas que llevaron de S a S' durante el intervalo que acaba de terminar
                        agent.learn(current_agent_state_dict, actions, reward_for_agent_info, next_agent_state_dict, done)

                        # --- Log agent state AFTER learning ---
                        metrics_collector.log_td_errors(agent.get_last_td_errors())

                        # --- 6. Apply the NEW gains (calculated in step 3) to the controller ---
                        if not done:
                            controller.update_params(
                                gains_after_applying_actions_prime['kp'],
                                gains_after_applying_actions_prime['ki'],
                                gains_after_applying_actions_prime['kd']
                            )
                            logging.debug(f"SM Action A' Applied Ep {episode} Dec {agent_decision_count}. New Gains: kp={controller.kp:.4f}, ki={controller.ki:.4f}, kd={controller.kd:.4f}")

                        # --- Log Gains AFTER actual update ---
                        gains_after_dict = controller.get_params() # Las ganancias reales después de aplicar A'
                        #logging.info(f"Ep {episode} Decision {agent_decision_count}: Gains_after = {gains_after_dict}")


                        # --- 7. Update 'actions' variable for the *next* dt step's logging/use ---
                        if not done:
                            actions = actions_prime # Actualizar 'actions' con A' para el próximo intervalo
                            # Loguear la acción A' que se aplicará ahora
                            metrics_collector.log('action_kp', actions.get('kp', np.nan))
                            metrics_collector.log('action_ki', actions.get('ki', np.nan))
                            metrics_collector.log('action_kd', actions.get('kd', np.nan))
                        else:
                            # Si termina, la acción no importa para el próximo intervalo, pero loguear NaNs
                            actions = {'kp': np.nan, 'ki': np.nan, 'kd': np.nan}
                            metrics_collector.log('action_kp', np.nan); metrics_collector.log('action_ki', np.nan); metrics_collector.log('action_kd', np.nan)


                        # --- Reset Interval Variables ---
                        interval_reward = 0.0
                        interval_stability_scores = []
                        current_agent_state_dict = next_agent_state_dict # S <- S'
                        next_decision_time = round((np.floor(time_for_decision_check / decision_interval) + 1) * decision_interval, 6)

                except Exception as e:
                    logging.error(f"Error during agent learn/select {t_step} ep {episode}: {e}", exc_info=True)
                    actions = {'kp': 1, 'ki': 1, 'kd': 1} # Default neutral action
                    metrics_collector.log('action_kp', actions['kp']); metrics_collector.log('action_ki', actions['ki']); metrics_collector.log('action_kd', actions['kd'])
                    # Log NaNs for agent metrics on error
                    metrics_collector.log_q_values(agent, current_agent_state_dict) # Log previous state's values? Or NaNs?
                    metrics_collector.log_q_visit_counts(agent, current_agent_state_dict)
                    metrics_collector.log_baselines(agent, current_agent_state_dict)
                    metrics_collector.log('td_error_kp', np.nan); metrics_collector.log('td_error_ki', np.nan); metrics_collector.log('td_error_kd', np.nan)
                    metrics_collector.log('virtual_reward_kp', np.nan); metrics_collector.log('virtual_reward_ki', np.nan); metrics_collector.log('virtual_reward_kd', np.nan)

                finally:
                    metrics_collector.log('learn_select_duration_ms', (time.time() - learn_start_time) * 1000)

            else: # --- If not a decision step ---
                # Log NaN for metrics only calculated during decision step
                metrics_collector.log('learn_select_duration_ms', np.nan)
                metrics_collector.log('id_agent_decision', np.nan) # No decision ID
                # Log NaNs for agent state metrics
                metrics_collector.log('q_value_max_kp', np.nan); metrics_collector.log('q_value_max_ki', np.nan); metrics_collector.log('q_value_max_kd', np.nan)
                metrics_collector.log('q_visit_count_state_kp', np.nan); metrics_collector.log('q_visit_count_state_ki', np.nan); metrics_collector.log('q_visit_count_state_kd', np.nan)
                metrics_collector.log('baseline_value_kp', np.nan); metrics_collector.log('baseline_value_ki', np.nan); metrics_collector.log('baseline_value_kd', np.nan)
                metrics_collector.log('td_error_kp', np.nan); metrics_collector.log('td_error_ki', np.nan); metrics_collector.log('td_error_kd', np.nan)
                metrics_collector.log('virtual_reward_kp', np.nan); metrics_collector.log('virtual_reward_ki', np.nan); metrics_collector.log('virtual_reward_kd', np.nan)


            if done:
                logging.info(f"Episode {episode} terminated: {termination_reason} at t={current_time:.3f}")
                break

            state_vector = next_state_vector

        # --- End of Episode Processing ---
        if not done:
            current_time = max_steps * dt
            termination_reason = "time_limit"
            logging.info(f"Episode {episode} finished: Reached max steps at t={current_time:.3f}")

        metrics_collector.log('episode_duration_s', time.time() - episode_start_time)
        metrics_collector.log('total_agent_decisions', agent_decision_count) # Log total decisions
        # Log final gains
        final_gains = controller.get_params() if hasattr(controller, 'get_params') else {}
        metrics_collector.log('final_kp', final_gains.get('kp', np.nan))
        metrics_collector.log('final_ki', final_gains.get('ki', np.nan))
        metrics_collector.log('final_kd', final_gains.get('kd', np.nan))


        # --- Collect Episode Data ---
        episode_data = metrics_collector.get_metrics()
        episode_data['termination_reason'] = termination_reason # Ensure reason is set
        episode_data['avg_stability_score'] = np.nanmean(episode_data.get('stability_score', [np.nan])) # Use nanmean


        # --- Update Adaptive Stats ---
        try:
            if hasattr(env, 'update_reward_calculator_stats'):
                env.update_reward_calculator_stats(episode_data, episode)
                # --- Log Adaptive Stats ---
                if hasattr(env.reward_function, 'stability_calculator') and \
                   hasattr(env.reward_function.stability_calculator, 'get_current_adaptive_stats'):
                    adaptive_stats = env.reward_function.stability_calculator.get_current_adaptive_stats()
                    metrics_collector.log_adaptive_stats(adaptive_stats)
                else: # Log NaNs if not available
                    metrics_collector.log_adaptive_stats({}) # Pass empty dict, logger handles it
        except Exception as e:
            logging.error(f"Error updating/logging adaptive stats ep {episode}: {e}", exc_info=True)
            metrics_collector.log_adaptive_stats({}) # Log NaNs

        # --- Finalize episode data (after potential adaptive stat logging) ---
        episode_data = metrics_collector.get_metrics() # Get metrics again to include logged stats
        episode_data['termination_reason'] = termination_reason
        episode_data['avg_stability_score'] = np.nanmean(episode_data.get('stability_score', [np.nan]))


        # --- Add to data lists ---
        episode_batch.append(episode_data)
        all_episodes_data.append(episode_data)

        # --- Summarize episode ---
        from utils.data_processing import summarize_episode
        try:
             summary = summarize_episode(episode_data) # Updated in Step 3
             summary_data.append(summary)
        except Exception as e:
             logging.error(f"Error summarizing episode {episode}: {e}", exc_info=True)
             summary_data.append({'episode': episode, 'error': 'summary_failed'})


        # --- Periodic Saving (Delegated to result_handler) ---
        if (episode + 1) % episodes_per_file == 0 or episode == max_episodes - 1:
            if episode_batch:
                logging.info(f"Saving episode batch ending with ep {episode}...")
                try:
                     save_episode_batch(episode_batch, results_folder, episode)
                     episode_batch = []
                except NameError: logging.error("Func 'save_episode_batch' not found.")
                except Exception as e: logging.error(f"Error saving batch: {e}", exc_info=True); episode_batch = []
            else: logging.debug(f"Skipping save ep {episode}, batch empty.")

        if simulation_cfg.get('save_agent_state', False) and \
           (episode + 1) % simulation_cfg.get('agent_state_save_frequency', 1000) == 0:
            logging.info(f"Saving periodic agent state ep {episode}...")
            try:
                save_agent_state(agent, episode, results_folder)
                # Tracking last saved path handled by result_handler now
            except NameError: logging.error("Func 'save_agent_state' not found.")
            except Exception as e: logging.error(f"Error saving periodic agent state: {e}", exc_info=True)

    # --- End of Training Loop ---
    logging.info("--- Simulation Run Finished ---")
    if current_reward_mode == 'echo_baseline':
        logging.info(f"Total time in Echo virtual sims: {total_virtual_sim_time:.2f}s")

    total_sim_duration = time.time() - start_time_sim
    logging.info(f"Total simulation duration: {total_sim_duration:.2f} seconds.")

    return all_episodes_data, summary_data