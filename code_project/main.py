import os
import yaml
from datetime import datetime
import logging
import numpy as np
import pandas as pd

from factories.environment_factory import EnvironmentFactory
from components.analysis.simple_metrics_collector import SimpleMetricsCollector
from utils.data_processing import summarize_episode, save_summary_table
from utils.visualization import generate_plots
from utils.episode_saver import save_episode_batch, save_metadata
from utils.agent_state_manager import save_agent_state, convert_json_agent_state_to_excel

# Setup logging (INFO level shows progress, ERROR shows critical issues)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    dirname = os.path.dirname(__file__)
    config_filename = os.path.join(dirname, 'config.yaml')
    vis_config = None
    try:
        with open(config_filename, 'r') as file:
            config = yaml.safe_load(file)
            logging.info("Configuration file loaded successfully.")
        
        vis_config_settings = config.get('visualization', {})
        if vis_config_settings.get('enabled', False):
            vis_config_filename_rel = vis_config_settings.get('config_file')
            if vis_config_filename_rel:
                vis_config_filename_abs = os.path.join(dirname, vis_config_filename_rel)
                try:
                    with open(vis_config_filename_abs, 'r') as vis_file:
                        vis_config = yaml.safe_load(vis_file)
                        logging.info(f"Visualization configuration loaded from: {vis_config_filename_abs}")
                except FileNotFoundError:
                    logging.warning(f"Visualization config file not found at {vis_config_filename_abs}. Plotting disabled.")
                    vis_config = None # Disable plotting if file not found
                except yaml.YAMLError as e_vis:
                    logging.error(f"Error parsing visualization config file {vis_config_filename_abs}: {e_vis}. Plotting disabled.")
                    vis_config = None
                except Exception as e_vis:
                    logging.error(f"Unexpected error loading visualization config {vis_config_filename_abs}: {e_vis}. Plotting disabled.", exc_info=True)
                    vis_config = None
            else:
                logging.warning("Visualization enabled but 'config_file' not specified in main config. Plotting disabled.")
        else:
            logging.info("Visualization generation disabled in main config.")
    
    except FileNotFoundError:
        logging.error(f"CRITICAL: Configuration file not found at {config_filename}. Exiting.")
        return
    except yaml.YAMLError as e:
        logging.error(f"CRITICAL: Error parsing configuration file: {e}. Exiting.")
        return
    except Exception as e:
        logging.error(f"CRITICAL: Unexpected error loading config: {e}. Exiting.", exc_info=True)
        return


    try:
        env = EnvironmentFactory.create_environment(config) # Pass the whole config
        metrics_collector = SimpleMetricsCollector()
        logging.info("Environment and Metrics Collector created.")
    except Exception as e:
        # Environment creation errors are often critical (config issues, factory errors)
        logging.error(f"CRITICAL: Error creating environment: {e}. Exiting.", exc_info=True)
        return

    # --- Setup Results Folder ---
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    base_results_folder = config.get('environment', {}).get('results_folder', 'results_history')
    results_folder = os.path.join(dirname, base_results_folder, timestamp)
    try:
        os.makedirs(results_folder, exist_ok=True)
        logging.info(f"Results will be saved in: {results_folder}")
    except OSError as e:
        # Failure to create results dir is critical
        logging.error(f"CRITICAL: Error creating results directory {results_folder}: {e}. Exiting.")
        return

    # --- Save Metadata ---
    metadata_to_save = {
        "environment_details": {
             "code_version": "1.9.0", # <<< UPDATE VERSION >>>
             "timestamp": datetime.now().isoformat(),
        },
        "config_parameters": config,
        "visualization_config": vis_config
    }
    # Keep try-except around file I/O
    try:
        save_metadata(metadata_to_save, results_folder) # Assumes function exists and handles internal errors/logging
    except NameError:
         logging.warning("Function 'save_metadata' not found/imported. Skipping metadata save.")
    except Exception as e:
         logging.error(f"Error saving metadata: {e}", exc_info=True)
    # --- End Save Metadata ---

    # --- Simulation Parameters (using .get for safer access) ---
    try:
        simulation_cfg = config.get('simulation', {})
        env_cfg = config.get('environment', {})
        # Agent config access might be needed for state building later, retrieve it if necessary
        agent_cfg_full = env_cfg.get('agent', {}) # Get the whole agent config section
        agent_params_cfg = agent_cfg_full.get('params', {}) # Get just the params sub-dict

        init_cond_cfg = config.get('initial_conditions', {})

        episodes_per_file = simulation_cfg.get('episodes_per_file', 200) # Default 200
        decision_interval = env_cfg.get('decision_interval', 0.01) # Default 0.01
        dt = env_cfg.get('dt', 0.001) # Default 0.001
        total_time = env_cfg.get('total_time', 5.0) # Default 5.0
        max_steps = int(total_time / dt) if dt > 0 else 0
        max_episodes = env_cfg.get('max_episodes', 1000) # Default 1000
        initial_state_vector = init_cond_cfg.get('x0') # Get initial state
        if initial_state_vector is None: raise KeyError("'initial_conditions: x0' is missing in config")

        save_agent_state_flag = simulation_cfg.get('save_agent_state', False)
        agent_state_save_frequency = simulation_cfg.get('agent_state_save_frequency', 1000)

    except KeyError as e:
        logging.error(f"CRITICAL: Missing essential configuration key: {e}. Exiting.")
        return
    except Exception as e:
        logging.error(f"CRITICAL: Error reading simulation parameters: {e}. Exiting.", exc_info=True)
        return

    episode_batch, summary_data, all_episodes_data = [], [], []
    last_saved_agent_json = None

    # --- Main Simulation Loop ---
    for episode in range(max_episodes):
        logging.info(f"--- Starting Episode: {episode}/{max_episodes-1} ---")

        # --- Reset Environment ---
        try:
            state_vector = env.reset(initial_state_vector) # Returns the numeric state vector
        except Exception as e:
            # Error during reset might be recoverable, log and skip episode
            logging.error(f"Error during environment reset for episode {episode}: {e}. Skipping episode.", exc_info=True)
            continue # Skip to next episode

        cumulative_reward, interval_reward, next_decision_time = 0, 0, decision_interval
        metrics_collector.reset()

        # --- Log Initial State (t=0) ---
        # Basic logging calls are unlikely to fail critically
        metrics_collector.log('time', 0.0)
        metrics_collector.log('cart_position', state_vector[0])
        metrics_collector.log('cart_velocity', state_vector[1])
        metrics_collector.log('pendulum_angle', state_vector[2])
        metrics_collector.log('pendulum_velocity', state_vector[3])
        try: # Keep minimal try-except for attributes that might not exist if env/agent init failed partially
             metrics_collector.log('error', state_vector[2] - env.controller.setpoint)
             metrics_collector.log('kp', env.controller.kp)
             metrics_collector.log('ki', env.controller.ki)
             metrics_collector.log('kd', env.controller.kd)
             metrics_collector.log('epsilon', env.agent.epsilon)
             metrics_collector.log('learning_rate', env.agent.learning_rate)
        except AttributeError as ae:
             logging.warning(f"Could not log initial controller/agent state: {ae}")
        metrics_collector.log('reward', 0.0)
        metrics_collector.log('cumulative_reward', 0.0)
        metrics_collector.log('force', 0.0)

        # Initial action selection
        try:
            # Build the dictionary state required by the agent
            current_agent_state_dict = env.agent.build_agent_state(state_vector, env.controller, agent_params_cfg.get('state_config', {}))
            actions = env.agent.select_action(current_agent_state_dict)
            # Log the actions chosen
            metrics_collector.log('action_kp', actions.get('kp', np.nan))
            metrics_collector.log('action_ki', actions.get('ki', np.nan))
            metrics_collector.log('action_kd', actions.get('kd', np.nan))
        except Exception as e:
            # Error in agent logic might be critical, log and consider stopping/default action
            logging.error(f"Error during initial action selection episode {episode}: {e}. Using default action.", exc_info=True)
            actions = {'kp': 1, 'ki': 1, 'kd': 1} # Default neutral action (maintain)
            metrics_collector.log('action_kp', actions['kp'])
            metrics_collector.log('action_ki', actions['ki'])
            metrics_collector.log('action_kd', actions['kd'])
        # --- End Log Initial State ---

        done = False
        termination_reason = "unknown"
        current_time = 0.0 # Initialize for final logging scope

        # --- Inner Step Loop ---
        for t_step in range(max_steps):
            current_time = (t_step + 1) * dt

            # --- Environment Step (Keep try-except here, step is complex) ---
            try:
                # Env.step now takes the dictionary of actions
                next_state_vector, reward, force = env.step(actions)
            except Exception as e:
                logging.error(f"CRITICAL: Error during env step {t_step} ep {episode}: {e}. Terminating episode.", exc_info=True)
                done = True
                termination_reason = "step_error"
                # Log minimal info before break
                metrics_collector.log('time', current_time)
                metrics_collector.log('reward', np.nan)
                metrics_collector.log('cumulative_reward', cumulative_reward)
                break # Exit step loop

            cumulative_reward += reward
            interval_reward += reward

            # --- Logging Metrics (Generally safe, no complex logic) ---
            metrics_collector.log('time', current_time)
            metrics_collector.log('cart_position', next_state_vector[0])
            metrics_collector.log('cart_velocity', next_state_vector[1])
            metrics_collector.log('pendulum_angle', next_state_vector[2])
            metrics_collector.log('pendulum_velocity', next_state_vector[3])
            try: # Minimal try-except for attributes
                 metrics_collector.log('error', next_state_vector[2] - env.controller.setpoint)
                 metrics_collector.log('kp', env.controller.kp) # Log actual applied gains
                 metrics_collector.log('ki', env.controller.ki)
                 metrics_collector.log('kd', env.controller.kd)
                 metrics_collector.log('epsilon', env.agent.epsilon)
                 metrics_collector.log('learning_rate', env.agent.learning_rate)
            except AttributeError as ae:
                 logging.debug(f"Could not log ctrl/agent state step {t_step}: {ae}")
            metrics_collector.log('reward', reward)
            metrics_collector.log('cumulative_reward', cumulative_reward)
            metrics_collector.log('force', force)
            # Actions logged after potential update below
            # --- End Logging ---

            # --- Check Termination (Keep try-except, depends on env implementation) ---
            try:
                # Pass state vector to check_termination if it needs it (or env uses internal state)
                angle_exc, cart_exc, stab = env.check_termination(config) # Assumes env uses its internal self.state
                time_limit_reached = (current_time >= total_time - dt / 2)
                if angle_exc or cart_exc or stab or time_limit_reached:
                    done = True
                    # Determine reason immediately if done
                    if termination_reason == "unknown": # Avoid overwriting error reasons
                        if angle_exc: termination_reason = "angle_limit"
                        elif cart_exc: termination_reason = "cart_limit"
                        elif stab: termination_reason = "stabilized"
                        elif time_limit_reached: termination_reason = "time_limit"
                        else: termination_reason = "unknown_done"
            except Exception as e:
                 logging.error(f"Error checking termination step {t_step} ep {episode}: {e}", exc_info=True)
                 done = True
                 termination_reason = "termination_check_error"

            # --- Agent Learning & Next Action Selection (Keep try-except, agent logic) ---
            time_for_decision_check = current_time # Check against time at end of step
            if time_for_decision_check >= next_decision_time or done:
                try:
                    # Build the dictionary state for the *next* state
                    next_agent_state_dict = env.agent.build_agent_state(next_state_vector, env.controller, agent_params_cfg.get('state_config', {}))

                    # Learn based on the *previous* state_dict, actions taken, reward, and the next_state_dict
                    env.agent.learn(current_agent_state_dict, actions, interval_reward, next_agent_state_dict, done)

                    if not done:
                        # Select the *next* actions based on the next_state_dict
                        actions = env.agent.select_action(next_agent_state_dict)
                        metrics_collector.log('action_kp', actions.get('kp', np.nan))
                        metrics_collector.log('action_ki', actions.get('ki', np.nan))
                        metrics_collector.log('action_kd', actions.get('kd', np.nan))
                    else:
                         # Log NaN for actions if done, as no action is selected/taken
                         metrics_collector.log('action_kp', np.nan)
                         metrics_collector.log('action_ki', np.nan)
                         metrics_collector.log('action_kd', np.nan)

                    # Reset parameters for next window decision
                    interval_reward = 0 # Reset reward accumulator for the next decision interval
                    # Schedule next decision accurately
                    next_decision_time = (np.floor(time_for_decision_check / decision_interval) + 1) * decision_interval

                    # Update the state dictionary for the next iteration's 'learn' call
                    current_agent_state_dict = next_agent_state_dict

                except Exception as e:
                    logging.error(f"Error during agent learn/action selection step {t_step} ep {episode}: {e}", exc_info=True)
                    # Decide if this error terminates episode:
                    # done = True # Optional: terminate on agent error
                    # termination_reason = "agent_error"
                    # If not terminating, ensure 'actions' has a safe default for the next env.step
                    actions = {'kp': 1, 'ki': 1, 'kd': 1} # Default neutral action
                    # Log default actions if error occurred
                    metrics_collector.log('action_kp', actions['kp'])
                    metrics_collector.log('action_ki', actions['ki'])
                    metrics_collector.log('action_kd', actions['kd'])

            else:
                 # Log the *currently active* actions if it wasn't a decision step
                 metrics_collector.log('action_kp', actions.get('kp', np.nan))
                 metrics_collector.log('action_ki', actions.get('ki', np.nan))
                 metrics_collector.log('action_kd', actions.get('kd', np.nan))


            if done:
                 logging.info(f"Episode {episode} terminated: {termination_reason} at t={current_time:.3f}")
                 break # Exit inner loop

            # Update state vector for next iteration (used in logging, termination check, building agent state dict)
            state_vector = next_state_vector
            # current_agent_state_dict updated in decision block

        # --- End of Episode Processing ---
        if not done: # If loop finished naturally by reaching max_steps
            current_time = max_steps * dt # Ensure current_time is max time
            termination_reason = "time_limit" # Overwrite reason if it finished due to steps
            logging.info(f"Episode {episode} finished: Reached max steps, reason: {termination_reason} at t={current_time:.3f}")

        # --- Process Episode Data ---
        # This part is generally safe unless metrics_collector fails fundamentally
        episode_data = metrics_collector.get_metrics()
        episode_data['episode'] = episode
        episode_data['termination_reason'] = termination_reason

        episode_batch.append(episode_data)
        all_episodes_data.append(episode_data) # For final plots

        # Summarize (assuming summarize_episode is robust)
        summary = summarize_episode(episode_data)
        summary_data.append(summary)

        # --- Save Batch Data Periodically (Keep try-except for file I/O) ---
        if (episode + 1) % episodes_per_file == 0 or episode == max_episodes - 1:
            if episode_batch:
                logging.info(f"Saving episode batch ending with ep {episode} ({len(episode_batch)} eps)...")
                try:
                     # Pass the actual last episode number of the batch being saved
                     last_ep_in_batch = episode_batch[-1]['episode']
                     save_episode_batch(episode_batch, results_folder, last_ep_in_batch)
                except NameError:
                     logging.error("'save_episode_batch' function not found.")
                except Exception as e:
                     logging.error(f"Error saving episode batch: {e}", exc_info=True)
                episode_batch = [] # Clear batch regardless of save success? Or only on success? Let's clear it.
                logging.info("Batch processed.")
            else:
                logging.debug(f"Skipping save for ep {episode}, batch empty.")
        # --- End Save Batch ---

        # --- Guardado Periódico del Estado del Agente (Usa la nueva estructura JSON) ---
        if save_agent_state_flag and (episode + 1) % agent_state_save_frequency == 0:
            logging.info(f"Attempting to save periodic agent state at episode {episode}...")
            try:
                # save_agent_state ahora guarda el formato Pandas-friendly y devuelve el path
                saved_filepath = save_agent_state(env.agent, episode, results_folder)
                if saved_filepath:
                     last_saved_agent_json = saved_filepath # Actualiza el último archivo guardado
            except NameError:
                 logging.error("Function 'save_agent_state' not found/imported.")
            except Exception as e:
                 logging.error(f"Error saving periodic agent state at episode {episode}: {e}", exc_info=True)
        # --- Fin Guardado Periódico ---


    # --- End of Training ---
    # --- End of Training Loop ---
    logging.info("Training finished.")

    # --- Guardar Estado Final del Agente (JSON formato Pandas-friendly) ---
    # Siempre guarda el estado final, independientemente de la frecuencia periódica,
    # a menos que 'save_agent_state_flag' sea False.
    if save_agent_state_flag and last_saved_agent_json is None:
        logging.info(f"Attempting to save FINAL agent state at episode {max_episodes - 1}...")
        try:
            # Llama a save_agent_state una última vez para el estado final
            final_episode_num = max_episodes - 1
            saved_filepath = save_agent_state(env.agent, final_episode_num, results_folder)
            if saved_filepath:
                 last_saved_agent_json = saved_filepath # Asegura que last_saved_agent_json apunte al último
        except NameError:
             logging.error("Function 'save_agent_state' not found/imported.")
        except Exception as e:
             logging.error(f"Error saving FINAL agent state JSON: {e}", exc_info=True)
    else:
        logging.info("Skipping final agent state saving because 'save_agent_state' is disabled in config.")


    # --- <<< NUEVO: Convertir el ÚLTIMO JSON de estado del agente a Excel >>> ---
    # Solo si se guardó algún estado
    if last_saved_agent_json:
         logging.info(f"Attempting to convert last saved agent state JSON ('{os.path.basename(last_saved_agent_json)}') to Excel...")
         excel_output_filename = os.path.join(results_folder, 'final_agent_tables.xlsx')
         try:
              convert_json_agent_state_to_excel(last_saved_agent_json, excel_output_filename)
              # El logging de éxito/error está dentro de la función de conversión
         except NameError:
              logging.error("Function 'convert_json_agent_state_to_excel' not found/imported.")
         except Exception as e:
              logging.error(f"Error during JSON to Excel conversion call: {e}", exc_info=True)
    elif save_agent_state_flag:
         logging.warning("Agent state saving was enabled, but no state file seems to have been saved correctly. Cannot convert to Excel.")
    else:
         logging.info("Agent state saving was disabled, skipping JSON to Excel conversion.")
    # --- <<< FIN NUEVO: Convertir a Excel >>> ---

    # --- Saving summary and plots ---
    logging.info("Saving summary and plotting results...")
    if summary_data:
        summary_df = pd.DataFrame(summary_data) # Create DataFrame first
        # --- Save Summary (Keep try-except for file I/O) ---
        try:
            summary_file_path = os.path.join(results_folder, 'summary.xlsx')
            # Using the name specified by user
            save_summary_table(summary_data, summary_file_path) # Pass list of dicts directly
            logging.info(f"Summary saved to {summary_file_path}")
        except NameError:
             logging.error("Function 'save_summary_table' not found.") # Corrected function name check
        except Exception as e:
            logging.error(f"Error saving summary file: {e}", exc_info=True)

        # --- Plotting (Keep try-except, plotting can fail) ---
        if vis_config and vis_config.get('plots'):
            logging.info("Generating plots...")
            try:
                # Pass summary_df and all_episodes_data to the generator
                generate_plots(
                    plot_configs=vis_config['plots'],
                    summary_df=summary_df, # Pass the DataFrame
                    detailed_data=all_episodes_data,
                    results_folder=results_folder
                )
                logging.info("Configurable plot generation finished.")
            except NameError:
                logging.error("Function 'generate_plots' not found/imported from visualization module.")
            except Exception as plot_e:
                logging.error(f"Error during configurable plot generation: {plot_e}", exc_info=True)
        elif vis_config is not None:
             logging.warning("Visualization config loaded, but no 'plots' section found or it's empty.")
        else:
             logging.info("Visualization config not loaded or disabled, skipping automatic plot generation.")
        # --- End Plotting ---
    else:
        logging.warning("No summary data generated. Cannot save summary or generate summary-based plots.")

    logging.info(f"Simulation finished. Results: {results_folder}")

if __name__ == "__main__":
    main()