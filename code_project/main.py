import time
import logging
import os

# --- Configuration and Initialization ---
from config_loader import load_and_validate_config
from world_initializer import initialize_simulation_components

# --- Simulation Execution ---
from simulation_manager import run_simulation

# --- Results Handling ---
from result_handler import setup_results_folder, finalize_results, save_metadata

# --- Visualization ---
from visualization_runner import run_visualizations

# --- Utilities ---
from datetime import datetime

# Setup detailed logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

def main():
    """ Main function orchestrating the RL simulation """
    # --- [1] Start Timer ---
    start_time_main = time.time()
    logging.info("--- Simulation Starting ---")

    # --- [2] Load Configuration ---
    config, vis_config = load_and_validate_config(config_filename='config.yaml')
    if config is None:
        return
    
    # --- [3] Setup Results Folder ---
    try:
        results_folder = setup_results_folder(config.get('environment', {}).get('results_folder', 'results_history'))
    except Exception as e:
         logging.error(f"Failed to set up results folder: {e}. Exiting.")
         return

    # --- [4] Initialize Simulation Components ---
    components = initialize_simulation_components(config)
    if components is None:
        return

    # --- [5] Save Metadata ---
    code_version = "3.1.0" # Update as appropriate
    try:
        # --- Obtener reward_mode desde la nueva estructura ---
        env_cfg = config.get('environment', {})
        reward_setup_cfg = env_cfg.get('reward_setup', {})
        current_reward_mode = reward_setup_cfg.get('learning_strategy', 'N/A') # Leer de reward_setup

        metadata_to_save = {
            "environment_details": {
                "code_version": code_version,
                "run_timestamp": datetime.now().isoformat(),
                "results_folder": results_folder,
                "reward_mode": current_reward_mode # Usar el valor leído
            },
            "config_parameters": config, # Guardar config completo con nueva estructura
            "visualization_config": vis_config
        }
        save_metadata(metadata_to_save, results_folder)
    except Exception as e:
        logging.error(f"Error saving metadata: {e}", exc_info=True)

    # --- [6] Run Simulation ---
    all_episodes_data, summary_data = [], [] # Initialize in case of error
    try:
         all_episodes_data, summary_data = run_simulation(components, config, results_folder)
    except Exception as e:
         logging.error(f"Critical error during simulation run: {e}. Attempting finalization.", exc_info=True)
         # Use initialized empty lists if run failed early

    # --- [7] Finalize Results ---
    try:
        finalize_results(
            config=config,
            summary_data=summary_data,
            # *** Pass all_episodes_data HERE ***
            all_episodes_data=all_episodes_data,
            # ************************************
            agent=components.get('agent'),
            results_folder=results_folder
        )
    except Exception as e:
         logging.error(f"Error during results finalization: {e}", exc_info=True)

    # --- [8] Run Visualizations ---
    try:
        run_visualizations(
            vis_config=vis_config,
            summary_data=summary_data,
            all_episodes_data=all_episodes_data,
            results_folder=results_folder
        )
    except Exception as e:
        logging.error(f"Error during visualization generation: {e}", exc_info=True)

    # --- [9] End Timer and Log Summary ---
    end_time_main = time.time()
    total_duration = end_time_main - start_time_main
    logging.info(f"--- Simulation Finished ---")
    logging.info(f"Total execution time: {total_duration:.2f} seconds.")
    logging.info(f"Results saved in: {results_folder}")

if __name__ == "__main__":
    main()