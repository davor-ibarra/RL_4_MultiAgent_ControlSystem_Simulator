import os
import json
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional

try:
     from components.agents.pid_qlearning_agent import PIDQLearningAgent
except ImportError:
     PIDQLearningAgent = None

from utils.numpy_encoder import NumpyEncoder
# Import heatmap generator function
from heatmap_generator import generate_heatmap_data, find_latest_simulation_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

_last_saved_agent_json_path: Optional[str] = None

def setup_results_folder(base_results_folder: str = 'results_history') -> str:
    """Creates the timestamped results folder."""
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assumes result_handler is in a 'handlers' or 'utils' dir, adjust if needed
    project_root = os.path.dirname(script_dir) + "\V7_code"
    results_folder = os.path.join(project_root, base_results_folder, timestamp)
    try:
        os.makedirs(results_folder, exist_ok=True)
        logging.info(f"Results will be saved in: {results_folder}")
        return results_folder
    except OSError as e:
        logging.error(f"CRITICAL: Error creating results directory {results_folder}: {e}.")
        raise

def save_episode_batch(batch_data: List[Dict], results_folder: str, last_episode_number_in_batch: int):
    """Saves a batch of episode data to a JSON file, named with episode range."""
    # ... (code unchanged from Step 1) ...
    if not batch_data:
        logging.warning("Attempted to save an empty episode batch.")
        return
    try:
        first_episode_in_batch = batch_data[0].get('episode', 'unknown')
        try:
            f_ep = int(first_episode_in_batch); l_ep = int(last_episode_number_in_batch)
            filename_range = f"{f_ep}_to_{l_ep}"
        except (ValueError, TypeError): filename_range = f"{first_episode_in_batch}_to_{last_episode_number_in_batch}"
        filename = os.path.join(results_folder, f'simulation_data_{filename_range}.json')
        with open(filename, 'w') as f: json.dump(batch_data, f, cls=NumpyEncoder, indent=2)
        logging.info(f"Saved batch episodes {filename_range} to {filename}")
    except IndexError: logging.error("Cannot determine filename range: batch_data empty.")
    except TypeError as e: logging.error(f"Error serializing batch data to JSON: {e}.")
    except OSError as e: logging.error(f"OS error saving episode batch to {filename}: {e}")
    except Exception as e: logging.error(f"Unexpected error saving episode batch to {filename}: {e}")


def save_metadata(metadata_dict: Dict, results_folder: str):
    """Saves the simulation metadata dictionary to metadata.json."""
    # ... (code unchanged from Step 1) ...
    filename = os.path.join(results_folder, 'metadata.json')
    try:
        with open(filename, 'w') as f: json.dump(metadata_dict, f, cls=NumpyEncoder, indent=4)
        logging.info(f"Metadata saved successfully to {filename}")
    except TypeError as e: logging.error(f"Error serializing metadata to JSON: {e}.")
    except OSError as e: logging.error(f"OS error saving metadata to {filename}: {e}")
    except Exception as e: logging.error(f"Unexpected error saving metadata to {filename}: {e}")


def save_agent_state(agent: Any, episode: int, results_folder: str) -> Optional[str]:
    """ Saves the agent's state to JSON. """
    # ... (code unchanged from Step 1, updates _last_saved_agent_json_path) ...
    global _last_saved_agent_json_path
    if not hasattr(agent, 'get_agent_state_for_saving'):
         logging.error("Agent has no 'get_agent_state_for_saving' method.")
         return None
    agent_state_data = agent.get_agent_state_for_saving()
    filename = f"agent_state_episode_{episode}.json"
    filepath = os.path.join(results_folder, filename)
    try:
        with open(filepath, 'w') as f: json.dump(agent_state_data, f, cls=NumpyEncoder, indent=4)
        logging.info(f"Agent state saved to {filepath}")
        _last_saved_agent_json_path = filepath
        return filepath
    except TypeError as e: logging.error(f"Serialization error saving agent state: {e}.")
    except IOError as e: logging.error(f"I/O error saving agent state: {e}")
    except Exception as e: logging.error(f"Unexpected error saving agent state: {e}", exc_info=True)
    return None


def convert_json_agent_state_to_excel(json_filepath: str, excel_filepath: str):
    """ Converts saved agent state JSON to Excel with separate sheets. """
    # ... (code mostly unchanged from Step 1, ensure baseline_tables handled) ...
    # This could be made into a separate script/function for independence
    try:
        with open(json_filepath, 'r') as f: agent_state_data = json.load(f)
        logging.info(f"Loaded agent state data from {json_filepath}")

        with pd.ExcelWriter(excel_filepath, engine='openpyxl') as writer:
            # Q-Tables
            if "q_tables" in agent_state_data:
                for gain, q_table_list in agent_state_data["q_tables"].items():
                     if q_table_list:
                        df_q = pd.DataFrame(q_table_list)
                        state_vars = [col for col in df_q.columns if col not in ['0', '1', '2']]
                        if state_vars and all(var in df_q.columns for var in state_vars):
                             try: df_q = df_q.set_index(state_vars)
                             except KeyError: logging.warning(f"Could not set index Q-table '{gain}'.")
                        df_q.to_excel(writer, sheet_name=f"q_table_{gain}")
                     else: logging.warning(f"Q-table data for gain '{gain}' is empty.")

            # Visit Counts
            if "visit_counts" in agent_state_data:
                for gain, visit_count_list in agent_state_data["visit_counts"].items():
                    if visit_count_list:
                        df_v = pd.DataFrame(visit_count_list)
                        state_vars = [col for col in df_v.columns if col not in ['0', '1', '2']]
                        if state_vars and all(var in df_v.columns for var in state_vars):
                            try: df_v = df_v.set_index(state_vars)
                            except KeyError: logging.warning(f"Could not set index Visit Count '{gain}'.")
                        df_v.to_excel(writer, sheet_name=f"visit_counts_{gain}")
                    else: logging.warning(f"Visit Count data for gain '{gain}' is empty.")

            # Baseline Tables
            if "baseline_tables" in agent_state_data:
                for gain, baseline_list in agent_state_data["baseline_tables"].items():
                    if baseline_list:
                        df_b = pd.DataFrame(baseline_list)
                        state_vars = [col for col in df_b.columns if col != 'baseline_value']
                        if state_vars and all(var in df_b.columns for var in state_vars):
                            try: df_b = df_b.set_index(state_vars)
                            except KeyError: logging.warning(f"Could not set index Baseline '{gain}'.")
                        df_b.to_excel(writer, sheet_name=f"baseline_{gain}")
                    else: logging.warning(f"Baseline table data for gain '{gain}' is empty.")

        logging.info(f"Agent state successfully converted to Excel: {excel_filepath}")

    except FileNotFoundError: logging.error(f"JSON agent state file not found: {json_filepath}")
    except json.JSONDecodeError as e: logging.error(f"Error decoding JSON {json_filepath}: {e}")
    except KeyError as e: logging.error(f"Error processing agent state data: Missing key {e}.")
    except ImportError: logging.error("Install 'openpyxl': pip install openpyxl")
    except Exception as e: logging.error(f"Unexpected error converting JSON to Excel: {e}", exc_info=True)


def save_summary_table(summary_list: List[Dict], filename: str):
    """Saves the list of episode summaries to an Excel file."""
    # ... (code unchanged from Step 1) ...
    if not summary_list:
        logging.warning("Summary list is empty. Skipping summary file save.")
        return
    try:
        df = pd.DataFrame(summary_list)
        if 'episode' in df.columns: df['episode'] = df['episode'].astype(int)
        df.to_excel(filename, index=False, engine='openpyxl')
        logging.info(f"Summary saved successfully to {filename}")
    except ImportError: logging.error("Install 'openpyxl': pip install openpyxl")
    except Exception as e: logging.error(f"Failed to save summary to {filename}: {e}", exc_info=True)


# --- Final Results Handling ---
def finalize_results(config: Dict[str, Any],
                     summary_data: List[Dict],
                     all_episodes_data: List[Dict], # Needed for heatmap data gen
                     agent: Any,
                     results_folder: str):
    """Handles final saving steps: agent state, summary, Excel conversion, heatmap data."""
    global _last_saved_agent_json_path

    simulation_cfg = config.get('simulation', {})
    save_agent_state_flag = simulation_cfg.get('save_agent_state', False)
    max_episodes = config.get('environment', {}).get('max_episodes', 0)

    # --- [1] Final Agent State Saving ---
    if save_agent_state_flag:
        logging.info(f"Attempting to save FINAL agent state ep {max_episodes - 1}...")
        try:
            final_episode_num = max_episodes - 1 if max_episodes > 0 else 0
            save_agent_state(agent, final_episode_num, results_folder)
            if not _last_saved_agent_json_path:
                logging.warning("Final agent state save function ran but didn't set path.")
        except Exception as e: logging.error(f"Error saving FINAL agent state: {e}", exc_info=True)
    else: logging.info("Skipping final agent state saving (disabled).")

    # --- [2] Convert Last Agent State JSON to Excel ---
    if _last_saved_agent_json_path:
        logging.info(f"Converting last saved agent state to Excel...")
        excel_output_filename = os.path.join(results_folder, 'agent_state_tables.xlsx')
        try:
            convert_json_agent_state_to_excel(_last_saved_agent_json_path, excel_output_filename)
        except Exception as e: logging.error(f"Error during JSON to Excel conversion: {e}", exc_info=True)
    elif save_agent_state_flag: logging.warning("Agent state saving enabled, but no state file saved? Cannot convert.")
    else: logging.info("Agent state saving disabled, skipping JSON to Excel conversion.")

    # --- [3] Saving Summary Table ---
    logging.info("Saving summary table...")
    if summary_data:
        summary_file_path = os.path.join(results_folder, 'summary.xlsx')
        save_summary_table(summary_data, summary_file_path)
    else: logging.warning("No summary data to save.")

    # --- [4] Generate Heatmap Data ---
    vis_config = config.get('visualization_config') # Use the metadata version passed down
    if vis_config and vis_config.get('enabled') and vis_config.get('plots'):
         logging.info("Generating heatmap data...")
         heatmap_configs = [p for p in vis_config['plots'] if p.get('type') == 'heatmap' and p.get('enabled', True)]
         if heatmap_configs:
             # Find the data file to process (assume last batch for simplicity)
             # A more robust approach might combine all batch files first
             data_file_path = find_latest_simulation_data(results_folder)
             if data_file_path:
                  output_heatmap_excel = os.path.join(results_folder, 'data_heatmaps.xlsx')
                  try:
                       generate_heatmap_data(
                           detailed_data_filepath=data_file_path,
                           heatmap_configs=heatmap_configs,
                           output_excel_filepath=output_heatmap_excel
                       )
                  except Exception as e:
                       logging.error(f"Failed to generate heatmap data: {e}", exc_info=True)
             else:
                  logging.warning("Could not find simulation data file to generate heatmap data.")
         else:
             logging.info("No enabled heatmap configurations found in vis_config.")
    else:
         logging.info("Skipping heatmap data generation (visualization disabled or no heatmap plots).")