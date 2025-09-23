import json
import os
import logging
import pandas as pd
import numpy as np
from typing import Optional
from components.agents.pid_qlearning_agent import PIDQLearningAgent # Import agent for type hinting

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)): # Handle arrays if needed
            return obj.tolist() # Convert arrays to lists
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)): # Handle void types if necessary
            return None
        return json.JSONEncoder.default(self, obj)

def save_agent_state(agent: PIDQLearningAgent, episode: int, results_folder: str) -> Optional[str]:
    """
    Saves the agent's Q-tables, visit counts, and baseline tables to a JSON file
    using the Pandas-friendly format provided by the agent.

    Args:
        agent: The PIDQLearningAgent instance.
        episode: The current episode number.
        results_folder: The folder where results are stored.

    Returns:
        The path to the saved file, or None if saving failed.
    """
    agent_state_data = agent.get_agent_state_for_saving() # Gets the structured dict
    filename = f"agent_state_episode_{episode}.json"
    filepath = os.path.join(results_folder, filename)

    try:
        with open(filepath, 'w') as f:
            # Use NumpyEncoder to handle potential numpy types within the agent state
            json.dump(agent_state_data, f, cls=NumpyEncoder, indent=4)
        logging.info(f"Agent state (Q, Visit, Baseline tables) saved to {filepath}")
        return filepath
    except TypeError as e:
        logging.error(f"Serialization error saving agent state to {filepath}: {e}. "
                      f"Check if agent state contains unsupported types.")
    except IOError as e:
        logging.error(f"I/O error saving agent state to {filepath}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error saving agent state to {filepath}: {e}", exc_info=True)

    return None # Return None if saving failed

def convert_json_agent_state_to_excel(json_filepath: str, excel_filepath: str):
    """
    Converts the saved agent state JSON (Pandas-friendly format) into an Excel file
    with separate sheets for Q-tables, visit counts, and baseline tables per gain.

    Args:
        json_filepath: Path to the input JSON file.
        excel_filepath: Path for the output Excel file.
    """
    try:
        with open(json_filepath, 'r') as f:
            agent_state_data = json.load(f)
        logging.info(f"Loaded agent state data from {json_filepath}")

        with pd.ExcelWriter(excel_filepath) as writer:
            # --- Process Q-Tables ---
            if "q_tables" in agent_state_data:
                for gain, q_table_list in agent_state_data["q_tables"].items():
                    if q_table_list: # Check if list is not empty
                        df_q = pd.DataFrame(q_table_list)
                        # Try to set state variables as index if they exist consistently
                        state_vars = [col for col in df_q.columns if col not in ['0', '1', '2']] # Adjust if num_actions changes
                        if state_vars and all(var in df_q.columns for var in state_vars):
                             try:
                                  df_q = df_q.set_index(state_vars)
                             except KeyError:
                                  logging.warning(f"Could not set index for Q-table '{gain}'. Keeping default index.")
                        df_q.to_excel(writer, sheet_name=f"q_table_{gain}")
                        logging.debug(f"Saved Q-table for '{gain}' to Excel.")
                    else:
                         logging.warning(f"Q-table data for gain '{gain}' is empty. Skipping Excel sheet.")


            # --- Process Visit Counts ---
            if "visit_counts" in agent_state_data:
                for gain, visit_count_list in agent_state_data["visit_counts"].items():
                     if visit_count_list:
                         df_v = pd.DataFrame(visit_count_list)
                         state_vars = [col for col in df_v.columns if col not in ['0', '1', '2']]
                         if state_vars and all(var in df_v.columns for var in state_vars):
                              try:
                                   df_v = df_v.set_index(state_vars)
                              except KeyError:
                                   logging.warning(f"Could not set index for Visit Count table '{gain}'. Keeping default index.")
                         df_v.to_excel(writer, sheet_name=f"visit_counts_{gain}")
                         logging.debug(f"Saved Visit Counts for '{gain}' to Excel.")
                     else:
                          logging.warning(f"Visit Count data for gain '{gain}' is empty. Skipping Excel sheet.")

            # --- NEW: Process Baseline Tables ---
            if "baseline_tables" in agent_state_data:
                for gain, baseline_list in agent_state_data["baseline_tables"].items():
                    if baseline_list:
                        df_b = pd.DataFrame(baseline_list)
                        # State variables should be all columns except 'baseline_value'
                        state_vars = [col for col in df_b.columns if col != 'baseline_value']
                        if state_vars and all(var in df_b.columns for var in state_vars):
                            try:
                                df_b = df_b.set_index(state_vars)
                            except KeyError:
                                logging.warning(f"Could not set index for Baseline table '{gain}'. Keeping default index.")
                        df_b.to_excel(writer, sheet_name=f"baseline_{gain}")
                        logging.debug(f"Saved Baseline table for '{gain}' to Excel.")
                    else:
                         logging.warning(f"Baseline table data for gain '{gain}' is empty or not found. Skipping Excel sheet.")


        logging.info(f"Agent state successfully converted to Excel: {excel_filepath}")

    except FileNotFoundError:
        logging.error(f"JSON agent state file not found at: {json_filepath}")
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON file {json_filepath}: {e}")
    except KeyError as e:
         logging.error(f"Error processing agent state data: Missing expected key {e}. Check JSON structure.")
    except ImportError:
         logging.error("Error writing Excel file: Optional dependency 'openpyxl' not found. Install it using 'pip install openpyxl'")
    except Exception as e:
        logging.error(f"Unexpected error converting JSON to Excel: {e}", exc_info=True)