import json
import os
import logging
import numpy as np # Import numpy
from .numpy_encoder import NumpyEncoder # Use the existing encoder
import pandas as pd

def save_agent_state(agent, episode, results_folder):
    """
    Saves the agent's Q-tables and visit counts to a JSON file.

    Args:
        agent: The agent instance (must have a get_agent_state_for_saving method).
        episode (int): The current episode number (used for filename).
        results_folder (str): The path to the directory where results are stored.
    """
    if not hasattr(agent, 'get_agent_state_for_saving'):
        logging.error("Agent object does not have 'get_agent_state_for_saving' method. Cannot save state.")
        return

    filename = os.path.join(results_folder, f'agent_state_episode_{episode}.json')
    logging.info(f"Saving agent state (Q-tables & Visit Counts) for episode {episode} to {filename}")

    try:
        agent_state_data = agent.get_agent_state_for_saving()

        # Add episode number to the dictionary being saved for context
        data_to_save = {
            "episode": episode,
            **agent_state_data # Unpack the dictionary from the agent
        }

        with open(filename, 'w') as f:
            # Use indent=None for smaller file size, or indent=2 for readability
            json.dump(data_to_save, f, cls=NumpyEncoder, indent=None)

        logging.info(f"Agent state (JSON) saved successfully to {filename}")
        return filename

    except TypeError as e:
        logging.error(f"Error serializing agent state to JSON: {e}. Check data types within Q-tables/Visit Counts.")
        return None
    except OSError as e:
        logging.error(f"OS error saving agent state to {filename}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error saving agent state to {filename}: {e}", exc_info=True)
        return None

def convert_json_agent_state_to_excel(json_filepath, excel_filepath):
    """
    Reads an agent state JSON file (with the list-of-dicts structure)
    and saves the Q-tables and Visit Counts into an Excel file with MultiIndex.

    Args:
        json_filepath (str): Path to the input JSON agent state file.
        excel_filepath (str): Path for the output Excel file.
    """
    logging.info(f"Converting agent state from JSON '{json_filepath}' to Excel '{excel_filepath}'...")

    try:
        # 1. Leer el archivo JSON
        with open(json_filepath, 'r') as f:
            agent_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"JSON file not found: {json_filepath}. Cannot convert to Excel.")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON file {json_filepath}: {e}. Cannot convert to Excel.")
        return
    except Exception as e:
        logging.error(f"Error reading JSON file {json_filepath}: {e}", exc_info=True)
        return

    q_tables_data = agent_data.get('q_tables', {})
    visit_counts_data = agent_data.get('visit_counts', {})

    if not q_tables_data and not visit_counts_data:
        logging.warning(f"JSON file {json_filepath} does not contain 'q_tables' or 'visit_counts'. Skipping Excel conversion.")
        return

    try:
        with pd.ExcelWriter(excel_filepath, engine='openpyxl') as writer:
            # Procesar Q-Tables
            for gain, table_list in q_tables_data.items():
                if not table_list:
                    logging.warning(f"Q-table for gain '{gain}' is empty in JSON. Skipping sheet.")
                    continue
                try:
                    df = pd.DataFrame(table_list)
                    # Identificar columnas de estado (las que no son índices de acción como '0', '1', '2')
                    action_cols = [col for col in df.columns if col.isdigit()]
                    state_cols = [col for col in df.columns if col not in action_cols]
                    if not state_cols:
                         logging.warning(f"Could not identify state columns for Q-table '{gain}'. Saving without index.")
                    else:
                        df = df.set_index(state_cols)
                    sheet_name = f'Q_Table_{gain}'
                    df.to_excel(writer, sheet_name=sheet_name, index=True)
                    logging.debug(f"Q-table DataFrame for '{gain}' added to sheet '{sheet_name}'.")
                except Exception as e_df:
                     logging.error(f"Error processing Q-table for gain '{gain}' into DataFrame/Excel: {e_df}", exc_info=True)

            # Procesar Visit Counts
            for gain, table_list in visit_counts_data.items():
                if not table_list:
                    logging.warning(f"Visit Counts table for gain '{gain}' is empty in JSON. Skipping sheet.")
                    continue
                try:
                    df = pd.DataFrame(table_list)
                    action_cols = [col for col in df.columns if col.isdigit()]
                    state_cols = [col for col in df.columns if col not in action_cols]
                    if not state_cols:
                        logging.warning(f"Could not identify state columns for Visit Counts '{gain}'. Saving without index.")
                    else:
                        df = df.set_index(state_cols)
                    sheet_name = f'Visits_{gain}'
                    df.to_excel(writer, sheet_name=sheet_name, index=True)
                    logging.debug(f"Visit Counts DataFrame for '{gain}' added to sheet '{sheet_name}'.")
                except Exception as e_df:
                     logging.error(f"Error processing Visit Counts for gain '{gain}' into DataFrame/Excel: {e_df}", exc_info=True)

        logging.info(f"Agent state successfully converted to Excel: {excel_filepath}")

    except ImportError:
        logging.error("`openpyxl` library not found. Cannot save to .xlsx. Please install it: pip install openpyxl")
    except OSError as e:
        logging.error(f"OS error saving agent state Excel {excel_filepath}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error converting/saving agent state Excel {excel_filepath}: {e}", exc_info=True)