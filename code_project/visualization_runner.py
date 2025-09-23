import logging
import pandas as pd
import os
from typing import List, Dict, Optional

# Assuming visualization utilities are in utils
from utils.visualization import generate_plots

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_visualizations(vis_config: Optional[Dict],
                       summary_data: List[Dict],
                       all_episodes_data: List[Dict],
                       results_folder: str):
    """
    Generates plots based on the visualization configuration and simulation data.

    Args:
        vis_config: The loaded visualization configuration dictionary, or None.
        summary_data: The list of summary dictionaries for each episode.
        all_episodes_data: The list of detailed data dictionaries for each episode.
        results_folder: The path to save the generated plots.
    """
    logging.info("--- Generating Visualizations ---")

    if not vis_config or not vis_config.get('plots'):
        if vis_config is not None:
             logging.warning("Visualization config loaded, but no 'plots' section found or empty. Skipping plot generation.")
        else:
             logging.info("Visualization config not loaded or visualization disabled. Skipping plot generation.")
        return

    if not summary_data and not all_episodes_data:
         logging.warning("No summary or detailed data available. Cannot generate plots.")
         return

    # Convert summary data list to DataFrame for plotting functions that expect it
    summary_df = pd.DataFrame(summary_data) if summary_data else pd.DataFrame()

    # --- Plotting ---
    logging.info("Generating plots based on visualization config...")
    try:
        # generate_plots expects the list of plot configs from the vis_config file
        generate_plots(
            plot_configs=vis_config['plots'],
            summary_df=summary_df,           # Pass DataFrame
            detailed_data=all_episodes_data, # Pass list of dicts
            results_folder=results_folder
        )
        logging.info("Configurable plot generation finished.")
    except NameError:
        logging.error("Function 'generate_plots' not found/imported from visualization module.")
    except KeyError:
         logging.error("Error accessing 'plots' key in visualization config. Check config structure.")
    except Exception as plot_e:
        logging.error(f"Error during configurable plot generation: {plot_e}", exc_info=True)