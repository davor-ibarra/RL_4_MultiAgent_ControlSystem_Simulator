import numpy as np
import pandas as pd
import logging
import os
import yaml
import json
import argparse # For command-line execution
from typing import List, Dict, Optional, Tuple

# Configure logging for standalone use
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

def _extract_heatmap_data(
    detailed_data: List[Dict],
    x_var: str,
    y_var: str,
    filter_reasons: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts and concatenates X, Y data for heatmaps from detailed episode list,
    with optional filtering by termination reason. Handles NaNs.
    """
    x_all, y_all = [], []

    if not isinstance(detailed_data, list):
         logging.error("Invalid detailed_data format: Expected a list of dictionaries.")
         return np.array([]), np.array([])

    filtered_episodes = detailed_data
    if filter_reasons:
        original_count = len(filtered_episodes)
        try:
            filtered_episodes = [ep for ep in detailed_data if isinstance(ep, dict) and ep.get('termination_reason') in filter_reasons]
            logging.info(f"Heatmap filtering on {filter_reasons}: Kept {len(filtered_episodes)}/{original_count} episodes.")
        except Exception as e:
             logging.error(f"Error during episode filtering: {e}")
             return np.array([]), np.array([])
        if not filtered_episodes:
            logging.warning(f"No episodes left after filtering for reasons: {filter_reasons}")
            return np.array([]), np.array([])

    for i, episode in enumerate(filtered_episodes):
        if not isinstance(episode, dict):
             logging.warning(f"Skipping episode {i} in heatmap data extraction: not a dictionary.")
             continue

        x_raw = episode.get(x_var)
        y_raw = episode.get(y_var)

        if isinstance(x_raw, (list, np.ndarray)) and isinstance(y_raw, (list, np.ndarray)):
            min_len = min(len(x_raw), len(y_raw))
            if min_len > 0:
                # Convert to numeric, coercing errors, crucial for handling None/'null'/str
                x_num = pd.to_numeric(x_raw[:min_len], errors='coerce')
                y_num = pd.to_numeric(y_raw[:min_len], errors='coerce')
                # Create a mask for valid (non-NaN) pairs
                valid_mask = ~np.isnan(x_num) & ~np.isnan(y_num)
                if np.any(valid_mask):
                    x_all.append(x_num[valid_mask])
                    y_all.append(y_num[valid_mask])
                # else:
                    # logging.debug(f"No valid (x,y) pairs found for {x_var} vs {y_var} in episode {episode.get('episode', i)}")
        # else:
            # logging.debug(f"Missing or invalid data type for {x_var} or {y_var} in episode {episode.get('episode', i)}")


    if not x_all or not y_all:
         logging.warning(f"No data points found for heatmap {y_var} vs {x_var} after processing all episodes.")
         return np.array([]), np.array([])

    try:
        x_combined = np.concatenate(x_all)
        y_combined = np.concatenate(y_all)
        logging.info(f"Extracted {len(x_combined)} valid data points for heatmap {y_var} vs {x_var}.")
        return x_combined, y_combined
    except ValueError as e:
         logging.error(f"Error concatenating data arrays for heatmap {y_var} vs {x_var}: {e}")
         return np.array([]), np.array([])


def generate_heatmap_data(
    detailed_data_filepath: str,
    heatmap_configs: List[Dict],
    output_excel_filepath: str
):
    """
    Generates 2D histogram data for specified variable pairs and saves to Excel.

    Args:
        detailed_data_filepath: Path to the JSON file containing the list of
                                 detailed episode dictionaries (e.g., simulation_data*.json).
        heatmap_configs: List of configuration dictionaries for each heatmap, typically
                          from the 'plots' section of sub_config_visualization.yaml where type='heatmap'.
                          Each dict must contain 'x_variable', 'y_variable', and 'output_filename'.
                          Optional keys from 'config': 'filter_termination_reason', 'bins',
                          'xmin', 'xmax', 'ymin', 'ymax'.
        output_excel_filepath: Path to save the output Excel file (.xlsx).
    """
    logging.info(f"Starting heatmap data generation. Input: {detailed_data_filepath}, Output: {output_excel_filepath}")

    # --- [1] Load Detailed Data ---
    try:
        with open(detailed_data_filepath, 'r') as f:
            # Load potentially large JSON file incrementally if needed in future
            detailed_data = json.load(f)
        if not isinstance(detailed_data, list):
            raise TypeError("Loaded data is not a list.")
        logging.info(f"Successfully loaded detailed data for {len(detailed_data)} episodes from {detailed_data_filepath}.")
    except FileNotFoundError:
        logging.error(f"Detailed data file not found: {detailed_data_filepath}")
        return
    except (json.JSONDecodeError, TypeError) as e:
        logging.error(f"Error loading or parsing JSON data from {detailed_data_filepath}: {e}")
        return
    except Exception as e:
         logging.error(f"Unexpected error loading detailed data: {e}", exc_info=True)
         return

    # --- [2] Process Each Heatmap Config ---
    try:
        with pd.ExcelWriter(output_excel_filepath, engine='openpyxl') as writer:
            logging.info(f"Processing {len(heatmap_configs)} heatmap configurations...")
            processed_count = 0
            for i, plot_cfg in enumerate(heatmap_configs):
                if not isinstance(plot_cfg, dict):
                    logging.warning(f"Skipping heatmap config #{i+1}: Not a dictionary.")
                    continue
                if plot_cfg.get('type') != 'heatmap' or not plot_cfg.get('enabled', True):
                    # logging.debug(f"Skipping config #{i+1}: Not an enabled heatmap.")
                    continue

                cfg = plot_cfg.get('config', {})
                x_var = plot_cfg.get('x_variable')
                y_var = plot_cfg.get('y_variable')
                sheet_name_base = os.path.splitext(plot_cfg.get('output_filename', f"heatmap_{i+1}"))[0]
                # Shorten sheet name if too long for Excel limits (31 chars)
                sheet_name = sheet_name_base[:31]
                if len(sheet_name_base) > 31:
                     logging.warning(f"Heatmap sheet name '{sheet_name_base}' truncated to '{sheet_name}'.")


                if not x_var or not y_var:
                    logging.warning(f"Skipping heatmap config #{i+1} ('{sheet_name}'): Missing 'x_variable' or 'y_variable'.")
                    continue

                logging.info(f"--- Processing Heatmap: {sheet_name} ({y_var} vs {x_var}) ---")

                # --- [2a] Extract Data ---
                filter_reasons = cfg.get('filter_termination_reason')
                x_data, y_data = _extract_heatmap_data(detailed_data, x_var, y_var, filter_reasons)

                if x_data.size == 0 or y_data.size == 0:
                    logging.warning(f"No valid data points found for heatmap '{sheet_name}'. Skipping sheet.")
                    continue

                # --- [2b] Define Histogram Bins and Range ---
                bins = cfg.get('bins', 100)
                # Get range limits from config, use None if not specified for auto-ranging
                xmin_cfg, xmax_cfg = cfg.get('xmin'), cfg.get('xmax')
                ymin_cfg, ymax_cfg = cfg.get('ymin'), cfg.get('ymax')
                hist_range = None
                # Only set range if ALL limits are provided, otherwise let numpy determine
                if all(v is not None for v in [xmin_cfg, xmax_cfg, ymin_cfg, ymax_cfg]):
                    hist_range = [[xmin_cfg, xmax_cfg], [ymin_cfg, ymax_cfg]]
                    logging.debug(f"Using specified range: X=[{xmin_cfg}, {xmax_cfg}], Y=[{ymin_cfg}, {ymax_cfg}]")
                else:
                     logging.debug("Using automatic range for histogram.")

                # --- [2c] Calculate 2D Histogram ---
                try:
                    counts, xedges, yedges = np.histogram2d(x_data, y_data, bins=bins, range=hist_range)
                    logging.info(f"Calculated histogram for '{sheet_name}'. Shape: {counts.shape}")
                except Exception as e_hist:
                    logging.error(f"Error calculating histogram for '{sheet_name}': {e_hist}", exc_info=True)
                    continue

                # --- [2d] Prepare DataFrames for Excel ---
                # Metadata DataFrame
                metadata = {
                    'Parameter': ['X Variable', 'Y Variable', 'Bins', 'X Range Min', 'X Range Max', 'Y Range Min', 'Y Range Max', 'Filter Reasons', 'Data Points'],
                    'Value': [
                        x_var, y_var, bins,
                        xedges[0] if hist_range else np.min(x_data), # Actual min used
                        xedges[-1] if hist_range else np.max(x_data),# Actual max used
                        yedges[0] if hist_range else np.min(y_data), # Actual min used
                        yedges[-1] if hist_range else np.max(y_data),# Actual max used
                        str(filter_reasons) if filter_reasons else 'None',
                        len(x_data)
                     ]
                }
                df_meta = pd.DataFrame(metadata)

                # Histogram Counts DataFrame (Y = rows, X = columns)
                # Use bin centers or edges for index/columns if preferred, but edges might be long
                # Using integer index/columns is simpler for basic export
                # Use yedges[:-1] as index (lower bounds of y bins)
                # Use xedges[:-1] as columns (lower bounds of x bins)
                df_counts = pd.DataFrame(counts, index=yedges[:-1], columns=xedges[:-1])
                df_counts.index.name = f"{y_var}_bin_start"
                df_counts.columns.name = f"{x_var}_bin_start"


                # --- [2e] Write to Excel Sheet ---
                df_meta.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
                # Add some space before the main data table
                df_counts.to_excel(writer, sheet_name=sheet_name, startrow=df_meta.shape[0] + 2)
                logging.info(f"Saved data for heatmap '{sheet_name}' to Excel.")
                processed_count += 1

    except ImportError:
        logging.error("Error writing Excel file: Optional dependency 'openpyxl' not found. Install: pip install openpyxl")
    except Exception as e:
        logging.error(f"Error writing heatmap data to Excel file {output_excel_filepath}: {e}", exc_info=True)

    logging.info(f"Heatmap data generation finished. Processed {processed_count} heatmap configurations.")


def find_latest_simulation_data(results_folder: str) -> Optional[str]:
     """Finds the simulation data JSON file with the highest episode number."""
     latest_file = None
     highest_ep = -1
     try:
         for filename in os.listdir(results_folder):
             if filename.startswith('simulation_data_') and filename.endswith('.json'):
                 parts = filename[:-5].split('_') # Remove .json and split by _
                 if len(parts) >= 4 and parts[-2] == 'to':
                     try:
                         end_ep = int(parts[-1])
                         if end_ep > highest_ep:
                             highest_ep = end_ep
                             latest_file = os.path.join(results_folder, filename)
                     except ValueError:
                         continue # Ignore files not matching the naming scheme
     except FileNotFoundError:
          logging.error(f"Results folder not found for finding latest data: {results_folder}")
          return None
     except Exception as e:
          logging.error(f"Error finding latest simulation data file: {e}")
          return None

     if latest_file:
          logging.info(f"Found latest simulation data file: {latest_file}")
     else:
          logging.warning(f"No simulation data files found in: {results_folder}")
     return latest_file


if __name__ == "__main__":
    # --- Standalone Execution Logic ---
    parser = argparse.ArgumentParser(description="Generate Heatmap Data from Simulation Results")
    parser.add_argument(
        "-d", "--datafile",
        help="Path to the detailed simulation data JSON file (e.g., results/.../simulation_data_0_to_999.json)."
             " If not provided, tries to find the latest in the results folder."
    )
    parser.add_argument(
        "-r", "--resultsfolder", required=True,
        help="Path to the main results folder containing simulation data and configs."
    )
    parser.add_argument(
        "-v", "--visconfig", default='sub_config_visualization.yaml',
        help="Filename of the visualization config (relative to results folder or absolute path)."
    )
    parser.add_argument(
        "-o", "--output", default='data_heatmaps.xlsx',
        help="Output Excel filename (will be saved in results folder)."
    )

    args = parser.parse_args()

    # --- Determine Input Data File ---
    data_file = args.datafile
    if not data_file:
        logging.info("Data file not specified, attempting to find latest simulation data...")
        data_file = find_latest_simulation_data(args.resultsfolder)
        if not data_file:
            logging.error("Could not find simulation data file. Exiting.")
            exit(1)
    elif not os.path.isabs(data_file):
         data_file = os.path.join(args.resultsfolder, data_file) # Assume relative to results folder

    # --- Determine Visualization Config File ---
    vis_config_path = args.visconfig
    if not os.path.isabs(vis_config_path):
         vis_config_path = os.path.join(args.resultsfolder, vis_config_path) # Assume relative

    # --- Determine Output File ---
    output_file = os.path.join(args.resultsfolder, args.output)

    # --- Load Visualization Config ---
    try:
        with open(vis_config_path, 'r') as f:
            vis_config = yaml.safe_load(f)
        if not vis_config or 'plots' not in vis_config:
            logging.error(f"Visualization config '{vis_config_path}' is empty or missing 'plots' section.")
            exit(1)
        heatmap_configs = [p for p in vis_config['plots'] if p.get('type') == 'heatmap' and p.get('enabled', True)]
        if not heatmap_configs:
             logging.warning("No enabled heatmap configurations found in visualization config.")
             exit(0) # Not an error, just nothing to do
    except FileNotFoundError:
        logging.error(f"Visualization config file not found: {vis_config_path}")
        exit(1)
    except yaml.YAMLError as e:
         logging.error(f"Error parsing visualization config {vis_config_path}: {e}")
         exit(1)
    except Exception as e:
         logging.error(f"Error loading visualization config: {e}", exc_info=True)
         exit(1)


    # --- Run Heatmap Generation ---
    generate_heatmap_data(
        detailed_data_filepath=data_file,
        heatmap_configs=heatmap_configs,
        output_excel_filepath=output_file
    )