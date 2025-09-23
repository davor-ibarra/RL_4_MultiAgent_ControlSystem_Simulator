import os
import yaml
import logging
from typing import Dict, Any, Optional, Tuple

# Setup logging (puede ser configurado más centralmente en main.py)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_validate_config(config_filename: str = 'config.yaml', 
                             vis_config_filename_rel: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Loads the main configuration and optionally the visualization configuration.

    Args:
        config_filename (str): Path to the main configuration file (relative to script location).
        vis_config_filename_rel (Optional[str]): Relative path to the visualization config file,
                                                  as specified in the main config. Can be None.

    Returns:
        Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
            A tuple containing (main_config, visualization_config).
            Returns (None, None) if the main config file is not found or fails to parse.
            visualization_config will be None if not enabled, not specified, or fails to load.
    """
    dirname = os.path.dirname(os.path.abspath(__file__)) # Use absolute path of this file
    main_config_path = os.path.join(dirname, config_filename)
    main_config = None
    vis_config = None

    # --- Load Main Configuration ---
    try:
        with open(main_config_path, 'r') as file:
            main_config = yaml.safe_load(file)
            logging.info(f"Main configuration file loaded successfully from: {main_config_path}")
    except FileNotFoundError:
        logging.error(f"CRITICAL: Main configuration file not found at {main_config_path}. Exiting.")
        return None, None
    except yaml.YAMLError as e:
        logging.error(f"CRITICAL: Error parsing main configuration file {main_config_path}: {e}. Exiting.")
        return None, None
    except Exception as e:
        logging.error(f"CRITICAL: Unexpected error loading main config {main_config_path}: {e}. Exiting.", exc_info=True)
        return None, None

    # --- Load Visualization Configuration (if enabled and specified) ---
    vis_config_settings = main_config.get('visualization', {})
    if vis_config_settings.get('enabled', False):
        # Use the relative path provided in the main config if it exists
        specified_vis_filename = vis_config_settings.get('config_file')
        if specified_vis_filename:
            vis_config_path_abs = os.path.join(dirname, specified_vis_filename)
            try:
                with open(vis_config_path_abs, 'r') as vis_file:
                    vis_config = yaml.safe_load(vis_file)
                    logging.info(f"Visualization configuration loaded from: {vis_config_path_abs}")
            except FileNotFoundError:
                logging.warning(f"Visualization config file not found at {vis_config_path_abs}. Plotting disabled.")
            except yaml.YAMLError as e_vis:
                logging.error(f"Error parsing visualization config file {vis_config_path_abs}: {e_vis}. Plotting disabled.")
            except Exception as e_vis:
                logging.error(f"Unexpected error loading visualization config {vis_config_path_abs}: {e_vis}. Plotting disabled.", exc_info=True)
        elif vis_config_filename_rel:
             # Fallback to argument if config_file is missing in main_config but provided as arg
             logging.warning(f"'config_file' not specified under 'visualization' in main config. Trying provided argument '{vis_config_filename_rel}'.")
             vis_config_path_abs = os.path.join(dirname, vis_config_filename_rel)
             try:
                 with open(vis_config_path_abs, 'r') as vis_file:
                     vis_config = yaml.safe_load(vis_file)
                     logging.info(f"Visualization configuration loaded from argument-specified path: {vis_config_path_abs}")
             except FileNotFoundError:
                  logging.warning(f"Visualization config file (from argument) not found at {vis_config_path_abs}. Plotting disabled.")
             except yaml.YAMLError as e_vis:
                  logging.error(f"Error parsing visualization config file (from argument) {vis_config_path_abs}: {e_vis}. Plotting disabled.")
             except Exception as e_vis:
                  logging.error(f"Unexpected error loading visualization config (from argument) {vis_config_path_abs}: {e_vis}. Plotting disabled.", exc_info=True)
        else:
            logging.warning("Visualization enabled but 'config_file' not specified in main config and no fallback filename provided. Plotting disabled.")
    else:
        logging.info("Visualization generation disabled in main config.")
        
    # TODO: Add schema validation here if needed using jsonschema or similar

    return main_config, vis_config