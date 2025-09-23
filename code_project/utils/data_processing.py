import numpy as np
import pandas as pd
import logging

def summarize_episode(episode_data):
    """Generates a summary dictionary for a single episode's data."""
    summary = {'episode': episode_data.get('episode', -1)} # Use .get for safety

    # List of numeric metrics expected in the episode data lists
    numeric_metrics = [
        'kp', 'ki', 'kd', 'pendulum_angle', 'pendulum_velocity',
        'cart_position', 'cart_velocity', 'force', 'reward', 'error'
    ]

    # --- Specific Summary Fields ---
    # Total reward (last value of cumulative_reward list)
    cumulative_reward = episode_data.get('cumulative_reward')
    summary['total_reward'] = float(cumulative_reward[-1]) if isinstance(cumulative_reward, (list, np.ndarray)) and len(cumulative_reward) > 0 else np.nan

    # Episode time (last value of time list)
    time = episode_data.get('time')
    summary['episode_time'] = float(time[-1]) if isinstance(time, (list, np.ndarray)) and len(time) > 0 else np.nan

    # Performance = Total Reward/Episode Time
    summary['performance'] = summary['total_reward']/summary['episode_time']

    # Termination Reason (replace 'finished')
    summary['termination_reason'] = episode_data.get('termination_reason', 'unknown')

    # Remove the old 'finished' key if present
    if 'finished' in summary:
        del summary['finished']
    if 'finished' in episode_data: # Also check original data dict if needed
         pass # Don't need to delete from source, just exclude from summary

    # Final Epsilon/LR (use last value from lists if available)
    epsilon_list = episode_data.get('epsilon')
    summary['final_epsilon'] = float(epsilon_list[-1]) if isinstance(epsilon_list, (list, np.ndarray)) and len(epsilon_list) > 0 else np.nan

    lr_list = episode_data.get('learning_rate')
    summary['final_learning_rate'] = float(lr_list[-1]) if isinstance(lr_list, (list, np.ndarray)) and len(lr_list) > 0 else np.nan


    for metric in numeric_metrics:
        values = episode_data.get(metric) # Use .get for safety

        # Check if values exist and is a list/array
        if values is not None and isinstance(values, (list, np.ndarray)) and len(values) > 0:
            try:
                # Convert to numeric, coercing errors (like potential None/str) to NaN
                numeric_values = pd.to_numeric(values, errors='coerce')
                # Filter out NaN values before calculation
                valid_values = numeric_values[~np.isnan(numeric_values)]

                if len(valid_values) > 0:
                    summary[f'{metric}_mean'] = float(np.mean(valid_values)) # Ensure float type
                    summary[f'{metric}_std'] = float(np.std(valid_values))
                    summary[f'{metric}_min'] = float(np.min(valid_values))
                    summary[f'{metric}_max'] = float(np.max(valid_values))
                else:
                    # Handle case where all values became NaN
                    summary[f'{metric}_mean'] = np.nan
                    summary[f'{metric}_std'] = np.nan
                    summary[f'{metric}_min'] = np.nan
                    summary[f'{metric}_max'] = np.nan
            except Exception as e:
                logging.warning(f"Could not summarize metric '{metric}' for episode {summary['episode']}: {e}")
                summary[f'{metric}_mean'] = np.nan
                summary[f'{metric}_std'] = np.nan
                summary[f'{metric}_min'] = np.nan
                summary[f'{metric}_max'] = np.nan
        else:
            # Handle case where metric key is missing, not a list, or empty
            # logging.debug(f"Metric '{metric}' not found or invalid for summary in episode {summary['episode']}.")
            summary[f'{metric}_mean'] = np.nan
            summary[f'{metric}_std'] = np.nan
            summary[f'{metric}_min'] = np.nan
            summary[f'{metric}_max'] = np.nan

    return summary

def save_summary_table(summary_list, filename):
    """Saves the list of episode summaries to an Excel file."""
    if not summary_list:
        logging.warning("Summary list is empty. Skipping summary file save.")
        return
    try:
        df = pd.DataFrame(summary_list)
        # Ensure episode column is integer and set as index if desired, otherwise keep as column
        if 'episode' in df.columns:
            df['episode'] = df['episode'].astype(int)
            # df = df.set_index('episode') # Optional: set episode as index

        # Use openpyxl engine for .xlsx format
        df.to_excel(filename, index=False, engine='openpyxl') # index=False keeps 'episode' as a column
        logging.info(f"Summary saved successfully to {filename}")
    except ImportError:
        logging.error("`openpyxl` library not found. Cannot save to .xlsx. Please install it: pip install openpyxl")
        # Fallback to CSV?
        # csv_filename = filename.replace('.xlsx', '.csv')
        # try:
        #     df.to_csv(csv_filename, index=False)
        #     logging.info(f"Saved summary as CSV fallback: {csv_filename}")
        # except Exception as e_csv:
        #     logging.error(f"Failed to save summary to CSV fallback {csv_filename}: {e_csv}")
    except Exception as e:
        logging.error(f"Failed to save summary to {filename}: {e}")
