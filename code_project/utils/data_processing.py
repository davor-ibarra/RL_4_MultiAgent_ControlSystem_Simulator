import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any

def _safe_agg(series: pd.Series, agg_func) -> float:
    """Helper to apply aggregation safely, returning NaN on error or empty."""
    if series.empty:
        return np.nan
    try:
        # Drop NaN before aggregation
        valid_series = series.dropna()
        if valid_series.empty:
            return np.nan
        result = agg_func(valid_series)
        # Ensure result is a standard Python float
        return float(result) if not pd.isna(result) else np.nan
    except Exception as e:
        # Log the error appropriately if needed
        # logging.warning(f"Could not aggregate series: {e}")
        return np.nan


def summarize_episode(episode_data: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    Generates a comprehensive summary dictionary for a single episode's detailed data,
    including statistics for all collected metrics.

    Args:
        episode_data: Dictionary where keys are metric names and values are lists
                      of measurements collected during the episode.

    Returns:
        Dictionary containing summary statistics for the episode.
    """
    summary = {'episode': episode_data.get('episode', -1)} # Use .get for safety

    # --- Direct Summary Fields (usually last value or pre-calculated) ---
    summary['termination_reason'] = episode_data.get('termination_reason', 'unknown')

    time_list = episode_data.get('time', [])
    summary['episode_time'] = float(time_list[-1]) if time_list else np.nan

    cumulative_reward_list = episode_data.get('cumulative_reward', [])
    summary['total_reward'] = float(cumulative_reward_list[-1]) if cumulative_reward_list else np.nan

    summary['performance'] = (summary['total_reward'] / summary['episode_time']) \
                             if summary['episode_time'] and summary['episode_time'] > 0 and not pd.isna(summary['total_reward']) \
                             else np.nan

    epsilon_list = episode_data.get('epsilon', [])
    summary['final_epsilon'] = float(epsilon_list[-1]) if epsilon_list else np.nan # Use final value before decay

    lr_list = episode_data.get('learning_rate', [])
    summary['final_learning_rate'] = float(lr_list[-1]) if lr_list else np.nan # Use final value before decay

    ep_duration_list = episode_data.get('episode_duration_s', [])
    summary['episode_duration_s'] = float(ep_duration_list[-1]) if ep_duration_list else np.nan

    avg_stab_list = episode_data.get('avg_stability_score', []) # This might be calculated post-collection
    summary['avg_stability_score'] = float(avg_stab_list) if isinstance(avg_stab_list, (float, int)) else (float(avg_stab_list[-1]) if avg_stab_list else np.nan)


    decisions_list = episode_data.get('total_agent_decisions', [])
    summary['total_agent_decisions'] = int(decisions_list[-1]) if decisions_list else 0

    summary['final_kp'] = float(episode_data.get('final_kp', [np.nan])[-1])
    summary['final_ki'] = float(episode_data.get('final_ki', [np.nan])[-1])
    summary['final_kd'] = float(episode_data.get('final_kd', [np.nan])[-1])


    # --- Aggregated Statistics (Mean, Std, Min, Max) ---
    # List of metrics to apply standard aggregations
    metrics_to_aggregate = [
        # System State & Control
        'cart_position', 'cart_velocity', 'pendulum_angle', 'pendulum_velocity',
        'force', 'error', 'integral_error', 'derivative_error',
        # Controller & Agent Params (over time)
        'kp', 'ki', 'kd', 'epsilon', 'learning_rate',
        # Actions (can be NaN between decisions)
        'action_kp', 'action_ki', 'action_kd',
        # Reward & Stability
        'reward', 'stability_score',
        # Training Internals (can be NaN between decisions)
        'q_value_max_kp', 'q_value_max_ki', 'q_value_max_kd',
        'q_visit_count_state_kp', 'q_visit_count_state_ki', 'q_visit_count_state_kd',
        'baseline_value_kp', 'baseline_value_ki', 'baseline_value_kd',
        'td_error_kp', 'td_error_ki', 'td_error_kd',
        'virtual_reward_kp', 'virtual_reward_ki', 'virtual_reward_kd',
        'learn_select_duration_ms',
        # Gain Step (constant or variable)
        'gain_step', 'gain_step_kp', 'gain_step_ki', 'gain_step_kd'
    ]

    for metric in metrics_to_aggregate:
        values = episode_data.get(metric)

        if values is not None:
            # Convert to pandas Series for easier handling of types and NaN
            # Ensure we handle potential errors during conversion
            try:
                # Attempt conversion, coercing errors to NaT/NaN
                s = pd.Series(values, dtype=float if not metric.startswith(('action','termination','id_')) else object)
                s_numeric = pd.to_numeric(s, errors='coerce')
            except (ValueError, TypeError):
                # If conversion fails entirely, treat as empty
                logging.warning(f"Could not convert metric '{metric}' to numeric for episode {summary['episode']}. Skipping aggregation.")
                s_numeric = pd.Series(dtype=float) # Empty series

            # Calculate aggregates using the helper
            summary[f'{metric}_mean'] = _safe_agg(s_numeric, np.mean)
            summary[f'{metric}_std'] = _safe_agg(s_numeric, np.std)
            summary[f'{metric}_min'] = _safe_agg(s_numeric, np.min)
            summary[f'{metric}_max'] = _safe_agg(s_numeric, np.max)
        else:
            # Handle case where metric key is missing
            # logging.debug(f"Metric '{metric}' not found for summary in episode {summary['episode']}.")
            summary[f'{metric}_mean'] = np.nan
            summary[f'{metric}_std'] = np.nan
            summary[f'{metric}_min'] = np.nan
            summary[f'{metric}_max'] = np.nan
        
        adaptive_vars = ['angle', 'angular_velocity', 'cart_position', 'cart_velocity']
        
        for var_name in adaptive_vars:
         # Extract last mu value
         mu_key = f'adaptive_mu_{var_name}'
         mu_list = episode_data.get(mu_key, [np.nan])
         summary[mu_key] = float(mu_list[-1]) if mu_list else np.nan

         # Extract last sigma value
         sigma_key = f'adaptive_sigma_{var_name}'
         sigma_list = episode_data.get(sigma_key, [np.nan])
         summary[sigma_key] = float(sigma_list[-1]) if sigma_list else np.nan

    return summary

def save_summary_table(summary_list: List[Dict], filename: str):
    """Saves the list of episode summaries to an Excel file."""
    if not summary_list:
        logging.warning("Summary list is empty. Skipping summary file save.")
        return
    try:
        df = pd.DataFrame(summary_list)
        if 'episode' in df.columns:
            df['episode'] = df['episode'].astype(int)
        # Ensure columns are in a somewhat logical order (optional)
        # cols = ['episode', 'termination_reason', 'total_reward', 'performance', ...] + sorted([c for c in df.columns if c not in [...]})
        # df = df[cols]
        df.to_excel(filename, index=False, engine='openpyxl')
        logging.info(f"Summary saved successfully to {filename}")
    except ImportError:
        logging.error("`openpyxl` library not found. Install: pip install openpyxl")
    except Exception as e:
        logging.error(f"Failed to save summary to {filename}: {e}", exc_info=True)