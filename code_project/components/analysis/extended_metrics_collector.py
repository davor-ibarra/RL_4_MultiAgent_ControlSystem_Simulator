from collections import defaultdict
from interfaces.metrics_collector import MetricsCollector
import numpy as np # Import numpy for nan handling
import logging
from typing import Dict, Any, TYPE_CHECKING
if TYPE_CHECKING:
    from components.agents.pid_qlearning_agent import PIDQLearningAgent


class ExtendedMetricsCollector(MetricsCollector):
    """
    Collects detailed metrics during the simulation, storing them in lists
    within a dictionary keyed by metric name. Includes episode tracking
    and methods for logging agent-specific data.
    """
    def __init__(self):
        self.metrics = defaultdict(list)
        self.episode_id = -1 # Track current episode

    def log(self, metric_name: str, metric_value: Any):
        """Logs a single metric value for the current episode."""
        # Convert None to np.nan for consistency in numerical analysis later
        value_to_log = np.nan if metric_value is None else metric_value
        self.metrics[metric_name].append(value_to_log)

    def get_metrics(self) -> dict:
        """Returns a copy of the collected metrics for the current episode."""
        current_metrics = dict(self.metrics)
        current_metrics['episode'] = self.episode_id
        return current_metrics

    def reset(self, episode_id: int = -1):
        """Clears all metrics and sets the current episode ID."""
        self.metrics.clear()
        self.episode_id = episode_id
        # logging.debug(f"Metrics collector reset for episode {self.episode_id}")

    # --- Methods for logging specific agent/training data ---

    def log_q_values(self, agent: 'PIDQLearningAgent', agent_state_dict: Dict):
        """Logs the Q-values for the given state for each gain."""
        try:
            q_values_dict = agent.get_q_values_for_state(agent_state_dict)
            for gain, q_vals in q_values_dict.items():
                # Log all Q-values for the state (or NaN if error)
                # Naming convention: q_value_{gain}_{action_index} might be too verbose
                # Option 1: Log the array (becomes list in JSON)
                # self.log(f'q_values_{gain}', q_vals.tolist() if isinstance(q_vals, np.ndarray) else q_vals)
                # Option 2: Log max Q-value (useful for tracking convergence)
                if isinstance(q_vals, np.ndarray) and q_vals.size > 0:
                     self.log(f'q_value_max_{gain}', np.nanmax(q_vals)) # Use nanmax
                else:
                     self.log(f'q_value_max_{gain}', np.nan)
                # Option 3: Log specific Q-value for action taken (if action known here)
                # Requires passing action index, might be better logged elsewhere

        except Exception as e:
            logging.warning(f"Could not log Q-values: {e}", exc_info=True)
            for gain in getattr(agent, 'gain_variables', ['kp', 'ki', 'kd']):
                self.log(f'q_value_max_{gain}', np.nan)

    def log_q_visit_counts(self, agent: 'PIDQLearningAgent', agent_state_dict: Dict):
        """Logs the visit counts for the given state for each gain."""
        try:
            visit_counts_dict = agent.get_visit_counts_for_state(agent_state_dict)
            for gain, visits in visit_counts_dict.items():
                 # Log total visits to the state (sum across actions)
                 if isinstance(visits, np.ndarray) and visits.size > 0:
                      self.log(f'q_visit_count_state_{gain}', np.sum(visits[visits>=0])) # Sum valid visits
                 else:
                      self.log(f'q_visit_count_state_{gain}', np.nan)
        except Exception as e:
            logging.warning(f"Could not log Q visit counts: {e}", exc_info=True)
            for gain in getattr(agent, 'gain_variables', ['kp', 'ki', 'kd']):
                self.log(f'q_visit_count_state_{gain}', np.nan)


    def log_baselines(self, agent: 'PIDQLearningAgent', agent_state_dict: Dict):
        """Logs the baseline B(s) value for the given state for each gain."""
        try:
            baselines_dict = agent.get_baseline_value_for_state(agent_state_dict)
            for gain, value in baselines_dict.items():
                self.log(f'baseline_value_{gain}', value) # Value is already float or NaN
             # Optional: Log baseline visit count if implemented
             # baseline_visits = agent.get_baseline_visit_counts(agent_state_dict)
             # for gain, visits in baseline_visits.items(): self.log(...)
        except Exception as e:
            logging.warning(f"Could not log baseline values: {e}", exc_info=True)
            for gain in getattr(agent, 'gain_variables', ['kp', 'ki', 'kd']):
                self.log(f'baseline_value_{gain}', np.nan)

    def log_virtual_rewards(self, echo_rewards: Dict[str, float]):
        """Logs the differential rewards calculated by the Echo Baseline method."""
        for gain, value in echo_rewards.items():
            self.log(f'virtual_reward_{gain}', value)

    def log_td_errors(self, td_errors: Dict[str, float]):
        """Logs the TD errors calculated during the last agent learn step."""
        for gain, value in td_errors.items():
            self.log(f'td_error_{gain}', value)
        # Log NaNs for gains where TD error wasn't calculated (e.g., skipped update)
        for gain in ['kp', 'ki', 'kd']:
             if gain not in td_errors:
                  self.log(f'td_error_{gain}', np.nan)


    def log_adaptive_stats(self, stats_dict: Dict[str, Dict[str, float]]):
        """Logs the updated mu and sigma from an adaptive stability calculator."""
        # Example stats_dict: {'angle': {'mu': 0.1, 'sigma': 0.5}, ...}
        for var_name, stats in stats_dict.items():
            self.log(f'adaptive_mu_{var_name}', stats.get('mu', np.nan))
            self.log(f'adaptive_sigma_{var_name}', stats.get('sigma', np.nan))