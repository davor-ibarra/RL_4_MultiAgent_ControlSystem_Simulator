import numpy as np
import pandas as pd
import math
import logging
from interfaces.stability_calculator import BaseStabilityCalculator # Import Base Class
from typing import Any, Dict, Optional
import copy # Para copiar el dict inicial

class IRAStabilityCalculator(BaseStabilityCalculator): # Inherit from Base
    """
    Calculates instantaneous stability score (w_stab) and stability-based reward
    based on normalized state deviations (z-scores). Can use fixed initial
    reference stats (mu, sigma) or adapt them per episode.
    Assumes state is a numpy array or list: [cart_pos, cart_vel, angle, angular_vel].
    """
    def __init__(self, ira_params: Dict[str, Any]):
        """
        Initializes the calculator with parameters from config.

        Args:
            params (Dict): Dictionary containing 'weights', 'lambda', 'normalization_epsilon',
                           'adaptive_stats', 'initial_reference_stats'.
        """
        logging.info("Initializing IRAStabilityCalculator...")
        try:
            self.weights = ira_params['weights']
            # Lambda solo se usa si reward_setup.calculation.method == 'stability_calculator'
            # Lo guardamos aquí pero GaussianReward decide si lo usa o no.
            self.lambda_param = ira_params.get('lambda', 1.0) # Default si no está
            self.z_score_epsilon = ira_params.get('z_score_epsilon', 1e-6)

            adaptive_cfg = ira_params.get('adaptive_stats', {})
            self.adaptive_enabled = adaptive_cfg.get('enabled', False)
            self.adaptive_min_episode = adaptive_cfg.get('min_episode', 0)
            self.min_sigma = adaptive_cfg.get('min_sigma', self.z_score_epsilon)

            self.initial_ref_stats = ira_params['initial_reference_stats']
            self.current_ref_stats = copy.deepcopy(self.initial_ref_stats)

            # Map state variable names used in config to metric names logged
            self.var_to_metric_map = {
                'angle': 'pendulum_angle',
                'angular_velocity': 'pendulum_velocity',
                'cart_position': 'cart_position',
                'cart_velocity': 'cart_velocity'
            }
            # Map variable names to state vector indices
            self.var_indices = {
                'cart_position': 0,
                'cart_velocity': 1,
                'angle': 2,
                'angular_velocity': 3
            }

            # Validate initial stats keys and weights
            expected_vars = set(self.var_indices.keys())
            if set(self.weights.keys()) != expected_vars:
                logging.warning(f"IRAStabilityCalculator: Mismatch between expected state variables {expected_vars} "
                                f"and keys in weights ({set(self.weights.keys())}). Check config.")
            if set(self.initial_ref_stats.keys()) != expected_vars:
                logging.warning(f"IRAStabilityCalculator: Mismatch between expected state variables {expected_vars} "
                                f"and keys in initial_ref_stats ({set(self.initial_ref_stats.keys())}). Check config.")
            # Ensure all expected vars have initial stats
            for var in expected_vars:
                 if var not in self.current_ref_stats:
                      raise ValueError(f"Missing initial_reference_stats for variable '{var}'")
                 if 'mu' not in self.current_ref_stats[var] or 'sigma' not in self.current_ref_stats[var]:
                      raise ValueError(f"Initial stats for '{var}' must contain 'mu' and 'sigma'.")


            logging.info(f"IRAStabilityCalculator initialized. Adaptive stats: {self.adaptive_enabled} "
                         f"(min_episode={self.adaptive_min_episode}, min_sigma={self.min_sigma})")
            logging.info(f"Initial reference stats: {self.current_ref_stats}")

        except KeyError as e:
            logging.error(f"IRAStabilityCalculator: Missing required parameter key: {e} in params: {ira_params}")
            raise ValueError(f"Missing required parameter for IRAStabilityCalculator: {e}") from e
        except Exception as e:
             logging.error(f"IRAStabilityCalculator: Error during initialization: {e}", exc_info=True)
             raise

    def _normalize_state_variable(self, value: float, var_name: str) -> float:
        """Normalizes a single state variable using CURRENT reference stats."""
        if var_name not in self.current_ref_stats:
            logging.warning(f"Reference stats not found for '{var_name}'. Returning 0 normalized value.")
            return 0.0
        stats = self.current_ref_stats[var_name]
        mu = stats.get('mu', 0.0)
        # Use max of configured sigma, min_std_dev, and norm_epsilon to ensure stability
        sigma = max(stats.get('sigma', 1.0), self.min_sigma, self.z_score_epsilon)
        return (value - mu) / sigma

    # @formula: w_stab = exp(- sum(w_s * z_s^2)) where z_s = (value_s - mu_s) / sigma_s
    def calculate_instantaneous_stability(self, state: Any) -> float:
        """Calculates w_stab = exp(- sum(w_s * z_s^2))."""
        if not isinstance(state, (np.ndarray, list)) or len(state) < 4:
            logging.error(f"IRAStabilityCalculator: Invalid state format for stability calc: {state}")
            return 0.0

        deviation_sum_sq_weighted = 0.0
        for var_name, index in self.var_indices.items():
            weight = self.weights.get(var_name, 0.0)
            if weight > 0: # Only include variables with positive weight
                try:
                    value = state[index]
                    z_s = self._normalize_state_variable(value, var_name)
                    deviation_sum_sq_weighted += weight * (z_s ** 2)
                except IndexError:
                    logging.error(f"IRAStabilityCalculator: State index {index} OOB for '{var_name}' in state {state}.")
                    deviation_sum_sq_weighted += weight * 100 # Penalty scaled by weight
                except Exception as e:
                    logging.error(f"IRAStabilityCalculator: Error processing var '{var_name}' for stability: {e}")
                    deviation_sum_sq_weighted += weight * 100 # Penalty

        try:
            # Clamp exponent argument to prevent overflow with large deviations
            exponent = -min(deviation_sum_sq_weighted, 700) # exp(-700) is already extremely small
            stability_score = math.exp(exponent)
        except OverflowError:
            logging.warning(f"IRAStabilityCalculator: Overflow calculating exp(-{deviation_sum_sq_weighted}). Returning 0.")
            stability_score = 0.0

        return max(0.0, min(stability_score, 1.0)) # Ensure score is [0, 1]

    # @formula: reward = exp(-lambda * sum(w_s * z_s^2))
    def calculate_stability_based_reward(self, state: Any) -> float:
        """Calculates reward = exp(-lambda * sum(w_s * z_s^2))."""
        if not isinstance(state, (np.ndarray, list)) or len(state) < 4:
            logging.error(f"IRAStabilityCalculator (reward): Invalid state format: {state}")
            return -10.0 # Low reward

        deviation_sum_sq_weighted = 0.0
        for var_name, index in self.var_indices.items():
            weight = self.weights.get(var_name, 0.0)
            if weight > 0:
                try:
                    value = state[index]
                    z_s = self._normalize_state_variable(value, var_name)
                    deviation_sum_sq_weighted += weight * (z_s ** 2)
                except IndexError:
                    logging.error(f"IRAStabilityCalculator (reward): State index {index} OOB for '{var_name}'.")
                    deviation_sum_sq_weighted += weight * 100
                except Exception as e:
                    logging.error(f"IRAStabilityCalculator (reward): Error processing var '{var_name}': {e}")
                    deviation_sum_sq_weighted += weight * 100

        try:
            # Clamp exponent argument based on lambda sensitivity
            max_exponent_arg = 700 / max(abs(self.lambda_param), 1e-6)
            exponent = -self.lambda_param * min(deviation_sum_sq_weighted, max_exponent_arg)
            reward = math.exp(exponent)
        except OverflowError:
            logging.warning(f"IRAStabilityCalculator (reward): Overflow calculating exp(-{self.lambda_param} * {deviation_sum_sq_weighted:.4f}).")
            reward = 0.0

        # Reward should probably be non-negative if it's based on exponential decay
        return max(0.0, reward)

    def update_reference_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """
        Updates mu and sigma for each relevant variable based on the metrics
        from the completed episode, if adaptive stats are enabled and min_episode is reached.
        """
        if not self.adaptive_enabled or current_episode < self.adaptive_min_episode:
            return # Do nothing if not enabled or too early

        logging.debug(f"Attempting to update reference stats after episode {current_episode}...")
        updated_any = False
        for var_name_cfg, metric_name in self.var_to_metric_map.items():
            if var_name_cfg in self.current_ref_stats: # Only update if it's configured
                if metric_name in episode_metrics_dict:
                    values = episode_metrics_dict[metric_name]
                    # Ensure values are numeric and filter out potential NaNs/None
                    try:
                         numeric_values = pd.to_numeric(values, errors='coerce')
                         valid_values = numeric_values[~np.isnan(numeric_values)]
                    except NameError: # Fallback if pandas is not available (should be)
                         valid_values = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]


                    if len(valid_values) > 1: # Need at least 2 points for std dev
                        try:
                            new_mu = np.mean(valid_values)
                            new_sigma = np.std(valid_values)

                            # Update internal stats, applying minimum standard deviation
                            effective_sigma = float(max(new_sigma, self.min_std_dev))
                            self.current_ref_stats[var_name_cfg]['mu'] = float(new_mu)
                            self.current_ref_stats[var_name_cfg]['sigma'] = effective_sigma
                            updated_any = True
                            # logging.debug(f"  Updated '{var_name_cfg}': mu={new_mu:.4f}, sigma={effective_sigma:.4f}")
                        except Exception as e:
                            logging.error(f"Error calculating stats for metric '{metric_name}' (var '{var_name_cfg}'): {e}")
                    else:
                        logging.warning(f"Not enough valid data points ({len(valid_values)}) for metric '{metric_name}' in episode {current_episode} to update stats.")
                else:
                    logging.warning(f"Metric '{metric_name}' needed for adaptive stats of '{var_name_cfg}' not found in episode data.")

        #if updated_any:
            #logging.info(f"Reference stats updated after episode {current_episode}. Current stats: {self.current_ref_stats}")
        # else:
             # logging.debug(f"No reference stats updated after episode {current_episode}.")

    def get_current_adaptive_stats(self) -> Dict:
         """Returns the current internal reference statistics (mu, sigma)."""
         # Return a deep copy to prevent external modification
         return copy.deepcopy(self.current_ref_stats)