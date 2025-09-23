import numpy as np
import math
import logging
from interfaces.stability_calculator import StabilityCalculator
from typing import Any, Dict, Optional
import copy # Para copiar el dict inicial

class IRAStabilityCalculator(StabilityCalculator):
    """
    Calculates instantaneous stability score and reward based on normalized
    state deviations. Can use fixed initial reference stats or adapt them
    per episode based on collected metrics.
    Assumes state is a numpy array or list: [cart_pos, cart_vel, angle, angular_vel].
    """
    def __init__(self, params: Dict[str, Any]):
        """
        Initializes the calculator with parameters from config.

        Args:
            params (Dict): Dictionary possibly containing 'weights', 'lambda',
                           'normalization_epsilon', 'adaptive_stats', 'initial_reference_stats'.
        """
        try:
            self.weights = params['weights']
            self.lambda_param = params['lambda']
            # --- RENOMBRADO y valor por defecto ---
            self.norm_epsilon = params.get('normalization_epsilon', 1e-6)

            # --- NUEVO: Manejo de Estadísticas Adaptativas ---
            adaptive_cfg = params.get('adaptive_stats', {})
            self.adaptive_enabled = adaptive_cfg.get('enabled', False)
            self.adaptive_min_episode = adaptive_cfg.get('min_episode', 0)
            self.min_std_dev = adaptive_cfg.get('min_std_dev', self.norm_epsilon) # Usar un mínimo para sigma

            self.initial_ref_stats = params['initial_reference_stats']
            # Deep copy para no modificar el original si se reutiliza la config
            self.current_ref_stats = copy.deepcopy(self.initial_ref_stats)

            # Map state variable names used in config to metric names logged by SimpleMetricsCollector
            # Esto es crucial para que update_reference_stats funcione
            self.var_to_metric_map = {
                'angle': 'pendulum_angle',
                'angular_velocity': 'pendulum_velocity',
                'cart_position': 'cart_position',
                'cart_velocity': 'cart_velocity'
            }
             # Map variable names to state vector indices for instantaneous calculations
            self.var_indices = {
                'cart_position': 0,
                'cart_velocity': 1,
                'angle': 2,
                'angular_velocity': 3
            }

            # Validate initial stats keys
            expected_vars = set(self.var_indices.keys())
            if set(self.weights.keys()) != expected_vars or set(self.initial_ref_stats.keys()) != expected_vars:
                 logging.warning(f"IRAStabilityCalculator: Mismatch between expected state variables {expected_vars} "
                                 f"and keys in weights ({set(self.weights.keys())}) or "
                                 f"initial_ref_stats ({set(self.initial_ref_stats.keys())}). Check config.")

            logging.info(f"IRAStabilityCalculator initialized. Adaptive stats: {self.adaptive_enabled} (min_episode={self.adaptive_min_episode})")
            logging.info(f"Initial reference stats: {self.current_ref_stats}")

        except KeyError as e:
            logging.error(f"IRAStabilityCalculator: Missing required parameter key: {e} in params: {params}")
            raise ValueError(f"Missing required parameter for IRAStabilityCalculator: {e}") from e

    def _normalize_state_variable(self, value: float, var_name: str) -> float:
        """Normalizes a single state variable using CURRENT reference stats."""
        if var_name not in self.current_ref_stats:
             logging.warning(f"Reference stats not found for '{var_name}'. Returning 0 normalized value.")
             return 0.0
        stats = self.current_ref_stats[var_name]
        mu = stats.get('mu', 0.0)
        # --- Usa self.norm_epsilon y self.min_std_dev ---
        sigma = max(stats.get('sigma', 1.0), self.min_std_dev, self.norm_epsilon)
        # sigma = stats.get('sigma', 1.0)
        # effective_sigma = max(sigma, self.min_std_dev) # Aplicar sigma mínimo
        return (value - mu) / sigma #(effective_sigma + self.norm_epsilon) # Epsilon aquí es menos necesario si sigma tiene mínimo

    def calculate_instantaneous_stability(self, state: Any) -> float:
        """Calculates w_stab = exp(- sum(w_s * z_s^2))."""
        if not isinstance(state, (np.ndarray, list)) or len(state) < 4:
             logging.error(f"IRAStabilityCalculator: Invalid state format: {state}")
             return 0.0

        deviation_sum_sq = 0.0
        for var_name, index in self.var_indices.items():
            if var_name in self.weights and self.weights[var_name] > 0:
                try:
                    value = state[index]
                    z_s = self._normalize_state_variable(value, var_name)
                    deviation_sum_sq += self.weights[var_name] * (z_s ** 2)
                except IndexError:
                     logging.error(f"IRAStabilityCalculator: State index {index} OOB for '{var_name}' in state {state}.")
                     deviation_sum_sq += self.weights.get(var_name, 1.0) * 100 # Penalty
                except Exception as e:
                     logging.error(f"IRAStabilityCalculator: Error processing var '{var_name}': {e}")
                     deviation_sum_sq += self.weights.get(var_name, 1.0) * 100 # Penalty

        try:
            exponent = -min(deviation_sum_sq, 700) # Clamp exponent arg
            stability_score = math.exp(exponent)
        except OverflowError:
             logging.warning(f"IRAStabilityCalculator: Overflow calculating exp(-{deviation_sum_sq}). Returning 0.")
             stability_score = 0.0

        return max(0.0, min(stability_score, 1.0))

    def calculate_stability_based_reward(self, state: Any) -> float:
        """Calculates reward = exp(-lambda * deviation_sum_sq)."""
        if not isinstance(state, (np.ndarray, list)) or len(state) < 4:
             logging.error(f"IRAStabilityCalculator (reward): Invalid state format: {state}")
             return -10.0 # Low reward

        deviation_sum_sq = 0.0
        for var_name, index in self.var_indices.items():
             if var_name in self.weights and self.weights[var_name] > 0:
                 try:
                     value = state[index]
                     z_s = self._normalize_state_variable(value, var_name)
                     deviation_sum_sq += self.weights[var_name] * (z_s ** 2)
                 except IndexError:
                      logging.error(f"IRAStabilityCalculator (reward): State index {index} OOB for '{var_name}'.")
                      deviation_sum_sq += self.weights.get(var_name, 1.0) * 100
                 except Exception as e:
                      logging.error(f"IRAStabilityCalculator (reward): Error processing var '{var_name}': {e}")
                      deviation_sum_sq += self.weights.get(var_name, 1.0) * 100

        try:
            # Clamp exponent arg based on lambda sensitivity
            exponent = -self.lambda_param * min(deviation_sum_sq, 700 / max(self.lambda_param, 1e-6))
            reward = math.exp(exponent)
        except OverflowError:
             logging.warning(f"IRAStabilityCalculator (reward): Overflow calculating exp(-{self.lambda_param} * {deviation_sum_sq}).")
             reward = 0.0

        return max(0.0, reward) # Ensure non-negative

    # --- NUEVO: Actualización de Estadísticas ---
    def update_reference_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """
        Updates mu and sigma for each relevant variable based on the metrics
        from the completed episode, if adaptive stats are enabled and min_episode is reached.
        """
        if not self.adaptive_enabled or current_episode < self.adaptive_min_episode:
            return # Do nothing if not enabled or too early

        logging.debug(f"Updating reference stats after episode {current_episode}...")
        updated_any = False
        for var_name_cfg, metric_name in self.var_to_metric_map.items():
            if var_name_cfg in self.current_ref_stats: # Only update if it's configured
                if metric_name in episode_metrics_dict:
                    values = episode_metrics_dict[metric_name]
                    if len(values) > 1: # Need at least 2 points for std dev
                        try:
                            new_mu = np.mean(values)
                            new_sigma = np.std(values)

                            # Update internal stats
                            self.current_ref_stats[var_name_cfg]['mu'] = float(new_mu)
                            # Apply minimum standard deviation check here too
                            self.current_ref_stats[var_name_cfg]['sigma'] = float(max(new_sigma, self.min_std_dev))
                            updated_any = True
                            # logging.debug(f"  Updated '{var_name_cfg}': mu={new_mu:.4f}, sigma={max(new_sigma, self.min_std_dev):.4f}")
                        except Exception as e:
                            logging.error(f"Error calculating stats for metric '{metric_name}' (var '{var_name_cfg}'): {e}")
                    else:
                        logging.warning(f"Not enough data points ({len(values)}) for metric '{metric_name}' in episode {current_episode} to update stats.")
                else:
                    logging.warning(f"Metric '{metric_name}' needed for adaptive stats of '{var_name_cfg}' not found in episode data.")
        #if updated_any:
        #     logging.info(f"Reference stats updated after episode {current_episode}: {self.current_ref_stats}")
        #else:
        #     logging.debug(f"No reference stats updated after episode {current_episode}.")