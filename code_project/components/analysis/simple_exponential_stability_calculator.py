import numpy as np
import math
import logging
from interfaces.stability_calculator import StabilityCalculator
from typing import Any, Dict

class SimpleExponentialStabilityCalculator(StabilityCalculator):
    """
    Calculates a simple stability score w_stab based on the formula:
    w_stab = exp(- sum(lambda_s * (state_s / scale_s)**2))
    Does not perform adaptive updates.
    Assumes state is a numpy array or list: [cart_pos, cart_vel, angle, angular_vel].
    """
    def __init__(self, params: Dict[str, Any]):
        """
        Initializes the calculator with lambda weights and scales.

        Args:
            params (Dict): Dictionary containing 'lambda_weights' and 'scales'.
        """
        try:
            self.lambda_weights = params['lambda_weights']
            self.scales = params['scales']

            # Map state variable names used in config to state vector indices
            self.var_indices = {
                'cart_position': 0,
                'cart_velocity': 1,
                'angle': 2,
                'angular_velocity': 3
            }
            # Validate keys
            expected_vars = set(self.var_indices.keys())
            if set(self.lambda_weights.keys()) != expected_vars or set(self.scales.keys()) != expected_vars:
                 logging.warning(f"SimpleExponentialStabilityCalculator: Mismatch between expected state variables {expected_vars} "
                                 f"and keys in lambda_weights ({set(self.lambda_weights.keys())}) or "
                                 f"scales ({set(self.scales.keys())}). Check config.")

            logging.info("SimpleExponentialStabilityCalculator initialized.")

        except KeyError as e:
            logging.error(f"SimpleExponentialStabilityCalculator: Missing required parameter key: {e} in params: {params}")
            raise ValueError(f"Missing required parameter for SimpleExponentialStabilityCalculator: {e}") from e

    def _scale_state_variable(self, value: float, var_name: str) -> float:
        """Scales a state variable."""
        scale = self.scales.get(var_name, 1.0)
        if scale <= 1e-9: # Avoid division by zero or very small scales
            # logging.warning(f"Scale for '{var_name}' is near zero ({scale}). Using 1.0.")
            scale = 1.0
        return value / scale

    def calculate_instantaneous_stability(self, state: Any) -> float:
        """Calculates w_stab = exp(- sum(lambda_s * scaled_state_s^2))."""
        if not isinstance(state, (np.ndarray, list)) or len(state) < 4:
             logging.error(f"SimpleExponentialStabilityCalculator: Invalid state format: {state}")
             return 0.0 # Minimum stability

        deviation_sum_sq_weighted = 0.0
        for var_name, index in self.var_indices.items():
            # Use lambda weight configured, default to 0 if missing for safety
            lambda_w = self.lambda_weights.get(var_name, 0.0)
            if lambda_w > 0: # Only process if weight is positive
                try:
                    value = state[index]
                    scaled_value = self._scale_state_variable(value, var_name)
                    deviation_sum_sq_weighted += lambda_w * (scaled_value ** 2)
                except IndexError:
                     logging.error(f"SimpleExponentialStabilityCalculator: State index {index} OOB for '{var_name}'.")
                     # Penalize if state is malformed and weight > 0
                     deviation_sum_sq_weighted += lambda_w * 100 # Add large penalty scaled by weight
                except Exception as e:
                     logging.error(f"SimpleExponentialStabilityCalculator: Error processing var '{var_name}': {e}")
                     deviation_sum_sq_weighted += lambda_w * 100 # Penalty

        try:
            exponent = -min(deviation_sum_sq_weighted, 700) # Clamp exponent arg
            stability_score = math.exp(exponent)
        except OverflowError:
             logging.warning(f"SimpleExponentialStabilityCalculator: Overflow calculating exp(-{deviation_sum_sq_weighted}). Ret 0.")
             stability_score = 0.0

        return max(0.0, min(stability_score, 1.0)) # Ensure score is [0, 1]

    def calculate_stability_based_reward(self, state: Any) -> float:
        """This simple calculator does not define a stability-based reward."""
        logging.warning("calculate_stability_based_reward called on SimpleExponentialStabilityCalculator. Returning 0.")
        return 0.0