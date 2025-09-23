import numpy as np
import math
import logging
from interfaces.stability_calculator import BaseStabilityCalculator # Import Base Class
from typing import Any, Dict

class SimpleExponentialStabilityCalculator(BaseStabilityCalculator): # Inherit from Base
    """
    Calculates a simple stability score w_stab based on the formula:
    w_stab = exp(- sum(lambda_s * (state_s / scale_s)**2))
    This calculator is not adaptive.
    Assumes state is a numpy array or list: [cart_pos, cart_vel, angle, angular_vel].
    """
    def __init__(self, simple_exp_params: Dict[str, Any]):
        """
        Initializes the calculator with lambda weights and scales.

        Args:
            simple_exp_params (Dict): Dictionary containing 'lambda_weights' and 'scales'.
        """
        logging.info("Initializing SimpleExponentialStabilityCalculator...")
        try:
            self.lambda_weights = simple_exp_params['lambda_weights']
            self.scales = simple_exp_params['scales']

            # Map state variable names used in config to state vector indices
            self.var_indices = {
                'cart_position': 0,
                'cart_velocity': 1,
                'angle': 2,
                'angular_velocity': 3
            }
            # Validate keys
            expected_vars = set(self.var_indices.keys())
            if set(self.lambda_weights.keys()) != expected_vars:
                logging.warning(f"SimpleExponentialStabilityCalculator: Mismatch between expected state variables {expected_vars} "
                                f"and keys in lambda_weights ({set(self.lambda_weights.keys())}). Check config.")
            if set(self.scales.keys()) != expected_vars:
                 logging.warning(f"SimpleExponentialStabilityCalculator: Mismatch between expected state variables {expected_vars} "
                              f"and keys in scales ({set(self.scales.keys())}). Check config.")
            # Ensure all expected vars have weights and scales
            for var in expected_vars:
                if var not in self.lambda_weights:
                     logging.warning(f"Missing lambda_weights for variable '{var}'. Defaulting to 0.")
                     self.lambda_weights[var] = 0.0
                if var not in self.scales:
                     logging.warning(f"Missing scales for variable '{var}'. Defaulting to 1.0.")
                     self.scales[var] = 1.0


            logging.info(f"SimpleExponentialStabilityCalculator initialized with weights: {self.lambda_weights} and scales: {self.scales}")

        except KeyError as e:
            logging.error(f"SimpleExponentialStabilityCalculator: Missing required parameter key: {e} in params: {simple_exp_params}")
            raise ValueError(f"Missing required parameter for SimpleExponentialStabilityCalculator: {e}") from e
        except Exception as e:
             logging.error(f"SimpleExponentialStabilityCalculator: Error during initialization: {e}", exc_info=True)
             raise

    def _scale_state_variable(self, value: float, var_name: str) -> float:
        """Scales a state variable."""
        scale = self.scales.get(var_name, 1.0)
        if abs(scale) <= 1e-9: # Avoid division by zero or very small scales
            # logging.warning(f"Scale for '{var_name}' is near zero ({scale}). Using 1.0.")
            return value # Return unscaled value if scale is invalid
        return value / scale

    # @formula: w_stab = exp(- sum(lambda_s * (state_s / scale_s)^2))
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
            # Clamp exponent argument to prevent overflow
            exponent = -min(deviation_sum_sq_weighted, 700)
            stability_score = math.exp(exponent)
        except OverflowError:
            logging.warning(f"SimpleExponentialStabilityCalculator: Overflow calculating exp(-{deviation_sum_sq_weighted:.4f}). Ret 0.")
            stability_score = 0.0

        return max(0.0, min(stability_score, 1.0)) # Ensure score is [0, 1]

    def calculate_stability_based_reward(self, state: Any) -> float:
        """
        This simple calculator does not define a stability-based reward itself.
        It might return the stability score or 0, depending on desired behavior.
        Returning 0 as it doesn't use a lambda parameter like IRA.
        """
        # Option 1: Return 0
        # return 0.0
        # Option 2: Return the stability score itself as reward
        return self.calculate_instantaneous_stability(state)


    def update_reference_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """This calculator is not adaptive, so this method does nothing."""
        pass # No adaptive stats to update

    # get_current_adaptive_stats inherited from base returns {}