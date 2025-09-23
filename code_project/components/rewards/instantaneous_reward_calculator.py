# components/rewards/instantaneous_reward_calculator.py
import numpy as np
import pandas as pd
import math
import logging
from typing import Tuple, Any, Optional, Dict
from interfaces.reward_function import RewardFunction # Importar Interfaz
from interfaces.stability_calculator import BaseStabilityCalculator # Importar interfaz base

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class InstantaneousRewardCalculator(RewardFunction): # Implementar Interfaz RewardFunction
    """
    Calculates the instantaneous reward R(s,a,s') and the stability score w_stab per step.
    The reward calculation method ('gaussian' or 'stability_calculator') is determined by config.
    The stability score w_stab is always calculated if a StabilityCalculator is provided.
    """
    def __init__(self,
                 calculation_config: Dict[str, Any], # Config de reward_setup.calculation
                 stability_calculator: Optional[BaseStabilityCalculator] = None # Dependencia Inyectada
                 ):
        """
        Initializes the instantaneous reward calculator.

        Args:
            calculation_config (Dict[str, Any]): The 'calculation' section from config,
                                                 must contain 'method' and relevant params
                                                 (e.g., 'gaussian_params').
            stability_calculator (Optional[BaseStabilityCalculator]): Injected stability
                                       calculator instance (can be None).
        """
        logger.info("Initializing InstantaneousRewardCalculator...")
        try:
            # --- Store Dependencies and Process Config ---
            self.stability_calculator = stability_calculator
            self.calculation_config = calculation_config
            self.method = calculation_config.get('method')

            if not self.method:
                raise ValueError("Missing 'method' in calculation_config.")

            logger.info(f"InstantaneousRewardCalculator mode: {self.method}")

            # --- Gaussian Mode Specifics ---
            self.gaussian_params = {}
            self.weights = {}
            self.scales = {}
            self.state_indices = {}
            self.required_gaussian_keys = []
            if self.method == 'gaussian':
                self.gaussian_params = calculation_config.get('gaussian_params', {})
                if not isinstance(self.gaussian_params, dict):
                     raise TypeError("gaussian_params must be a dictionary for method 'gaussian'.")
                self.weights = self.gaussian_params.get('weights', {})
                self.scales = self.gaussian_params.get('scales', {})
                self.state_indices = { # Hardcoded for pendulum, make dynamic if needed
                    'cart_position': 0, 'cart_velocity': 1,
                    'angle': 2, 'angular_velocity': 3
                }
                self.required_gaussian_keys = list(self.state_indices.keys()) + ['force', 'time']
                self._validate_gaussian_params() # Validate weights and scales

            # --- Stability Calculator Mode Specifics ---
            elif self.method == 'stability_calculator':
                if self.stability_calculator is None:
                    msg = "CRITICAL: method='stability_calculator' selected, but no StabilityCalculator was provided!"
                    logger.error(msg)
                    raise ValueError(msg)
                if not hasattr(self.stability_calculator, 'calculate_stability_based_reward'):
                    msg = f"CRITICAL: StabilityCalculator ({type(self.stability_calculator).__name__}) missing 'calculate_stability_based_reward' method."
                    logger.error(msg)
                    raise AttributeError(msg)
                logger.info(f"Reward will be calculated by: {type(self.stability_calculator).__name__}")

            else:
                raise ValueError(f"Unknown calculation method specified: {self.method}")

            # --- Validate w_stab calculation capability ---
            if self.stability_calculator and not hasattr(self.stability_calculator, 'calculate_instantaneous_stability'):
                 msg = f"CRITICAL: Provided StabilityCalculator ({type(self.stability_calculator).__name__}) missing 'calculate_instantaneous_stability' method."
                 logger.error(msg)
                 raise AttributeError(msg)
            elif self.stability_calculator is None:
                 logger.warning("InstantaneousRewardCalculator: No StabilityCalculator provided. w_stab will default to 1.0.")

        except Exception as e:
            logger.error(f"InstantaneousRewardCalculator: Error during initialization: {e}", exc_info=True)
            raise RuntimeError("Failed to initialize InstantaneousRewardCalculator") from e

    def _validate_gaussian_params(self):
        """Validates weights and scales for Gaussian mode."""
        logger.debug("Validating parameters for Gaussian reward calculation...")
        valid = True
        for key in self.required_gaussian_keys:
            # Validate Weight
            if key not in self.weights:
                logger.warning(f"Gaussian Mode: Missing weight for '{key}'. Using 0.0.")
                self.weights[key] = 0.0
            elif not isinstance(self.weights[key], (int, float)):
                logger.error(f"Gaussian Mode: Weight for '{key}' ({self.weights[key]}) is not numeric. Invalid config.")
                self.weights[key] = 0.0
                valid = False
            # Validate Scale
            if key not in self.scales:
                logger.warning(f"Gaussian Mode: Missing scale for '{key}'. Using 1.0.")
                self.scales[key] = 1.0
            elif not isinstance(self.scales[key], (int, float)) or self.scales[key] <= 0:
                logger.error(f"Gaussian Mode: Scale for '{key}' ({self.scales[key]}) must be a positive number. Invalid config.")
                self.scales[key] = 1.0
                valid = False
        if not valid:
            raise ValueError("Invalid parameters found in gaussian_params configuration.")

    # --- Implementation of RewardFunction Interface ---

    def calculate(self, state: Any, action: Any, next_state: Any, t: float) -> Tuple[float, float]:
        """Calculates reward_value and w_stab based on the configured method and stability calculator."""
        stability_score = 1.0 # Default w_stab
        reward_value = 0.0    # Default reward

        # 1. Calculate Stability Score (w_stab) if calculator is available
        if self.stability_calculator:
            try:
                w_stab = self.stability_calculator.calculate_instantaneous_stability(next_state)
                # Clip and handle NaN/inf
                stability_score = float(np.clip(w_stab if pd.notna(w_stab) and np.isfinite(w_stab) else 0.0, 0.0, 1.0))
            except Exception as e:
                logger.error(f"Error calculating w_stab from {type(self.stability_calculator).__name__}: {e}", exc_info=True)
                stability_score = 0.0 # Default to 0 on error

        # 2. Calculate Reward Value based on method
        try:
            if self.method == 'gaussian':
                if not isinstance(next_state, (np.ndarray, list)) or len(next_state) < len(self.state_indices):
                     raise IndexError(f"Invalid next_state format or length for Gaussian calculation: {next_state}")

                # Helper for safe exponentiation
                def safe_exp(arg):
                     try: return math.exp(-min(float(arg)**2, 700.0)) # Ensure arg is float
                     except (OverflowError, ValueError, TypeError): return 0.0

                # Calculate Gaussian terms
                terms = {}
                for key, index in self.state_indices.items():
                    terms[key] = safe_exp(next_state[index] / self.scales[key])
                terms['force'] = safe_exp(float(action) / self.scales['force']) # Ensure action is float
                terms['time'] = safe_exp(float(t) / self.scales['time'])     # Ensure time is float

                # Weighted sum
                reward_calc = sum(self.weights[key] * terms.get(key, 0.0) for key in self.required_gaussian_keys)
                reward_value = float(reward_calc if pd.notna(reward_calc) and np.isfinite(reward_calc) else 0.0)

            elif self.method == 'stability_calculator':
                # Assumes stability_calculator is not None and has the method (validated in __init__)
                reward_calc = self.stability_calculator.calculate_stability_based_reward(next_state) # type: ignore
                reward_value = float(reward_calc if pd.notna(reward_calc) and np.isfinite(reward_calc) else 0.0)

            # else: Handled in __init__

        except IndexError as e: logger.error(f"Reward Calc ({self.method}): IndexError accessing state/action: {e}"); reward_value = 0.0
        except KeyError as e: logger.error(f"Reward Calc ({self.method}): KeyError accessing scales/weights: {e}"); reward_value = 0.0
        except TypeError as e: logger.error(f"Reward Calc ({self.method}): TypeError (likely invalid state/action value): {e}"); reward_value = 0.0
        except Exception as e: logger.error(f"Reward Calc ({self.method}): Unexpected error: {e}", exc_info=True); reward_value = 0.0

        # logger.debug(f"Calculate Result: Reward={reward_value:.4f}, Stability={stability_score:.4f}")
        return reward_value, stability_score

    def update_calculator_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """Delegates stats update to the stability calculator, if it exists and supports it."""
        if self.stability_calculator and hasattr(self.stability_calculator, 'update_reference_stats'):
            try:
                # logger.debug(f"Delegating update_stats to {type(self.stability_calculator).__name__}")
                self.stability_calculator.update_reference_stats(episode_metrics_dict, current_episode)
            except Exception as e:
                logger.error(f"Error calling update_reference_stats on calculator: {e}", exc_info=True)
        # else: logger.debug("No stability calculator or it lacks update_reference_stats method.")