import math
import logging
from typing import Tuple, Any, Optional, Dict
from interfaces.reward_function import RewardFunction
from interfaces.stability_calculator import StabilityCalculator # Import interface

class GaussianReward(RewardFunction):
    """
    Calculates reward based on Gaussian functions of state variables,
    or optionally uses a stability-based reward if configured.
    Also calculates a stability score (w_stab).
    """
    def __init__(self, reward_config: Dict[str, Any], stability_calculator: Optional[StabilityCalculator] = None):
        """
        Initializes the GaussianReward function.

        Args:
            reward_config (Dict): The 'reward' section from the main config.
            stability_calculator (Optional[StabilityCalculator]): An instance of
                a stability calculator, or None.
        """
        try:
             self.params = reward_config['params']
             self.weights = self.params['weights']
             self.scales = self.params['scales']
             self.stability_calculator = stability_calculator
             self.use_stability_reward = reward_config.get('use_stability_based_reward', False)

             if self.use_stability_reward and self.stability_calculator is None:
                  logging.warning("Config requests 'use_stability_based_reward' but no valid stability calculator was provided. Falling back to Gaussian reward.")
                  self.use_stability_reward = False # Force fallback

             logging.info(f"GaussianReward initialized. Using stability-based reward: {self.use_stability_reward}. "
                          f"Stability calculator present: {self.stability_calculator is not None}")

        except KeyError as e:
             logging.error(f"GaussianReward: Missing required key in reward_config['params']: {e}")
             raise ValueError(f"Missing required parameter for GaussianReward: {e}") from e

    def calculate(self, state: Any, action: Any, next_state: Any, t: float) -> Tuple[float, float]:
        """
        Calculates the reward and stability score.

        Args:
            state: State before action.
            action: Action taken (force).
            next_state: State after action.
            t: Current time.

        Returns:
            Tuple[float, float]: (reward, stability_score)
        """
        stability_score = 1.0 # Default stability score
        reward = 0.0

        # --- Calculate Stability Score (if calculator exists) ---
        if self.stability_calculator:
            try:
                # Use the next_state to evaluate stability after the action
                stability_score = self.stability_calculator.calculate_instantaneous_stability(next_state)
            except Exception as e:
                logging.error(f"Error calculating stability score: {e}", exc_info=True)
                stability_score = 0.0 # Penalize stability on error

        # --- Calculate Reward ---
        if self.use_stability_reward and self.stability_calculator:
            # Calculate reward based on stability metric
            try:
                reward = self.stability_calculator.calculate_stability_based_reward(next_state)
            except Exception as e:
                 logging.error(f"Error calculating stability-based reward: {e}", exc_info=True)
                 reward = 0.0 # Assign neutral reward on error
        else:
            # Calculate reward using the original Gaussian method
            try:
                # Ensure next_state has the expected structure (adjust indices if needed)
                # Assumes next_state = [cart_pos, cart_vel, angle, angular_vel]
                angle_norm = next_state[2] / self.scales.get('angle', 1.0)
                vel_norm = next_state[3] / self.scales.get('angular_velocity', 1.0)
                force_norm = action / self.scales.get('force', 1.0)
                pos_cart_norm = next_state[0] / self.scales.get('cart_position', 1.0)
                vel_cart_norm = next_state[1] / self.scales.get('cart_velocity', 1.0) # Corrected key
                time_norm = t / self.scales.get('time', 1.0)

                pendulum_stability_reward = self.weights.get('stability', 0.0) * math.exp(-angle_norm**2 - vel_norm**2)
                force_penalty = self.weights.get('force', 0.0) * math.exp(-force_norm**2) # Often a penalty, so weight might be negative in config later
                cart_stability_reward = self.weights.get('cart', 0.0) * math.exp(-pos_cart_norm**2 - vel_cart_norm**2) # Include position
                time_reward = self.weights.get('time', 0.0) * math.exp(-time_norm**2) # Can be reward or penalty

                reward = pendulum_stability_reward + force_penalty + cart_stability_reward + time_reward # Adjust signs based on penalty/reward intent

            except IndexError:
                 logging.error(f"GaussianReward: next_state {next_state} does not have expected length/indices.")
                 reward = 0.0 # Neutral reward on state format error
            except KeyError as e:
                 logging.error(f"GaussianReward: Missing scale or weight key in params: {e}")
                 reward = 0.0
            except Exception as e:
                 logging.error(f"Error calculating Gaussian reward: {e}", exc_info=True)
                 reward = 0.0

        return reward, stability_score
    
    def update_calculator_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """
        Calls the update method on the internal stability calculator, if it exists.
        """
        if self.stability_calculator and hasattr(self.stability_calculator, 'update_reference_stats'):
             try:
                 # Delegate the call
                 self.stability_calculator.update_reference_stats(episode_metrics_dict, current_episode)
             except Exception as e:
                  logging.error(f"Error calling update_reference_stats on stability calculator: {e}", exc_info=True)