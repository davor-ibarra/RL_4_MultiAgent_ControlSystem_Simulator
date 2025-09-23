import numpy as np
import math
import logging
from typing import Tuple, Any, Optional, Dict
from interfaces.reward_function import RewardFunction
from interfaces.stability_calculator import BaseStabilityCalculator # Import base class

class GaussianReward(RewardFunction):
    """
    Calculates reward based on Gaussian functions of state variables,
    or optionally uses a stability-based reward provided by a StabilityCalculator.
    Also calculates a stability score (w_stab) using the provided calculator.
    """
    def __init__(self, init_config: Dict[str, Any], stability_calculator: Optional[BaseStabilityCalculator] = None):
        logging.info("Initializing GaussianReward...")
        try:
            # --- Leer desde init_config ---
            self.params = init_config.get('params', {})
            self.weights = self.params.get('weights', {})
            self.scales = self.params.get('scales', {})
            self.stability_calculator = stability_calculator
            self.use_stability_reward = init_config.get('use_stability_based_reward', False)
            # Guardar config del stability calculator por si acaso se necesita info extra (e.g., lambda de IRA)
            self.stability_calculator_config = init_config.get('stability_calculator_config', {})


            # State mapping
            self.state_indices = {
                'cart_position': 0, 'cart_velocity': 1,
                'angle': 2, 'angular_velocity': 3
            }
            self.required_gaussian_keys = ['angle', 'angular_velocity', 'cart_position', 'cart_velocity', 'force', 'time']

            if self.use_stability_reward:
                if self.stability_calculator is None:
                    # Este caso debería ser prevenido por la fábrica, pero añadir warning por si acaso
                    logging.error("GaussianReward CRITICAL: Initialized to use stability reward, but no calculator instance provided!")
                    # Forzar a no usarlo para evitar errores en calculate()
                    self.use_stability_reward = False
                else:
                    logging.info("GaussianReward: Initialized to use stability-based reward from calculator.")
            else:
                # Validar pesos y escalas Gaussianas si se usan
                missing_weights = [k for k in self.required_gaussian_keys if k not in self.weights]
                missing_scales = [k for k in self.required_gaussian_keys if k not in self.scales]
                if missing_weights:
                     logging.warning(f"GaussianReward: Using Gaussian method, but weights missing for: {missing_weights}. Defaulting to 0.0.")
                     for k in missing_weights: self.weights[k] = 0.0
                if missing_scales:
                     logging.warning(f"GaussianReward: Using Gaussian method, but scales missing for: {missing_scales}. Defaulting to 1.0.")
                     for k in missing_scales: self.scales[k] = 1.0
                logging.info("GaussianReward: Initialized to use Gaussian reward calculation.")

            if self.stability_calculator is None:
                logging.warning("GaussianReward: No stability calculator provided. Stability score (w_stab) will default to 1.0.")


        except Exception as e:
            logging.error(f"GaussianReward: Error during initialization: {e}", exc_info=True)
            raise

    # @formula (Gaussian): reward = w_angle * exp(-(angle/s_a)^2)  w_angle_vel * exp(-(vel/s_v)^2) + w_cart * exp(-(pos/s_p)^2) w_cart_vel * exp(-(cart_vel/s_cv)^2) + w_force * exp(-(force/s_f)^2) + w_time * exp(-(t/s_t)^2)
    # @formula (Stability): reward = stability_calculator.calculate_stability_based_reward(next_state)
    def calculate(self, state: Any, action: Any, next_state: Any, t: float) -> Tuple[float, float]:
        """
        Calculates the reward and stability score for the transition.

        Args:
            state: State before action (unused in this version).
            action: Action taken (force).
            next_state: State after action [cart_pos, cart_vel, angle, angular_vel].
            t: Current time (used only for Gaussian time component).

        Returns:
            Tuple[float, float]: (reward, stability_score)
        """
        stability_score = 1.0 # Default stability score
        reward = 0.0

        # --- [1] Calculate Stability Score (using next_state) ---
        if self.stability_calculator:
            try:
                stability_score = self.stability_calculator.calculate_instantaneous_stability(next_state)
            except Exception as e:
                logging.error(f"Error calculating stability score: {e}", exc_info=True)
                stability_score = 0.0 # Penalize stability on error
        # else: stability_score remains 1.0 (or could be set to NaN)

        # --- [2] Calculate Reward ---
        if self.use_stability_reward and self.stability_calculator:
            # --- [2a] Stability-Based Reward ---
            try:
                reward = self.stability_calculator.calculate_stability_based_reward(next_state)
            except AttributeError:
                 logging.error(f"Configured to use stability reward, but calculator {type(self.stability_calculator).__name__} lacks 'calculate_stability_based_reward' method. Using 0 reward.")
                 reward = 0.0
            except Exception as e:
                logging.error(f"Error calculating stability-based reward: {e}", exc_info=True)
                reward = 0.0
        else:
            # --- [2b] Gaussian Reward ---
            try:
                # Check state length
                if not isinstance(next_state, (np.ndarray, list)) or len(next_state) < 4:
                     raise IndexError(f"Expected next_state with at least 4 elements, got {next_state}")

                # --- Calcular términos normalizados individuales ---
                angle_norm = next_state[self.state_indices['angle']] / max(abs(self.scales.get('angle', 1.0)), 1e-9)
                vel_norm = next_state[self.state_indices['angular_velocity']] / max(abs(self.scales.get('angular_velocity', 1.0)), 1e-9)
                pos_cart_norm = next_state[self.state_indices['cart_position']] / max(abs(self.scales.get('cart_position', 1.0)), 1e-9)
                vel_cart_norm = next_state[self.state_indices['cart_velocity']] / max(abs(self.scales.get('cart_velocity', 1.0)), 1e-9)
                force_norm = float(action) / max(abs(self.scales.get('force', 1.0)), 1e-9)
                time_norm = t / max(abs(self.scales.get('time', 1.0)), 1e-9)

                # --- Calcular exponenciales individuales ---
                term_angle = math.exp(-angle_norm**2)
                term_ang_vel = math.exp(-vel_norm**2)
                term_cart_pos = math.exp(-pos_cart_norm**2)
                term_cart_vel = math.exp(-vel_cart_norm**2)
                term_force = math.exp(-force_norm**2)
                term_time = math.exp(-time_norm**2)

                # --- Combinar términos con pesos individuales ---
                reward = (self.weights.get('angle', 0.0) * term_angle +
                          self.weights.get('angular_velocity', 0.0) * term_ang_vel +
                          self.weights.get('cart_position', 0.0) * term_cart_pos +
                          self.weights.get('cart_velocity', 0.0) * term_cart_vel +
                          self.weights.get('force', 0.0) * term_force +
                          self.weights.get('time', 0.0) * term_time)

            except IndexError as e:
                logging.error(f"GaussianReward: Error accessing next_state elements: {e}. State: {next_state}")
                reward = 0.0
            except KeyError as e:
                # Menos probable ahora con .get, pero mantenido por si acaso
                logging.error(f"GaussianReward: Unexpected KeyError: {e}")
                reward = 0.0
            except Exception as e:
                logging.error(f"Error calculating Gaussian reward: {e}", exc_info=True)
                reward = 0.0

        return float(reward), float(stability_score)

    def update_calculator_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """
        Calls the update method on the internal stability calculator, if it exists
        and supports updating.
        """
        if self.stability_calculator and hasattr(self.stability_calculator, 'update_reference_stats'):
            try:
                # Delegate the call
                self.stability_calculator.update_reference_stats(episode_metrics_dict, current_episode)
            except Exception as e:
                logging.error(f"Error calling update_reference_stats on stability calculator: {e}", exc_info=True)