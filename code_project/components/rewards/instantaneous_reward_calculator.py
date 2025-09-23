# components/rewards/instantaneous_reward_calculator.py
import numpy as np
import pandas as pd
import math
import logging
from typing import Tuple, Any, Optional, Dict
from interfaces.reward_function import RewardFunction
from interfaces.stability_calculator import BaseStabilityCalculator

logger = logging.getLogger(__name__) # Logger específico del módulo

class InstantaneousRewardCalculator(RewardFunction):
    def __init__(self,
                 calculation_config: Dict[str, Any], # Recibe environment.reward_setup.calculation
                 stability_calculator: Optional[BaseStabilityCalculator]
                 ):
        logger.info(f"[InstantRewardCalc] Initializing with calculation_config keys: {list(calculation_config.keys())}")

        if not isinstance(calculation_config, dict):
            raise ValueError("calculation_config must be a dictionary.")
        self.calculation_config = calculation_config # Guardar para referencia si es necesario

        self.method = self.calculation_config.get('method')
        if not self.method or not isinstance(self.method, str):
            raise ValueError("Missing or invalid 'method' in calculation_config.")
        logger.info(f"[InstantRewardCalc] Reward calculation method: {self.method}")

        # Validar stability_calculator si se proporciona
        if stability_calculator is not None and not isinstance(stability_calculator, BaseStabilityCalculator):
            raise TypeError("Provided stability_calculator does not implement BaseStabilityCalculator.")
        self.stability_calculator = stability_calculator

        # --- Cargar y validar parámetros específicos del método ---
        self.gaussian_params: Dict[str, Any] = {} # Para método 'gaussian'
        self.weights: Dict[str, float] = {}       # Para método 'gaussian'
        self.scales: Dict[str, float] = {}        # Para método 'gaussian'
        self.state_indices = {'cart_position': 0, 'cart_velocity': 1, 'angle': 2, 'angular_velocity': 3}

        if self.method == 'gaussian':
            self.gaussian_params = self.calculation_config.get('gaussian_params', {})
            if not isinstance(self.gaussian_params, dict):
                raise TypeError("'gaussian_params' must be a dictionary for method 'gaussian'.")
            self._load_and_validate_gaussian_params() # Llama a helper para validar
        elif self.method == 'stability_calculator':
            if self.stability_calculator is None:
                msg = "Method 'stability_calculator' selected, but no StabilityCalculator instance was provided."
                logger.critical(f"[InstantRewardCalc] {msg}")
                raise ValueError(msg)
            if not hasattr(self.stability_calculator, 'calculate_stability_based_reward'):
                msg = f"Provided StabilityCalculator ({type(self.stability_calculator).__name__}) is missing 'calculate_stability_based_reward' method."
                logger.critical(f"[InstantRewardCalc] {msg}")
                raise AttributeError(msg)
            logger.info(f"[InstantRewardCalc] Reward will be determined by: {type(self.stability_calculator).__name__}")
        else:
            # Esto no debería ocurrir si config_loader/RewardFactory valida el método
            raise ValueError(f"Unknown reward calculation method specified: '{self.method}'")

        if self.stability_calculator:
             logger.info(f"[InstantRewardCalc] Using StabilityCalculator: {type(self.stability_calculator).__name__} for w_stab.")
        else:
             logger.info("[InstantRewardCalc] No StabilityCalculator provided. w_stab will default to 1.0.")
        logger.info("[InstantRewardCalc] Initialization complete.")

    def _load_and_validate_gaussian_params(self):
        """Valida los parámetros 'weights' y 'scales' dentro de gaussian_params."""
        logger.debug("[InstantRewardCalc:_load_gaussian] Validating gaussian_params...")
        self.weights = self.gaussian_params.get('weights', {})
        self.scales = self.gaussian_params.get('scales', {})

        if not isinstance(self.weights, dict): raise TypeError("gaussian_params.weights must be a dictionary.")
        if not isinstance(self.scales, dict): raise TypeError("gaussian_params.scales must be a dictionary.")

        # Claves esperadas para pesos y escalas en modo gaussiano
        expected_gaussian_keys = list(self.state_indices.keys()) + ['force', 'time']
        valid_params = True

        for key in expected_gaussian_keys:
            # Validar pesos (default a 0.0 si falta)
            w_val = self.weights.get(key, 0.0)
            if not isinstance(w_val, (int, float)):
                logger.error(f"[InstantRewardCalc:_load_gaussian] Weight for '{key}' ({w_val}) is not numeric. Defaulting to 0.0.")
                self.weights[key] = 0.0; valid_params = False
            else: self.weights[key] = float(w_val)

            # Validar escalas (default a 1.0 si falta, debe ser > 0)
            s_val = self.scales.get(key, 1.0)
            if not isinstance(s_val, (int, float)) or s_val <= 0:
                logger.error(f"[InstantRewardCalc:_load_gaussian] Scale for '{key}' ({s_val}) must be a positive number. Defaulting to 1.0.")
                self.scales[key] = 1.0; valid_params = False
            else: self.scales[key] = float(s_val)

        if not valid_params:
            raise ValueError("Invalid values found during gaussian_params validation.")
        logger.debug(f"[InstantRewardCalc:_load_gaussian] Gaussian params validated. Weights: {self.weights}, Scales: {self.scales}")

    def calculate(self, state: Any, action: Any, next_state: Any, t: float) -> Tuple[float, float]:
        stability_score = 1.0
        reward_value = 0.0

        if self.stability_calculator:
            try:
                w_stab_calc = self.stability_calculator.calculate_instantaneous_stability(next_state)
                if pd.notna(w_stab_calc) and np.isfinite(w_stab_calc):
                    stability_score = float(np.clip(w_stab_calc, 0.0, 1.0))
                else:
                    logger.warning(f"[InstantRewardCalc:calculate] StabilityCalculator ({type(self.stability_calculator).__name__}) returned invalid w_stab ({w_stab_calc}). Defaulting to 0.0.")
                    stability_score = 0.0 # Default a 0 si es inválido
            except Exception as e:
                logger.error(f"[InstantRewardCalc:calculate] Error calculating w_stab from {type(self.stability_calculator).__name__}: {e}", exc_info=True)
                stability_score = 0.0
        # else: stability_score remains 1.0

        try:
            if self.method == 'gaussian':
                if not isinstance(next_state, (np.ndarray, list)) or len(next_state) < 4:
                     logger.warning(f"[InstantRewardCalc:calculate_gaussian] Invalid next_state: {next_state}. Reward set to 0.")
                     return 0.0, stability_score
                action_f = float(action) if np.isfinite(action) else 0.0

                reward_calc_terms = []
                for key, index in self.state_indices.items():
                    val = next_state[index]
                    if pd.isna(val) or not np.isfinite(val):
                        #logger.debug(f"[InstantRewardCalc:calculate_gaussian] Invalid value for '{key}' in next_state: {val}. Term is 0.")
                        continue # Skip this term
                    term = self.weights[key] * math.exp(-min( (val / self.scales[key])**2, 700.0)) # Limitar exponente
                    reward_calc_terms.append(term)

                reward_calc_terms.append(self.weights['force'] * math.exp(-min( (action_f / self.scales['force'])**2, 700.0)))
                reward_calc_terms.append(self.weights['time'] * math.exp(-min( (float(t) / self.scales['time'])**2, 700.0)))
                reward_value = float(np.nansum(reward_calc_terms)) # nansum para robustez si algún término es NaN (no debería)

            elif self.method == 'stability_calculator':
                if self.stability_calculator: # Ya validado en init
                    reward_calc = self.stability_calculator.calculate_stability_based_reward(next_state)
                    reward_value = float(reward_calc) if pd.notna(reward_calc) and np.isfinite(reward_calc) else 0.0
                else: # Should not happen due to init checks
                    logger.error("[InstantRewardCalc:calculate] Method 'stability_calculator' but no instance. Reward=0.")
                    reward_value = 0.0

        except IndexError as e: logger.error(f"[InstantRewardCalc:calculate] IndexError ({self.method}): {e}"); reward_value = 0.0
        except KeyError as e: logger.error(f"[InstantRewardCalc:calculate] KeyError ({self.method}) accessing weights/scales: {e}"); reward_value = 0.0
        except Exception as e: logger.error(f"[InstantRewardCalc:calculate] Unexpected error ({self.method}): {e}", exc_info=True); reward_value = 0.0

        # Asegurar que reward_value sea finito
        if not np.isfinite(reward_value):
            logger.warning(f"[InstantRewardCalc:calculate] Calculated reward_value is not finite ({reward_value}). Defaulting to 0.0.")
            reward_value = 0.0

        #logger.debug(f"[InstantRewardCalc:calculate] Result: Reward={reward_value:.4f}, Stability={stability_score:.4f}")
        return reward_value, stability_score

    def update_calculator_stats(self, episode_metrics_dict: Dict, current_episode: int):
        if self.stability_calculator and hasattr(self.stability_calculator, 'update_reference_stats'):
            try:
                #logger.debug(f"[InstantRewardCalc:update_stats] Delegating to {type(self.stability_calculator).__name__}")
                self.stability_calculator.update_reference_stats(episode_metrics_dict, current_episode)
            except Exception as e:
                logger.error(f"[InstantRewardCalc:update_stats] Error calling update_reference_stats on calculator: {e}", exc_info=True)