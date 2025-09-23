# components/analysis/simple_exponential_stability_calculator.py
import numpy as np
import math
import logging
from interfaces.stability_calculator import BaseStabilityCalculator
from typing import Any, Dict
import pandas as pd

logger = logging.getLogger(__name__) # Logger específico del módulo

class SimpleExponentialStabilityCalculator(BaseStabilityCalculator):
    def __init__(self, simple_exp_params: Dict[str, Any]):
        # simple_exp_params es la sección config[...]['simple_exponential_params']
        logger.info(f"[SimpleExpStabilityCalc] Initializing with params: {list(simple_exp_params.keys())}")

        if not isinstance(simple_exp_params, dict):
            raise TypeError("simple_exp_params must be a dictionary.")

        # Lambda Weights
        raw_lambda_weights = simple_exp_params.get('lambda_weights')
        if not isinstance(raw_lambda_weights, dict):
            raise ValueError("'lambda_weights' missing or not a dictionary in simple_exp_params.")
        self.lambda_weights: Dict[str, float] = {}

        # Scales
        raw_scales = simple_exp_params.get('scales')
        if not isinstance(raw_scales, dict):
            raise ValueError("'scales' missing or not a dictionary in simple_exp_params.")
        self.scales: Dict[str, float] = {}

        self.var_indices = {'cart_position': 0, 'cart_velocity': 1, 'angle': 2, 'angular_velocity': 3}
        expected_vars = set(self.var_indices.keys())

        # Validar y establecer lambda_weights y scales
        for var_name in expected_vars:
            # Lambda weights (default 0.0 si falta la clave específica para una variable)
            lw = raw_lambda_weights.get(var_name, 0.0)
            if not isinstance(lw, (int, float)):
                raise TypeError(f"Lambda weight for '{var_name}' must be numeric, got {type(lw).__name__}.")
            self.lambda_weights[var_name] = float(lw)

            # Scales (default 1.0 si falta la clave específica, debe ser > 0)
            sc = raw_scales.get(var_name, 1.0)
            if not isinstance(sc, (int, float)):
                raise TypeError(f"Scale for '{var_name}' must be numeric, got {type(sc).__name__}.")
            if sc <= 0:
                raise ValueError(f"Scale for '{var_name}' must be positive, got {sc}.")
            self.scales[var_name] = float(sc)

        logger.info("[SimpleExpStabilityCalc] Initialized.")
        logger.debug(f"[SimpleExpStabilityCalc] Lambda Weights: {self.lambda_weights}, Scales: {self.scales}")


    def _scale_state_variable(self, value: float, var_name: str) -> float:
        if var_name not in self.scales: # Debería existir por validación en init
            logger.warning(f"[SimpleExpStabilityCalc:_scale] Scale for '{var_name}' not found. Returning unscaled value.")
            return value
        if pd.isna(value) or not np.isfinite(value):
             #logger.debug(f"[SimpleExpStabilityCalc:_scale] Invalid value (NaN/inf) for '{var_name}': {value}. Scaled to 0.")
             return 0.0
        return value / self.scales[var_name]

    def calculate_instantaneous_stability(self, state: Any) -> float:
        if not isinstance(state, (np.ndarray, list)) or len(state) < 4:
            logger.warning(f"[SimpleExpStabilityCalc:calc_inst_stab] Invalid or short state: {state}. Returning 0.0.")
            return 0.0
        deviation_sum_sq_weighted = 0.0
        try:
            for var_name, index in self.var_indices.items():
                lambda_w = self.lambda_weights.get(var_name, 0.0) # Ya validado
                if lambda_w > 0: # Solo procesar si el peso es significativo
                    value = float(state[index])
                    scaled_value = self._scale_state_variable(value, var_name)
                    deviation_sum_sq_weighted += lambda_w * (scaled_value ** 2)
            exponent_arg = -min(deviation_sum_sq_weighted, 700.0)
            stability_score = math.exp(exponent_arg)
        except IndexError:
            logger.error(f"[SimpleExpStabilityCalc:calc_inst_stab] IndexError. State: {state}")
            return 0.0
        except Exception as e:
            logger.error(f"[SimpleExpStabilityCalc:calc_inst_stab] Unexpected error: {e}", exc_info=True)
            return 0.0
        return float(np.clip(stability_score, 0.0, 1.0))

    def calculate_stability_based_reward(self, state: Any) -> float:
        # Para este calculador simple, la recompensa basada en estabilidad es igual a w_stab.
        return self.calculate_instantaneous_stability(state)

    def update_reference_stats(self, episode_metrics_dict: Dict, current_episode: int):
        pass # No es adaptativo

    def get_current_adaptive_stats(self) -> Dict:
         return {} # No es adaptativo