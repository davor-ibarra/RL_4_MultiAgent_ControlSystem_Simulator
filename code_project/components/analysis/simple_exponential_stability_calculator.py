# components/analysis/simple_exponential_stability_calculator.py
import numpy as np
import math
import logging
from interfaces.stability_calculator import BaseStabilityCalculator
from typing import Any, Dict
import pandas as pd # Para pd.notna/isna

logger = logging.getLogger(__name__)

class SimpleExponentialStabilityCalculator(BaseStabilityCalculator):
    def __init__(self, exp_decay_metric_params: Dict[str, Any]): # Parámetros de config...stability_measure.exp_decay_metric_params
        logger.info(f"[SimpleExpStabilityCalc] Initializing...")

        # Asignación directa, asumiendo claves y tipos correctos (validados externamente).
        self.decay_coeffs: Dict[str, float] = exp_decay_metric_params['feature_decay_coefficients']
        self.scales: Dict[str, float] = exp_decay_metric_params['feature_scales']

        # Mapeo de características (fijo para péndulo, podría ser configurable)
        self.system_feature_map: Dict[str, int] = {'cart_position': 0, 'cart_velocity': 1, 'angle': 2, 'angular_velocity': 3}
        
        # Validar que las escalas sean positivas (pequeña auto-corrección si es necesario, aunque idealmente validado antes)
        for feat_key in self.scales:
            if self.scales[feat_key] <= 0:
                logger.warning(f"[SimpleExpStabilityCalc] Scale for '{feat_key}' is not positive ({self.scales[feat_key]}). Using 1.0 as fallback.")
                self.scales[feat_key] = 1.0
        
        logger.info("[SimpleExpStabilityCalc] Initialized.")
        # logger.debug(f"[SimpleExpStabilityCalc] DecayCoeffs: {self.decay_coeffs}, Scales: {self.scales}")

    def _scale_feature_val(self, value: float, feature_name: str) -> float:
        # Asume que feature_name está en self.scales.
        # Asume que value es un float (puede ser NaN/inf).
        if pd.isna(value) or not np.isfinite(value): return 0.0 # Valor escalado 0 para inválidos
        return value / self.scales[feature_name]

    def calculate_instantaneous_stability(self, current_state: Any) -> float:
        # Asume que current_state es un array/lista con suficientes elementos.
        sum_weighted_sq_terms = 0.0
        for feat_name, feat_idx in self.system_feature_map.items():
            decay_c = self.decay_coeffs.get(feat_name, 0.0) # Default a 0 si no está en config
            if abs(decay_c) > 1e-9: # Solo si el coeficiente es significativo
                # Si current_state es muy corto, dará IndexError.
                # Si current_state[feat_idx] no es float, _scale_feature_val lo manejará o fallará.
                scaled_val = self._scale_feature_val(float(current_state[feat_idx]), feat_name)
                sum_weighted_sq_terms += decay_c * (scaled_val ** 2)
        
        exponent = -min(sum_weighted_sq_terms, 700.0) # Limitar exponente
        stability = math.exp(exponent)
        return float(np.clip(stability, 0.0, 1.0)) # Asegurar [0,1]

    def calculate_stability_based_reward(self, current_state: Any) -> float:
        # Para este calculador, la recompensa basada en estabilidad es igual al stability_score.
        return self.calculate_instantaneous_stability(current_state)

    def update_reference_stats(self, episode_metrics_dict: Dict, current_episode_idx: int):
        pass # No es adaptativo

    def get_current_adaptive_stats(self) -> Dict:
         return {} # No es adaptativo