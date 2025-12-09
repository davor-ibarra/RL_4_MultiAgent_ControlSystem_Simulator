# components/analysis/simple_exponential_stability_calculator.py
import numpy as np
import math
import logging
from interfaces.stability_calculator import BaseStabilityCalculator
from typing import Any, Dict
import pandas as pd

logger = logging.getLogger(__name__)

class SimpleExponentialStabilityCalculator(BaseStabilityCalculator):
    """
    Calculates stability using a simple exponential decay model based on scaled
    system features. It is self-configuring from a global config dictionary.
    """
    def __init__(self, config: Dict[str, Any]):
        logger.info(f"[SimpleExpStabilityCalc] Initializing...")
        
        # --- Self-configuration from global config ---
        try:
            exp_params = config['environment']['reward_setup']['calculation']['stability_measure']['exp_decay_metric_params']
            feature_configs = exp_params.get('features', {})
        except KeyError as e:
            raise KeyError(f"SimpleExponentialStabilityCalculator: Missing required configuration path: {e}")

        if not isinstance(feature_configs, dict):
            raise ValueError("SimpleExponential 'features' config must be a dictionary.")
        
        # --- Dynamic parameter extraction ---
        self.decay_coeffs: Dict[str, float] = {}
        self.scales: Dict[str, float] = {}
        
        for feat, params in feature_configs.items():
            if not isinstance(params, dict): continue
            self.decay_coeffs[feat] = float(params.get('weight', 0.0))
            self.scales[feat] = float(params.get('scaled', 1.0))
            if self.scales[feat] <= 0:
                logger.warning(f"[SimpleExpStabilityCalc] Scale for '{feat}' is not positive ({self.scales[feat]}). Using 1.0 as fallback.")
                self.scales[feat] = 1.0
        
        logger.info(f"[SimpleExpStabilityCalc] Initialized. Features: {list(self.decay_coeffs.keys())}")

    def _scale_feature_val(self, value: float, feature_name: str) -> float:
        if pd.isna(value) or not np.isfinite(value): return 0.0
        return value / self.scales[feature_name]

    def calculate_instantaneous_stability(self, state_dict: Dict[str, Any]) -> float:
        sum_weighted_sq_terms = 0.0
        for feat_name, decay_c in self.decay_coeffs.items():
            if abs(decay_c) > 1e-9:
                value = state_dict.get(feat_name)
                if value is None: continue # Skip if feature not in state_dict

                scaled_val = self._scale_feature_val(float(value), feat_name)
                sum_weighted_sq_terms += decay_c * (scaled_val ** 2)
        
        exponent = -min(sum_weighted_sq_terms, 700.0)
        stability = math.exp(exponent)
        return float(np.clip(stability, 0.0, 1.0))

    def calculate_stability_based_reward(self, state_dict: Dict[str, Any]) -> float:
        return self.calculate_instantaneous_stability(state_dict)

    def update_reference_stats(self, episode_metrics_dict: Dict, current_episode_idx: int):
        pass # Not adaptive

    def get_current_adaptive_stats(self) -> Dict:
         return {} # Not adaptive