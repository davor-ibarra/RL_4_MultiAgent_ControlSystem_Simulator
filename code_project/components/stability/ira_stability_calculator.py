# components/analysis/ira_stability_calculator.py
import numpy as np
import pandas as pd
import math
import logging
from interfaces.stability_calculator import BaseStabilityCalculator
from typing import Any, Dict, List
import copy

logger = logging.getLogger(__name__)

class IRAStabilityCalculator(BaseStabilityCalculator):
    """
    Calculates stability (w_stab) using an Inverse Reward Amplitude (IRA) approach
    based on Z-scores of system features. It is self-configuring from a global config dictionary.
    """
    def __init__(self, config: Dict[str, Any]):
        logger.info(f"[IRAStabilityCalc] Initializing...")
        
        # --- Self-configuration from global config ---
        try:
            ira_params = config['environment']['reward_setup']['calculation']['stability_measure']['ira_zscore_metric_params']
            feature_configs = ira_params.get('features', {})
        except KeyError as e:
            raise KeyError(f"IRAStabilityCalculator: Missing required configuration path: {e}")

        if not isinstance(feature_configs, dict):
            raise ValueError("IRA 'features' config must be a dictionary.")

        # --- Dynamic parameter extraction ---
        self.feature_weights: Dict[str, float] = {}
        initial_ref_stats: Dict[str, Dict[str, float]] = {}

        for feat, params in feature_configs.items():
            if not isinstance(params, dict): continue
            self.feature_weights[feat] = float(params.get('weight', 0.0))
            initial_ref_stats[feat] = {
                'mu': float(params.get('mu', 0.0)),
                'sigma': float(params.get('sigma', 1e-6))
            }

        self.lambda_factor = float(ira_params.get('lambda_factor', 1.0))
        self.epsilon_denom = float(ira_params.get('epsilon_zscore_denominator', 1e-6))
        if self.epsilon_denom <= 0: self.epsilon_denom = 1e-6

        adaptive_cfg = ira_params.get('adaptive_stats', {})
        self.is_adaptive_enabled = bool(adaptive_cfg.get('enabled', False))
        self.min_episode_for_adapt = int(adaptive_cfg.get('min_episode', 0))
        self.min_sigma_threshold = float(adaptive_cfg.get('min_sigma', 1e-4))
        if self.min_sigma_threshold <= 0: self.min_sigma_threshold = 1e-4
        
        # --- Initialize current reference stats with validation ---
        self.current_ref_stats: Dict[str, Dict[str, float]] = {}
        for feat in self.feature_weights.keys():
            if feat not in initial_ref_stats:
                raise ValueError(f"Missing initial_reference_stats for feature '{feat}' in IRA config.")
            self.current_ref_stats[feat] = {
                'mu': float(initial_ref_stats[feat]['mu']),
                'sigma': max(abs(float(initial_ref_stats[feat]['sigma'])), self.min_sigma_threshold)
            }

        logger.info(f"[IRAStabilityCalc] Initialized. Adaptive: {self.is_adaptive_enabled}, Features: {list(self.feature_weights.keys())}")

    def _normalize_feature_value(self, value: float, feature_name: str) -> float:
        stats = self.current_ref_stats[feature_name]
        mu, sigma = stats['mu'], stats['sigma']
        effective_denom = max(sigma, self.epsilon_denom) 
        
        if pd.isna(value) or not np.isfinite(value): return 0.0
        return (value - mu) / effective_denom

    def calculate_instantaneous_stability(self, state_dict: Dict[str, Any]) -> float:
        sum_weighted_sq_z = 0.0
        for feat_name, weight in self.feature_weights.items():
            if abs(weight) > 1e-9:
                value = state_dict.get(feat_name)
                if value is None: continue # Skip if feature not in state_dict
                
                z_score = self._normalize_feature_value(float(value), feat_name)
                sum_weighted_sq_z += weight * (z_score ** 2)
        
        exponent = -min(sum_weighted_sq_z, 700.0)
        stability = math.exp(exponent)
        return float(np.clip(stability, 0.0, 1.0))

    def calculate_stability_based_reward(self, state_dict: Dict[str, Any]) -> float:
        sum_weighted_sq_z_reward = 0.0
        for feat_name, weight in self.feature_weights.items():
            if abs(weight) > 1e-9:
                value = state_dict.get(feat_name)
                if value is None: continue

                z_score = self._normalize_feature_value(float(value), feat_name)
                sum_weighted_sq_z_reward += weight * (z_score ** 2)
        
        eff_lambda = max(abs(self.lambda_factor), 1e-9)
        exponent_reward = -self.lambda_factor * min(sum_weighted_sq_z_reward, 700.0 / eff_lambda)
        reward = math.exp(exponent_reward)
        return float(max(0.0, reward))

    def update_reference_stats(self, episode_metrics: Dict[str, List[Any]], episode_num: int):
        if not self.is_adaptive_enabled or episode_num < self.min_episode_for_adapt:
            return

        for feat_name in self.current_ref_stats.keys():
            # The metric key in episode_metrics must match the feature name (e.g., 'angle').
            # The ExtendedMetricsCollector config must be aligned to save metrics with these keys.
            metric_values_list = episode_metrics.get(feat_name)
            if not isinstance(metric_values_list, list) or len(metric_values_list) < 2: continue

            valid_data_points = np.array([float(v) for v in metric_values_list if pd.notna(v) and np.isfinite(v)])
            if len(valid_data_points) < 2: continue
            
            new_mu_calc = np.mean(valid_data_points)
            new_sigma_calc = np.std(valid_data_points)
            effective_new_sigma = max(new_sigma_calc, self.min_sigma_threshold)
            
            if not np.isclose(self.current_ref_stats[feat_name]['mu'], new_mu_calc) or \
               not np.isclose(self.current_ref_stats[feat_name]['sigma'], effective_new_sigma):
                self.current_ref_stats[feat_name]['mu'] = new_mu_calc
                self.current_ref_stats[feat_name]['sigma'] = effective_new_sigma

    def get_current_adaptive_stats(self) -> Dict[str, Dict[str, float]]:
         return copy.deepcopy(self.current_ref_stats)