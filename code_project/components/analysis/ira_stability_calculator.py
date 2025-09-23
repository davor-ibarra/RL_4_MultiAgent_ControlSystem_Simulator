# components/analysis/ira_stability_calculator.py
import numpy as np
import pandas as pd # Solo para pd.notna/isna
import math
import logging
from interfaces.stability_calculator import BaseStabilityCalculator
from typing import Any, Dict, List # List
import copy

logger = logging.getLogger(__name__)

class IRAStabilityCalculator(BaseStabilityCalculator):
    def __init__(self, ira_zscore_metric_params: Dict[str, Any]): # Parámetros de config.reward_setup.calculation.stability_measure.ira_zscore_metric_params
        logger.info(f"[IRAStabilityCalc] Initializing...")

        # Asignación directa. Se asume que las claves existen y los tipos son correctos (validados externamente).
        self.feature_weights: Dict[str, float] = ira_zscore_metric_params['feature_weights']
        initial_ref_stats: Dict[str, Dict[str, float]] = ira_zscore_metric_params['initial_reference_stats']
        
        self.lambda_factor = float(ira_zscore_metric_params.get('lambda_factor', 1.0))
        self.epsilon_denom = float(ira_zscore_metric_params.get('epsilon_zscore_denominator', 1e-6))
        if self.epsilon_denom <= 0: self.epsilon_denom = 1e-6 # Pequeña auto-corrección

        adaptive_cfg = ira_zscore_metric_params.get('adaptive_stats', {})
        self.is_adaptive_enabled = bool(adaptive_cfg.get('enabled', False))
        self.min_episode_for_adapt = int(adaptive_cfg.get('min_episode', 0))
        self.min_sigma_threshold = float(adaptive_cfg.get('min_sigma', 1e-4))
        if self.min_sigma_threshold <= 0: self.min_sigma_threshold = 1e-4

        # Mapeo de características (fijo para péndulo, podría ser configurable)
        self.system_feature_map: Dict[str, int] = {'cart_position': 0, 'cart_velocity': 1, 'angle': 2, 'angular_velocity': 3}
        
        # Inicializar current_ref_stats asegurando que sigma sea >= min_sigma_threshold
        self.current_ref_stats: Dict[str, Dict[str, float]] = {}
        for feat, idx in self.system_feature_map.items():
            if feat not in initial_ref_stats: # Error de config si falta
                raise ValueError(f"Missing initial_reference_stats for feature '{feat}' in IRAStabilityCalculator config.")
            mu_init = float(initial_ref_stats[feat]['mu'])
            sigma_init = float(initial_ref_stats[feat]['sigma'])
            self.current_ref_stats[feat] = {
                'mu': mu_init,
                'sigma': max(abs(sigma_init), self.min_sigma_threshold) # abs() por si se configura negativo
            }
        
        logger.info(f"[IRAStabilityCalc] Initialized. Adaptive: {self.is_adaptive_enabled}, Lambda: {self.lambda_factor}")
        # logger.debug(f"[IRAStabilityCalc] Weights: {self.feature_weights}, InitRefStats: {self.current_ref_stats}")

    def _normalize_feature_value(self, value: float, feature_name: str) -> float:
        # Asume que feature_name está en self.current_ref_stats.
        # Asume que value es un float (puede ser NaN/inf).
        stats = self.current_ref_stats[feature_name]
        mu, sigma = stats['mu'], stats['sigma'] # sigma ya es >= min_sigma_threshold
        
        # Denominador efectivo, siempre positivo
        effective_denom = max(sigma, self.epsilon_denom) 
        
        if pd.isna(value) or not np.isfinite(value): return 0.0 # Z-score de 0 para inválidos
        return (value - mu) / effective_denom

    def calculate_instantaneous_stability(self, current_state: Any) -> float:
        # Asume que current_state es un array/lista con suficientes elementos.
        sum_weighted_sq_z = 0.0
        for feat_name, feat_idx in self.system_feature_map.items():
            weight = self.feature_weights.get(feat_name, 0.0) # Default a 0 si no está en config
            if abs(weight) > 1e-9: # Solo si el peso es significativo
                # Si current_state es muy corto, esto dará IndexError.
                # Si current_state[feat_idx] no es float, _normalize_feature_value lo manejará (o fallará si no es convertible).
                z_score = self._normalize_feature_value(float(current_state[feat_idx]), feat_name)
                sum_weighted_sq_z += weight * (z_score ** 2)
        
        exponent = -min(sum_weighted_sq_z, 700.0) # Limitar exponente
        stability = math.exp(exponent)
        return float(np.clip(stability, 0.0, 1.0)) # Asegurar [0,1]

    def calculate_stability_based_reward(self, current_state: Any) -> float:
        # Similar a calculate_instantaneous_stability pero aplicando lambda_factor.
        sum_weighted_sq_z_reward = 0.0
        for feat_name, feat_idx in self.system_feature_map.items():
            weight = self.feature_weights.get(feat_name, 0.0)
            if abs(weight) > 1e-9:
                z_score = self._normalize_feature_value(float(current_state[feat_idx]), feat_name)
                sum_weighted_sq_z_reward += weight * (z_score ** 2)
        
        # Aplicar lambda_factor
        eff_lambda = max(abs(self.lambda_factor), 1e-9) # Evitar lambda cero
        exponent_reward = -self.lambda_factor * min(sum_weighted_sq_z_reward, 700.0 / eff_lambda)
        reward = math.exp(exponent_reward)
        return float(max(0.0, reward)) # Recompensa no negativa

    def update_reference_stats(self, episode_metrics: Dict[str, List[Any]], episode_num: int):
        if not self.is_adaptive_enabled or episode_num < self.min_episode_for_adapt:
            return

        # logger.info(f"[IRAStabilityCalc:update_ref_stats] Updating adaptive stats after ep {episode_num}...")
        stats_did_change = False
        # Mapeo de nombres de características internas a nombres de métricas en episode_metrics
        feat_to_metric_key = {
            'angle': 'pendulum_angle', 'angular_velocity': 'pendulum_velocity',
            'cart_position': 'cart_position', 'cart_velocity': 'cart_velocity'
        }
        for feat_name_update, metric_data_key in feat_to_metric_key.items():
            # Asumir que feat_name_update está en self.current_ref_stats.
            metric_values_list = episode_metrics.get(metric_data_key)
            if not isinstance(metric_values_list, list) or len(metric_values_list) < 2: continue

            # Convertir a numérico y quitar NaNs/infs directamente
            valid_data_points = np.array([float(v) for v in metric_values_list if pd.notna(v) and np.isfinite(v)])
            if len(valid_data_points) < 2: continue
            
            new_mu_calc = np.mean(valid_data_points)
            new_sigma_calc = np.std(valid_data_points)
            effective_new_sigma_val = max(new_sigma_calc, self.min_sigma_threshold)
            
            # Actualizar si hay cambio (opcional, pero bueno para evitar logs innecesarios)
            if not np.isclose(self.current_ref_stats[feat_name_update]['mu'], new_mu_calc) or \
               not np.isclose(self.current_ref_stats[feat_name_update]['sigma'], effective_new_sigma_val):
                self.current_ref_stats[feat_name_update]['mu'] = new_mu_calc
                self.current_ref_stats[feat_name_update]['sigma'] = effective_new_sigma_val
                stats_did_change = True
                # logger.debug(f"[IRAStabilityCalc] Stats for '{feat_name_update}' updated: mu={new_mu_calc:.4f}, sigma={effective_new_sigma_val:.4f}")
        
        # if stats_did_change: logger.info(f"[IRAStabilityCalc] Adaptive reference stats updated after ep {episode_num}.")

    def get_current_adaptive_stats(self) -> Dict[str, Dict[str, float]]:
         return copy.deepcopy(self.current_ref_stats) # Devolver copia para evitar modificación externa