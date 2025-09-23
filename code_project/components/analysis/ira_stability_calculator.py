# components/analysis/ira_stability_calculator.py
import numpy as np
import pandas as pd
import math
import logging
from interfaces.stability_calculator import BaseStabilityCalculator
from typing import Any, Dict, Optional
import copy

logger = logging.getLogger(__name__) # Logger específico del módulo

class IRAStabilityCalculator(BaseStabilityCalculator):
    def __init__(self, ira_params: Dict[str, Any]):
        # ira_params es la sección config['environment']['reward_setup']['calculation']['stability_calculator']['ira_params']
        logger.info(f"[IRAStabilityCalc] Initializing with params: {list(ira_params.keys())}")

        # --- Validar estructura y tipos de ira_params ---
        if not isinstance(ira_params, dict):
            raise TypeError("ira_params must be a dictionary.")

        # Weights
        self.weights = ira_params.get('weights')
        if not isinstance(self.weights, dict):
            raise ValueError("'weights' missing or not a dictionary in ira_params.")

        # Initial Reference Stats
        initial_ref_stats_raw = ira_params.get('initial_reference_stats')
        if not isinstance(initial_ref_stats_raw, dict):
            raise ValueError("'initial_reference_stats' missing or not a dictionary in ira_params.")

        # Lambda, z_score_epsilon (con defaults si faltan)
        self.lambda_param = float(ira_params.get('lambda', 1.0)) # Default 1.0
        self.z_score_epsilon = float(ira_params.get('z_score_epsilon', 1e-6)) # Default 1e-6
        if self.z_score_epsilon <= 0:
            logger.warning(f"[IRAStabilityCalc] z_score_epsilon ({self.z_score_epsilon}) must be > 0. Using 1e-6.")
            self.z_score_epsilon = 1e-6

        # Adaptive Stats Config
        adaptive_cfg = ira_params.get('adaptive_stats', {}) # Default a dict vacío si falta
        if not isinstance(adaptive_cfg, dict):
            raise TypeError("'adaptive_stats' must be a dictionary if provided.")
        self.adaptive_enabled = bool(adaptive_cfg.get('enabled', False))
        self.adaptive_min_episode = int(adaptive_cfg.get('min_episode', 0)) # Default 0
        self.min_sigma = float(adaptive_cfg.get('min_sigma', 1e-4)) # Default 1e-4
        if self.min_sigma <= 0:
            logger.warning(f"[IRAStabilityCalc] min_sigma ({self.min_sigma}) must be > 0. Using 1e-4.")
            self.min_sigma = 1e-4

        # --- Validar contenido de weights y initial_reference_stats ---
        self.var_indices = {'cart_position': 0, 'cart_velocity': 1, 'angle': 2, 'angular_velocity': 3}
        expected_vars = set(self.var_indices.keys())

        # Validar weights
        if set(self.weights.keys()) != expected_vars:
            # Permitir si solo algunas están, pero advertir. Si faltan todas, error.
            if not expected_vars.issubset(self.weights.keys()): # Check if all expected are present
                 logger.warning(f"[IRAStabilityCalc] 'weights' keys ({self.weights.keys()}) do not exactly match expected state variables ({expected_vars}). Ensure all relevant variables have weights.")
            # Validar que los pesos sean numéricos
        for var_name in expected_vars:
            w = self.weights.get(var_name, 0.0) # Default a 0 si alguna variable falta
            if not isinstance(w, (int, float)):
                raise TypeError(f"Weight for '{var_name}' in 'weights' is not numeric: {w}")
            self.weights[var_name] = float(w) # Asegurar float

        # Validar y copiar profundamente initial_reference_stats
        self.current_ref_stats = self._validate_and_copy_stats(initial_ref_stats_raw, expected_vars)

        logger.info(f"[IRAStabilityCalc] Initialized. Adaptive: {self.adaptive_enabled} (Min Ep: {self.adaptive_min_episode}).")
        logger.debug(f"[IRAStabilityCalc] Weights: {self.weights}, Lambda: {self.lambda_param}, Init Stats: {self.current_ref_stats}")

    def _validate_and_copy_stats(self, stats_dict: Dict, expected_vars: set) -> Dict:
        """Valida y copia profundamente las estadísticas de referencia."""
        validated_stats = {}
        if set(stats_dict.keys()) != expected_vars:
            # Permitir stats parciales pero advertir si faltan todas las esperadas
            if not expected_vars.issubset(stats_dict.keys()):
                raise ValueError(f"'initial_reference_stats' keys ({stats_dict.keys()}) do not match expected state variables ({expected_vars}).")
            logger.warning(f"[IRAStabilityCalc:_validate_stats] 'initial_reference_stats' keys do not exactly match. Processing available.")

        for var_name in expected_vars:
            if var_name not in stats_dict: # Si una variable esperada no tiene stats, es un error
                raise ValueError(f"Missing initial reference stats for expected variable '{var_name}'.")

            stats = stats_dict[var_name]
            if not isinstance(stats, dict) or 'mu' not in stats or 'sigma' not in stats:
                raise ValueError(f"Invalid entry for '{var_name}' in 'initial_reference_stats': {stats}. Must be dict with 'mu' and 'sigma'.")

            mu = stats['mu']
            sigma = stats['sigma']
            if not isinstance(mu, (int, float)):
                raise TypeError(f"'mu' for '{var_name}' is not numeric: {type(mu).__name__}.")
            if not isinstance(sigma, (int, float)):
                raise TypeError(f"'sigma' for '{var_name}' is not numeric: {type(sigma).__name__}.")

            # Asegurar que sigma sea positivo y >= min_sigma
            effective_sigma = float(sigma)
            if effective_sigma <= 0:
                logger.warning(f"[IRAStabilityCalc:_validate_stats] Initial sigma for '{var_name}' ({sigma}) is not positive. Using min_sigma ({self.min_sigma}).")
                effective_sigma = self.min_sigma
            else:
                effective_sigma = max(effective_sigma, self.min_sigma)

            validated_stats[var_name] = {'mu': float(mu), 'sigma': effective_sigma}
        return copy.deepcopy(validated_stats) # Usar deepcopy

    def _normalize_state_variable(self, value: float, var_name: str) -> float:
        if var_name not in self.current_ref_stats:
            logger.warning(f"[IRAStabilityCalc:_normalize] Stats for '{var_name}' not found. Returning 0 for z-score.")
            return 0.0
        stats = self.current_ref_stats[var_name]
        mu = stats.get('mu', 0.0)
        sigma_actual = stats.get('sigma', 1.0)
        sigma_effective = max(sigma_actual, self.min_sigma, self.z_score_epsilon)
        if pd.isna(value) or not np.isfinite(value):
            #logger.debug(f"[IRAStabilityCalc:_normalize] Invalid value (NaN/inf) for '{var_name}': {value}. Returning 0 for z-score.")
            return 0.0
        return (value - mu) / sigma_effective

    def calculate_instantaneous_stability(self, state: Any) -> float:
        if not isinstance(state, (np.ndarray, list)) or len(state) < 4:
            logger.warning(f"[IRAStabilityCalc:calc_inst_stab] Invalid or short state: {state}. Returning 0.0.")
            return 0.0
        deviation_sum_sq_weighted = 0.0
        try:
            for var_name, index in self.var_indices.items():
                weight = self.weights.get(var_name, 0.0)
                if weight > 0: # Solo procesar si el peso es significativo
                    value = float(state[index])
                    z_s = self._normalize_state_variable(value, var_name)
                    deviation_sum_sq_weighted += weight * (z_s ** 2)
            exponent_arg = -min(deviation_sum_sq_weighted, 700.0) # Limitar exponente
            stability_score = math.exp(exponent_arg)
        except IndexError:
            logger.error(f"[IRAStabilityCalc:calc_inst_stab] IndexError. State: {state}")
            return 0.0
        except Exception as e:
            logger.error(f"[IRAStabilityCalc:calc_inst_stab] Unexpected error: {e}", exc_info=True)
            return 0.0
        return float(np.clip(stability_score, 0.0, 1.0))

    def calculate_stability_based_reward(self, state: Any) -> float:
        if not isinstance(state, (np.ndarray, list)) or len(state) < 4:
            logger.warning(f"[IRAStabilityCalc:calc_reward] Invalid or short state: {state}. Returning 0.0.")
            return 0.0
        deviation_sum_sq_weighted = 0.0
        try:
            for var_name, index in self.var_indices.items():
                weight = self.weights.get(var_name, 0.0)
                if weight > 0:
                    value = float(state[index])
                    z_s = self._normalize_state_variable(value, var_name)
                    deviation_sum_sq_weighted += weight * (z_s ** 2)
            lambda_eff = max(abs(self.lambda_param), 1e-9) # Evitar lambda cero
            exponent_arg = -self.lambda_param * min(deviation_sum_sq_weighted, 700.0 / lambda_eff)
            reward = math.exp(exponent_arg)
        except IndexError:
            logger.error(f"[IRAStabilityCalc:calc_reward] IndexError. State: {state}")
            return 0.0
        except Exception as e:
            logger.error(f"[IRAStabilityCalc:calc_reward] Unexpected error: {e}", exc_info=True)
            return 0.0
        return float(max(0.0, reward)) # Recompensa no negativa

    def update_reference_stats(self, episode_metrics_dict: Dict, current_episode: int):
        if not self.adaptive_enabled or current_episode < self.adaptive_min_episode:
            return
        logger.info(f"[IRAStabilityCalc:update_stats] Updating adaptive stats after ep {current_episode}...")
        updated_any = False
        var_to_metric_map = {'angle': 'pendulum_angle', 'angular_velocity': 'pendulum_velocity', 'cart_position': 'cart_position', 'cart_velocity': 'cart_velocity'}
        for var_name_cfg, metric_name in var_to_metric_map.items():
            if var_name_cfg not in self.current_ref_stats: continue
            values = episode_metrics_dict.get(metric_name)
            if not isinstance(values, list) or len(values) < 2: continue

            try:
                numeric_values = pd.to_numeric(pd.Series(values), errors='coerce')
                valid_values = numeric_values.dropna()[np.isfinite(numeric_values.dropna())]
                if len(valid_values) < 2: continue
                new_mu = float(np.mean(valid_values))
                new_sigma = float(np.std(valid_values))
                effective_sigma = max(new_sigma, self.min_sigma)
                current_mu = self.current_ref_stats[var_name_cfg]['mu']
                current_sigma = self.current_ref_stats[var_name_cfg]['sigma']
                if not np.isclose(current_mu, new_mu) or not np.isclose(current_sigma, effective_sigma):
                    self.current_ref_stats[var_name_cfg]['mu'] = new_mu
                    self.current_ref_stats[var_name_cfg]['sigma'] = effective_sigma
                    updated_any = True
                    #logger.debug(f"[IRAStabilityCalc:update_stats] Stats '{var_name_cfg}' updated: mu={new_mu:.4f}, sigma={effective_sigma:.4f}")
            except Exception as e:
                logger.error(f"[IRAStabilityCalc:update_stats] Error updating stats for '{metric_name}': {e}", exc_info=True)
        if updated_any: logger.info(f"[IRAStabilityCalc:update_stats] Adaptive stats updated after ep {current_episode}.")

    def get_current_adaptive_stats(self) -> Dict:
         return copy.deepcopy(self.current_ref_stats)