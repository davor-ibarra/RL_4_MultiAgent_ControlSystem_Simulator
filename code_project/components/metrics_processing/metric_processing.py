# components/metrics_processing/metric_processing.py
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
import logging
from collections import deque

logger = logging.getLogger(__name__)

class MetricProcessing:
    """
    Centralized component for data transformation, aggregation, and derived metric calculation.
    Agnostic to the specific domain, operating on dictionaries of features.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processing_cfg = config.get('environment', {}).get('reward_setup', {}).get('reward_config', {}).get('reward_calculation', {}).get('reward_low_observability', {}).get('metrics_processing', {})
        
        self.enabled = self.processing_cfg.get('enabled', True)
        self.window_mode = self.processing_cfg.get('window', 'agent_decision_interval')
        
        # Buffers for history (for standardization/normalization)
        self.history_buffers: Dict[str, deque] = {}
        self.history_horizon = self.processing_cfg.get('standardization', {}).get('history_horizon_intervals', 100)
        
        # Normalization params
        self.norm_cfg = self.processing_cfg.get('normalization', {})
        self.quantile_low = self.norm_cfg.get('quantile_low', 0.05)
        self.quantile_high = self.norm_cfg.get('quantile_high', 0.95)
        self.epsilon_norm = float(self.norm_cfg.get('epsilon_norm', 1e-6))
        
        # Standardization params
        self.std_cfg = self.processing_cfg.get('standardization', {})
        self.std_method = self.std_cfg.get('method', 'median_mad')
        self.epsilon_std = float(self.std_cfg.get('epsilon_std', 1e-6))

        logger.info(f"[MetricProcessing] Initialized. Window: {self.window_mode}, StdMethod: {self.std_method}")

    def process_step_metrics(self, env_id: str, current_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Processes metrics for a single step for a given env_id.
        Updates history and calculates Z-scores/Norms on the fly if configured.
        """
        if not self.enabled or not current_metrics:
            return {}
            
        # 1. Update History
        self._update_history(env_id, current_metrics)
        
        # 2. Transform (Normalization / Standardization) based on current history
        processed_metrics = self._transform_metrics(env_id, current_metrics)
        
        return processed_metrics

    def process_interval_metrics(self, 
                                 env_id: str,
                                 raw_metrics_buffer: List[Dict[str, float]], 
                                 interval_duration: float) -> Dict[str, float]:
        """
        Processes a buffer of raw metrics (e.g., from a simulation interval) into aggregated and transformed metrics.
        """
        if not self.enabled or not raw_metrics_buffer:
            return {}

        # 1. Aggregation (Mean, Sum, RMS, etc.)
        aggregated_metrics = self._aggregate_metrics(raw_metrics_buffer, interval_duration)
        
        # 2. Derived Metrics (Energy, Smoothness, etc.) - Placeholder for now
        # derived_metrics = self._calculate_derived_metrics(aggregated_metrics)
        # aggregated_metrics.update(derived_metrics)

        # 3. Update History (for adaptive normalization/standardization)
        self._update_history(env_id, aggregated_metrics)

        # 4. Transformation (Normalization / Standardization)
        processed_metrics = self._transform_metrics(env_id, aggregated_metrics)

        return processed_metrics

    def _aggregate_metrics(self, buffer: List[Dict[str, float]], duration: float) -> Dict[str, float]:
        """Aggregates a list of dicts into a single dict using defined methods."""
        if not buffer:
            return {}
            
        # Convert list of dicts to dict of lists for easier processing
        keys = buffer[0].keys()
        values_map = {k: [] for k in keys}
        for item in buffer:
            for k, v in item.items():
                values_map[k].append(v)
        
        aggregated = {}
        # Default aggregation: mean for most, sum for some? 
        # Ideally this should be configurable per feature. For now, we calculate common stats.
        
        for k, values in values_map.items():
            arr = np.array(values, dtype=float)
            # Basic aggregations
            aggregated[f"{k}_mean"] = float(np.mean(arr))
            aggregated[f"{k}_sum"] = float(np.sum(arr))
            aggregated[f"{k}_max"] = float(np.max(arr))
            aggregated[f"{k}_min"] = float(np.min(arr))
            aggregated[f"{k}_std"] = float(np.std(arr))
            aggregated[f"{k}_rms"] = float(np.sqrt(np.mean(arr**2)))
            
            # Integral approximation (sum * dt) - assuming uniform dt within buffer if not provided
            # If duration is provided, we can approximate integral
            aggregated[f"{k}_integral"] = float(np.sum(arr)) * (duration / len(arr)) if len(arr) > 0 else 0.0

        return aggregated

    def _update_history(self, env_id: str, metrics: Dict[str, float]):
        """Updates the history buffers with new aggregated metrics."""
        if env_id not in self.history_buffers:
            self.history_buffers[env_id] = {}
        
        env_buffer = self.history_buffers[env_id]
        
        for k, v in metrics.items():
            if k not in env_buffer:
                env_buffer[k] = deque(maxlen=self.history_horizon)
            env_buffer[k].append(v)

    def _transform_metrics(self, env_id: str, metrics: Dict[str, float]) -> Dict[str, float]:
        """Applies standardization and normalization to metrics."""
        transformed = metrics.copy()
        
        env_buffer = self.history_buffers.get(env_id, {})
        
        for k, v in metrics.items():
            history = list(env_buffer.get(k, []))
            if len(history) < 2:
                continue
                
            # Standardization
            if self.std_method == 'median_mad':
                median = np.median(history)
                mad = np.median(np.abs(np.array(history) - median))
                z_score = (v - median) / (mad + self.epsilon_std)
                transformed[f"{k}_zscore"] = float(z_score)
            elif self.std_method == 'mean_std':
                mean = np.mean(history)
                std = np.std(history)
                z_score = (v - mean) / (std + self.epsilon_std)
                transformed[f"{k}_zscore"] = float(z_score)

            # Normalization (Min-Max / Quantile)
            if self.norm_cfg.get('enabled', False):
                q_low = np.quantile(history, self.quantile_low)
                q_high = np.quantile(history, self.quantile_high)
                rng = q_high - q_low
                norm_val = (v - q_low) / (rng + self.epsilon_norm)
                # Clip to [0, 1] or keep raw? Usually clip for NN inputs
                norm_val = np.clip(norm_val, 0.0, 1.0) 
                transformed[f"{k}_norm"] = float(norm_val)

        return transformed

    def reset(self):
        """Resets history buffers."""
        self.history_buffers.clear()
