# components/rewards/instantaneous_reward_calculator.py
import numpy as np
import pandas as pd # Para pd.notna/isna
import math
import logging
from typing import Tuple, Any, Dict
from interfaces.reward_function import RewardFunction
from interfaces.stability_calculator import BaseStabilityCalculator

logger = logging.getLogger(__name__)

class InstantaneousRewardCalculator(RewardFunction):
    def __init__(self,
                 reward_setup_config: Dict[str, Any], # Recibe toda la sección reward_setup
                 stability_calculator: BaseStabilityCalculator # Inyección del StabilityCalculator
                 ):
        self.calculation_config_data = reward_setup_config.get('calculation', {})
        self.reward_method = self.calculation_config_data.get('method')
        logger.info(f"[InstantRewardCalc] Initializing. Method: {self.reward_method}")
        
        self.stability_calc_instance = stability_calculator # Almacenarla

        # --- Type of Reward Configuración ---
        if self.reward_method == 'stability_measure_based':
            if not hasattr(self.stability_calc_instance, 'calculate_stability_based_reward'):
                raise AttributeError(f"Provided StabilityCalculator for 'stability_measure_based' reward is missing 'calculate_stability_based_reward' method.")

        if self.reward_method == 'weighted_exponential':
            weighted_exp_cfg = self.calculation_config_data.get('weighted_exponential_params', {})
            self.feature_weights = weighted_exp_cfg.get('feature_weights', {})
            self.feature_scales = weighted_exp_cfg.get('feature_scales', {})
            # Mapeo fijo para el péndulo. Podría ser configurable.
            self.sys_state_indices_map = {'cart_position': 0, 'cart_velocity': 1, 'angle': 2, 'angular_velocity': 3}
            logger.debug(f"[InstantRewardCalc] WeightedExp Params: Weights={self.feature_weights}, Scales={self.feature_scales}")
        
        # --- Penalty Configuración ---
        penalty_cfg = reward_setup_config.get('penalty_instantaneous_reward', {})
        self.penalty_enabled = bool(penalty_cfg.get('enabled', False))
        self.penalty_method = penalty_cfg.get('method')
        self.penalty_type = penalty_cfg.get('type')
        self.penalty_value_coeff = 0.0
        if self.penalty_enabled and self.penalty_method == 'lineal' and self.penalty_type == 'fixed':
            lineal_params = penalty_cfg.get('penalty_lineal_params', {})
            self.penalty_value_coeff = float(lineal_params.get('time', 0.2))
        
        # --- Goal Bonus Reward Configuration---
        goal_bonus_cfg = reward_setup_config.get('goal_bonus_reward', {})
        self.goal_bonus_enabled = bool(goal_bonus_cfg.get('enabled', False))
        self.goal_bonus_type = goal_bonus_cfg.get('type', 'static')  # "static" | "decay"
        # Estático
        self.goal_bonus_static_value = float(goal_bonus_cfg.get('static_value', 0.0))
        # Decay
        self.goal_bonus_decay_type = goal_bonus_cfg.get('decay_type', 'exponential')  # "exponential" | "linear" | "none"
        self.goal_bonus_base_value = float(goal_bonus_cfg.get('base_value', 0.0))
        self.goal_bonus_decay_tau = float(goal_bonus_cfg.get('decay_time_constant', 1.0))
        self.goal_bonus_min_value = float(goal_bonus_cfg.get('min_value', 0.0))
        # Approach Bonus
        approach_cfg = goal_bonus_cfg.get('approach_bonus', {})
        self.approach_bonus_enabled = bool(approach_cfg.get('enabled', False))
        self.approach_angle_range = float(approach_cfg.get('angle_range', 0.05))
        self.approach_angvel_range = float(approach_cfg.get('angular_velocity_range', 0.4))
        self.approach_cart_position_range = float(approach_cfg.get('cart_position_range', 0.1))
        self.approach_cart_velocity_range = float(approach_cfg.get('cart_velocity_range', 0.2))
        self.approach_per_step_bonus = float(approach_cfg.get('per_step_bonus', 0.0))
        self.approach_max_total_bonus = approach_cfg.get('max_total_bonus', None)
        self.approach_max_total_bonus = float(self.approach_max_total_bonus) if self.approach_max_total_bonus is not None else None

        # Estado interno: sólo para el approach_bonus
        self._approach_cumulative_bonus = 0.0

        logger.info(f"[InstantRewardCalc] GoalBonus: enabled={self.goal_bonus_enabled}, type={self.goal_bonus_type}, static={self.goal_bonus_static_value}, decay={self.goal_bonus_decay_type}, base={self.goal_bonus_base_value}, tau={self.goal_bonus_decay_tau}, min={self.goal_bonus_min_value}")

        logger.info(f"[InstantRewardCalc] Initialized. PenaltyEnabled: {self.penalty_enabled}, BonusEnabled: {self.goal_bonus_enabled}")

        logger.info("[InstantRewardCalc] Initialization complete.")

    def calculate(self, 
                  state_s: Any,
                  action_a: Any,
                  next_state_s_prime: Any, 
                  current_episode_time_sec: float,
                  dt_sec: float,
                  goal_reached_in_step: bool
                  ) -> float:
        """
        Calcula la recompensa total considerando:
        - Recompensa base (según método configurado)
        - Penalización temporal
        - Bono de meta (goal_bonus) según modo (estático o decaimiento)
        - Bono por tiempo dentro de la zona ampliada ("approach bonus")
        """
        # 1. Calcular el valor de la recompensa (calculated_reward_value)
        if self.reward_method == 'weighted_exponential':
            action_val_float = float(action_a)
            reward_terms_list = []
            for feat_name, feat_idx in self.sys_state_indices_map.items():
                # Si feat_name no está en feature_weights o feature_scales, ocurrirá KeyError (error de config).
                weight = self.feature_weights.get(feat_name, 0.0)
                val_s_prime = next_state_s_prime[feat_idx]
                scale = self.feature_scales.get(feat_name, 1.0)
                if scale == 0.0: 
                    scale = 1.0
                if not (pd.notna(val_s_prime) and np.isfinite(val_s_prime)):
                    reward_terms_list.append(0.0)
                    continue
                exp_arg_val = (val_s_prime / scale)**2
                term_val = weight * math.exp(-exp_arg_val) # Limitar exp_arg?? min(exp_arg_val, 700.0)
                reward_terms_list.append(term_val)
            # Force Gaussian Reward
            force_weight = self.feature_weights.get('force', 0.0)
            force_scale = self.feature_scales.get('force', 1.0)
            if force_scale == 0: 
                force_scale = 1.0
            exp_arg_force_val = (action_val_float / force_scale)**2
            reward_terms_list.append(force_weight * math.exp(-exp_arg_force_val))
            # Time Gaussian Reward
            time_weight_exp = self.feature_weights.get('time', 0.0)
            time_scale_exp = self.feature_scales.get('time', 1.0)
            if time_scale_exp == 0: 
                time_scale_exp = 1.0
            exp_arg_time_val_exp = (float(current_episode_time_sec) / time_scale_exp)**2
            reward_terms_list.append(time_weight_exp * math.exp(-exp_arg_time_val_exp))
            
            base_calculated_reward = float(np.nansum(reward_terms_list)) # nansum para manejar posibles NaNs de math.exp

        elif self.reward_method == 'stability_measure_based':
            base_calculated_reward = float(self.stability_calc_instance.calculate_stability_based_reward(next_state_s_prime))
        else:
            base_calculated_reward = 0.0

        final_reward_value = base_calculated_reward

        # --- 2. Lineal Penalty Reward for time ---
        final_reward_value += -self.penalty_value_coeff * current_episode_time_sec

        # --- 3. Goal Bonus Reward ---
        if self.goal_bonus_enabled and goal_reached_in_step:
            bonus = 0.0
            if self.goal_bonus_type == "static":
                bonus = self.goal_bonus_static_value
            elif self.goal_bonus_type == "decay":
                if self.goal_bonus_decay_type == "exponential":
                    bonus = self.goal_bonus_base_value * math.exp(-current_episode_time_sec / max(self.goal_bonus_decay_tau, 1e-8))
                elif self.goal_bonus_decay_type == "linear":
                    bonus = self.goal_bonus_base_value * max(0.0, 1.0 - current_episode_time_sec / max(self.goal_bonus_decay_tau, 1e-8))
                else:  # none
                    bonus = self.goal_bonus_base_value
                bonus = max(self.goal_bonus_min_value, min(self.goal_bonus_base_value, bonus))
            final_reward_value += bonus
            #logger.debug(f"[InstantReward:Calculation] Goal reached at t={current_episode_time_sec:.3f}s. Goal bonus applied: {bonus:.3f} (type={self.goal_bonus_type}, decay={self.goal_bonus_decay_type})")

        # --- 4. Approach Bonus: por cada paso dentro de la zona ampliada (acumulativo) ---
        if self.approach_bonus_enabled and self._is_in_approach_band(next_state_s_prime):
            self._approach_cumulative_bonus += self.approach_per_step_bonus
            # Tope máximo opcional por episodio
            if self.approach_max_total_bonus is not None:
                self._approach_cumulative_bonus = min(self._approach_cumulative_bonus, self.approach_max_total_bonus)
            final_reward_value += self.approach_per_step_bonus
            
        return final_reward_value

    def update_calculator_stats(self, episode_metrics_data_dict: Dict, completed_episode_index: int):
        pass # No es adaptativo (por ahora)

    def reset(self):
        """Llamar desde environment.reset() al inicio de cada episodio."""
        self._approach_cumulative_bonus = 0.0
    
    def _is_in_approach_band(self, state: Any) -> bool:
        """Retorna True si el estado está dentro de la zona ampliada definida."""
        angle_ok = abs(state[2]) < self.approach_angle_range
        angvel_ok = abs(state[3]) < self.approach_angvel_range
        cartpos_ok = abs(state[0]) < self.approach_cart_position_range
        cartvel_ok = abs(state[1]) < self.approach_cart_velocity_range
        return angle_ok and angvel_ok and cartpos_ok and cartvel_ok