# components/rewards/instantaneous_reward_calculator.py
import numpy as np
import pandas as pd
import math
import logging
from typing import Any, Dict
from interfaces.reward_function import RewardFunction
from interfaces.stability_calculator import BaseStabilityCalculator

logger = logging.getLogger(__name__)

class InstantaneousRewardCalculator(RewardFunction):
    """
    Calculates the instantaneous reward based on a declarative configuration.
    It is self-configuring from the global config and operates on a contextual
    state dictionary provided by the environment.
    """
    def __init__(self,
                 config: Dict[str, Any],
                 stability_calculator: BaseStabilityCalculator
                 ):
        logger.info("[InstantRewardCalc] Initializing...")
        
        # --- Self-configuration from global config ---
        try:
            reward_setup = config['environment']['reward_setup']
            self.calculation_config = reward_setup.get('calculation', {})
            self.reward_method = self.calculation_config.get('method')
        except KeyError as e:
            raise KeyError(f"InstantaneousRewardCalculator: Missing required configuration path: {e}")

        self.stability_calc_instance = stability_calculator
        
        # --- Reward Method Specific Setup ---
        self.feature_weights: Dict[str, float] = {}
        self.feature_scales: Dict[str, float] = {}
        self.feature_setpoints: Dict[str, float] = {}
        self.log_reward_params: Dict[str, float] = {}

        if self.reward_method == 'weighted_exponential':
            weighted_exp_params = self.calculation_config.get('weighted_exponential_params', {})
            feature_configs = weighted_exp_params.get('features', {})
            if not isinstance(feature_configs, dict):
                raise ValueError("weighted_exponential 'features' config must be a dictionary.")
            
            for feat, params in feature_configs.items():
                if not isinstance(params, dict): continue
                self.feature_weights[feat] = float(params.get('weight', 0.0))
                self.feature_scales[feat] = float(params.get('scaled', 1.0))
                self.feature_setpoints[feat] = float(params.get('setpoint', 0.0))

        elif self.reward_method == 'stability_measure_based':
            if not hasattr(self.stability_calc_instance, 'calculate_stability_based_reward'):
                raise AttributeError("Provided StabilityCalculator for 'stability_measure_based' reward is missing 'calculate_stability_based_reward' method.")
        
        # --- Penalty Configuration ---
        penalty_cfg = reward_setup['penalty_approach']
        # --- Penalty Instantaneous Reward Configuration ---
        penalty_inst_cfg = penalty_cfg.get('penalty_instantaneous_reward', {})
        self.penalty_method = penalty_inst_cfg.get('method', 'lineal')
        self.penalty_enabled = bool(penalty_inst_cfg.get('enabled', False))
        self.penalty_value_coeff = 0.0
        if self.penalty_enabled:
            if self.penalty_method == 'lineal':
                lineal_params = penalty_inst_cfg.get('penalty_lineal_params', {})
                self.penalty_value_coeff = float(lineal_params.get('time', 0.0))
            elif self.penalty_method == 'quadratic':
                lineal_params = penalty_inst_cfg.get('penalty_quadratic_params', {})
                self.penalty_value_coeff = float(lineal_params.get('time', 0.0))

        # --- Bonus Configuration ---
        bonus_cfg = reward_setup['bonus_approach']
        # --- Goal Bonus Configuration ---
        goal_bonus_cfg = bonus_cfg.get('goal_bonus_reward', {})
        self.goal_bonus_enabled = bool(goal_bonus_cfg.get('enabled', False))
        self.goal_bonus_type = goal_bonus_cfg.get('type', 'static')
        self.goal_bonus_static_value = float(goal_bonus_cfg.get('static_value', 0.0))
        self.goal_bonus_decay_type = goal_bonus_cfg.get('decay_type', 'exponential')
        self.goal_bonus_base_value = float(goal_bonus_cfg.get('base_value', 0.0))
        self.goal_bonus_decay_tau = float(goal_bonus_cfg.get('decay_time_constant', 1.0))
        self.goal_bonus_min_value = float(goal_bonus_cfg.get('min_value', 0.0))
        
        # --- Approach Bonus Configuration ---
        bandwidth_bonus_cfg = bonus_cfg.get('bandwidth_bonus', {})
        self.bandwidth_bonus_enabled = bool(bandwidth_bonus_cfg.get('enabled', False))
        self.bandwidth_per_step_bonus = float(bandwidth_bonus_cfg.get('per_step_bonus', 0.0))
        self.bandwidth_ranges = bandwidth_bonus_cfg.get('ranges', {})
        self.bandwidth_max_bonus = bandwidth_bonus_cfg.get('max_total_bonus', 1000)
        self.bandwidth_cumulative_bonus = 0.0

        logger.info(f"[InstantRewardCalc] Initialized. Method: {self.reward_method}, Penalty: {self.penalty_enabled}, GoalBonus: {self.goal_bonus_enabled}")

    def calculate(self, 
                  state_dict: Dict[str, Any],
                  action_a: Any,
                  next_state_dict: Dict[str, Any], 
                  current_episode_time_sec: float,
                  dt_sec: float,
                  goal_reached_in_step: bool
                  ) -> float:
        
        self.log_reward_params.clear()
        base_calculated_reward = 0.0

        if self.reward_method == 'weighted_exponential':
            reward_terms_list = []
            for feat_name, weight in self.feature_weights.items():
                # Determine value source (next state, action, or time)
                if feat_name in next_state_dict:
                    value = next_state_dict.get(feat_name)
                elif feat_name == 'control_action':
                    value = action_a
                elif feat_name == 'time':
                    value = current_episode_time_sec
                else:
                    continue

                if value is None or pd.isna(value) or not np.isfinite(value): continue
                
                feat_setpoint = self.feature_setpoints.get(feat_name, 0.0)
                value_error = float(value) - feat_setpoint
                exp_arg_val = (value_error / self.feature_scales.get(feat_name, 1.0))**2
                reward_term = weight * math.exp(-exp_arg_val)
                reward_terms_list.append(reward_term)
                self.log_reward_params[f'reward_{feat_name}'] = reward_term
            
            base_calculated_reward = float(np.nansum(reward_terms_list))

        elif self.reward_method == 'stability_measure_based':
            base_calculated_reward = self.stability_calc_instance.calculate_stability_based_reward(next_state_dict)
            self.log_reward_params['reward_by_stability'] = base_calculated_reward
            

        final_reward_value = base_calculated_reward
        
        if self.penalty_enabled:
            if self.penalty_method == 'lineal':
                # restar κ·Δt en cada paso
                penalty_term = -self.penalty_value_coeff * dt_sec
                final_reward_value += penalty_term
                self.log_reward_params['lineal_penalty_reward'] = penalty_term
            elif self.penalty_method in 'quadratic':
                # restar κ·t en cada paso
                penalty_term = -self.penalty_value_coeff * current_episode_time_sec
                final_reward_value += penalty_term
                self.log_reward_params['quadratic_penalty_reward'] = penalty_term

        if self.goal_bonus_enabled and goal_reached_in_step:
            bonus = 0.0
            if self.goal_bonus_type == "static":
                bonus = self.goal_bonus_static_value
            elif self.goal_bonus_type == "decay":
                if self.goal_bonus_decay_type == "exponential":
                    bonus = self.goal_bonus_base_value * math.exp(-current_episode_time_sec / max(self.goal_bonus_decay_tau, 1e-8))
                elif self.goal_bonus_decay_type == "linear":
                    bonus = self.goal_bonus_base_value * max(0.0, 1.0 - current_episode_time_sec / max(self.goal_bonus_decay_tau, 1e-8))
                else:
                    bonus = self.goal_bonus_base_value
                bonus = max(self.goal_bonus_min_value, min(self.goal_bonus_base_value, bonus))
                self.log_reward_params['bonus_reward'] = bonus
            final_reward_value += bonus

        if self.bandwidth_bonus_enabled and self._is_in_bandwidth_ranges(next_state_dict):
            self.bandwidth_cumulative_bonus += self.bandwidth_per_step_bonus
            final_reward_value += self.bandwidth_per_step_bonus if self.bandwidth_cumulative_bonus <= self.bandwidth_max_bonus else 0.0
            self.log_reward_params['bandwidth_per_step_bonus'] = self.bandwidth_per_step_bonus
            
        return final_reward_value

    def update_calculator_stats(self, episode_metrics_data_dict: Dict, completed_episode_index: int):
        pass # Not adaptive

    def reset(self):
        self.bandwidth_cumulative_bonus = 0.0
    
    def _is_in_bandwidth_ranges(self, state_dict: Dict[str, Any]) -> bool:
        if not self.bandwidth_ranges: 
            return False
        for feat_name, range_list in self.bandwidth_ranges.items():
            value = state_dict.get(feat_name)
            if value is not None: # Considerate feature only if exist
                if not (float(range_list[0]) <= float(value) <= float(range_list[1])):
                    return False # If any feature is out of range, not in band
        return True # If all features are within their ranges
    
    def get_params_log(self) -> Dict[str, float]:
        return self.log_reward_params