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
    The calculate() method acts as an orchestrator for different reward components.
    """
    def __init__(self,
                 config: Dict[str, Any],
                 stability_calculator: BaseStabilityCalculator
                 ):
        logger.info("[InstantRewardCalc] Initializing...")
        
        try:
            reward_setup = config['environment']['reward_setup']
        except KeyError as e:
            raise KeyError(f"InstantaneousRewardCalculator: Missing required configuration path: {e}")

        self.stability_calc_instance = stability_calculator
        self.log_reward_params: Dict[str, float] = {}

        # --- Cargar todas las configuraciones --- ### NUEVO ###
        self._init_base_calculation(reward_setup)
        self._init_penalties_and_bonuses(reward_setup)
        self._init_conditional_rewards(reward_setup)
        
        # --- Variables de estado internas ---
        self.bandwidth_cumulative_bonus = 0.0
        self.last_action_a = 0.0

        logger.info(f"[InstantRewardCalc] Initialization complete.")

    ### NUEVO: Métodos de inicialización para limpieza de código ###
    def _init_base_calculation(self, reward_setup: Dict):
        """Initializes the main reward calculation method."""
        self.cfg_base = reward_setup.get('calculation', {})
        self.reward_method = self.cfg_base.get('method')
        
    def _init_penalties_and_bonuses(self, reward_setup: Dict):
        """Initializes fixed penalties and bonuses."""
        # --- Penalty Configuration ---
        self.cfg_penalty = reward_setup.get('penalty_approach', {}).get('penalty_instantaneous_reward', {})
        self.penalty_enabled = self.cfg_penalty.get('enabled', False)

        # --- Delta Var Penalty Configuration ---
        self.cfg_delta_penalty = reward_setup.get('penalty_approach', {}).get('penalty_delta_var', {})
        self.delta_penalty_enabled = self.cfg_delta_penalty.get('enabled', False)

        # --- Bonus Configuration ---
        bonus_cfg = reward_setup.get('bonus_approach', {})
        self.cfg_goal_bonus = bonus_cfg.get('goal_bonus_reward', {})
        self.goal_bonus_enabled = self.cfg_goal_bonus.get('enabled', False)
        
        self.cfg_bw_bonus = bonus_cfg.get('bandwidth_bonus', {})
        self.bw_bonus_enabled = self.cfg_bw_bonus.get('enabled', False)

    def _init_conditional_rewards(self, reward_setup: Dict):
        """Initializes conditional and dynamic reward components."""
        cond_cfg = reward_setup.get('conditional_approach', {})
        # --- Dynamic Penalty ---
        self.cfg_dp = cond_cfg.get('dynamic_penalty', {})
        self.dp_enabled = self.cfg_dp.get('enabled', False)
        self.dp_method = self.cfg_dp.get('method')
        self.dp_params = self.cfg_dp.get('dynamic_penalty_params', {})

        # --- Dynamic Incentive ---
        self.cfg_di = cond_cfg.get('dynamic_incentive', {})
        self.di_enabled = self.cfg_di.get('enabled', False)
        self.di_method = self.cfg_di.get('method')
        self.di_lineal_params = self.cfg_di.get('dynamic_incentive_lineal_params', {})
        self.di_tanh_params = self.cfg_di.get('dynamic_incentive_adapt_params', {})

    ### MODIFICADO: calculate() ahora es un orquestador ###
    def calculate(self, 
                  state_dict: Dict[str, Any],
                  action_a: Any,
                  next_state_dict: Dict[str, Any], 
                  current_episode_time_sec: float,
                  dt_sec: float,
                  goal_reached_in_step: bool
                  ) -> float:
        
        self.log_reward_params.clear()
        reward_terms = []

        # 1. Recompensa base (cálculo principal)
        reward_terms.append(self._calculate_base_reward(next_state_dict, action_a))

        #if next_state_dict['cart_position'] > 0.0:
        
        # 2. Incentivo dinámico (shaping condicional)
        reward_terms.append(self._calculate_dynamic_incentive(next_state_dict))

        # 3. Penalización dinámica (esfuerzo condicional)
        reward_terms.append(self._calculate_dynamic_penalty(next_state_dict, action_a))

        # 4. Penalización por tiempo fija
        reward_terms.append(self._calculate_fixed_penalty(current_episode_time_sec, dt_sec))

        # 5. Penalización por variación excesiva
        reward_terms.append(self._calculate_delta_var_penalty(action_a))
        
        # Suma de todos los componentes de recompensa por paso
        total_reward = float(np.nansum(reward_terms))

        # 6. Bonos de evento (aditivos sobre el total)
        total_reward += self._calculate_goal_bonus(current_episode_time_sec, goal_reached_in_step)
        total_reward += self._calculate_bandwidth_bonus(next_state_dict)
            
        return total_reward

    ### NUEVO: Métodos de cálculo desacoplados ###
    
    def _calculate_base_reward(self, next_state_dict: Dict, action_a: Any) -> float:
        """Calculates the main reward based on the 'calculation' config section."""
        if self.reward_method == 'weighted_exponential':
            reward_terms_list = []
            features_cfg = self.cfg_base.get('weighted_exponential_params', {}).get('features', {})
            for feat_name, params in features_cfg.items():
                value = next_state_dict.get(feat_name)
                if value is None or pd.isna(value) or not np.isfinite(value): continue
                
                feat_setpoint = params.get('setpoint', 0.0)
                value_error = float(value) - feat_setpoint
                exp_arg_val = (value_error / params.get('scaled', 1.0))**2
                reward_term = params.get('weight', 0.0) * math.exp(-exp_arg_val)
                reward_terms_list.append(reward_term)
                self.log_reward_params[f'reward_base_{feat_name}'] = reward_term
            return float(np.nansum(reward_terms_list))

        elif self.reward_method == 'stability_measure_based':
            reward = self.stability_calc_instance.calculate_stability_based_reward(next_state_dict)
            self.log_reward_params['reward_by_stability'] = reward
            return reward
        
        return 0.0

    def _calculate_dynamic_penalty(self, next_state_dict: Dict, action_a: Any) -> float:
        """Calculates penalty on a variable, conditional on another system state."""
        if not self.dp_enabled:
            return 0.0
        
        total_penalty = 0.0
        for penalized_var, params in self.dp_params.items():
            # Obtener valor a penalizar
            penalized_value = action_a if penalized_var == 'control_action' else next_state_dict.get(penalized_var)
            if penalized_value is None: continue

            # Calcular la condición
            cond_params = params.get('condition', {})
            cond_feature_val = next_state_dict.get(cond_params.get('feature'))
            if cond_feature_val is None: continue
            
            cond_multiplier = 0.0
            if cond_params.get('type') == 'exp':
                error = cond_feature_val - cond_params.get('setpoint', 0.0)
                cond_multiplier = math.exp(-(error / cond_params.get('scaled', 1.0))**2)
            
            # Calcular la penalización base
            base_penalty = 0.0
            weight = params.get('weight', 0.0)
            if self.dp_method == 'lineal':
                base_penalty = -weight * abs(penalized_value)
            elif self.dp_method == 'quadratic':
                base_penalty = -weight * (penalized_value**2)

            final_penalty = base_penalty * cond_multiplier
            total_penalty += final_penalty
            self.log_reward_params[f'reward_dp_{penalized_var}'] = final_penalty
            
        return total_penalty

    def _calculate_dynamic_incentive(self, next_state_dict: Dict) -> float:
        """Calculates reward for following a dynamic target."""
        if not self.di_enabled:
            return 0.0

        total_incentive = 0.0
        if self.di_method == 'adaptative': # Metodo Tanh
            for y_var, params in self.di_tanh_params.items():
                y_val = next_state_dict.get(y_var)
                x_val = next_state_dict.get(params.get('x'))
                if y_val is None or x_val is None: continue

                error = params.get('x_sp', 0.0) - x_val
                target_y_val = params.get('y_max', 0.0) * math.tanh(params.get('strength', 1.0) * error)
                
                y_error = y_val - target_y_val
                
                f_reward = params.get('f_reward', {})
                exp_arg = (y_error / f_reward.get('scaled', 1.0))**2
                incentive = f_reward.get('weight', 0.0) * math.exp(-exp_arg)
                total_incentive += incentive
                self.log_reward_params[f'reward_di_tanh_{y_var}'] = incentive
                self.log_reward_params[f'target_{y_var}'] = target_y_val

        elif self.di_method == 'lineal':
            for y_var, params in self.di_lineal_params.items():
                y_val = next_state_dict.get(y_var)
                x_val = next_state_dict.get(params.get('x'))
                if y_val is None or x_val is None: continue

                x_max = params.get('x_max', 1.0)
                y_max = params.get('y_max', 1.0)
                
                # Termino 1: Mapea x_val [0, x_max] a [-1, 1]
                term1 = (2 * x_val / x_max) - 1.0
                # Termino 2: Normaliza y_val
                term2 = y_val / y_max

                incentive = params.get('weight', 0.0) * term1 * term2
                total_incentive += incentive
                self.log_reward_params[f'reward_di_lineal_{y_var}'] = incentive

        return total_incentive

    def _calculate_fixed_penalty(self, time: float, dt: float) -> float:
        """Calculates a fixed penalty per time step."""
        if not self.penalty_enabled:
            return 0.0

        penalty = 0.0
        penalty_method = self.cfg_penalty.get('method', 'lineal')
        if penalty_method == 'lineal':
            params = self.cfg_penalty.get('penalty_lineal_params', {})
            penalty = -params.get('time', 0.0) * dt
        elif penalty_method == 'quadratic':
            params = self.cfg_penalty.get('penalty_quadratic_params', {})
            penalty = -params.get('time', 0.0) * time
            
        self.log_reward_params['fixed_penalty'] = penalty
        return penalty

    def _calculate_delta_var_penalty(self, action_a: Any) -> float:
        """Calculates penalty on the change of a variable from the previous step."""
        if not self.delta_penalty_enabled:
            return 0.0

        total_penalty = 0.0
        method = self.cfg_delta_penalty.get('method', 'quadratic')
        params = self.cfg_delta_penalty.get('penalty_delta_var_params', {})

        for var_name, var_params in params.items():
            if var_name == 'delta_control_action':
                delta_value = action_a - self.last_action_a
                
                penalty = 0.0
                weight = var_params.get('weight', 0.0)
                if method == 'quadratic':
                    penalty = -weight * (delta_value ** 2)
                elif method == 'lineal':
                    penalty = -weight * abs(delta_value)
                
                total_penalty += penalty
                self.log_reward_params[f'reward_penalty_{var_name}'] = penalty
        
        self.last_action_a = action_a

        return total_penalty

    def _calculate_goal_bonus(self, time: float, goal_reached: bool) -> float:
        """Calculates bonus upon reaching the goal state."""
        if not self.goal_bonus_enabled or not goal_reached:
            return 0.0
            
        bonus = 0.0
        bonus_type = self.cfg_goal_bonus.get('type', 'static')
        if bonus_type == "static":
            bonus = self.cfg_goal_bonus.get('static_value', 0.0)
        elif bonus_type == "decay":
            decay_type = self.cfg_goal_bonus.get('decay_type', 'exponential')
            base_val = self.cfg_goal_bonus.get('base_value', 0.0)
            tau = max(self.cfg_goal_bonus.get('decay_time_constant', 1.0), 1e-8)

            if decay_type == "exponential":
                bonus = base_val * math.exp(-time / tau)
            elif decay_type == "linear":
                bonus = base_val * max(0.0, 1.0 - time / tau)
            
            bonus = max(self.cfg_goal_bonus.get('min_value', 0.0), bonus)
        
        self.log_reward_params['goal_bonus'] = bonus
        return bonus
        
    def _calculate_bandwidth_bonus(self, next_state_dict: Dict) -> float:
        """Calculates bonus for staying within specific state ranges."""
        if not self.bw_bonus_enabled:
            return 0.0

        if self._is_in_bandwidth_ranges(next_state_dict):
            per_step_band_bonus = self.cfg_bw_bonus.get('per_step_band_bonus', 0.0)
            max_total_band_bonus = self.cfg_bw_bonus.get('max_total_band_bonus', float('inf'))
            if self.bandwidth_cumulative_bonus < max_total_band_bonus:
                self.bandwidth_cumulative_bonus += per_step_band_bonus
                self.log_reward_params['bandwidth_bonus'] = per_step_band_bonus
                return per_step_band_bonus
        return 0.0
    
    # --- Métodos de utilidad y de interfaz ---
    
    def update_calculator_stats(self, episode_metrics_data_dict: Dict, completed_episode_index: int):
        pass # Not adaptive

    def reset(self):
        self.bandwidth_cumulative_bonus = 0.0
        self.last_action_a = 0.0
    
    def _is_in_bandwidth_ranges(self, state_dict: Dict[str, Any]) -> bool:
        ranges = self.cfg_bw_bonus.get('ranges', {})
        if not ranges: 
            return False
        for feat_name, range_list in ranges.items():
            value = state_dict.get(feat_name)
            if value is not None:
                if not (float(range_list[0]) <= float(value) <= float(range_list[1])):
                    return False
        return True
    
    def get_params_log(self) -> Dict[str, float]:
        return self.log_reward_params