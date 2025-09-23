# components/environments/water_tank_environment.py
from interfaces.environment import Environment
from interfaces.dynamic_system import DynamicSystem
from interfaces.controller import Controller
from interfaces.rl_agent import RLAgent
from interfaces.reward_function import RewardFunction
from interfaces.stability_calculator import BaseStabilityCalculator

import numpy as np
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

class WaterTankEnvironment(Environment):
    """
    Orchestrates the simulation for the Water Tank system, handling state
    transitions, termination, and interaction with all system components.
    """
    def __init__(self,
                 system: DynamicSystem,
                 controller: Controller,
                 agent: RLAgent,
                 reward_function: RewardFunction,
                 stability_calculator: BaseStabilityCalculator,
                 config: Dict[str, Any]
                 ):
        logger.info("[WaterTankEnvironment] Initializing...")
        self.system = system
        self.controller = controller
        self.agent = agent
        self.reward_function = reward_function
        self.stability_calculator = stability_calculator
        self.config = config

        env_cfg = self.config.get('environment', {})
        sim_params_cfg = env_cfg.get('simulation', {})
        stabilization_cfg = sim_params_cfg.get('stabilization_criteria', {})

        dt_val = sim_params_cfg.get('dt_sec')
        if not isinstance(dt_val, (float, int)) or dt_val <= 0 or not np.isfinite(dt_val):
            raise ValueError(f"WaterTankEnvironment: dt_sec ({dt_val}) must be a positive finite number.")
        self._dt_val = float(dt_val)
        
        self.current_sim_time_sec: float = 0.0
        self.current_episode_state: np.ndarray
        self.error: float = 0.0
        self.last_known_level_rate: float = 0.0
        self.control_action_u: float = 0.0

        # --- System ---
        self.level_limit_m = sim_params_cfg.get('level_limit', 2.0)
        self.use_level_limit = sim_params_cfg.get('enable_level_limit', True)
        _lvl_rng = stabilization_cfg.get('level_threshold', [self.target_level - 0.01, self.target_level + 0.01])
        self.level_lo, self.level_hi = float(_lvl_rng[0]), float(_lvl_rng[1])
        _lvldot_rng = stabilization_cfg.get('level_rate_threshold', [-0.005, 0.005])
        self.level_rate_lo, self.level_rate_hi = float(_lvldot_rng[0]), float(_lvldot_rng[1])
        # --- Controlador ---
        # target
        self.target_level = self.controller.get_target()
        # reset
        self.controller_reset_level = env_cfg.get('controller', {}).get('pid_adaptation', {}).get('reset_policy_on_episode_end', 'full_params_and_state')

        logger.info(f"[WaterTankEnvironment] Initialized with dt={self._dt_val:.4f}, TargetLevel={self.target_level}m.")

    @property
    def dt(self) -> float:
        return self._dt_val

    def _create_state_dict(self, state_vector: np.ndarray) -> Dict[str, float]:
        """Creates a named dictionary from the state vector for this environment."""
        return {
            'level': state_vector[0],
            'level_rate': self.last_known_level_rate,
            'time': self.current_sim_time_sec,
            'error': self.error
        }

    def step(self) -> Tuple[np.ndarray, float, float, float]:
        state_s_vec = self.current_episode_state
        state_s_dict = self._create_state_dict(state_s_vec)

        # El controlador calcula su acción de salida ideal.
        self.control_action_u = self.controller.compute_action(state_s_dict)

        # El sistema dinámico recibe la acción traducida por el actuador
        next_state_s_prime_vec = self.system.apply_action(state_s_vec, self.control_action_u, self.current_sim_time_sec, self._dt_val)
        self.error = next_state_s_prime_vec[0] - state_s_vec[0]
        self.last_known_level_rate = (self.error) / self._dt_val
        next_state_s_prime_dict = self._create_state_dict(next_state_s_prime_vec)
        
        goal_condition_met = self._evaluate_if_state_is_goal(next_state_s_prime_dict)

        reward_value = self.reward_function.calculate(
            state_dict=state_s_dict,
            action_a=self.control_action_u,                  # Pasar la acción original del PID a la función de recompensa?? probar***
            next_state_dict=next_state_s_prime_dict,
            current_episode_time_sec=self.current_sim_time_sec,
            dt_sec=self._dt_val,
            goal_reached_in_step=goal_condition_met
        )
        stability_score = self.stability_calculator.calculate_instantaneous_stability(next_state_s_prime_dict)

        self.current_episode_state = next_state_s_prime_vec
        self.current_sim_time_sec += self._dt_val

        return self.current_episode_state, reward_value, stability_score, {}    # In case others params are required

    def reset(self, initial_conditions: Dict[str, float]) -> np.ndarray:
        logger.debug(f"[WaterTankEnvironment:reset] Resetting with initial conditions: {initial_conditions}")
        self.current_episode_state = self.system.reset(initial_conditions)
        self.current_sim_time_sec = 0.0
        self.last_known_level_rate = 0.0
        self.control_action_u = 0.0

        self.controller.reset_policy(self.controller_reset_level)
        self.agent.reset_agent()
        self.reward_function.reset()
        logger.debug(f"[WaterTankEnvironment:reset] Initial state after reset: {self.current_episode_state}")
        return np.copy(self.current_episode_state)

    def check_termination(self) -> Tuple[bool, bool, bool]:
        state_vec = self.current_episode_state
        state_dict = self._create_state_dict(state_vec)

        level_exceeded = self.use_level_limit and (state_vec[0] >= self.level_limit_m or state_vec[0] <= 0.001)
        goal_reached = self._evaluate_if_state_is_goal(state_dict)
        agent_requested_termination = self.agent.should_episode_terminate_early()
        
        return level_exceeded, goal_reached, agent_requested_termination

    def _evaluate_if_state_is_goal(self, state_to_evaluate: Dict[str, float]) -> bool:
        lvl = state_to_evaluate['level']
        lvldot = state_to_evaluate['level_rate']
        is_level_stable = self.level_lo <= lvl <= self.level_hi
        is_rate_stable  = self.level_rate_lo <= lvldot <= self.level_rate_hi
        return is_level_stable and is_rate_stable

    def update_reward_and_stability_calculator_stats(self, episode_metrics_dict: Dict, current_episode: int):
        self.reward_function.update_calculator_stats(episode_metrics_dict, current_episode)     # Implementar si se requiere adaptative params
        if hasattr(self.stability_calculator, 'update_reference_stats'):
            self.stability_calculator.update_reference_stats(episode_metrics_dict, current_episode)
    
    def get_params_log(self) -> Dict[str, Any]:
        """Expone los parámetros internos del entorno para el logging."""
        log_system_params = self.system.get_log_system_params()
        return log_system_params