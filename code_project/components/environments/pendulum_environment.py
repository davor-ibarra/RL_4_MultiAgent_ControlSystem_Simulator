# components/environments/pendulum_environment.py
from interfaces.environment import Environment
from interfaces.dynamic_system import DynamicSystem
from interfaces.controller import Controller
from interfaces.rl_agent import RLAgent
from interfaces.reward_function import RewardFunction
from interfaces.stability_calculator import BaseStabilityCalculator

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Any, Optional # Optional no es necesario para current_episode_state

logger = logging.getLogger(__name__)

class PendulumEnvironment(Environment):
    def __init__(self,
                 system: DynamicSystem,
                 controller: Controller,
                 agent: RLAgent,
                 reward_function: RewardFunction,
                 stability_calculator: BaseStabilityCalculator,
                 config: Dict[str, Any]
                 ):
        logger.info("[PendulumEnvironment] Initializing...")
        self.system = system
        self.controller = controller
        self.agent = agent
        self.reward_function = reward_function
        self.stability_calculator = stability_calculator
        self.config = config

        self.env_cfg = self.config.get('environment', {})
        self.sim_params_cfg = self.env_cfg.get('simulation', {})
        self.stabilization_criteria_cfg = self.sim_params_cfg.get('stabilization_criteria', {})
        self.ctrl_adapt_cfg = self.env_cfg.get('controller', {})

        dt_val = self.sim_params_cfg.get('dt_sec')
        if not isinstance(dt_val, (float, int)) or dt_val <= 0 or not np.isfinite(dt_val):
            raise ValueError(f"PendulumEnvironment: dt_sec ({dt_val}) must be a positive finite number.")
        self._dt_val = float(dt_val)

        self.controller_reset_level = self.ctrl_adapt_cfg.get('pid_adaptation', {}).get('reset_policy_on_episode_end', 'full_params_and_state')
        
        self.current_episode_state: np.ndarray # Se inicializa en reset()
        self.current_sim_time_sec: float = 0.0

        # Get parameters from config
        # Condiciones de error
        self.target_angle = float(self.ctrl_adapt_cfg.get('setpoint', 0.0))
        # Condiciones de estabilización
        self.pendulum_and_cart_objetive = self.stabilization_criteria_cfg.get('pendulum_and_cart',False)
        self.angle_thresh = float(self.stabilization_criteria_cfg.get('angle_threshold', 0.001))
        self.vel_thresh = float(self.stabilization_criteria_cfg.get('velocity_threshold', 0.005))
        self.cart_pos_thresh = float(self.stabilization_criteria_cfg.get('cart_position_threshold', 0.05))
        self.cart_vel_thresh = float(self.stabilization_criteria_cfg.get('cart_velocity_threshold', 0.05))
        # Condiciones de borde
        self.angle_lim_rad = self.sim_params_cfg.get('pendulum_angle_limit_rad', np.pi/3.0)
        self.use_angle_lim = self.sim_params_cfg.get('enable_angle_limit', True)
        self.cart_pos_lim_m = self.sim_params_cfg.get('cart_pos_limit_m', 5.0)
        self.use_cart_pos_lim = self.sim_params_cfg.get('enable_cart_pos_limit', True)

        logger.info(f"[PendulumEnvironment] Initialized with dt_sec={self._dt_val:.4f}, controller_reset_level='{self.controller_reset_level}'.")

    @property
    def dt(self) -> float:
        return self._dt_val

    def step(self) -> Tuple[np.ndarray, float, float, float]:
        """
        Avanza la simulación un paso dt.
        Retorna: (next_state_vector, reward, stability_score, control_force_applied)
        """
        state_s = self.current_episode_state # S
        # 1. Obtener acción del controlador
        control_force_a = self.controller.compute_action(state_s)
        # 2. Aplicar acción al sistema y obtener estado siguiente S'
        next_state_s_prime = self.system.apply_action(
            state_s, control_force_a, self.current_sim_time_sec, self._dt_val
        )
        # 3. Calcular recompensa y estabilidad para la transición (S, A, S')
        # Primero se debe determinar si S' (el estado resultante de este paso) es un estado meta.
        goal_condition_met_by_s_prime = self._evaluate_if_state_is_goal(next_state_s_prime)
        
        reward_value_for_this_step = self.reward_function.calculate(
            state_s=state_s, action_a=control_force_a,
            next_state_s_prime=next_state_s_prime, 
            current_episode_time_sec=self.current_sim_time_sec,
            dt_sec=self._dt_val,
            goal_reached_in_step=goal_condition_met_by_s_prime # << Flag for Bonus
        )
        stability_w = self.stability_calculator.calculate_instantaneous_stability(next_state_s_prime)

        # 4. Actualizar estado interno y tiempo del entorno
        self.current_episode_state = next_state_s_prime
        self.current_sim_time_sec += self._dt_val

        return self.current_episode_state, reward_value_for_this_step, stability_w, control_force_a

    def reset(self, initial_conditions: Any) -> np.ndarray:
        logger.debug(f"[PendulumEnvironment:reset] Resetting with initial conditions: {initial_conditions}")
        
        self.current_episode_state = self.system.reset(initial_conditions)
        self.current_sim_time_sec = 0.0
        self.controller.reset_policy(self.controller_reset_level)
        self.agent.reset_agent()
        self.reward_function.reset()

        # El constructor de System o su reset debería asegurar que current_episode_state es ndarray.
        # Si no, fallará aquí, lo cual es un error de contrato del DynamicSystem.
        logger.debug(f"[PendulumEnvironment:reset] Initial state after reset: {np.round(self.current_episode_state[:4], 4)}")
        return np.copy(self.current_episode_state)

    def check_termination(self) -> Tuple[bool, bool, bool]: # Eliminado config_param_check_term
        """
        Verifica condiciones de terminación.
        Retorna: (limit_exceeded, goal_reached, agent_requested_early_termination)
        """
        # Determinar si no se han corrompido las condiciones de borde
        angle_exceeded = self.use_angle_lim and (abs(self.current_episode_state[2]) > self.angle_lim_rad)
        cart_exceeded = self.use_cart_pos_lim and (abs(self.current_episode_state[0]) > self.cart_pos_lim_m)
        limit_exceeded_flag = angle_exceeded or cart_exceeded

        # Determinar si el estado actual del entorno (S') es un estado meta para la terminación.
        goal_reached_flag = self._evaluate_if_state_is_goal(self.current_episode_state)
        # Determinar si al agente se le acaba la paciencia para early termination.
        agent_requested_termination = self.agent.should_episode_terminate_early()
        
        # logger.debug(f"[PendulumEnvironment:check_termination] LimitExceeded={limit_exceeded_flag}, GoalReached={goal_reached_flag}, AgentReqET={agent_requested_termination}")
        return limit_exceeded_flag, goal_reached_flag, agent_requested_termination
    
    def _evaluate_if_state_is_goal(self, state_to_evaluate: np.ndarray) -> bool:
        """
        Helper para evaluar si un estado dado cumple los criterios de estabilización.
        """        
        angle_stable = abs(state_to_evaluate[2] - self.target_angle) < self.angle_thresh
        ang_vel_stable = abs(state_to_evaluate[3]) < self.vel_thresh
        cart_pos = abs(state_to_evaluate[0]) < 0.05
        cart_vel = abs(state_to_evaluate[1]) < 0.05
        return angle_stable and ang_vel_stable and cart_pos and cart_vel if self.pendulum_and_cart_objetive else angle_stable and ang_vel_stable

    def update_reward_and_stability_calculator_stats(self, episode_metrics_log_dict: Dict, episode_idx_completed: int):
        # logger.debug(f"[PendulumEnvironment:update_reward_stats] Delegating stats update to {type(self.reward_function).__name__}")
        self.reward_function.update_calculator_stats(episode_metrics_log_dict, episode_idx_completed)
        # Delegar también a StabilityCalculator si es adaptativo
        if hasattr(self.stability_calculator, 'update_reference_stats'):
            try:
                # logger.debug(f"[PendulumEnvironment:update_reward_stats] Delegating stats update to {type(self.stability_calculator).__name__}")
                self.stability_calculator.update_reference_stats(episode_metrics_log_dict, episode_idx_completed) # type: ignore
            except Exception as e_update_stab_stats:
                logger.error(f"[PendulumEnvironment:update_reward_stats] Error calling update_reference_stats on StabilityCalculator: {e_update_stab_stats}", exc_info=True)