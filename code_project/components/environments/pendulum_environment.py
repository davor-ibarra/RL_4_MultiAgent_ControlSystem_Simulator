# components/environments/pendulum_environment.py
from interfaces.environment import Environment
from interfaces.dynamic_system import DynamicSystem
from interfaces.controller import Controller
from interfaces.rl_agent import RLAgent
from interfaces.reward_function import RewardFunction

import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__) # Logger específico del módulo

class PendulumEnvironment(Environment):
    def __init__(self,
                 system: DynamicSystem,
                 controller: Controller,
                 agent: RLAgent,
                 reward_function: RewardFunction,
                 dt: Optional[float], # Viene de environment.simulation.dt
                 reset_gains: Optional[bool], # Viene de environment.controller.pid_adaptation.reset_gains_each_episode
                 config: Dict[str, Any]
                 ):
        logger.info("[PendulumEnvironment] Initializing...")

        if not isinstance(system, DynamicSystem): raise TypeError("system must implement DynamicSystem.")
        if not isinstance(controller, Controller): raise TypeError("controller must implement Controller.")
        if not isinstance(agent, RLAgent): raise TypeError("agent must implement RLAgent.")
        if not isinstance(reward_function, RewardFunction): raise TypeError("reward_function must implement RewardFunction.")
        self.system = system
        self.controller = controller
        self.agent = agent
        self.reward_function = reward_function

        if dt is None or not isinstance(dt, (float, int)) or dt <= 0 or not np.isfinite(dt):
            raise ValueError(f"dt ({dt}) must be a positive finite number.")
        self._dt = float(dt)

        if reset_gains is None or not isinstance(reset_gains, bool):
             raise ValueError(f"reset_gains ({reset_gains}) must be a boolean.")
        self.reset_gains = reset_gains

        if not isinstance(config, dict): raise TypeError("config must be a dictionary.")
        self.config = config # Guardar config completa para check_termination, etc.

        self.state: Optional[np.ndarray] = None
        self.t: float = 0.0
        logger.info(f"[PendulumEnvironment] Initialized with dt={self._dt:.4f}, reset_gains={self.reset_gains}.")

    @property
    def dt(self) -> float:
        return self._dt

    def step(self) -> Tuple[Any, Tuple[float, float], Any]:
        if self.state is None:
            msg = "[PendulumEnvironment:step] CRITICAL: step() called before reset()."
            logger.critical(msg); raise RuntimeError(msg)

        current_state_for_reward = np.copy(self.state) # Copia para RewardFunction

        try:
            force = self.controller.compute_action(self.state) # Usa self.state
            force_f = float(force) if np.isfinite(force) else 0.0

            next_state_vector = self.system.apply_action(self.state, force_f, self.t, self._dt)
            if not isinstance(next_state_vector, np.ndarray) or not np.all(np.isfinite(next_state_vector)):
                 logger.error(f"[PendulumEnvironment:step] System returned invalid next_state: {next_state_vector}. Reverting to current state for this step.")
                 # Devolver recompensa y estabilidad basadas en el estado anterior si el nuevo es inválido
                 reward, stability_score = self.reward_function.calculate(current_state_for_reward, force_f, self.state, self.t)
                 # No avanzar tiempo ni estado
            else:
                 reward, stability_score = self.reward_function.calculate(current_state_for_reward, force_f, next_state_vector, self.t)
                 self.state = next_state_vector # Actualizar solo si es válido
                 self.t += self._dt

            reward_f = float(reward) if np.isfinite(reward) else 0.0
            stability_score_f = float(stability_score) if np.isfinite(stability_score) else 0.0 # Default 0 si NaN/inf

            return self.state, (reward_f, stability_score_f), force_f

        except Exception as e:
             logger.critical(f"[PendulumEnvironment:step] CRITICAL error during step at t={self.t:.4f}: {e}", exc_info=True)
             raise RuntimeError(f"Critical failure in PendulumEnvironment step at t={self.t:.4f}") from e

    def reset(self, initial_conditions: Any) -> Any:
        logger.debug(f"[PendulumEnvironment:reset] Resetting with initial conditions: {initial_conditions}")
        try:
            self.state = self.system.reset(initial_conditions)
            self.t = 0.0
            if self.reset_gains: self.controller.reset()
            else: self.controller.reset_internal_state()
            self.agent.reset_agent()
            logger.debug(f"[PendulumEnvironment:reset] Initial state after reset: {np.round(self.state, 4) if self.state is not None else 'None'}")
            return np.copy(self.state) if self.state is not None else np.zeros(4) # Devolver copia
        except Exception as e:
            logger.critical(f"[PendulumEnvironment:reset] CRITICAL error during reset: {e}", exc_info=True)
            raise RuntimeError(f"Critical failure during environment reset: {e}") from e

    def check_termination(self, config_param: Dict[str, Any]) -> Tuple[bool, bool, bool]:
        # Usa self.config guardado en __init__ o el config_param pasado
        # Para consistencia, usamos self.config.
        if self.state is None or len(self.state) < 4:
            logger.warning("[PendulumEnvironment:check_termination] Called with invalid internal state."); return False, False, False

        sim_limits_cfg = self.config.get('environment', {}).get('simulation', {})
        stabilization_cfg = self.config.get('environment', {}).get('simulation', {}).get('stabilization_criteria', {})
        controller_params_cfg = self.config.get('environment', {}).get('controller', {}).get('params', {})
        setpoint = controller_params_cfg.get('setpoint', 0.0)

        # Límites
        angle_limit_val = sim_limits_cfg.get('angle_limit', np.pi / 2.0)
        use_angle_limit_flag = sim_limits_cfg.get('use_angle_limit', True)
        angle_exceeded = use_angle_limit_flag and (abs(self.state[2]) > angle_limit_val)

        cart_limit_val = sim_limits_cfg.get('cart_limit', 2.4)
        use_cart_limit_flag = sim_limits_cfg.get('use_cart_limit', True)
        cart_exceeded = use_cart_limit_flag and (abs(self.state[0]) > cart_limit_val)
        limit_exceeded = angle_exceeded or cart_exceeded

        # Estabilización
        stabilized = False
        if isinstance(stabilization_cfg, dict) and stabilization_cfg:
            angle_thresh = stabilization_cfg.get('angle_threshold', 0.05)
            velocity_thresh = stabilization_cfg.get('velocity_threshold', 0.05)
            angle_is_stable = abs(self.state[2] - setpoint) < angle_thresh
            ang_vel_is_stable = abs(self.state[3]) < velocity_thresh
            stabilized = angle_is_stable and ang_vel_is_stable

        #logger.debug(f"[PendulumEnvironment:check_termination] LimitEx={limit_exceeded}, GoalReached={stabilized}")
        return limit_exceeded, stabilized, False # (limit_exceeded, goal_reached, other_condition)

    def update_reward_calculator_stats(self, episode_metrics_dict: Dict, current_episode: int):
        try:
            #logger.debug(f"[PendulumEnvironment:update_reward_stats] Delegating to {type(self.reward_function).__name__}")
            self.reward_function.update_calculator_stats(episode_metrics_dict, current_episode)
        except Exception as e:
            logger.error(f"[PendulumEnvironment:update_reward_stats] Error calling update_calculator_stats on RewardFunction: {e}", exc_info=True)