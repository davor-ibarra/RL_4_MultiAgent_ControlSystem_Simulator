# components/environments/pendulum_environment.py

import logging
import numpy as np
from typing import Tuple, Dict, Any, Optional

from interfaces.environment import Environment
from interfaces.dynamic_system import DynamicSystem
from interfaces.controller import Controller
from interfaces.rl_agent import RLAgent
from interfaces.reward_function import RewardFunction
from interfaces.stability_calculator import BaseStabilityCalculator
from interfaces.metrics_collector import MetricsCollector

from components.metrics_processing.metric_processing import MetricProcessing

logger = logging.getLogger(__name__)


class PendulumEnvironment(Environment):
    def __init__(
        self,
        system: DynamicSystem,
        controllers: Dict[str, Controller],
        agent: RLAgent,
        reward_function: RewardFunction,
        stability_calculator: BaseStabilityCalculator,
        config: Dict[str, Any],
    ):
        logger.info("[PendulumEnvironment] Initializing...")
        self.system = system
        self.controllers = controllers
        self.agent = agent
        self.reward_function = reward_function
        self.stability_calculator = stability_calculator
        self.config = config

        self.env_cfg = self.config.get('environment', {})
        self.env_runtime_cfg = self._select_environment_section(self.env_cfg)
        self.sim_params_cfg = self.env_runtime_cfg.get('simulation', {})
        self.stabilization_criteria_cfg = self.sim_params_cfg.get('stabilization_criteria', {})

        dt_val = self.sim_params_cfg.get('dt_sec')
        if not isinstance(dt_val, (float, int)) or dt_val <= 0 or not np.isfinite(dt_val):
            raise ValueError(f"PendulumEnvironment: dt_sec ({dt_val}) must be a positive finite number.")
        self._dt_val = float(dt_val)

        controller_cfg = self.env_cfg.get('controller', {})
        self.multi_controller = controller_cfg.get('global_actuator', False)
        self.mixing_policy = controller_cfg.get('mixing_policy', 'sum')
        self.select_method = controller_cfg.get('select_method', 'abs_max')
        self.cascade_outer_key = controller_cfg.get('cascade_outer')
        self.cascade_inner_key = controller_cfg.get('cascade_inner')
        self.use_global_actuator = controller_cfg.get('global_actuator', False)
        if self.use_global_actuator:
            global_limits = controller_cfg.get('global_actuator_limits', [-1.0, 1.0])
            self.global_actuator_min = float(global_limits[0])
            self.global_actuator_max = float(global_limits[1])
        self.controller_reset_policies: Dict[str, str] = {}
        for name, ctrl_instance in self.controllers.items():
            obj_var = ctrl_instance.name_objective_var
            ctrl_config_section = next(
                (cfg for cfg in controller_cfg.values() if isinstance(cfg, dict) and cfg.get('params', {}).get('name_objective_var') == obj_var),
                None,
            )
            if ctrl_config_section:
                reset_policy = ctrl_config_section.get('pid_adaptation', {}).get('reset_policy_on_episode_end', 'full_params_and_state')
                self.controller_reset_policies[name] = reset_policy
            else:
                self.controller_reset_policies[name] = 'internal_state_only'

        self.current_episode_state: np.ndarray
        self.current_sim_time_sec: float = 0.0

        self.pendulum_and_cart_objetive = self.stabilization_criteria_cfg.get('pendulum_and_cart', False)
        _angle_rng = self.stabilization_criteria_cfg.get('angle_threshold', [-0.001, 0.001])
        self.angle_lo, self.angle_hi = float(_angle_rng[0]), float(_angle_rng[1])
        _angvel_rng = self.stabilization_criteria_cfg.get('velocity_threshold', [-0.005, 0.005])
        self.ang_vel_lo, self.ang_vel_hi = float(_angvel_rng[0]), float(_angvel_rng[1])
        _cartpos_rng = self.stabilization_criteria_cfg.get('cart_position_threshold', [-0.05, 0.05])
        self.cart_pos_lo, self.cart_pos_hi = float(_cartpos_rng[0]), float(_cartpos_rng[1])
        _cartvel_rng = self.stabilization_criteria_cfg.get('cart_velocity_threshold', [-0.05, 0.05])
        self.cart_vel_lo, self.cart_vel_hi = float(_cartvel_rng[0]), float(_cartvel_rng[1])

        self.angle_lim_rad = self.sim_params_cfg.get('pendulum_angle_limit_rad', np.pi / 3.0)
        self.use_angle_lim = self.sim_params_cfg.get('enable_angle_limit', True)
        self.cart_pos_lim_m = self.sim_params_cfg.get('cart_pos_limit_m', 5.0)
        self.use_cart_pos_lim = self.sim_params_cfg.get('enable_cart_pos_limit', True)

        # Metric processing placeholder
        self.metric_processing = MetricProcessing(config)

        # Lagrangian metric tracking
        self.prev_error: float = 0.0
        self.integral_error: float = 0.0
        self.prev_action: float = 0.0
        self.saturation_counter: int = 0
        self.saturation_time: float = 0.0

        logger.info(
            f"[PendulumEnvironment] Initialized with dt_sec={self._dt_val:.4f}, controller_reset_level='{self.controller_reset_policies}'."
        )

    def _select_environment_section(self, env_root_cfg: Dict[str, Any]) -> Dict[str, Any]:
        creator_mode = env_root_cfg.get('environment_creator', 'single')
        if creator_mode == 'single':
            return env_root_cfg.get('environment_single', env_root_cfg)
        if creator_mode == 'multi_single':
            multi_cfg = env_root_cfg.get('environment_multi_single', {})
            for cfg in multi_cfg.values():
                if isinstance(cfg, dict) and cfg.get('module_name'):
                    return cfg
        if creator_mode == 'multi_equal':
            base_cfg = env_root_cfg.get('environment_multi_equal', {})
            if isinstance(base_cfg, dict):
                selected = base_cfg.get('environment_base')
                if isinstance(selected, dict):
                    return selected
        return env_root_cfg

    @property
    def dt(self) -> float:
        return self._dt_val

    def _create_state_dict(self, state_vector: np.ndarray) -> Dict[str, float]:
        """Creates a named dictionary from the state vector for this environment."""
        return {
            'cart_position': state_vector[0],
            'cart_velocity': state_vector[1],
            'pendulum_angle': state_vector[2],
            'pendulum_velocity': state_vector[3],
        }

    def _compute_lagrangian_metrics(
        self,
        state_dict: Dict[str, float],
        next_state_dict: Dict[str, float],
        action: float,
    ) -> Dict[str, float]:
        """Compute Lagrangian terms used by the reward calculator.

        Returns a dictionary with keys matching the expected metric names, e.g.
        ``L_e``, ``L_edot``, ``L_x``, ``L_xdot``, ``L_u``, ``L_udot``.
        """
        # Error (e) based on pendulum angle deviation from zero
        error = next_state_dict.get('pendulum_angle', 0.0)
        error_dot = (error - self.prev_error) if self.prev_error is not None else 0.0
        self.integral_error += error
        # Control effort and its derivative
        effort = action
        effort_dot = (effort - self.prev_action) if self.prev_action is not None else 0.0
        # Gain inertia – detect sign change of action
        gain_inertia = 1.0 if np.sign(effort) != np.sign(self.prev_action) else 0.0
        # Saturation fraction – simple count of steps where action hits limits
        saturated = 0.0
        if self.use_global_actuator:
            if effort <= self.global_actuator_min or effort >= self.global_actuator_max:
                saturated = 1.0
                self.saturation_counter += 1
                self.saturation_time += self._dt_val
        # Update previous values for next step
        self.prev_error = error
        self.prev_action = effort

        return {
            'L_e': error,
            'L_edot': error_dot,
            'L_x': next_state_dict.get('cart_position', 0.0),
            'L_xdot': next_state_dict.get('cart_velocity', 0.0),
            'L_u': effort,
            'L_udot': effort_dot,
            'gain_inertia': gain_inertia,
            'saturation': saturated,
        }

    def step(self) -> Tuple[np.ndarray, float, float, float]:
        """Execute one simulation step.

        Returns the next state, total reward, stability score, and applied action.
        """
        s = self.current_episode_state
        s_dict = self._create_state_dict(s)

        # 1) Control action
        if self.multi_controller:
            u = self._compute_control_action_by_policy(s_dict)
        else:
            u = list(self.controllers.items())[0][1].compute_action(s_dict)

        # 2) System dynamics
        s_next = self.system.apply_action(s, u, self.current_sim_time_sec, self._dt_val)

        # 3) Compute metrics and reward
        s_next_dict = self._create_state_dict(s_next)
        goal = self._evaluate_if_state_is_goal(s_next_dict)
        metrics = self._compute_lagrangian_metrics(s_dict, s_next_dict, float(u))
        r = self.reward_function.calculate(
            state_dict=s_dict,
            action_a=u,
            next_state_dict=s_next_dict,
            current_episode_time_sec=self.current_sim_time_sec,
            dt_sec=self._dt_val,
            goal_reached_in_step=goal,
            reward_components=metrics,
+        )
        w_stab = self.stability_calculator.calculate_instantaneous_stability(s_next_dict)

        # 4) Advance state and time
        self.current_episode_state = s_next
        self.current_sim_time_sec += self._dt_val

        return self.current_episode_state, r, w_stab, u

    def _compute_control_action_by_policy(self, state_s_dict: Dict[str, float]) -> float:
        """Encapsulates all coordination of control according to the active policy."""
        if self.mixing_policy == 'cascade-setpoint':
            outer = self.controllers.get(self.cascade_outer_key)
            inner = self.controllers.get(self.cascade_inner_key)
            if not (outer and inner and hasattr(inner, 'set_target')):
                logger.error("Cascade misconfigured or inner controller missing 'set_target'.")
                return 0.0
            sp_inner = outer.compute_action(state_s_dict)
            inner.set_target(sp_inner)
            u = inner.compute_action(state_s_dict)
            if hasattr(inner, 'track_actuator_output'):
                inner.track_actuator_output(u)
            if hasattr(outer, 'track_actuator_output'):
                outer.track_actuator_output(sp_inner)
            return float(u)
        # Parallel policies
        u_individual_dict = {name: ctrl.compute_action(state_s_dict) for name, ctrl in self.controllers.items()}
        u, u_effective = self._mix_actions(u_individual_dict)
        for name, ctrl in self.controllers.items():
            if hasattr(ctrl, 'track_actuator_output'):
                ctrl.track_actuator_output(u_effective.get(name, 0.0))
        return float(u)

    def _mix_actions(self, u_individual_dict: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Apply mixing policy and global saturation.

        Returns the final actuator action and the effective contributions for anti‑windup.
        """
        if self.mixing_policy == 'select':
            if not u_individual_dict:
                return 0.0, {}
            if self.select_method == 'high':
                winner_name, winner_action = max(u_individual_dict.items(), key=lambda kv: kv[1])
            elif self.select_method == 'low':
                winner_name, winner_action = min(u_individual_dict.items(), key=lambda kv: kv[1])
            else:
                winner_name, winner_action = max(u_individual_dict.items(), key=lambda kv: abs(kv[1]))
            final_actuator_action = (
                np.clip(winner_action, self.global_actuator_min, self.global_actuator_max)
                if self.use_global_actuator
                else winner_action
            )
            effective_actions = {name: 0.0 for name in u_individual_dict}
            effective_actions[winner_name] = final_actuator_action
            return final_actuator_action, effective_actions
        elif self.mixing_policy == 'sum':
            total_action = sum(u_individual_dict.values())
            final_actuator_action = (
                np.clip(total_action, self.global_actuator_min, self.global_actuator_max)
                if self.use_global_actuator
                else total_action
            )
            overage = total_action - final_actuator_action
            effective_actions = {}
            if not np.isclose(overage, 0.0):
                total_abs_actions = sum(abs(v) for v in u_individual_dict.values())
                if np.isclose(total_abs_actions, 0.0):
                    return final_actuator_action, {name: 0.0 for name in u_individual_dict}
                for name, action in u_individual_dict.items():
                    proportion = abs(action) / total_abs_actions
                    reduction = overage * proportion
                    effective_actions[name] = action - reduction
            else:
                effective_actions = u_individual_dict.copy()
            return final_actuator_action, effective_actions
        return 0.0, {name: 0.0 for name in u_individual_dict}

    def reset(self, initial_conditions: Dict[str, float]) -> np.ndarray:
        logger.debug(f"[PendulumEnvironment:reset] Resetting with initial conditions: {initial_conditions}")
        self.current_episode_state = self.system.reset(initial_conditions)
        self.current_sim_time_sec = 0.0
        for name, ctrl in self.controllers.items():
            reset_level = self.controller_reset_policies.get(name, 'internal_state_only')
            ctrl.reset_policy(reset_level)
        self.agent.reset_agent()
        self.reward_function.reset()
        # Reset metric tracking
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.prev_action = 0.0
        self.saturation_counter = 0
        self.saturation_time = 0.0
        logger.debug(f"[PendulumEnvironment:reset] Initial state after reset: {np.round(self.current_episode_state, 4)}")
        return np.copy(self.current_episode_state)

    def check_termination(self) -> Tuple[bool, bool, bool]:
        state_vec = self.current_episode_state
        state_dict = self._create_state_dict(state_vec)
        angle_exceeded = self.use_angle_lim and (abs(state_dict['pendulum_angle']) > self.angle_lim_rad)
        cart_exceeded = self.use_cart_pos_lim and (abs(state_dict['cart_position']) > self.cart_pos_lim_m)
        limit_exceeded_flag = angle_exceeded or cart_exceeded
        goal_reached_flag = self._evaluate_if_state_is_goal(state_dict)
        agent_requested_termination = self.agent.should_episode_terminate_early()
        return limit_exceeded_flag, goal_reached_flag, agent_requested_termination

    def _evaluate_if_state_is_goal(self, state_to_evaluate: Dict[str, float]) -> bool:
        angle_val = state_to_evaluate['pendulum_angle']
        ang_vel_val = state_to_evaluate['pendulum_velocity']
        angle_stable = self.angle_lo <= angle_val <= self.angle_hi
        ang_vel_stable = self.ang_vel_lo <= ang_vel_val <= self.ang_vel_hi
        if self.pendulum_and_cart_objetive:
            cart_pos_val = state_to_evaluate['cart_position']
            cart_vel_val = state_to_evaluate['cart_velocity']
            cart_pos_stable = self.cart_pos_lo <= cart_pos_val <= self.cart_pos_hi
            cart_vel_stable = self.cart_vel_lo <= cart_vel_val <= self.cart_vel_hi
            return angle_stable and ang_vel_stable and cart_pos_stable and cart_vel_stable
        return angle_stable and ang_vel_stable

    def update_reward_and_stability_calculator_stats(self, episode_metrics_log_dict: Dict, episode_idx_completed: int):
        self.reward_function.update_calculator_stats(episode_metrics_log_dict, episode_idx_completed)
        if hasattr(self.stability_calculator, 'update_reference_stats'):
            self.stability_calculator.update_reference_stats(episode_metrics_log_dict, episode_idx_completed)

    def get_params_log(self) -> Dict[str, Any]:
        """Expose internal environment parameters for logging."""
        log_system_params = self.system.get_log_system_params()
        return log_system_params

    def get_controllers(self) -> Dict[str, Controller]:
        """Return the dictionary of controllers managed by the environment."""
        return self.controllers