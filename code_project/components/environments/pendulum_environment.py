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
from typing import Tuple, Dict, Any, List

logger = logging.getLogger(__name__)

class PendulumEnvironment(Environment):
    def __init__(self,
                 system: DynamicSystem,
                 controllers: Dict[str, Controller],
                 agent: RLAgent,
                 reward_function: RewardFunction,
                 stability_calculator: BaseStabilityCalculator,
                 config: Dict[str, Any]
                 ):
        logger.info("[PendulumEnvironment] Initializing...")
        self.system = system
        self.controllers = controllers
        self.agent = agent
        self.reward_function = reward_function
        self.stability_calculator = stability_calculator
        self.config = config

        self.env_cfg = self.config.get('environment', {})
        self.sim_params_cfg = self.env_cfg.get('simulation', {})
        self.stabilization_criteria_cfg = self.sim_params_cfg.get('stabilization_criteria', {})
        
        dt_val = self.sim_params_cfg.get('dt_sec')
        if not isinstance(dt_val, (float, int)) or dt_val <= 0 or not np.isfinite(dt_val):
            raise ValueError(f"PendulumEnvironment: dt_sec ({dt_val}) must be a positive finite number.")
        self._dt_val = float(dt_val)

        controller_cfg = self.env_cfg.get('controller', {})
        self.multi_controller = controller_cfg.get('global_actuator', False)
        self.mixing_policy = controller_cfg.get('mixing_policy', 'sum')
        # --- Configuración para la política 'select' ---
        self.select_method = controller_cfg.get('select_method', 'abs_max') # Opciones: 'high', 'low', 'abs_max'
        # --- Configuración para la política 'cascade' ---
        self.cascade_outer_key = controller_cfg.get('cascade_outer')
        self.cascade_inner_key = controller_cfg.get('cascade_inner')
        # Lectura de la configuración del actuador global
        self.use_global_actuator = controller_cfg.get('global_actuator', False)
        if self.use_global_actuator:
            global_limits = controller_cfg.get('global_actuator_limits', [-1.0, 1.0])
            self.global_actuator_min = float(global_limits[0])
            self.global_actuator_max = float(global_limits[1])
        # Almacenamiento de políticas de reseteo individuales
        self.controller_reset_policies: Dict[str, str] = {}
        for name, ctrl_instance in self.controllers.items():
            obj_var = ctrl_instance.name_objective_var
            ctrl_config_section = next(
                (cfg for cfg in controller_cfg.values() if isinstance(cfg, dict) and cfg.get('params', {}).get('name_objective_var') == obj_var),
                None
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
        
        self.angle_lim_rad = self.sim_params_cfg.get('pendulum_angle_limit_rad', np.pi/3.0)
        self.use_angle_lim = self.sim_params_cfg.get('enable_angle_limit', True)
        self.cart_pos_lim_m = self.sim_params_cfg.get('cart_pos_limit_m', 5.0)
        self.use_cart_pos_lim = self.sim_params_cfg.get('enable_cart_pos_limit', True)

        logger.info(f"[PendulumEnvironment] Initialized with dt_sec={self._dt_val:.4f}, controller_reset_level='{self.controller_reset_policies}'.")

    @property
    def dt(self) -> float:
        return self._dt_val

    def _create_state_dict(self, state_vector: np.ndarray) -> Dict[str, float]:
        """Creates a named dictionary from the state vector for this environment."""
        return {
            'cart_position': state_vector[0],
            'cart_velocity': state_vector[1],
            'pendulum_angle': state_vector[2],
            'pendulum_velocity': state_vector[3]
        }

    def step(self) -> Tuple[np.ndarray, float, float, float]:
        """
        Un paso de simulación:
        - Orquesta el cálculo de la acción vía política de mezcla/cascada
        - Aplica acción al sistema
        - Calcula recompensa y estabilidad
        """
        s = self.current_episode_state
        s_dict = self._create_state_dict(s)  # cart_position, cart_velocity, pendulum_angle, pendulum_velocity

        # 1) Acción de control (toda la coordinación queda encapsulada aquí)
        if self.multi_controller is True:
            u = self._compute_control_action_by_policy(s_dict)
        else:
            u = list(self.controllers.items())[0][1].compute_action(s_dict)

        # 2) Dinámica del sistema
        s_next = self.system.apply_action(s, u, self.current_sim_time_sec, self._dt_val)

        # 3) Recompensa y estabilidad
        s_next_dict = self._create_state_dict(s_next)
        goal = self._evaluate_if_state_is_goal(s_next_dict)
        r = self.reward_function.calculate(
            state_dict=s_dict,
            action_a=u,
            next_state_dict=s_next_dict,
            current_episode_time_sec=self.current_sim_time_sec,
            dt_sec=self._dt_val,
            goal_reached_in_step=goal
        )
        w_stab = self.stability_calculator.calculate_instantaneous_stability(s_next_dict)

        # 4) Avance de estado y tiempo
        self.current_episode_state = s_next
        self.current_sim_time_sec += self._dt_val

        return self.current_episode_state, r, w_stab, u

    def _compute_control_action_by_policy(self, state_s_dict: Dict[str, float]) -> float:
        """
        Encapsula TODA la coordinación de control según la política activa.
        - 'cascade-setpoint': outer produce setpoint del inner (sin _mix_actions)
        - 'sum'/'select': cada controlador produce acción y se mezclan vía _mix_actions
        (con tracking por anti-windup a cada PID).
        """
        # --- CASCADE: outer -> setpoint(inner) -> acción final ---
        if self.mixing_policy == 'cascade-setpoint':
            outer = self.controllers.get(self.cascade_outer_key)
            inner = self.controllers.get(self.cascade_inner_key)

            if not (outer and inner and hasattr(inner, 'set_target')):
                logger.error("Cascade misconfigured or inner controller missing 'set_target'.")
                return 0.0

            sp_inner = outer.compute_action(state_s_dict)     # nuevo setpoint para el lazo interno
            inner.set_target(sp_inner)
            u = inner.compute_action(state_s_dict)

            # Tracking: solo el lazo interno acciona el actuador; al outer se le informa su salida
            if hasattr(inner, 'track_actuator_output'):
                inner.track_actuator_output(u)
            if hasattr(outer, 'track_actuator_output'):
                outer.track_actuator_output(sp_inner)

            return float(u)

        # --- PARALELO: sum/select con mezcla global ---
        # 1) Acciones locales (cada PID ya aplica su propio clipping/normalización)
        u_individual_dict = {name: ctrl.compute_action(state_s_dict) for name, ctrl in self.controllers.items()}

        # 2) Mezcla + saturación global
        u, u_effective = self._mix_actions(u_individual_dict)

        # 3) Tracking efectivo por PID (clave para anti-windup correcto)
        for name, ctrl in self.controllers.items():
            if hasattr(ctrl, 'track_actuator_output'):
                ctrl.track_actuator_output(u_effective.get(name, 0.0))

        return float(u)

    def _mix_actions(self, u_individual_dict: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        Aplica la política de mezcla y saturación global.
        Devuelve la acción final para el sistema y las contribuciones efectivas para el anti-windup.
        """
        # --- POLÍTICA: SELECT (High/Low/Abs_Max Selector) ---
        if self.mixing_policy == 'select':
            if not u_individual_dict: 
                return 0.0, {}
            # 1) elegir ganador y acción
            if self.select_method == 'high':
                winner_name, winner_action = max(u_individual_dict.items(), key=lambda kv: kv[1])
            elif self.select_method == 'low':
                winner_name, winner_action = min(u_individual_dict.items(), key=lambda kv: kv[1])
            else:  # abs_max
                winner_name, winner_action = max(u_individual_dict.items(), key=lambda kv: abs(kv[1]))
            # 2) clip global solo al ganador
            final_actuator_action = np.clip(winner_action, self.global_actuator_min, self.global_actuator_max) if self.use_global_actuator else winner_action
            # 3) el ganador “contribuye”, los demás 0.0 -> anti-windup correcto
            effective_actions = {name: 0.0 for name in u_individual_dict}
            effective_actions[winner_name] = final_actuator_action
            return final_actuator_action, effective_actions

        # --- POLÍTICA: SUM (Suma con escalado proporcional) ---
        elif self.mixing_policy == 'sum':
            total_action = sum(u_individual_dict.values())
            final_actuator_action = np.clip(total_action, self.global_actuator_min, self.global_actuator_max) if self.use_global_actuator else total_action
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
        
        # Fallback por si se define una política no implementada
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
        angle_val   = state_to_evaluate['pendulum_angle']
        ang_vel_val = state_to_evaluate['pendulum_velocity']
        angle_stable   = self.angle_lo   <= angle_val   <= self.angle_hi
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
        """Expone los parámetros internos del entorno para el logging."""
        log_system_params = self.system.get_log_system_params()
        return log_system_params
    
    def get_controllers(self) -> Dict[str, Controller]:
        """Devuelve el diccionario de controladores gestionados por el entorno."""
        return self.controllers