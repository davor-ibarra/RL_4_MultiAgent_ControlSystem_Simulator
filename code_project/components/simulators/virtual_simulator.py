# components/simulators/pendulum_virtual_simulator.py
import numpy as np
import pandas as pd # Para pd.notna/isna
import logging
from typing import Any, Dict, Optional, Tuple, List
import copy # Para deepcopy

from interfaces.virtual_simulator import VirtualSimulator
from interfaces.dynamic_system import DynamicSystem
from interfaces.controller import Controller
from interfaces.reward_function import RewardFunction
from interfaces.stability_calculator import BaseStabilityCalculator

logger = logging.getLogger(__name__)

class DynamicVirtualSimulator(VirtualSimulator):
    """
    Runs self-contained, isolated simulations of an environment interval.
    It is initialized with component templates and a state-to-name mapping
    to remain generic while supporting dictionary-based state logic.
    """
    def __init__(self,
                 system_template: DynamicSystem,
                 controller_template: Controller,
                 reward_function_template: RewardFunction,
                 stability_calculator_template: BaseStabilityCalculator,
                 state_variable_map: Dict[str, int],
                 dt_sec_value: float # dt_sec, no opcional
                ):
        logger.info("[PendulumVirtualSimulator] Initializing...")

        # Validaciones estructurales mínimas en el constructor (Fail-Fast para DI)
        if not all(hasattr(o, m) for o, m_list in [
            (system_template, ['apply_action', 'reset']),
            (controller_template, ['compute_action', 'update_params', 'reset_internal_state']),
            (reward_function_template, ['calculate']),
            (stability_calculator_template, ['calculate_instantaneous_stability'])
        ] for m in m_list):
            raise TypeError("A provided template does not conform to its interface.")

        self.system_tpl = system_template
        self.controller_tpl = controller_template
        self.reward_func_tpl = reward_function_template
        self.stability_calc_tpl = stability_calculator_template
        self.state_map = state_variable_map
        self.sim_dt_sec = float(dt_sec_value)

        logger.info(f"[DynamicVirtualSimulator] Initialized with dt={self.sim_dt_sec:.4f} and state map: {self.state_map}")

    def _create_state_dict(self, state_vector: np.ndarray) -> Dict[str, float]:
        """Creates a named dictionary from the state vector using the injected map."""
        return {key: state_vector[idx] for key, idx in self.state_map.items() if len(state_vector) > idx}
    
    def run_interval_simulation(self,
                                initial_state_vec: Any, # Numpy array esperado
                                interval_start_time: float,
                                interval_duration: float,
                                fixed_gains_map: Dict[str, float] # {'kp': val, 'ki': val, 'kd': val}
                               ) -> Tuple[float, float]: # (total_reward, avg_stability_score)
        
        num_steps = max(1, int(round(interval_duration / self.sim_dt_sec)))
        
        # Crear copias profundas para esta simulación virtual aislada
        virt_system = copy.deepcopy(self.system_tpl)
        virt_controller = copy.deepcopy(self.controller_tpl)
        virt_reward_func = copy.deepcopy(self.reward_func_tpl)
        virt_stability_calc = copy.deepcopy(self.stability_calc_tpl)

        # Configurar el controlador virtual
        virt_controller.reset_internal_state() # Limpiar estado interno (ej. integral error)
        virt_controller.update_params( # Asume que fixed_gains_map tiene kp, ki, kd
            fixed_gains_map.get('kp', 0.0),
            fixed_gains_map.get('ki', 0.0),
            fixed_gains_map.get('kd', 0.0)
        )

        current_virt_state_vec = np.array(initial_state_vec, dtype=float).flatten() # Asegurar que es un array numpy
        current_virt_time = float(interval_start_time)
        
        total_reward_accum = 0.0
        stability_scores_list: List[float] = []

        for _ in range(num_steps):
            state_s_vec = np.copy(current_virt_state_vec)
            state_s_dict = self._create_state_dict(state_s_vec)
            
            # 1. Acción del controlador virtual
            control_force_virt_a = virt_controller.compute_action(state_s_vec)
            
            # 2. Aplicar acción al sistema virtual
            next_virt_state_vec = virt_system.apply_action(state_s_vec, control_force_virt_a, current_virt_time, self.sim_dt_sec)
            next_virt_state_dict = self._create_state_dict(next_virt_state_vec)
            
            # 3. Calcular recompensa y estabilidad del paso virtual
            reward_step_virt = virt_reward_func.calculate(
                state_dict=state_s_dict, action_a=control_force_virt_a,
                next_state_dict=next_virt_state_dict, current_episode_time_sec=current_virt_time,
                dt_sec=self.sim_dt_sec, goal_reached_in_step=False # Goal bonus not relevant for virtual runs
            )
            stability_step_virt = virt_stability_calc.calculate_instantaneous_stability(next_virt_state_dict)
            total_reward_accum += float(reward_step_virt) # Asumir que reward_step_virt es float
            stability_scores_list.append(float(stability_step_virt)) # Asumir que stability_step_virt es float

            current_virt_state_vec = next_virt_state_vec
            current_virt_time += self.sim_dt_sec
        
        # Limpieza explícita de los clones (opcional, Python GC debería hacerlo, pero puede ayudar)
        del virt_system, virt_controller, virt_reward_func, virt_stability_calc
        # gc.collect() # Podría ser excesivo llamarlo aquí siempre
        
        avg_stability = np.nanmean(stability_scores_list) if stability_scores_list else 1.0
        
        final_reward = float(total_reward_accum) if pd.notna(total_reward_accum) else 0.0
        final_stability = float(avg_stability) if pd.notna(avg_stability) else 1.0
        # logger.debug(f"[PendulumVirtualSimulator] Virtual run complete. Gains={fixed_gains_map}. Reward={final_reward:.3f}, AvgStability={final_stability:.3f}")
        return final_reward, final_stability