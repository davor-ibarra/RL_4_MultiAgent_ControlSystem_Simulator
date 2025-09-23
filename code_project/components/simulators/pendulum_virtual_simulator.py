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

class PendulumVirtualSimulator(VirtualSimulator):
    def __init__(self,
                 system_template: DynamicSystem,
                 controller_template: Controller,
                 reward_function_template: RewardFunction,
                 stability_calculator_template: BaseStabilityCalculator,
                 dt_sec_value: float # dt_sec, no opcional
                ):
        logger.info("[PendulumVirtualSimulator] Initializing...")

        # Validaciones estructurales mínimas en el constructor (Fail-Fast para DI)
        if not hasattr(system_template, 'apply_action') or not hasattr(system_template, 'reset'):
            raise TypeError("system_template_instance must behave like DynamicSystem.")
        if not hasattr(controller_template, 'compute_action') or \
           not hasattr(controller_template, 'update_params') or \
           not hasattr(controller_template, 'reset_internal_state'): # Método esperado
            raise TypeError("controller_template_instance must provide core Controller methods.")
        if not hasattr(reward_function_template, 'calculate'):
            raise TypeError("reward_function_template_instance must behave like RewardFunction.")
        if not hasattr(stability_calculator_template, 'calculate_instantaneous_stability'): # <<< AÑADIR VALIDACIÓN
            raise TypeError("stability_calculator_template must behave like BaseStabilityCalculator.")
        
        self.system_tpl = system_template
        self.controller_tpl = controller_template
        self.reward_func_tpl = reward_function_template
        self.stability_calc_tpl = stability_calculator_template

        # dt_sec_value se asume validado por config_loader/DI (positivo, finito)
        self.sim_dt_sec = float(dt_sec_value)

        logger.info(f"[PendulumVirtualSimulator] Initialized with dt_sec={self.sim_dt_sec:.4f}.")

    def run_interval_simulation(self,
                                initial_state_vec: Any, # Numpy array esperado
                                interval_start_time: float,
                                interval_duration: float,
                                fixed_gains_map: Dict[str, float] # {'kp': val, 'ki': val, 'kd': val}
                               ) -> Tuple[float, float]: # (total_reward, avg_stability_score)
        
        # Asumir que las entradas son válidas (tipos y valores finitos).
        # Si interval_duration es muy pequeño, num_steps será 0 o 1, lo cual es manejado.
        
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

        current_virt_state = np.array(initial_state_vec, dtype=float).flatten() # Asegurar que es un array numpy
        current_virt_time = float(interval_start_time)
        
        total_reward_accum = 0.0
        stability_scores_list: List[float] = []

        for _ in range(num_steps):
            goal_reached_virtual_step = False
            state_at_virt_step_s = np.copy(current_virt_state) # S
            
            # 1. Acción del controlador virtual
            control_force_virt_a = virt_controller.compute_action(state_at_virt_step_s)
            # No validar control_force_virt_a; se asume que el controlador devuelve un float.
            
            # 2. Aplicar acción al sistema virtual
            next_virt_state_s_prime = virt_system.apply_action(
                state_at_virt_step_s, float(control_force_virt_a),
                current_virt_time, self.sim_dt_sec
            )
            # Si apply_action falla, propagará el error.

            # 3. Calcular recompensa y estabilidad del paso virtual
            reward_step_virt = virt_reward_func.calculate(
                state_s=state_at_virt_step_s, action_a=float(control_force_virt_a),
                next_state_s_prime=next_virt_state_s_prime,
                current_episode_time_sec=current_virt_time, dt_sec=self.sim_dt_sec,
                goal_reached_in_step=goal_reached_virtual_step # << Flag for Bonus
            )
            stability_step_virt = virt_stability_calc.calculate_instantaneous_stability(next_virt_state_s_prime) # type: ignore
            total_reward_accum += float(reward_step_virt) # Asumir que reward_step_virt es float
            stability_scores_list.append(float(stability_step_virt)) # Asumir que stability_step_virt es float

            current_virt_state = next_virt_state_s_prime
            current_virt_time += self.sim_dt_sec
        
        # Limpieza explícita de los clones (opcional, Python GC debería hacerlo, pero puede ayudar)
        del virt_system, virt_controller, virt_reward_func, virt_stability_calc
        # gc.collect() # Podría ser excesivo llamarlo aquí siempre

        avg_stability_score = np.nanmean(stability_scores_list) if stability_scores_list else 1.0
        # Asegurar que los retornos sean finitos
        final_total_reward = float(total_reward_accum) if pd.notna(total_reward_accum) and np.isfinite(total_reward_accum) else 0.0
        final_avg_stability = float(avg_stability_score) if pd.notna(avg_stability_score) and np.isfinite(avg_stability_score) else 1.0

        # logger.debug(f"[PendulumVirtualSimulator] Virtual run complete. Gains={fixed_gains_map}. Reward={final_total_reward:.3f}, AvgStability={final_avg_stability:.3f}")
        return final_total_reward, final_avg_stability