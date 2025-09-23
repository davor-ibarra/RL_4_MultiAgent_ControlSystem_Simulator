# components/simulators/pendulum_virtual_simulator.py
import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, Optional, Tuple
import copy

from interfaces.virtual_simulator import VirtualSimulator
from interfaces.dynamic_system import DynamicSystem
from interfaces.controller import Controller
from interfaces.reward_function import RewardFunction

logger = logging.getLogger(__name__) # Logger específico del módulo

class PendulumVirtualSimulator(VirtualSimulator):
    def __init__(self,
                 system: DynamicSystem,
                 controller: Controller, # Plantilla del controlador
                 reward_function: RewardFunction,
                 dt: Optional[float] # Viene de environment.simulation.dt
                ):
        logger.info("[PendulumVirtualSimulator] Initializing...")

        if not isinstance(system, DynamicSystem): raise TypeError("system must implement DynamicSystem.")
        if not isinstance(controller, Controller): raise TypeError("controller (template) must implement Controller.")
        if not isinstance(reward_function, RewardFunction): raise TypeError("reward_function must implement RewardFunction.")
        self.system_template = system # Guardar como plantilla
        self.controller_template = controller # Guardar como plantilla
        self.reward_function_template = reward_function # Guardar como plantilla

        if dt is None or not isinstance(dt, (float, int)) or dt <= 0 or not np.isfinite(dt):
             raise ValueError(f"dt ({dt}) must be a positive finite number.")
        self.dt = float(dt)

        # Validar métodos necesarios en la plantilla del controlador
        required_ctrl_methods = ['reset_internal_state', 'update_params', 'compute_action']
        missing_methods = [m for m in required_ctrl_methods if not hasattr(self.controller_template, m)]
        if missing_methods:
             raise AttributeError(f"Controller template ({type(controller).__name__}) is missing required methods: {missing_methods}")

        logger.info(f"[PendulumVirtualSimulator] Initialized with dt={self.dt:.4f}.")

    def run_interval_simulation(self,
                                initial_state_vector: Any,
                                start_time: float,
                                duration: float,
                                controller_gains_dict: Dict[str, float]
                               ) -> Tuple[float, float]:
        #logger.debug(f"[VirtualSim:run] Start: InitialState={np.round(initial_state_vector,3) if isinstance(initial_state_vector, np.ndarray) else initial_state_vector}, Duration={duration:.3f}, Gains={controller_gains_dict}")
        if duration <= 0: return 0.0, 1.0 # Recompensa 0, estabilidad perfecta si no hay duración

        try:
            virtual_state = np.array(initial_state_vector, dtype=float).flatten()
            if virtual_state.shape != (4,): raise ValueError(f"Invalid initial_state_vector shape: {virtual_state.shape}")
            if not np.all(np.isfinite(virtual_state)): raise ValueError(f"initial_state_vector contains NaN/inf: {virtual_state}")
            if not all(g in controller_gains_dict for g in ['kp', 'ki', 'kd']): raise ValueError(f"Missing gains in controller_gains_dict: {controller_gains_dict.keys()}")
            kp_v, ki_v, kd_v = map(float, [controller_gains_dict['kp'], controller_gains_dict['ki'], controller_gains_dict['kd']])
            if not all(np.isfinite(k) for k in [kp_v, ki_v, kd_v]): raise ValueError(f"Virtual gains contain NaN/inf: {controller_gains_dict}")
        except (ValueError, TypeError) as e:
             logger.error(f"[VirtualSim:run] Invalid input parameters: {e}. Returning (0.0, 1.0).")
             return 0.0, 1.0 # Default: 0 recompensa, perfecta estabilidad

        virtual_time = float(start_time)
        accumulated_reward: float = 0.0
        accumulated_w_stab: float = 0.0 # Suma de w_stab, se promediará
        num_steps = max(1, int(round(duration / self.dt)))

        # Crear copias profundas de los componentes para esta simulación virtual
        # Esto asegura aislamiento completo del estado del entorno real.
        virtual_system = copy.deepcopy(self.system_template)
        virtual_controller = copy.deepcopy(self.controller_template)
        virtual_reward_func = copy.deepcopy(self.reward_function_template)

        try:
            virtual_controller.reset_internal_state() # Resetear estado interno del controlador copiado
            virtual_controller.update_params(kp_v, ki_v, kd_v) # Aplicar ganancias fijas

            for _ in range(num_steps):
                current_virtual_state_for_reward = np.copy(virtual_state)
                virtual_force = virtual_controller.compute_action(virtual_state)
                virtual_force_f = float(virtual_force) if np.isfinite(virtual_force) else 0.0

                next_virtual_state = virtual_system.apply_action(virtual_state, virtual_force_f, virtual_time, self.dt)
                if not isinstance(next_virtual_state, np.ndarray) or not np.all(np.isfinite(next_virtual_state)):
                     logger.warning(f"[VirtualSim:run] System returned invalid next_state: {next_virtual_state}. Terminating virtual run.")
                     return 0.0, 1.0 # Fallo -> 0 recompensa, estabilidad neutra

                inst_reward, w_stab_v = virtual_reward_func.calculate(
                    state=current_virtual_state_for_reward,
                    action=virtual_force_f,
                    next_state=next_virtual_state,
                    t=virtual_time
                )
                accumulated_reward += inst_reward if np.isfinite(inst_reward) else 0.0
                accumulated_w_stab += w_stab_v if np.isfinite(w_stab_v) else 0.0 # Default a 0 si w_stab es inválido

                virtual_state = next_virtual_state
                virtual_time += self.dt
        except Exception as e:
            logger.error(f"[VirtualSim:run] Unexpected error during virtual simulation: {e}", exc_info=True)
            return 0.0, 1.0 # Fallo -> 0 recompensa, estabilidad neutra
        finally:
            # Asegurar limpieza de las copias
            del virtual_system, virtual_controller, virtual_reward_func

        final_reward = float(accumulated_reward)
        avg_w_stab = (accumulated_w_stab / num_steps) if num_steps > 0 else 1.0
        if not np.isfinite(final_reward):
            logger.warning(f"[VirtualSim:run] Final virtual reward is not finite ({final_reward}). Returning 0.0."); final_reward = 0.0
        if not np.isfinite(avg_w_stab):
            logger.warning(f"[VirtualSim:run] Average virtual w_stab is not finite ({avg_w_stab}). Returning 1.0."); avg_w_stab = 1.0

        #logger.debug(f"[VirtualSim:run] Finish: TotalReward={final_reward:.4f}, AvgWStab={avg_w_stab:.4f}")
        return final_reward, avg_w_stab