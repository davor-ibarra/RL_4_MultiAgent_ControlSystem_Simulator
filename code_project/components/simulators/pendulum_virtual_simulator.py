import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, Optional, Tuple
import copy

# Importar Interfaces necesarias
from interfaces.virtual_simulator import VirtualSimulator # Implementar Interfaz
from interfaces.dynamic_system import DynamicSystem
from interfaces.controller import Controller
from interfaces.reward_function import RewardFunction

# 11.1: Usar logger específico del módulo
logger = logging.getLogger(__name__)

class PendulumVirtualSimulator(VirtualSimulator): # Implementar Interfaz VirtualSimulator
    """
    Simulador virtual para el Péndulo Invertido. Implementa VirtualSimulator.
    Ejecuta simulaciones de intervalo usando copias independientes de los componentes.
    """
    def __init__(self,
                 system: DynamicSystem,
                 controller: Controller, # Plantilla del controlador real
                 reward_function: RewardFunction, # Instancia de RewardFunction inyectada
                 dt: float):
        """
        Inicializa el simulador virtual.

        Args:
            system: Instancia del sistema dinámico (inyectada).
            controller: Instancia plantilla del controlador (inyectada).
            reward_function: Instancia de la función de recompensa (inyectada).
            dt: Paso de tiempo para las simulaciones virtuales.

        Raises:
            TypeError: Si las dependencias no implementan las interfaces requeridas.
            ValueError: Si dt es inválido.
            AttributeError: Si falta algún método requerido en las dependencias.
        """
        logger.info("Inicializando PendulumVirtualSimulator...")
        # 11.2: Validar tipos de dependencias (Fail-Fast)
        if not isinstance(system, DynamicSystem): raise TypeError("system debe implementar DynamicSystem")
        if not isinstance(controller, Controller): raise TypeError("controller debe implementar Controller")
        if not isinstance(reward_function, RewardFunction): raise TypeError("reward_function debe implementar RewardFunction")
        self.system = system
        self.controller_template = controller
        self.reward_function = reward_function

        if not isinstance(dt, (float, int)) or dt <= 0:
             raise ValueError(f"dt inválido ({dt}) para PendulumVirtualSimulator.")
        self.dt = dt

        # Validar métodos necesarios en la plantilla del controlador
        required_ctrl_methods = ['reset_internal_state', 'update_params', 'compute_action']
        missing_methods = [m for m in required_ctrl_methods if not hasattr(self.controller_template, m)]
        if missing_methods:
             raise AttributeError(f"Plantilla controlador ({type(controller).__name__}) sin métodos: {missing_methods}")

        logger.info("PendulumVirtualSimulator inicializado.")

    def run_interval_simulation(self,
                                initial_state_vector: Any,
                                start_time: float,
                                duration: float,
                                controller_gains_dict: Dict[str, float]) -> Tuple[float, float]:
        """
        Ejecuta una simulación virtual autocontenida para un intervalo.
        Devuelve la recompensa acumulada y la estabilidad promedio (w_stab).
        """
        # 11.3: Validar inputs básicos
        if duration <= 0: return 0.0
        try:
            # Validar y copiar estado inicial
            virtual_state = np.array(initial_state_vector, dtype=float).flatten()
            if virtual_state.shape != (4,): # Asumiendo péndulo
                raise ValueError(f"Shape estado inicial inválido: {virtual_state.shape}")
            if not np.all(np.isfinite(virtual_state)):
                 raise ValueError(f"Estado inicial contiene NaN/inf: {virtual_state}")
            # Validar ganancias
            if not all(g in controller_gains_dict for g in ['kp', 'ki', 'kd']):
                 raise ValueError(f"Faltan ganancias en controller_gains_dict: {controller_gains_dict}")
            kp_v, ki_v, kd_v = map(float, [controller_gains_dict['kp'], controller_gains_dict['ki'], controller_gains_dict['kd']])
            if not all(np.isfinite(k) for k in [kp_v, ki_v, kd_v]):
                 raise ValueError(f"Ganancias virtuales contienen NaN/inf: {controller_gains_dict}")

        except (ValueError, TypeError) as e:
             logger.error(f"VirtualSim: Error en parámetros de entrada: {e}. Retornando 0.")
             return 0.0, 1.0

        virtual_time = float(start_time)
        accumulated_reward: float = 0.0
        accumulated_w_stab: float = 0.0
        num_steps = max(1, int(round(duration / self.dt)))
        #logger.debug(f"PendulumVirtualSimulator -> run_interval_simulation -> Start: t_v={virtual_time:.4f}, dur={duration:.4f}, steps={num_steps}")
        #logger.debug(f"PendulumVirtualSimulator -> run_interval_simulation -> Start: virtual_state={virtual_state}, gains={controller_gains_dict}")

        virtual_controller: Optional[Controller] = None
        try:
            # Crear copia INDEPENDIENTE del controlador
            virtual_controller = copy.deepcopy(self.controller_template)
            # Configurar controlador virtual
            virtual_controller.reset_internal_state()
            #logger.debug(f"PendulumVirtualSimulator -> run_interval_simulation -> Update virtual gains: Kp_v={kp_v}, Ki_v={ki_v}, Kd_v={kd_v}")
            virtual_controller.update_params(kp_v, ki_v, kd_v)

            # --- Bucle Simulación Virtual ---
            for step_idx in range(num_steps):
                current_virtual_state_copy = np.copy(virtual_state) # Copia para reward func

                #logger.debug(f"PendulumVirtualSimulator -> run_interval_simulation -> [Step {_+1}, dt={virtual_time}] Currrent state: S_v={current_virtual_state_copy}") # [DEBUG ADDED]

                # Calcular acción virtual (maneja errores internos)
                virtual_force = virtual_controller.compute_action(current_virtual_state_copy)
                virtual_force = float(virtual_force) if np.isfinite(virtual_force) else 0.0

                #logger.debug(f"PendulumVirtualSimulator -> run_interval_simulation -> Compute action: F_v={virtual_force:.3f}") # [DEBUG ADDED]

                # Aplicar acción al sistema (maneja errores internos)
                next_virtual_state = self.system.apply_action(current_virtual_state_copy, virtual_force, virtual_time, self.dt)
                # Validar estado resultante
                if not isinstance(next_virtual_state, np.ndarray) or next_virtual_state.shape != (4,) or not np.all(np.isfinite(next_virtual_state)):
                     logger.warning(f"VirtualSim: Estado inválido devuelto por system.apply_action: {next_virtual_state}. Terminando simulación virtual.")
                     # Devolver recompensa acumulada hasta ahora o 0? Mejor 0 para indicar fallo.
                     return 0.0, 1.0

                #logger.debug(f"PendulumVirtualSimulator -> run_interval_simulation -> Apply action: S'_v={next_virtual_state}") # [DEBUG ADDED]

                # Calcular recompensa instantánea (maneja errores internos)
                inst_reward, w_stab_v = self.reward_function.calculate( # Ignorar w_stab_v virtual
                    state=current_virtual_state_copy,
                    action=virtual_force,
                    next_state=next_virtual_state,
                    t=virtual_time
                )

                #logger.debug(f"PendulumVirtualSimulator -> run_interval_simulation -> Calculate reward: r_ins_v={inst_reward:.4f}, w_stab_v={w_stab_v:.4f}") # [DEBUG ADDED]
                
                # Acumular recompensa (asegurando finita)
                accumulated_reward += inst_reward if np.isfinite(inst_reward) else 0.0
                accumulated_w_stab += w_stab_v if np.isfinite(w_stab_v) else 0.0

                # Actualizar estado y tiempo
                virtual_state = next_virtual_state
                virtual_time += self.dt

        # 11.4: Simplificar manejo de errores, fallos deben ser capturados en los componentes
        except Exception as e:
            logger.error(f"VirtualSim: Error inesperado durante simulación: {e}", exc_info=True)
            accumulated_reward = 0.0 # Devolver 0 en error inesperado
            accumulated_w_stab = 0.0 # Reset w_stab sum on error
        finally:
             # Limpiar copia del controlador
             del virtual_controller

        #logger.debug(f"VirtualSim Finish: Accumulated Reward = {accumulated_reward:.4f}")
        # Asegurar valor final finito
        final_reward = float(accumulated_reward)
        if pd.isna(final_reward) or not np.isfinite(final_reward):
            logger.warning(f"VirtualSim: Recompensa final inválida ({final_reward}). Devolviendo 0.0.")
            final_reward = 0.0
        # Calcular estabilidad promedio
        avg_virtual_w_stab = (accumulated_w_stab / num_steps) if num_steps > 0 else 1.0
        if pd.isna(avg_virtual_w_stab) or not np.isfinite(avg_virtual_w_stab):
            logger.warning(f"VirtualSim: Estabilidad promedio virtual inválida ({avg_virtual_w_stab}). Devolviendo 1.0.")
            avg_virtual_w_stab = 1.0

        #logger.debug(f"VirtualSim Finish: Avg W_stab = {avg_virtual_w_stab:.4f}")

        return final_reward, float(avg_virtual_w_stab)