# components/simulators/pendulum_virtual_simulator.py
import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, Optional
import copy

# Importar Interfaces necesarias
from interfaces.virtual_simulator import VirtualSimulator # Implementar Interfaz
from interfaces.dynamic_system import DynamicSystem
from interfaces.controller import Controller
from interfaces.reward_function import RewardFunction # Interfaz que implementa InstantaneousRewardCalculator

logger = logging.getLogger(__name__)

class PendulumVirtualSimulator(VirtualSimulator): # Implementar Interfaz VirtualSimulator
    """
    Implementación de VirtualSimulator para el entorno del Péndulo Invertido.
    Ejecuta simulaciones virtuales de un intervalo de tiempo usando una copia
    independiente del controlador con ganancias específicas.
    Recibe sus componentes (System, Controller Template, RewardFunction) inyectados.
    """
    def __init__(self,
                 system: DynamicSystem,
                 controller: Controller, # Plantilla del controlador real
                 reward_function: RewardFunction, # Instancia de InstantaneousRewardCalculator
                 dt: float):
        """
        Inicializa el simulador virtual.

        Args:
            system: Instancia del sistema dinámico.
            controller: Instancia del controlador *real* (plantilla).
            reward_function: Instancia de la función de recompensa (InstantaneousRewardCalculator).
            dt: Paso de tiempo para las simulaciones virtuales.
        """
        logger.info("Inicializando PendulumVirtualSimulator...")
        self.system = system
        self.controller_template = controller
        self.reward_function = reward_function # Esta es la instancia de InstantaneousRewardCalculator
        self.dt = dt

        required_methods = ['reset_internal_state', 'update_params', 'compute_action', 'get_params']
        missing_methods = [m for m in required_methods if not hasattr(self.controller_template, m)]
        if missing_methods: raise AttributeError(f"Plantilla controlador ({type(controller).__name__}) sin métodos: {missing_methods}")
        if not isinstance(dt, (float, int)) or dt <= 0: raise ValueError(f"dt inválido ({dt}).")
        # Validar que reward_function tenga el método calculate (ya validado por interfaz)
        if not hasattr(self.reward_function, 'calculate'): raise AttributeError("reward_function inyectada no tiene método 'calculate'.")

        logger.info("PendulumVirtualSimulator inicializado.")

    def run_interval_simulation(self,
                                initial_state_vector: Any,
                                start_time: float,
                                duration: float,
                                controller_gains_dict: Dict[str, float]) -> float:
        """
        Ejecuta una simulación virtual autocontenida para un intervalo.
        Implementa el método de la interfaz. Utiliza la reward_function inyectada.
        """
        if duration <= 0: return 0.0

        virtual_state = np.array(initial_state_vector).flatten()
        if virtual_state.shape != (4,): # Asumiendo estado de 4 dimensiones para péndulo
             logger.error(f"VirtualSim: Estado inicial shape inválido: {virtual_state.shape}. Retornando 0.")
             return 0.0

        virtual_time = start_time
        accumulated_reward: float = 0.0
        num_steps = max(1, int(round(duration / self.dt)))
        # logger.debug(f"VirtualSim Start: t={start_time:.4f}, dur={duration:.4f}, steps={num_steps}, gains={controller_gains_dict}")

        virtual_controller: Optional[Controller] = None
        try:
            # --- Crear copia INDEPENDIENTE del controlador ---
            virtual_controller = copy.deepcopy(self.controller_template)

            # --- Configurar controlador virtual ---
            virtual_controller.reset_internal_state() # Limpiar errores integrales/derivativos
            kp_v, ki_v, kd_v = controller_gains_dict['kp'], controller_gains_dict['ki'], controller_gains_dict['kd']
            virtual_controller.update_params(kp_v, ki_v, kd_v) # Establecer ganancias fijas

            # --- Bucle Simulación Virtual ---
            for _ in range(num_steps):
                # Calcular acción virtual
                virtual_force = virtual_controller.compute_action(virtual_state)
                # Aplicar acción al sistema
                next_virtual_state = self.system.apply_action(virtual_state, virtual_force, virtual_time, self.dt)

                # Calcular recompensa instantánea usando la RewardFunction inyectada
                # La función calculate devuelve (reward_value, w_stab), solo necesitamos reward_value aquí.
                inst_reward, _ = self.reward_function.calculate(
                    state=virtual_state,
                    action=virtual_force,
                    next_state=next_virtual_state,
                    t=virtual_time
                )

                # Acumular recompensa (asegurando que sea finita)
                accumulated_reward += inst_reward if np.isfinite(inst_reward) else 0.0

                # Actualizar estado y tiempo
                virtual_state = next_virtual_state
                virtual_time += self.dt

        except KeyError as e: logger.error(f"VirtualSim: Falta clave ganancia: {e}. Gains: {controller_gains_dict}"); accumulated_reward = 0.0 # Devolver 0 en error
        except AttributeError as e: logger.error(f"VirtualSim: Error atributo controlador/reward_func virtual: {e}", exc_info=True); accumulated_reward = 0.0
        except IndexError as e: logger.error(f"VirtualSim: Error de índice (probablemente estado): {e}", exc_info=True); accumulated_reward = 0.0
        except Exception as e: logger.error(f"VirtualSim: Error inesperado: {e}", exc_info=True); accumulated_reward = 0.0
        finally: del virtual_controller # Ayudar GC

        # logger.debug(f"VirtualSim Finish: Accumulated Reward = {accumulated_reward:.4f}")
        final_reward = float(accumulated_reward)
        if pd.isna(final_reward) or not np.isfinite(final_reward):
             logger.warning(f"VirtualSim: Recompensa final inválida ({final_reward}). Devolviendo 0.0.")
             return 0.0
        return final_reward