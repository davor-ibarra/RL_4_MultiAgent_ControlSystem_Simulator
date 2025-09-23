import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, Optional
import copy # Para deepcopy del controlador

# Importar Interfaces necesarias
from interfaces.virtual_simulator import VirtualSimulator # Implementar Interfaz
from interfaces.dynamic_system import DynamicSystem
from interfaces.controller import Controller
from interfaces.reward_function import RewardFunction

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class PendulumVirtualSimulator(VirtualSimulator): # Implementar Interfaz VirtualSimulator
    """
    Implementación de VirtualSimulator para el entorno del Péndulo Invertido.
    Ejecuta simulaciones virtuales de un intervalo de tiempo usando una copia
    independiente del controlador con ganancias específicas.
    Recibe sus componentes (System, Controller Template, RewardFunction) inyectados.
    """
    def __init__(self,
                 # --- Dependencias Inyectadas ---
                 system: DynamicSystem,
                 controller: Controller, # Recibe el controlador REAL como plantilla
                 reward_function: RewardFunction,
                 # --- Parámetros desde Config ---
                 dt: float):
        """
        Inicializa el simulador virtual.

        Args:
            system: Instancia del sistema dinámico (compartida con entorno real).
            controller: Instancia del controlador *real* (se usará como plantilla para deepcopy).
            reward_function: Instancia de la función de recompensa (compartida).
            dt: Paso de tiempo para las simulaciones virtuales.
        """
        logger.info("Inicializando PendulumVirtualSimulator...")
        self.system = system
        # Guardar el controlador original como plantilla para copias
        self.controller_template = controller
        self.reward_function = reward_function
        self.dt = dt

        # --- Validaciones (Mantenidas) ---
        required_methods = ['reset_internal_state', 'update_params', 'compute_action', 'get_params']
        missing_methods = [m for m in required_methods if not hasattr(self.controller_template, m)]
        if missing_methods:
            msg = f"VirtualSimulator: Plantilla controlador ({type(controller).__name__}) sin métodos: {missing_methods}"
            logger.error(msg); raise AttributeError(msg)
        # No es necesario validar atributos Kp/Ki/Kd aquí, update_params/get_params es suficiente

        if not isinstance(dt, (float, int)) or dt <= 0:
             msg = f"VirtualSimulator: dt inválido ({dt})."; logger.error(msg); raise ValueError(msg)

        logger.info("PendulumVirtualSimulator inicializado.")


    def run_interval_simulation(self,
                                initial_state_vector: Any,
                                start_time: float,
                                duration: float,
                                controller_gains_dict: Dict[str, float]) -> float:
        """
        Ejecuta una simulación virtual autocontenida para un intervalo.
        Implementa el método de la interfaz.
        """
        # ... (lógica mantenida como estaba, usa componentes inyectados) ...
        if duration <= 0: return 0.0

        virtual_state = np.array(initial_state_vector).flatten()
        virtual_time = start_time
        accumulated_reward: float = 0.0
        num_steps = max(1, int(round(duration / self.dt)))
        # logger.debug(f"VirtualSim Start: t={start_time:.4f}, dur={duration:.4f}, steps={num_steps}, gains={controller_gains_dict}")

        virtual_controller: Optional[Controller] = None # Para finally
        try:
            # --- Crear copia INDEPENDIENTE del controlador ---
            virtual_controller = copy.deepcopy(self.controller_template)
            # logger.debug(f"VirtualSim: Copia controlador creada (ID: {id(virtual_controller)})")

            # --- Configurar controlador virtual ---
            virtual_controller.reset_internal_state()
            kp_v, ki_v, kd_v = controller_gains_dict['kp'], controller_gains_dict['ki'], controller_gains_dict['kd']
            virtual_controller.update_params(kp_v, ki_v, kd_v)
            # logger.debug(f"VirtualSim (ID: {id(virtual_controller)}): Gains set Kp={kp_v:.3f}, Ki={ki_v:.3f}, Kd={kd_v:.3f}")

            # --- Bucle Simulación Virtual ---
            for _ in range(num_steps):
                 # Calcular acción virtual
                 virtual_force = virtual_controller.compute_action(virtual_state)
                 # Aplicar acción al sistema
                 next_virtual_state = self.system.apply_action(virtual_state, virtual_force, virtual_time, self.dt)
                 # Calcular recompensa instantánea
                 inst_reward, _ = self.reward_function.calculate(virtual_state, virtual_force, next_virtual_state, virtual_time)
                 # Asegurar que la recompensa es finita
                 accumulated_reward += inst_reward if np.isfinite(inst_reward) else 0.0
                 # Actualizar estado y tiempo
                 virtual_state = next_virtual_state
                 virtual_time += self.dt

        except KeyError as e:
            logger.error(f"VirtualSim: Falta clave ganancia: {e}. Gains: {controller_gains_dict}"); accumulated_reward = -1.0e6
        except AttributeError as e:
             logger.error(f"VirtualSim: Error atributo controlador virtual (ID: {id(virtual_controller)}): {e}", exc_info=True); accumulated_reward = -1.0e6
        except Exception as e:
            sim_id_str = f"(ID: {id(virtual_controller)})" if virtual_controller else ""
            logger.error(f"VirtualSim {sim_id_str}: Error inesperado: {e}", exc_info=True); accumulated_reward = -1.0e6
        finally:
             del virtual_controller # Ayudar al GC

        # logger.debug(f"VirtualSim Finish: Accumulated Reward = {accumulated_reward:.4f}")
        final_reward = float(accumulated_reward)
        if pd.isna(final_reward) or not np.isfinite(final_reward):
             logger.warning(f"VirtualSim: Recompensa final inválida ({final_reward}). Devolviendo 0.0.")
             return 0.0
        return final_reward