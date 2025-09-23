import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, Optional
import copy # Para deepcopy del controlador

# Importar Interfaces necesarias
from interfaces.virtual_simulator import VirtualSimulator
from interfaces.dynamic_system import DynamicSystem
from interfaces.controller import Controller
from interfaces.reward_function import RewardFunction

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class PendulumVirtualSimulator(VirtualSimulator):
    """
    Implementación de VirtualSimulator para el entorno del Péndulo Invertido.
    Ejecuta simulaciones virtuales de un intervalo de tiempo usando una copia
    independiente del controlador con ganancias específicas, sin afectar el estado
    del controlador real.
    """
    def __init__(self,
                 system: DynamicSystem,
                 controller: Controller, # Recibe el controlador REAL como plantilla
                 reward_function: RewardFunction,
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

        # --- Validaciones Cruciales ---
        # Asegurar que la plantilla del controlador tiene los métodos necesarios
        required_methods = ['reset_internal_state', 'update_params', 'compute_action', 'get_params']
        missing_methods = [m for m in required_methods if not hasattr(self.controller_template, m)]
        if missing_methods:
            msg = f"VirtualSimulator: La plantilla del controlador ({type(controller).__name__}) NO tiene los métodos requeridos: {missing_methods}"
            logger.error(msg)
            raise AttributeError(msg)
        # Asegurar que tiene las ganancias como atributos (para get/set)
        required_attrs = ['kp', 'ki', 'kd']
        missing_attrs = [a for a in required_attrs if not hasattr(self.controller_template, a)]
        if missing_attrs:
            msg = f"VirtualSimulator: La plantilla del controlador ({type(controller).__name__}) NO tiene los atributos requeridos: {missing_attrs}"
            logger.error(msg)
            raise AttributeError(msg)

        if not isinstance(dt, (float, int)) or dt <= 0:
             msg = f"VirtualSimulator: dt inválido ({dt}). Debe ser número positivo."
             logger.error(msg)
             raise ValueError(msg)

        logger.info("PendulumVirtualSimulator inicializado exitosamente (usando plantilla de controlador).")


    def run_interval_simulation(self,
                                initial_state_vector: Any,
                                start_time: float,
                                duration: float,
                                controller_gains_dict: Dict[str, float]) -> float:
        """
        Ejecuta una simulación virtual autocontenida para un intervalo.

        Args:
            initial_state_vector: Estado inicial [cart_pos, cart_vel, angle, angular_vel].
            start_time: Tiempo inicial t.
            duration: Duración del intervalo a simular (e.g., decision_interval).
            controller_gains_dict: Diccionario {'kp': Kp_virt, 'ki': Ki_virt, 'kd': Kd_virt}
                                   a usar FIJAS durante esta simulación virtual.

        Returns:
            float: Recompensa total acumulada durante la simulación virtual.
                   Devuelve 0.0 si la duración es no positiva o si ocurren errores críticos.
        """
        # Validar duración
        if duration <= 0:
            logger.warning("VirtualSimulator: Duración no positiva solicitada. Devolviendo recompensa 0.")
            return 0.0

        # Clonar estado inicial para no modificar el original
        virtual_state = np.array(initial_state_vector).flatten()
        virtual_time = start_time
        accumulated_reward: float = 0.0
        # Calcular número de pasos (asegurar al menos 1 si duration > 0)
        num_steps = max(1, int(round(duration / self.dt)))
        # logger.debug(f"VirtualSim Start: t={start_time:.4f}, duration={duration:.4f}, steps={num_steps}, "
        #              f"Gains={controller_gains_dict}, InitState={np.round(virtual_state, 4)}")

        # --- Crear copia INDEPENDIENTE del controlador para esta simulación ---
        try:
            # Usar deepcopy para asegurar aislamiento total del estado interno (errores, integral)
            virtual_controller = copy.deepcopy(self.controller_template)
            # logger.debug(f"VirtualSim: Copia del controlador creada (ID: {id(virtual_controller)})")
        except Exception as e:
            logger.error(f"VirtualSimulator CRITICAL: Fallo al hacer deepcopy de la plantilla del controlador: {e}", exc_info=True)
            # Devolver recompensa muy negativa en caso de fallo crítico
            return -1.0e6 # Usar un valor grande negativo


        try:
            # --- Configurar el controlador virtual ---
            # 1. Resetear su estado interno (errores, integral)
            virtual_controller.reset_internal_state()

            # 2. Establecer las ganancias fijas para esta simulación virtual
            kp_virt = controller_gains_dict['kp']
            ki_virt = controller_gains_dict['ki']
            kd_virt = controller_gains_dict['kd']
            virtual_controller.update_params(kp_virt, ki_virt, kd_virt)
            # logger.debug(f"VirtualSim (ID: {id(virtual_controller)}): Gains set to Kp={kp_virt:.3f}, Ki={ki_virt:.3f}, Kd={kd_virt:.3f}")

            # --- Bucle de Simulación Virtual ---
            for _ in range(num_steps): # Iterar el número de pasos calculado

                # a. Calcular acción virtual usando el estado virtual y las ganancias virtuales fijas
                try:
                    virtual_force = virtual_controller.compute_action(virtual_state)
                except Exception as e:
                    logger.error(f"VirtualSim (ID: {id(virtual_controller)}): Error calculando acción virtual en t={virtual_time:.4f}: {e}", exc_info=True)
                    virtual_force = 0.0 # Acción neutral si falla

                # b. Aplicar acción al sistema para obtener siguiente estado virtual
                #    (El sistema es compartido, pero asumimos que apply_action es stateless respecto al historial)
                try:
                    next_virtual_state = self.system.apply_action(virtual_state, virtual_force, virtual_time, self.dt)
                except Exception as e:
                    logger.error(f"VirtualSim (ID: {id(virtual_controller)}): Error aplicando acción virtual al sistema en t={virtual_time:.4f}: {e}", exc_info=True)
                    next_virtual_state = virtual_state # Mantener estado si dinámica falla

                # c. Calcular recompensa instantánea para este paso virtual
                try:
                    # Usar la función de recompensa compartida
                    inst_reward, _ = self.reward_function.calculate(virtual_state, virtual_force, next_virtual_state, virtual_time)
                    accumulated_reward += inst_reward
                except Exception as e:
                    logger.error(f"VirtualSim (ID: {id(virtual_controller)}): Error calculando recompensa virtual en t={virtual_time:.4f}: {e}", exc_info=True)
                    # No sumar recompensa si el cálculo falla

                # d. Actualizar estado y tiempo virtuales para el siguiente paso
                virtual_state = next_virtual_state
                virtual_time += self.dt

            # --- Fin del Bucle Virtual ---

        except KeyError as e:
            logger.error(f"VirtualSimulator: Falta clave de ganancia en controller_gains_dict: {e}. Gains recibidos: {controller_gains_dict}")
            accumulated_reward = -1.0e6 # Penalizar fuertemente
        except AttributeError as e: # Si la copia del controlador no tiene un método esperado
             logger.error(f"VirtualSimulator: Error de atributo en controlador virtual (ID: {id(virtual_controller)}): {e}", exc_info=True)
             accumulated_reward = -1.0e6
        except Exception as e:
            logger.error(f"VirtualSimulator: Error inesperado durante simulación virtual (ID: {id(virtual_controller)}): {e}", exc_info=True)
            accumulated_reward = -1.0e6 # Penalizar fuertemente
        finally:
             # No es estrictamente necesario, pero ayuda al GC a liberar la copia
             del virtual_controller

        # logger.debug(f"VirtualSim Finish: Accumulated Reward = {accumulated_reward:.4f}")
        # Asegurar que devolvemos un float estándar
        final_reward = float(accumulated_reward)
        if pd.isna(final_reward) or not np.isfinite(final_reward):
             logger.warning(f"VirtualSimulator: Recompensa acumulada final es inválida ({final_reward}). Devolviendo 0.0.")
             return 0.0
        return final_reward
