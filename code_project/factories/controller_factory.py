# factories/controller_factory.py
import logging
from typing import Dict, Any
from interfaces.controller import Controller
from components.controllers.pid_controller import PIDController

logger = logging.getLogger(__name__) # Logger específico del módulo

class ControllerFactory:
    def __init__(self):
        logger.info("[ControllerFactory] Instance created.")

    def create_controller(self, controller_type: str, controller_params: Dict[str, Any]) -> Controller:
        # controller_params ya contiene 'dt' y otros params del controlador desde DI.
        logger.info(f"[ControllerFactory:create_controller] Attempting controller type: {controller_type}")
        logger.debug(f"[ControllerFactory:create_controller] Received controller_params keys: {list(controller_params.keys())}")
        controller: Controller

        try:
            if controller_type == 'pid':
                # PIDController validará internamente kp, ki, kd, setpoint, dt
                logger.debug(f"[ControllerFactory:create_controller] Creating PIDController with **params.")
                controller = PIDController(**controller_params)
            # --- Añadir otros tipos de controlador aquí ---
            # elif controller_type == 'lqr_controller':
            #     controller = LQRController(**controller_params)
            else:
                raise ValueError(f"Unknown controller type specified: {controller_type}")

            logger.info(f"[ControllerFactory:create_controller] Controller '{type(controller).__name__}' created.")
            return controller
        except (ValueError, TypeError) as e_constr: # Errores del constructor del controller
            logger.error(f"[ControllerFactory:create_controller] Error constructing controller '{controller_type}': {e_constr}", exc_info=True)
            raise
        except Exception as e_unexp:
            logger.error(f"[ControllerFactory:create_controller] Unexpected error creating controller '{controller_type}': {e_unexp}", exc_info=True)
            raise RuntimeError(f"Unexpected error creating controller '{controller_type}'") from e_unexp