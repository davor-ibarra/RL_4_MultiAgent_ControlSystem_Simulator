# factories/controller_factory.py
import logging
from typing import Dict, Any

# Import interfaces and specific controller implementations
from interfaces.controller import Controller
from components.controllers.pid_controller import PIDController
# Import other controller classes here

# 11.1: Usar logger específico del módulo
logger = logging.getLogger(__name__)

class ControllerFactory:
    """
    Factory service for creating controller instances.
    Ensures required parameters, including 'dt', are provided.
    """
    def __init__(self):
        logger.info("ControllerFactory instance created.")
        pass

    def create_controller(self, controller_type: str, controller_params: Dict[str, Any]) -> Controller:
        """
        Creates a controller instance based on type and parameters.

        Args:
            controller_type (str): Type of controller (e.g., 'pid').
            controller_params (Dict[str, Any]): Parameters for the constructor,
                                                 MUST include 'dt' provided by DI.

        Returns:
            An instance of a Controller subclass.

        Raises:
            ValueError: If type unknown or required parameters missing/invalid.
            TypeError: If parameter types are incorrect.
            RuntimeError: For unexpected errors.
        """
        logger.info(f"Attempting to create controller of type: {controller_type}")
        logger.debug(f"Controller params received by factory (keys): {list(controller_params.keys())}")
        controller: Controller # Type hint

        try:
            if controller_type == 'pid':
                # 11.2: Validar presencia de claves *esenciales* para PID, incluyendo 'dt'
                required_keys = ['kp', 'ki', 'kd', 'setpoint', 'dt']
                missing_keys = [key for key in required_keys if key not in controller_params]
                if missing_keys:
                    # Fail-Fast si falta algo esencial
                    raise ValueError(f"Missing required parameters {missing_keys} for 'pid' controller in controller_params.")

                # 11.3: Crear instancia usando desempaquetado (**), asumiendo que PIDController acepta estos args
                logger.debug(f"Creating PIDController with params: {controller_params}")
                controller = PIDController(**controller_params)

            # --- Add other controller types here ---
            # elif controller_type == 'lqr':
            #     required_lqr_keys = [...]
            #     missing_keys = [...]
            #     if missing_keys: raise ValueError(...)
            #     controller = LQRController(**controller_params)

            else:
                # Fail-Fast si el tipo es desconocido
                raise ValueError(f"Unknown controller type specified: {controller_type}")

            logger.info(f"Successfully created controller: {type(controller).__name__}")
            return controller

        # 11.4: Capturar errores específicos y relanzar
        except TypeError as e: # Error si los tipos de params no coinciden con el constructor
            logger.error(f"Type error creating controller '{controller_type}'. Check config param types: {e}", exc_info=True)
            raise TypeError(f"Parameter type mismatch for controller '{controller_type}': {e}") from e
        except KeyError as e: # Error si el constructor espera una clave no presente en **controller_params
            logger.error(f"Missing parameter key expected by controller '{controller_type}' constructor: {e}", exc_info=True)
            raise ValueError(f"Internal config error: Missing key '{e}' for controller '{controller_type}'.") from e
        except ValueError as e: # Captura errores de validación o tipo desconocido
            logger.error(f"Configuration or parameter error for controller '{controller_type}': {e}", exc_info=True)
            raise # Re-raise known config/value errors (Fail-Fast)
        except Exception as e:
            logger.error(f"Unexpected error creating controller '{controller_type}': {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error creating controller '{controller_type}'") from e