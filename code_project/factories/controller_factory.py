import logging
from typing import Dict, Any

# Import interfaces and specific controller implementations
from interfaces.controller import Controller
from components.controllers.pid_controller import PIDController
# Import other controller classes here if you add them, e.g.:
# from components.controllers.lqr_controller import LQRController

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class ControllerFactory:
    """
    Factory class (as a service) for creating controller instances.
    Instances of this factory will be registered in the DI container.
    """
    def __init__(self):
        """Constructor (puede estar vacío si la fábrica no necesita dependencias)."""
        logger.info("ControllerFactory instance created.")
        pass

    def create_controller(self, controller_type: str, controller_params: Dict[str, Any]) -> Controller:
        """
        Creates a controller instance based on the specified type and parameters.
        This is now an instance method.

        Args:
            controller_type (str): The type of controller to create (e.g., 'pid').
            controller_params (Dict[str, Any]): Dictionary containing parameters needed by the
                                                 controller's constructor (e.g., kp, ki, kd, setpoint).
                                                 The 'dt' is often handled by the environment, but if the
                                                 controller needs it at initialization, it should be
                                                 present here or obtained from a shared config.

        Returns:
            An instance of a Controller subclass.

        Raises:
            ValueError: If the controller type is unknown or parameters are missing/invalid.
            RuntimeError: For unexpected errors during creation.
        """
        logger.info(f"Attempting to create controller of type: {controller_type}")
        controller: Controller # Type hint

        try:
            if controller_type == 'pid':
                # Check for required parameters for PIDController
                required_keys = ['kp', 'ki', 'kd', 'setpoint']
                # Check if dt is provided, otherwise use a default or raise error
                # Assuming dt is NOT in controller_params but handled by Environment/SimManager
                # The PIDController might need a default dt for internal init?
                if 'dt' not in controller_params:
                     logger.warning("PID controller 'dt' not found in params. Using default 0.01 for init. Environment should manage actual dt.")
                     # Add a default dt if PIDController constructor requires it
                     controller_params['dt'] = 0.01 # Default placeholder if needed

                # Validate presence of required keys
                missing_keys = [key for key in required_keys if key not in controller_params]
                if missing_keys:
                     raise ValueError(f"Missing required parameters {missing_keys} for 'pid' controller in config.")

                # Parameters are valid, create the instance using dictionary unpacking
                logger.debug(f"Creating PIDController with params: {controller_params}")
                controller = PIDController(**controller_params)

            # --- Add other controller types here ---
            # elif controller_type == 'lqr':
            #     # Ensure LQRController specific params are present
            #     # required_lqr_keys = [...]
            #     # if not all(key in controller_params for key in required_lqr_keys):
            #     #    raise ValueError(...)
            #     logger.debug(f"Creating LQRController with params: {controller_params}")
            #     controller = LQRController(**controller_params)

            else:
                raise ValueError(f"Unknown controller type specified: {controller_type}")

            logger.info(f"Successfully created controller: {type(controller).__name__}")
            return controller

        except KeyError as e:
             # Should be caught by missing_keys check, but kept for safety
             logger.error(f"Unexpected KeyError during controller creation '{controller_type}': {e}", exc_info=True)
             raise ValueError(f"Configuration error: Missing parameter for controller '{controller_type}'") from e
        except ValueError as e: # Catch ValueErrors from checks or unknown type
             logger.error(f"Configuration or parameter error for controller '{controller_type}': {e}", exc_info=True)
             raise # Re-raise known config/value errors
        except TypeError as e:
             # Catches errors like providing wrong type arguments to the constructor
             logger.error(f"Type error creating controller '{controller_type}'. Check parameter types in config: {e}", exc_info=True)
             raise ValueError(f"Parameter type mismatch for controller '{controller_type}'.") from e
        except Exception as e:
            logger.error(f"Failed to create controller of type '{controller_type}': {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error creating controller '{controller_type}'") from e