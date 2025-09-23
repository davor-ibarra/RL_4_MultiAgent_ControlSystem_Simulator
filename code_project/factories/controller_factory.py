import logging
from typing import Dict, Any

# Import interfaces and specific controller implementations
from interfaces.controller import Controller
from components.controllers.pid_controller import PIDController
# Import other controller classes here if you add them, e.g.:
# from components.controllers.lqr_controller import LQRController

class ControllerFactory:
    """
    Factory class for creating controller instances.
    """
    @staticmethod # Ensure this decorator is present
    def create_controller(controller_type: str, controller_params: Dict[str, Any]) -> Controller:
        """
        Creates a controller instance based on the specified type and parameters.

        Args:
            controller_type (str): The type of controller to create (e.g., 'pid').
            controller_params (Dict[str, Any]): Dictionary containing parameters needed by the
                                                 controller's constructor (e.g., kp, ki, kd, setpoint).

        Returns:
            An instance of a Controller subclass.

        Raises:
            ValueError: If the controller type is unknown or parameters are missing/invalid.
            RuntimeError: For unexpected errors during creation.
        """
        logging.info(f"Attempting to create controller of type: {controller_type}")
        controller: Controller # Type hint

        try:
            if controller_type == 'pid':
                # Check for required parameters for PIDController
                required_keys = ['kp', 'ki', 'kd', 'setpoint'] # dt is usually from environment
                if not all(key in controller_params for key in required_keys):
                     # dt is often passed separately or obtained from env config, handle if needed
                     # For now, assume dt is handled by the environment constructor
                     raise ValueError(f"Missing required parameters {required_keys} for 'pid' controller.")

                # Extract parameters
                kp = controller_params['kp']
                ki = controller_params['ki']
                kd = controller_params['kd']
                setpoint = controller_params['setpoint']
                # dt is crucial but often comes from the environment config, not controller params directly
                # We will assume the PendulumEnvironment constructor handles passing dt correctly.
                # If dt were needed here, the factory signature or param dict would need adjustment.
                # Example: dt = controller_params.get('dt', None) # Check if passed in params

                # Placeholder dt - the actual dt will be used by PendulumEnvironment's step method
                # The PIDController needs a dt for its internal calculation state,
                # but it might be better if the environment updates it or passes it during compute_action.
                # Let's initialize with a placeholder or fetch from global config if absolutely necessary,
                # but ideally, the environment handles the dt usage.
                # Fetching dt from environment config within ControllerFactory tightly couples them.
                # Let's assume PIDController constructor expects dt, fetch it if possible, default otherwise.
                # This is slightly awkward design - dt is an environment property.
                # TEMPORARY WORKAROUND: Let's hardcode a default or raise error if needed by constructor
                # A better design would pass dt from EnvironmentFactory or have Controller get dt from Env.
                placeholder_dt = 0.01 # Example placeholder or get from config['environment']['dt']

                controller = PIDController(kp=kp, ki=ki, kd=kd, setpoint=setpoint, dt=placeholder_dt)

            # --- Add other controller types here ---
            # elif controller_type == 'lqr':
            #     controller = LQRController(**controller_params)

            else:
                raise ValueError(f"Unknown controller type specified: {controller_type}")

            logging.info(f"Successfully created controller: {type(controller).__name__}")
            return controller

        except KeyError as e:
             logging.error(f"Missing parameter key during controller creation '{controller_type}': {e}", exc_info=True)
             raise ValueError(f"Configuration error: Missing parameter for controller '{controller_type}'") from e
        except ValueError as e: # Catch ValueErrors from checks or unknown type
             logging.error(f"Configuration or parameter error for controller '{controller_type}': {e}", exc_info=True)
             raise # Re-raise config errors
        except Exception as e:
            logging.error(f"Failed to create controller of type '{controller_type}': {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error creating controller '{controller_type}'") from e