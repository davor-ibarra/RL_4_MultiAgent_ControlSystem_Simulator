from components.controllers.pid_controller import PIDController
import logging

class ControllerFactory:
    @staticmethod
    def create_controller(controller_config, dt): # Accept dt explicitly
        """
        Creates a controller instance.

        Args:
            controller_config (dict): Configuration specific to the controller type
                                      (e.g., {'type': 'pid', 'params': {...}}).
            dt (float): The time step for the controller.

        Returns:
            An instance of the specified controller.

        Raises:
            ValueError: If the controller_type is not recognized or params are missing.
        """
        controller_type = controller_config.get('type')
        params = controller_config.get('params', {})

        logging.info(f"Attempting to create controller of type '{controller_type}' with dt={dt} and params: {params}")

        if controller_type == 'pid':
            try:
                # Pass dt and unpack other specific params for PIDController
                # Ensure required params are present
                required_pid_params = ['kp', 'ki', 'kd', 'setpoint']
                if not all(p in params for p in required_pid_params):
                     missing = [p for p in required_pid_params if p not in params]
                     raise ValueError(f"Missing required PID parameters: {missing}")

                # Remove dt from params if it accidentally exists there
                if 'dt' in params:
                    logging.warning("Ignoring 'dt' found in controller params dictionary. Using explicitly passed dt.")
                    del params['dt']

                return PIDController(dt=dt, **params) # Pass dt explicitly

            except TypeError as e:
                # This might catch issues if params contains unexpected args for PIDController
                logging.error(f"Type error creating PIDController. Check parameter names and types: {e}")
                raise ValueError(f"Incorrect parameters provided for PIDController: {e}") from e
            except ValueError as e: # Catch missing param error from above check
                 logging.error(e)
                 raise
            except Exception as e:
                 logging.error(f"Unexpected error creating PIDController: {e}")
                 raise

        # Add other controller types here with 'elif'
        # elif controller_type == 'lqr':
        #     # return LQRController(dt=dt, **params)
        #     pass

        logging.error(f"Controller type '{controller_type}' not recognized.")
        raise ValueError(f"Controller type '{controller_type}' not recognized.")