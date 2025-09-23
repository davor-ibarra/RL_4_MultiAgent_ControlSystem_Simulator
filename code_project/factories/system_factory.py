import logging
from typing import Dict, Any

# Import interfaces and specific system implementations
from interfaces.dynamic_system import DynamicSystem
from components.systems.inverted_pendulum_system import InvertedPendulumSystem
# Import other system classes here if you add them, e.g.:
# from components.systems.another_system import AnotherSystem

class SystemFactory:
    """
    Factory class for creating dynamic system instances.
    """
    @staticmethod  # Ensure this decorator is present
    def create_system(system_type: str, system_params: Dict[str, Any]) -> DynamicSystem:
        """
        Creates a dynamic system instance based on the specified type and parameters.

        Args:
            system_type (str): The type of system to create (e.g., 'inverted_pendulum').
            system_params (Dict[str, Any]): Dictionary containing parameters needed by the
                                           system's constructor (e.g., m1, m2, l, g).

        Returns:
            An instance of a DynamicSystem subclass.

        Raises:
            ValueError: If the system type is unknown or parameters are missing/invalid.
            RuntimeError: For unexpected errors during creation.
        """
        logging.info(f"Attempting to create system of type: {system_type}")
        system: DynamicSystem # Type hint

        try:
            if system_type == 'inverted_pendulum':
                # Check for required parameters for this specific system
                required_keys = ['m1', 'm2', 'l', 'g']
                if not all(key in system_params for key in required_keys):
                    raise ValueError(f"Missing required parameters {required_keys} for 'inverted_pendulum' system.")

                # Extract parameters, providing defaults for optional ones if applicable
                m1 = system_params['m1']
                m2 = system_params['m2']
                l = system_params['l']
                g = system_params['g']
                cr = system_params.get('cr', 0.0) # Damping defaults to 0 if not provided
                ca = system_params.get('ca', 0.0) # Damping defaults to 0 if not provided

                system = InvertedPendulumSystem(m1=m1, m2=m2, l=l, g=g, cr=cr, ca=ca)

            # --- Add other system types here ---
            # elif system_type == 'another_system':
            #     # Example: Pass all params using dictionary unpacking
            #     system = AnotherSystem(**system_params)

            else:
                raise ValueError(f"Unknown system type specified: {system_type}")

            logging.info(f"Successfully created system: {type(system).__name__}")
            return system

        except KeyError as e:
             # This might happen if required_keys check fails or params are accessed incorrectly
             logging.error(f"Missing parameter key during system creation '{system_type}': {e}", exc_info=True)
             raise ValueError(f"Configuration error: Missing parameter for system '{system_type}'") from e
        except ValueError as e: # Catch ValueErrors from checks or unknown type
             logging.error(f"Configuration or parameter error for system '{system_type}': {e}", exc_info=True)
             raise # Re-raise config errors
        except Exception as e:
            logging.error(f"Failed to create system of type '{system_type}': {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error creating system '{system_type}'") from e
