import logging
from typing import Dict, Any

# Import interfaces and specific system implementations
from interfaces.dynamic_system import DynamicSystem
from components.systems.inverted_pendulum_system import InvertedPendulumSystem
# Import other system classes here if you add them, e.g.:
# from components.systems.another_system import AnotherSystem

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class SystemFactory:
    """
    Factory class (as a service) for creating dynamic system instances.
    Instances of this factory will be registered in the DI container.
    """
    def __init__(self):
        """Constructor (puede estar vacío si la fábrica no necesita dependencias)."""
        logger.info("SystemFactory instance created.")
        pass

    def create_system(self, system_type: str, system_params: Dict[str, Any]) -> DynamicSystem:
        """
        Creates a dynamic system instance based on the specified type and parameters.
        This is now an instance method.

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
        logger.info(f"Attempting to create system of type: {system_type}")
        system: DynamicSystem # Type hint

        try:
            if system_type == 'inverted_pendulum':
                # Check for required parameters for this specific system
                required_keys = ['m1', 'm2', 'l', 'g']
                missing_keys = [key for key in required_keys if key not in system_params]
                if missing_keys:
                    raise ValueError(f"Missing required parameters {missing_keys} for 'inverted_pendulum' system.")

                # Optional parameters with defaults
                system_params.setdefault('cr', 0.0) # Damping defaults to 0 if not provided
                system_params.setdefault('ca', 0.0) # Damping defaults to 0 if not provided

                # Create instance using dictionary unpacking
                logger.debug(f"Creating InvertedPendulumSystem with params: {system_params}")
                system = InvertedPendulumSystem(**system_params)

            # --- Add other system types here ---
            # elif system_type == 'another_system':
            #     # Ensure required params for AnotherSystem are present
            #     # required_another_keys = [...]
            #     # ... validation ...
            #     logger.debug(f"Creating AnotherSystem with params: {system_params}")
            #     system = AnotherSystem(**system_params)

            else:
                raise ValueError(f"Unknown system type specified: {system_type}")

            logger.info(f"Successfully created system: {type(system).__name__}")
            return system

        except KeyError as e:
             # Should be caught by missing_keys check, but kept for safety
             logger.error(f"Unexpected KeyError during system creation '{system_type}': {e}", exc_info=True)
             raise ValueError(f"Configuration error: Missing parameter for system '{system_type}'") from e
        except ValueError as e: # Catch ValueErrors from checks or unknown type
             logger.error(f"Configuration or parameter error for system '{system_type}': {e}", exc_info=True)
             raise # Re-raise known config/value errors
        except TypeError as e:
             # Catches errors like providing wrong type arguments to the constructor
             logger.error(f"Type error creating system '{system_type}'. Check parameter types in config: {e}", exc_info=True)
             raise ValueError(f"Parameter type mismatch for system '{system_type}'.") from e
        except Exception as e:
            logger.error(f"Failed to create system of type '{system_type}': {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error creating system '{system_type}'") from e