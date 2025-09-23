# factories/system_factory.py
import logging
from typing import Dict, Any

# Import interfaces and specific system implementations
from interfaces.dynamic_system import DynamicSystem
from components.systems.inverted_pendulum_system import InvertedPendulumSystem
# Import other system classes here

# 14.1: Usar logger específico del módulo
logger = logging.getLogger(__name__)

class SystemFactory:
    """
    Factory service for creating dynamic system instances.
    """
    def __init__(self):
        logger.info("SystemFactory instance created.")
        pass

    def create_system(self, system_type: str, system_params: Dict[str, Any]) -> DynamicSystem:
        """
        Creates a dynamic system instance based on type and parameters.

        Args:
            system_type (str): Type of system (e.g., 'inverted_pendulum').
            system_params (Dict[str, Any]): Parameters for the constructor.

        Returns:
            An instance of a DynamicSystem subclass.

        Raises:
            ValueError: If type unknown or required parameters missing/invalid.
            TypeError: If parameter types are incorrect.
            RuntimeError: For unexpected errors.
        """
        logger.info(f"Attempting to create system of type: {system_type}")
        logger.debug(f"System params received by factory (keys): {list(system_params.keys())}")
        system: DynamicSystem # Type hint

        try:
            if system_type == 'inverted_pendulum':
                # 14.2: Validar presencia de claves *esenciales* para este sistema
                required_keys = ['m1', 'm2', 'l', 'g']
                # Coeficientes cr, ca son opcionales con default en el constructor
                missing_keys = [key for key in required_keys if key not in system_params]
                if missing_keys:
                    # Fail-Fast si falta algo esencial
                    raise ValueError(f"Missing required parameters {missing_keys} for '{system_type}' system.")

                # 14.3: Crear instancia usando desempaquetado
                logger.debug(f"Creating InvertedPendulumSystem with params: {system_params}")
                system = InvertedPendulumSystem(**system_params)

            # --- Add other system types here ---
            # elif system_type == 'another_system':
            #     required_another_keys = [...]
            #     missing_keys = [...]
            #     if missing_keys: raise ValueError(...)
            #     system = AnotherSystem(**system_params)

            else:
                # Fail-Fast si el tipo es desconocido
                raise ValueError(f"Unknown system type specified: {system_type}")

            logger.info(f"Successfully created system: {type(system).__name__}")
            return system

        # 14.4: Capturar errores específicos y relanzar
        except TypeError as e:
            logger.error(f"Type error creating system '{system_type}'. Check config param types: {e}", exc_info=True)
            raise TypeError(f"Parameter type mismatch for system '{system_type}': {e}") from e
        except KeyError as e:
            logger.error(f"Missing parameter key expected by system '{system_type}' constructor: {e}", exc_info=True)
            raise ValueError(f"Internal config error: Missing key '{e}' for system '{system_type}'.") from e
        except ValueError as e: # Captura errores de validación o tipo desconocido
            logger.error(f"Configuration or parameter error for system '{system_type}': {e}", exc_info=True)
            raise # Re-raise known config/value errors (Fail-Fast)
        except Exception as e:
            logger.error(f"Unexpected error creating system '{system_type}': {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error creating system '{system_type}'") from e