# factories/system_factory.py
import logging
from typing import Dict, Any
from interfaces.dynamic_system import DynamicSystem
from components.systems.inverted_pendulum_system import InvertedPendulumSystem

logger = logging.getLogger(__name__) # Logger específico del módulo

class SystemFactory:
    def __init__(self):
        logger.info("[SystemFactory] Instance created.")

    def create_system(self, system_type: str, system_params: Dict[str, Any]) -> DynamicSystem:
        # system_params ya viene de config['environment']['system']['params'] vía DI.
        logger.info(f"[SystemFactory:create_system] Attempting system type: {system_type}")
        logger.debug(f"[SystemFactory:create_system] Received system_params keys: {list(system_params.keys())}")
        system: DynamicSystem

        try:
            if system_type == 'inverted_pendulum':
                # InvertedPendulumSystem validará m1, m2, l, g, cr, ca
                logger.debug(f"[SystemFactory:create_system] Creating InvertedPendulumSystem with **params.")
                system = InvertedPendulumSystem(**system_params)
            # --- Añadir otros tipos de sistema aquí ---
            # elif system_type == 'cart_pole_v2':
            #     system = CartPoleV2System(**system_params)
            else:
                raise ValueError(f"Unknown system type specified: {system_type}")

            logger.info(f"[SystemFactory:create_system] System '{type(system).__name__}' created.")
            return system
        except (ValueError, TypeError) as e_constr: # Errores del constructor del sistema
            logger.error(f"[SystemFactory:create_system] Error constructing system '{system_type}': {e_constr}", exc_info=True)
            raise
        except Exception as e_unexp:
            logger.error(f"[SystemFactory:create_system] Unexpected error creating system '{system_type}': {e_unexp}", exc_info=True)
            raise RuntimeError(f"Unexpected error creating system '{system_type}'") from e_unexp