import logging
from typing import Dict, Any, Optional

# Import interfaces and components
from interfaces.stability_calculator import BaseStabilityCalculator
from interfaces.reward_function import RewardFunction
# Import specific implementations
from components.analysis.ira_stability_calculator import IRAStabilityCalculator
from components.analysis.simple_exponential_stability_calculator import SimpleExponentialStabilityCalculator
# --- IMPORT RENOMBRADO ---
from components.rewards.instantaneous_reward_calculator import InstantaneousRewardCalculator
# -------------------------
# Import other reward/calculator types here

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class RewardFactory:
    """
    Factory class (as a service) for creating Stability Calculators and Reward Functions.
    Instances of this factory will be registered in the DI container.
    """
    def __init__(self):
        """Constructor."""
        logger.info("RewardFactory instance created.")
        pass

    def create_stability_calculator(self, stability_config: Dict[str, Any]) -> Optional[BaseStabilityCalculator]:
        """
        Creates a stability calculator instance based on the 'stability_calculator' config section.
        Returns None if disabled or configuration is invalid.

        Args:
            stability_config: The 'stability_calculator' section of the reward_setup config.

        Returns:
            An instance of a BaseStabilityCalculator subclass, or None.
        """
        logger.debug("Attempting to create StabilityCalculator...")
        calculator: Optional[BaseStabilityCalculator] = None

        try:
            # stability_config = reward_setup_config.get('stability_calculator', {}) # No, recibe ya la sección
            if not stability_config or not isinstance(stability_config, dict):
                logger.info("Stability calculator section missing or invalid in config. Calculator disabled.")
                return None
            if not stability_config.get('enabled', False):
                logger.info("Stability calculator is disabled in the configuration.")
                return None

            calc_type = stability_config.get('type')
            if not calc_type:
                 logger.error("Stability calculator enabled, but 'type' is not specified.")
                 return None

            logger.info(f"Stability calculator enabled. Type requested: {calc_type}")

            # --- Get parameters based on type ---
            params_key_map = {
                 'ira_instantaneous': 'ira_params',
                 'simple_exponential': 'simple_exponential_params',
            }
            params_key = params_key_map.get(calc_type)
            if params_key is None:
                 logger.error(f"Unknown stability calculator type specified: '{calc_type}'.")
                 return None

            params = stability_config.get(params_key)
            if params is None or not isinstance(params, dict):
                 logger.error(f"Parameters section ('{params_key}') invalid or not found for calculator type '{calc_type}'.")
                 return None

            # --- Create instance ---
            logger.debug(f"Creating StabilityCalculator '{calc_type}' with params: {list(params.keys())}")
            if calc_type == 'ira_instantaneous':
                calculator = IRAStabilityCalculator(params)
            elif calc_type == 'simple_exponential':
                calculator = SimpleExponentialStabilityCalculator(params)
            else:
                logger.error(f"Logic error: Unknown stability calculator type '{calc_type}' reached instance creation.")
                return None

            logger.info(f"Successfully created stability calculator: {type(calculator).__name__}")
            return calculator

        except (KeyError, ValueError, TypeError) as e:
             logger.error(f"Error processing stability calculator config for type '{calc_type}': {e}", exc_info=True)
             return None # Devuelve None en error de config
        except Exception as e:
            logger.error(f"Failed to create stability calculator '{calc_type}': {e}", exc_info=True)
            return None # Devuelve None en error inesperado

    def create_reward_function(self,
                               reward_setup_config: Dict[str, Any], # Pasar toda la sección reward_setup
                               stability_calculator: Optional[BaseStabilityCalculator]
                               ) -> RewardFunction:
        """
        Creates a reward function instance (InstantaneousRewardCalculator), injecting the
        pre-resolved stability calculator and the relevant calculation config.

        Args:
            reward_setup_config: The full 'reward_setup' section from the main config.
            stability_calculator: The resolved stability calculator instance (can be None).

        Returns:
            An instance of InstantaneousRewardCalculator.

        Raises:
            ValueError: If configuration is invalid.
            RuntimeError: For unexpected errors during creation.
        """
        logger.debug("Attempting to create RewardFunction (InstantaneousRewardCalculator)...")
        reward_function: Optional[RewardFunction] = None

        try:
            # Extraer la sub-sección 'calculation' para pasarla al constructor
            calculation_config = reward_setup_config.get('calculation')
            if not calculation_config or not isinstance(calculation_config, dict):
                raise ValueError("Configuration missing 'reward_setup.calculation' section or it's not a dictionary.")

            reward_method = calculation_config.get('method')
            logger.info(f"Reward function calculation method requested: {reward_method}")

            # --- Crear instancia ---
            # Actualmente solo existe InstantaneousRewardCalculator
            # pero la estructura permite añadir otros si fuera necesario.

            # Pasar solo la config de 'calculation' y el stability calculator resuelto
            logger.debug(f"Creating InstantaneousRewardCalculator with stability_calculator: {type(stability_calculator).__name__ if stability_calculator else 'None'}")
            reward_function = InstantaneousRewardCalculator(
                calculation_config=calculation_config,
                stability_calculator=stability_calculator
            )

            # --- Añadir otros tipos de RewardFunction aquí si se crean en el futuro ---
            # elif reward_method == 'other_reward_type':
            #     other_params = calculation_config.get('other_params', {})
            #     reward_function = OtherRewardFunction(...)

            if reward_function is None: # Si ninguna condición coincide (no debería pasar con validación config)
                 raise ValueError(f"Could not determine RewardFunction type for method '{reward_method}'")

            logger.info(f"Successfully created reward function: {type(reward_function).__name__} (using method: {reward_method})")
            return reward_function

        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.error(f"Error creating RewardFunction: {e}", exc_info=True)
            raise ValueError(f"Invalid configuration or component for RewardFunction: {e}") from e
        except Exception as e:
            logger.error(f"Failed to create reward function: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error creating reward function") from e