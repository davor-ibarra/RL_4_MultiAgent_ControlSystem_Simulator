import logging
from typing import Dict, Any, Optional

# Import interfaces and components
from interfaces.stability_calculator import BaseStabilityCalculator
from interfaces.reward_function import RewardFunction
# Import specific implementations
from components.analysis.ira_stability_calculator import IRAStabilityCalculator
from components.analysis.simple_exponential_stability_calculator import SimpleExponentialStabilityCalculator
from components.rewards.gaussian_reward import GaussianReward

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class RewardFactory:
    """
    Factory class (as a service) for creating Stability Calculators and Reward Functions.
    Instances of this factory will be registered in the DI container.
    """
    def __init__(self):
        """Constructor (puede estar vacío si la fábrica no necesita dependencias)."""
        logger.info("RewardFactory instance created.")
        pass

    def create_stability_calculator(self, reward_setup_config: Dict[str, Any]) -> Optional[BaseStabilityCalculator]:
        """
        Creates a stability calculator instance based on the 'stability_calculator'
        section within the 'reward_setup' config.
        Returns None if disabled or configuration is invalid.
        This is now an instance method.

        Args:
            reward_setup_config: The 'reward_setup' section of the main config.

        Returns:
            An instance of a BaseStabilityCalculator subclass, or None.
        """
        logger.debug("Attempting to create StabilityCalculator...")
        try:
            stability_config = reward_setup_config.get('stability_calculator', {})
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
            params: Optional[Dict] = None
            if calc_type == 'ira_instantaneous':
                 params = stability_config.get('ira_params')
            elif calc_type == 'simple_exponential':
                 params = stability_config.get('simple_exponential_params')
            # Add other types here
            # elif calc_type == 'other_calc':
            #    params = stability_config.get('other_params')

            if params is None:
                 # This case handles both unknown type and missing params section for a known type
                 logger.error(f"Parameters section ('{calc_type}_params') not found or calculator type '{calc_type}' is unknown.")
                 return None
            if not isinstance(params, dict):
                 logger.error(f"Parameters section for calculator type '{calc_type}' is not a valid dictionary.")
                 return None

            # --- Create instance ---
            calculator: Optional[BaseStabilityCalculator] = None
            logger.debug(f"Creating StabilityCalculator '{calc_type}' with params: {params.keys()}")
            if calc_type == 'ira_instantaneous':
                calculator = IRAStabilityCalculator(params)
            elif calc_type == 'simple_exponential':
                calculator = SimpleExponentialStabilityCalculator(params)
            # Add other calculator types here if needed
            # elif calc_type == 'other_type':
            #     calculator = OtherCalculator(params)
            else:
                # Should have been caught by params check, but for safety
                logger.error(f"Unknown stability calculator type reached instance creation: {calc_type}")
                return None

            logger.info(f"Successfully created stability calculator: {type(calculator).__name__}")
            return calculator

        except KeyError as e:
             # This indicates an issue within the specific calculator's __init__ accessing params
             logger.error(f"Missing required key within params for stability calculator '{calc_type}': {e}", exc_info=True)
             return None
        except ValueError as e:
             # This indicates a validation error within the specific calculator's __init__
             logger.error(f"Configuration value error for stability calculator '{calc_type}': {e}", exc_info=True)
             return None
        except Exception as e:
            logger.error(f"Failed to create stability calculator '{calc_type}': {e}", exc_info=True)
            return None


    def create_reward_function(self,
                               reward_setup_config: Dict[str, Any],
                               stability_calculator: Optional[BaseStabilityCalculator]
                               ) -> RewardFunction:
        """
        Creates a reward function instance based on the configuration, injecting the
        pre-resolved stability calculator instance (which might be None).
        This is now an instance method.

        Args:
            reward_setup_config: The 'reward_setup' section of the main config.
            stability_calculator: The resolved stability calculator instance (can be None).

        Returns:
            An instance of a RewardFunction subclass.

        Raises:
            ValueError: If the reward function type is unknown or configuration is invalid.
            RuntimeError: For unexpected errors during creation.
        """
        logger.debug("Attempting to create RewardFunction...")
        try:
            calc_config = reward_setup_config.get('calculation')
            if not calc_config or not isinstance(calc_config, dict):
                raise ValueError("Configuration missing 'reward_setup.calculation' section or it's not a dictionary.")

            reward_type = calc_config.get('method') # e.g., 'gaussian' or 'stability_calculator'
            if not reward_type:
                 raise ValueError("Missing 'method' in 'reward_setup.calculation' config section.")

            logger.info(f"Reward function calculation method requested: {reward_type}")

            reward_function: Optional[RewardFunction] = None

            # Prepare the init_config dict expected by GaussianReward constructor
            # This centralizes how GaussianReward gets its parameters, regardless of method
            gaussian_init_config = {
                'params': calc_config.get('gaussian_params', {}), # Weights/scales live here
                'use_stability_based_reward': (reward_type == 'stability_calculator'),
                # Pass stability calculator config subsection for potential future use
                'stability_calculator_config': reward_setup_config.get('stability_calculator', {})
            }

            if reward_type == 'gaussian':
                # Pass the potentially None stability calculator
                logger.debug(f"Creating GaussianReward (Gaussian method) with stability_calculator: {type(stability_calculator).__name__}")
                reward_function = GaussianReward(gaussian_init_config, stability_calculator)

            elif reward_type == 'stability_calculator':
                 if stability_calculator is None:
                      # Critical error: requested stability method but no calculator available
                      logger.error("Reward calculation method is 'stability_calculator', but no stability calculator instance was provided or enabled.")
                      raise ValueError("Cannot create reward function: 'stability_calculator' method requires an enabled stability calculator.")

                 # Create GaussianReward instance but configured to use stability method
                 logger.debug(f"Creating GaussianReward (Stability method) with stability_calculator: {type(stability_calculator).__name__}")
                 reward_function = GaussianReward(gaussian_init_config, stability_calculator)

            # Add other reward function types here if needed
            # elif reward_type == 'other_reward':
            #     other_params = calc_config.get('other_params', {})
            #     reward_function = OtherReward(other_params, stability_calculator)

            else:
                raise ValueError(f"Unknown reward calculation method specified: {reward_type}")

            logger.info(f"Successfully created reward function: {type(reward_function).__name__} (using method: {reward_type})")
            return reward_function

        except KeyError as e:
             # This indicates an issue within the specific reward function's __init__ accessing params
             logger.error(f"Missing required key in config for reward function method '{reward_type}': {e}", exc_info=True)
             raise ValueError(f"Configuration error for reward function '{reward_type}'") from e
        except ValueError as e: # Catch ValueErrors from checks or unknown type
            logger.error(f"Configuration error for reward function method '{reward_type}': {e}", exc_info=True)
            raise # Re-raise known config/value errors
        except TypeError as e:
             # E.g., if stability_calculator has the wrong type
             logger.error(f"Type error creating reward function '{reward_type}': {e}", exc_info=True)
             raise TypeError(f"Component type mismatch for reward function '{reward_type}'.") from e
        except Exception as e:
            logger.error(f"Failed to create reward function using method '{reward_type}': {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error creating reward function '{reward_type}'") from e