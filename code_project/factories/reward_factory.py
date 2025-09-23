import logging
from typing import Dict, Any, Optional

# Import interfaces and components
from interfaces.stability_calculator import BaseStabilityCalculator
from interfaces.reward_function import RewardFunction
# Import specific implementations
from components.analysis.ira_stability_calculator import IRAStabilityCalculator
from components.analysis.simple_exponential_stability_calculator import SimpleExponentialStabilityCalculator
from components.rewards.gaussian_reward import GaussianReward
# Import other reward/calculator types here
# from components.rewards.other_reward import OtherReward
# from components.analysis.other_calculator import OtherCalculator

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
        calculator: Optional[BaseStabilityCalculator] = None # Initialize

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
            params_key_map = {
                 'ira_instantaneous': 'ira_params',
                 'simple_exponential': 'simple_exponential_params',
                 # 'other_calc': 'other_params', # Add other types here
            }
            params_key = params_key_map.get(calc_type)
            if params_key is None:
                 logger.error(f"Unknown stability calculator type specified: '{calc_type}'.")
                 return None

            params = stability_config.get(params_key)
            if params is None:
                 logger.error(f"Parameters section ('{params_key}') not found for calculator type '{calc_type}'.")
                 return None
            if not isinstance(params, dict):
                 logger.error(f"Parameters section ('{params_key}') for calculator type '{calc_type}' is not a valid dictionary.")
                 return None

            # --- Create instance ---
            logger.debug(f"Creating StabilityCalculator '{calc_type}' with params: {list(params.keys())}")
            if calc_type == 'ira_instantaneous':
                calculator = IRAStabilityCalculator(params)
            elif calc_type == 'simple_exponential':
                calculator = SimpleExponentialStabilityCalculator(params)
            # Add other calculator types here if needed
            # elif calc_type == 'other_type':
            #     calculator = OtherCalculator(params)
            else:
                # Should have been caught earlier, but for safety
                logger.error(f"Logic error: Unknown stability calculator type '{calc_type}' reached instance creation.")
                return None

            logger.info(f"Successfully created stability calculator: {type(calculator).__name__}")
            return calculator

        except KeyError as e:
             logger.error(f"Missing required key within params for stability calculator '{calc_type}': {e}", exc_info=True)
             return None
        except ValueError as e:
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
        reward_function: Optional[RewardFunction] = None # Initialize

        try:
            calc_config = reward_setup_config.get('calculation')
            if not calc_config or not isinstance(calc_config, dict):
                raise ValueError("Configuration missing 'reward_setup.calculation' section or it's not a dictionary.")

            reward_method = calc_config.get('method') # e.g., 'gaussian' or 'stability_calculator'
            if not reward_method:
                 raise ValueError("Missing 'method' in 'reward_setup.calculation' config section.")

            logger.info(f"Reward function calculation method requested: {reward_method}")

            # --- Create Instance Based on Method ---
            # Currently, only GaussianReward exists, but it handles both methods internally.
            # If other RewardFunction types are added, expand this logic.

            if reward_method in ['gaussian', 'stability_calculator']:
                # Prepare the init_config dict expected by GaussianReward constructor
                gaussian_init_config = {
                    'params': calc_config.get('gaussian_params', {}),
                    'use_stability_based_reward': (reward_method == 'stability_calculator'),
                    'stability_calculator_config': reward_setup_config.get('stability_calculator', {})
                }

                # Validate stability calculator presence if needed
                if reward_method == 'stability_calculator' and stability_calculator is None:
                      logger.error("Reward method 'stability_calculator' requires an enabled stability calculator, but none was provided/created.")
                      raise ValueError("Cannot create reward function: 'stability_calculator' method requires an enabled stability calculator.")

                logger.debug(f"Creating GaussianReward (Method: {reward_method}) with stability_calculator: {type(stability_calculator).__name__}")
                reward_function = GaussianReward(gaussian_init_config, stability_calculator)

            # --- Add other reward function types ---
            # elif reward_method == 'other_reward_type':
            #     other_params = calc_config.get('other_params', {})
            #     # Validate params for OtherReward
            #     reward_function = OtherReward(other_params, stability_calculator)

            else:
                raise ValueError(f"Unknown reward calculation method specified: {reward_method}")

            logger.info(f"Successfully created reward function: {type(reward_function).__name__} (using method: {reward_method})")
            return reward_function

        except KeyError as e:
             logger.error(f"Missing required key in config for reward function method '{reward_method}': {e}", exc_info=True)
             raise ValueError(f"Configuration error for reward function '{reward_method}'") from e
        except ValueError as e: # Catch ValueErrors from checks or unknown type
            logger.error(f"Configuration error for reward function method '{reward_method}': {e}", exc_info=True)
            raise
        except TypeError as e:
             logger.error(f"Type error creating reward function '{reward_method}': {e}", exc_info=True)
             raise TypeError(f"Component type mismatch for reward function '{reward_method}'.") from e
        except Exception as e:
            logger.error(f"Failed to create reward function using method '{reward_method}': {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error creating reward function '{reward_method}'") from e