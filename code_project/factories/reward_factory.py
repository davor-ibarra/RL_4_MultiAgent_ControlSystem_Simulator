import logging
from typing import Dict, Any, Optional

# Import interfaces and components
from interfaces.stability_calculator import BaseStabilityCalculator
from interfaces.reward_function import RewardFunction
from components.analysis.ira_stability_calculator import IRAStabilityCalculator
from components.analysis.simple_exponential_stability_calculator import SimpleExponentialStabilityCalculator
from components.rewards.gaussian_reward import GaussianReward

class RewardFactory:
    """
    Factory class for creating Stability Calculators and Reward Functions.
    """
    @staticmethod
    def create_stability_calculator(reward_setup_config: Dict[str, Any]) -> Optional[BaseStabilityCalculator]:
        """
        Creates a stability calculator instance based on the configuration.

        Args:
            reward_config: The 'reward' section of the main configuration dictionary.

        Returns:
            An instance of a BaseStabilityCalculator subclass, or None if disabled or type unknown.
        """
        stability_config = reward_setup_config.get('stability_calculator', {})
        if not stability_config or not stability_config.get('enabled', False):
            logging.info("Stability calculator is disabled in the configuration.")
            return None

        calc_type = stability_config.get('type')
        # --- Obtener params según el tipo ---
        params = {}
        if calc_type == 'ira_instantaneous':
             params = stability_config.get('ira_params', {})
        elif calc_type == 'simple_exponential':
             params = stability_config.get('simple_exponential_params', {})
        else:
             # Log error temprano si el tipo es desconocido y no hay params definidos
             logging.error(f"Unknown stability calculator type specified: {calc_type} or params section missing.")
             return None # Return None for unknown types or missing params section
        
        logging.info(f"Attempting to create stability calculator of type: {calc_type}")

        calculator: Optional[BaseStabilityCalculator] = None
        try:
            if calc_type == 'ira_instantaneous':
                calculator = IRAStabilityCalculator(params)
            elif calc_type == 'simple_exponential':
                calculator = SimpleExponentialStabilityCalculator(params)
            # Add other calculator types here if needed
            # elif calc_type == 'other_type':
            #     calculator = OtherCalculator(params)
            else:
                logging.error(f"Unknown stability calculator type specified: {calc_type}")
                return None # Return None for unknown types

            logging.info(f"Successfully created stability calculator: {type(calculator).__name__}")
            return calculator

        except KeyError as e:
             logging.error(f"Missing required key within params for stability calculator '{calc_type}': {e}", exc_info=True)
             return None
        except ValueError as e:
             logging.error(f"Configuration error for stability calculator '{calc_type}': {e}", exc_info=True)
             return None
        except Exception as e:
            logging.error(f"Failed to create stability calculator '{calc_type}': {e}", exc_info=True)
            return None


    @staticmethod
    def create_reward_function(reward_setup_config: Dict[str, Any],
                               stability_calculator: Optional[BaseStabilityCalculator]) -> RewardFunction:
        """
        Creates a reward function instance based on the configuration, injecting the stability calculator.

        Args:
            reward_config: The 'reward' section of the main configuration dictionary.
            stability_calculator: The pre-created stability calculator instance (can be None).

        Returns:
            An instance of a RewardFunction subclass.

        Raises:
            ValueError: If the reward function type is unknown or configuration is invalid.
        """
        calc_config  = reward_setup_config.get('calculation')
        reward_type = calc_config.get('method') # 'gaussian' o 'stability_calculator'
        logging.info(f"Attempting to create reward function of type: {reward_type}")

        reward_function: Optional[RewardFunction] = None
        try:
            if reward_type == 'gaussian':
                # Pasar los parámetros gaussianos y el stability calculator (puede ser None)
                gaussian_params = calc_config.get('gaussian_params', {})
                # Crear un dict temporal con la info que espera GaussianReward.__init__
                # (incluye weights, scales y si debe usar stability reward - que ahora es implícito por el 'method')
                temp_gaussian_init_config = {
                    'params': gaussian_params, # Contiene weights y scales
                    'use_stability_based_reward': False, # Ya que el method es 'gaussian'
                     # Pasar el config completo de stability_calculator para cualquier referencia futura
                    'stability_calculator_config': reward_setup_config.get('stability_calculator', {})
                }
                reward_function = GaussianReward(temp_gaussian_init_config, stability_calculator)

            elif reward_type == 'stability_calculator':
                 if stability_calculator is None:
                      # Error crítico: se pide usar el calculador pero no existe
                      raise ValueError("Reward calculation method is 'stability_calculator', but no stability calculator instance was provided or enabled.")
                 # Reutilizar GaussianReward pero decirle que use el modo stability
                 temp_gaussian_init_config = {
                     'params': {}, # No necesita weights/scales gaussianos
                     'use_stability_based_reward': True, # Inicializar de todas formas
                     'stability_calculator_config': reward_setup_config.get('stability_calculator', {})
                 }
                 reward_function = GaussianReward(temp_gaussian_init_config, stability_calculator)
            # Add other reward function types here if needed
            # elif reward_type == 'other_reward':
            #     reward_function = OtherReward(...)
            else:
                raise ValueError(f"Unknown reward calculation method specified: {reward_type}")

            logging.info(f"Successfully created reward function: {type(reward_function).__name__} (using method: {reward_type})")
            return reward_function

        except KeyError as e:
             logging.error(f"Missing required key in config for reward function method '{reward_type}': {e}", exc_info=True)
             raise ValueError(f"Configuration error for reward function '{reward_type}'") from e
        except ValueError as e:
            logging.error(f"Configuration error for reward function method '{reward_type}': {e}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Failed to create reward function using method '{reward_type}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to create reward function '{reward_type}'") from e