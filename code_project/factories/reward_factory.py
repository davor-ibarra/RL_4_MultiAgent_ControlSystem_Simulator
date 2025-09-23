# factories/reward_factory.py
import logging
from typing import Dict, Any, Optional

from interfaces.stability_calculator import BaseStabilityCalculator
from interfaces.reward_function import RewardFunction
from components.analysis.ira_stability_calculator import IRAStabilityCalculator
from components.analysis.simple_exponential_stability_calculator import SimpleExponentialStabilityCalculator
from components.rewards.instantaneous_reward_calculator import InstantaneousRewardCalculator

logger = logging.getLogger(__name__)

class RewardFactory:
    def __init__(self):
        logger.info("[RewardFactory] Instance created.")

    def create_stability_calculator(self, stability_config: Dict[str, Any]) -> Optional[BaseStabilityCalculator]:
        logger.debug(f"[RewardFactory:create_stability_calculator] Attempting with config: {list(stability_config.keys()) if stability_config else 'None'}")

        # stability_config es la sección environment.reward_setup.calculation.stability_calculator
        if not isinstance(stability_config, dict) or not stability_config:
            logger.info("[RewardFactory:create_stability_calculator] Config section absent/invalid. No instance created.")
            return None

        calc_type = stability_config.get('type')
        if not calc_type or not isinstance(calc_type, str):
            logger.info(f"[RewardFactory:create_stability_calculator] 'type' missing or invalid in stability_config. No instance created (Type: {calc_type}).")
            return None # No se crea si el tipo no está definido

        # Obtener la clave del bloque de parámetros específico para este tipo de calculador
        # Ej: si type='ira_instantaneous', params_key='ira_params'
        params_key = f"{calc_type}_params"
        calculator_specific_params = stability_config.get(params_key)

        if not isinstance(calculator_specific_params, dict):
            # Si el tipo está definido, sus parámetros deben existir y ser un diccionario.
            msg = f"Parameters section '{params_key}' for stability calculator type '{calc_type}' is missing or not a dictionary."
            logger.error(f"[RewardFactory:create_stability_calculator] {msg}")
            raise ValueError(msg) # Fail-Fast si los params específicos no son un dict

        calculator: Optional[BaseStabilityCalculator] = None
        logger.info(f"[RewardFactory:create_stability_calculator] Creating StabilityCalculator of type: '{calc_type}' using params from '{params_key}'.")
        try:
            if calc_type == 'ira_instantaneous':
                calculator = IRAStabilityCalculator(calculator_specific_params)
            elif calc_type == 'simple_exponential':
                calculator = SimpleExponentialStabilityCalculator(calculator_specific_params)
            # --- Añadir otros tipos de calculadores aquí ---
            # elif calc_type == 'other_calculator_type':
            #     calculator = OtherCalculatorClass(calculator_specific_params)
            else:
                raise ValueError(f"Unknown stability calculator type: '{calc_type}'")

            logger.info(f"[RewardFactory:create_stability_calculator] Stability calculator '{type(calculator).__name__}' created.")
            return calculator
        except (ValueError, TypeError) as e_constr:
            # Errores de validación desde el constructor del calculador
            logger.error(f"[RewardFactory:create_stability_calculator] Error constructing '{calc_type}' with params from '{params_key}': {e_constr}", exc_info=True)
            raise # Re-lanzar para que DI falle
        except Exception as e_unexp:
            logger.error(f"[RewardFactory:create_stability_calculator] Unexpected error creating '{calc_type}': {e_unexp}", exc_info=True)
            raise RuntimeError(f"Unexpected error creating stability calculator '{calc_type}'") from e_unexp

    def create_reward_function(self,
                               reward_setup_config: Dict[str, Any], # environment.reward_setup
                               stability_calculator: Optional[BaseStabilityCalculator]
                               ) -> RewardFunction:
        logger.debug(f"[RewardFactory:create_reward_function] Attempting with reward_setup_config keys: {list(reward_setup_config.keys())}")
        reward_function: Optional[RewardFunction] = None

        # Extraer la sub-sección 'calculation' que necesita InstantaneousRewardCalculator
        calculation_config = reward_setup_config.get('calculation')
        if not isinstance(calculation_config, dict):
            raise ValueError("Config 'reward_setup.calculation' section missing or not a dictionary.")

        # Actualmente, solo se soporta InstantaneousRewardCalculator que usa 'method'
        # Si se añaden otras clases de RewardFunction, aquí se podría despachar por un 'type'
        # en reward_setup_config a nivel raíz, o mantenerlo simple si Instantaneous... es el orquestador.
        try:
            logger.info(f"[RewardFactory:create_reward_function] Creating InstantaneousRewardCalculator (StabilityCalc: {type(stability_calculator).__name__ if stability_calculator else 'None'}).")
            reward_function = InstantaneousRewardCalculator(
                calculation_config=calculation_config, # Pasa la sub-sección
                stability_calculator=stability_calculator
            )
            logger.info(f"[RewardFactory:create_reward_function] Reward function '{type(reward_function).__name__}' created (method: {calculation_config.get('method')}).")
            return reward_function
        except (ValueError, TypeError, AttributeError) as e_constr:
            # Errores de validación desde el constructor de InstantaneousRewardCalculator
            logger.critical(f"[RewardFactory:create_reward_function] Error constructing RewardFunction: {e_constr}", exc_info=True)
            raise RuntimeError(f"Failed to create RewardFunction: {e_constr}") from e_constr
        except Exception as e_unexp:
            logger.critical(f"[RewardFactory:create_reward_function] Unexpected error creating RewardFunction: {e_unexp}", exc_info=True)
            raise RuntimeError("Unexpected error creating RewardFunction") from e_unexp