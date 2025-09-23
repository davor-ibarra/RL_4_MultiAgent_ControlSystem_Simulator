# factories/reward_factory.py
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

# 13.1: Usar logger específico del módulo
logger = logging.getLogger(__name__)

class RewardFactory:
    """
    Factory service for creating Stability Calculators and Reward Functions.
    """
    def __init__(self):
        logger.info("RewardFactory instance created.")
        pass

    def create_stability_calculator(self, stability_config: Dict[str, Any]) -> Optional[BaseStabilityCalculator]:
        """
        Creates a stability calculator instance based on the 'stability_calculator' config section.
        Returns None if disabled or configuration is invalid/missing.

        Args:
            stability_config: The 'stability_calculator' subsection from reward_setup config.

        Returns:
            An instance of a BaseStabilityCalculator subclass, or None.
        """
        logger.debug("RewardFactory: Attempting to create StabilityCalculator...")
        # 2.2: Refinamiento de validaciones si está HABILITADO.
        if not isinstance(stability_config, dict) or not stability_config:
            logger.info("RewardFactory: Stability calculator config sección ausente/inválida. No se crea.")
            return None

        is_enabled = stability_config.get('enabled', False)
        if not isinstance(is_enabled, bool):
            logger.warning("RewardFactory: 'enabled' en stability_config no es booleano. Asumiendo deshabilitado.")
            is_enabled = False

        if not is_enabled:
            logger.info("RewardFactory: Stability calculator DESHABILITADO en config.")
            return None

        # --- Si está HABILITADO, 'type' y 'params' son mandatorios ---
        logger.info("RewardFactory: Stability calculator HABILITADO. Procediendo a crear...")
        calc_type = stability_config.get('type')
        if not calc_type or not isinstance(calc_type, str):
            raise ValueError("Stability calculator HABILITADO, pero 'type' falta o no es un string.")

        # Determinar la clave de los parámetros según el tipo
        params_key_map = {
            'ira_instantaneous': 'ira_params',
            'simple_exponential': 'simple_exponential_params',
        }
        params_key = params_key_map.get(calc_type)
        if params_key is None:
            raise ValueError(f"Tipo de stability calculator desconocido: '{calc_type}'. Opciones: {list(params_key_map.keys())}")

        params = stability_config.get(params_key) # Obtener params
        if params is None or not isinstance(params, dict): # Params debe ser un dict, no puede ser None si está habilitado
            raise ValueError(f"Sección de parámetros '{params_key}' para stability calculator HABILITADO tipo '{calc_type}' falta o no es un diccionario.")

        # --- Crear instancia ---
        calculator: Optional[BaseStabilityCalculator] = None
        try:
            logger.debug(f"RewardFactory: Creando StabilityCalculator '{calc_type}' con params (keys): {list(params.keys())}")
            if calc_type == 'ira_instantaneous':
                calculator = IRAStabilityCalculator(params)
            elif calc_type == 'simple_exponential':
                calculator = SimpleExponentialStabilityCalculator(params)
            # --- Añadir otros tipos aquí ---
            # elif calc_type == 'other_calculator':
            #     calculator = OtherStabilityCalculator(params)
            else:
                # Esta rama no debería alcanzarse debido a la validación de params_key
                raise ValueError(f"Lógica interna: Tipo de stability calculator '{calc_type}' no manejado después de validación.")

            logger.info(f"RewardFactory: Stability calculator '{type(calculator).__name__}' creado exitosamente.")
            return calculator
        except (ValueError, TypeError) as e_constr: # Errores del constructor del calculator
            logger.error(f"RewardFactory: Error en constructor de '{calc_type}' (params={params}): {e_constr}", exc_info=True)
            raise ValueError(f"Error en parámetros para '{calc_type}': {e_constr}") from e_constr
        except Exception as e:
            logger.error(f"RewardFactory: Error inesperado creando stability calculator '{calc_type}': {e}", exc_info=True)
            raise RuntimeError(f"Error inesperado creando stability calculator '{calc_type}'") from e

    def create_reward_function(self,
                               reward_setup_config: Dict[str, Any], # Recibe toda la sección reward_setup
                               stability_calculator: Optional[BaseStabilityCalculator] # Inyectado
                               ) -> RewardFunction:
        """
        Creates a reward function instance. Currently only supports
        InstantaneousRewardCalculator.

        Args:
            reward_setup_config: The full 'reward_setup' section from config.
            stability_calculator: The resolved stability calculator instance (can be None).

        Returns:
            An instance of a RewardFunction implementation.

        Raises:
            ValueError: If configuration is invalid or essential parts are missing.
            TypeError: If types in config are wrong.
            RuntimeError: For unexpected errors.
        """
        logger.debug("Attempting to create RewardFunction...")
        reward_function: Optional[RewardFunction] = None

        try:
            # 13.4: Extraer la sub-sección 'calculation'
            calculation_config = reward_setup_config.get('calculation')
            if not calculation_config or not isinstance(calculation_config, dict):
                # Fail-Fast si falta la sección de cálculo
                raise ValueError("Configuration missing 'reward_setup.calculation' section or it's not a dictionary.")

            reward_method = calculation_config.get('method')
            if not reward_method:
                 raise ValueError("Missing 'method' in 'reward_setup.calculation' config.")
            logger.info(f"Reward function calculation method requested: {reward_method}")

            logger.debug(f"create_stability_calculator -> Stability calculator instance is {stability_calculator}")
            # --- Crear instancia (Actualmente solo InstantaneousRewardCalculator) ---
            # La validación de params específicos ('gaussian_params') la hace el constructor
            logger.debug(f"Creating InstantaneousRewardCalculator (stability_calc: {type(stability_calculator).__name__ if stability_calculator else 'None'})")
            reward_function = InstantaneousRewardCalculator(
                calculation_config=calculation_config, # Pasar sub-diccionario
                stability_calculator=stability_calculator # Pasar instancia inyectada
            )

            # --- Añadir otros tipos de RewardFunction aquí si se crean ---
            # elif reward_method == 'other_reward_type':
            #     other_params = calculation_config.get('other_params', {})
            #     reward_function = OtherRewardFunction(...)

            if reward_function is None: # Si ninguna condición coincide
                 # Fail-Fast si el método es desconocido (aunque config_loader ya lo valida)
                 raise ValueError(f"Unknown or unhandled reward calculation method: '{reward_method}'")

            logger.info(f"Successfully created reward function: {type(reward_function).__name__} (using method: {reward_method})")
            return reward_function

        # 13.5: Capturar errores específicos y relanzar (RewardFunction es esencial)
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.critical(f"Error crítico creando RewardFunction: {e}", exc_info=True)
            # Relanzar como RuntimeError para indicar fallo crítico
            raise RuntimeError(f"Failed to create RewardFunction: {e}") from e
        except Exception as e:
            logger.critical(f"Unexpected error creating reward function: {e}", exc_info=True)
            raise RuntimeError("Unexpected error creating reward function") from e