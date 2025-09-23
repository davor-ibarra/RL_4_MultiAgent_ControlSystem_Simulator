# factories/reward_factory.py
import logging
from typing import Dict, Any, Optional, Callable # Added Callable

from interfaces.stability_calculator import BaseStabilityCalculator
from interfaces.reward_function import RewardFunction
from interfaces.reward_strategy import RewardStrategy
# No se importan implementaciones concretas aquí directamente.

logger = logging.getLogger(__name__)

# --- Null Object Pattern para StabilityCalculator ---
class NullStabilityCalculator(BaseStabilityCalculator):
    """Una implementación nula de BaseStabilityCalculator."""
    def __init__(self, params: Optional[Dict]=None): # Acepta params pero los ignora
        logger.info("[NullStabilityCalculator] Instance created. No stability calculations will be performed.")
    def calculate_instantaneous_stability(self, state: Any) -> float: return 1.0 # Neutral stability
    def calculate_stability_based_reward(self, state: Any) -> float: return 0.0 # Neutral reward
    def update_reference_stats(self, episode_metrics_dict: Dict, current_episode: int): pass
    def get_current_adaptive_stats(self) -> Dict: return {}

class RewardFactory:
    def __init__(self):
        self._reward_function_creators: Dict[str, Callable[..., RewardFunction]] = {}
        self._reward_strategy_creators: Dict[str, Callable[..., RewardStrategy]] = {}
        logger.info("[RewardFactory] Instance created. Ready to register creators.")
        # Registrar el NullStabilityCalculator por defecto para el caso "no configurado"
        #self.register_stability_calculator_type("__null__", NullStabilityCalculator)

    def register_reward_function_type(self, func_type_name: str, creator_func: Callable[..., RewardFunction]):
        if not isinstance(func_type_name, str) or not func_type_name:
            logger.error("[RewardFactory:register_reward_function] Invalid func_type_name.")
            return
        if not callable(creator_func):
            logger.error(f"[RewardFactory:register_reward_function] Creator for '{func_type_name}' not callable.")
            return
        if func_type_name in self._reward_function_creators:
            logger.warning(f"[RewardFactory:register_reward_function] Overwriting creator for type: {func_type_name}")
        self._reward_function_creators[func_type_name] = creator_func
        logger.info(f"[RewardFactory:register_reward_function] RewardFunction type '{func_type_name}' registered.")

    def register_reward_strategy_type(self, strategy_type_name: str, creator_func: Callable[..., RewardStrategy]):
        """Registra una función creadora para un tipo de RewardStrategy específico."""
        if not isinstance(strategy_type_name, str) or not strategy_type_name:
            logger.error("[RewardFactory:register_reward_strategy] Invalid strategy_type_name.")
            return
        if not callable(creator_func):
            logger.error(f"[RewardFactory:register_reward_strategy] Creator for '{strategy_type_name}' not callable.")
            return
        if strategy_type_name in self._reward_strategy_creators:
            logger.warning(f"[RewardFactory:register_reward_strategy] Overwriting creator for type: {strategy_type_name}")
        self._reward_strategy_creators[strategy_type_name] = creator_func
        logger.info(f"[RewardFactory:register_reward_strategy] RewardStrategy type '{strategy_type_name}' registered.")
    
    def create_reward_function(self,
                               reward_setup_config: Dict[str, Any], # Ya no es Optional, el helper se encarga
                               stability_calculator_instance: BaseStabilityCalculator
                               ) -> RewardFunction:

        # logger.debug(f"[RewardFactory:create_reward_function] Received reward_function_config (calculation section) keys: {list(reward_function_config.keys())}") # Mantener este log

        calculation_sub_config = reward_setup_config.get('calculation', {})
        rf_type = calculation_sub_config.get('reward_function_type', 'default_instantaneous_reward')
        creator_cls = self._reward_function_creators.get(rf_type)

        if not creator_cls:
            raise ValueError(f"Unknown RewardFunction type: '{rf_type}'")

        return creator_cls(reward_setup_config=reward_setup_config, stability_calculator=stability_calculator_instance)
    
    def create_reward_strategy(self, reward_strategy_config: Dict[str, Any]) -> RewardStrategy:
        """
        Crea una instancia de RewardStrategy basada en la configuración proporcionada.
        
        Args:
            reward_strategy_config (Dict[str, Any]): La sección 'reward_strategy' de la config,
                                                     que incluye 'type' y 'strategy_params'.
        Returns:
            RewardStrategy: Una instancia de la estrategia de recompensa solicitada.
        Raises:
            ValueError: Si el tipo de estrategia es desconocido o la config es inválida.
        """
        logger.debug(f"[RewardFactory:create_reward_strategy] Attempting with config keys: {list(reward_strategy_config.keys())}")

        if not isinstance(reward_strategy_config, dict):
            # Esto debería ser validado por config_loader o el helper del DI.
            raise ValueError("reward_strategy_config must be a dictionary.")

        strategy_type_name_from_config = reward_strategy_config.get('type')
        if not strategy_type_name_from_config or not isinstance(strategy_type_name_from_config, str):
            raise ValueError("Missing or invalid 'type' in reward_strategy_config.")

        strategy_creator_func = self._reward_strategy_creators.get(strategy_type_name_from_config)
        if not strategy_creator_func:
            error_msg_strat_type = f"Unknown reward_strategy.type: '{strategy_type_name_from_config}'. Available types: {list(self._reward_strategy_creators.keys())}"
            logger.error(f"[RewardFactory:create_reward_strategy] {error_msg_strat_type}")
            raise ValueError(error_msg_strat_type)

        # Obtener el bloque de parámetros específico para la estrategia elegida.
        # ej: strategy_params = {'weighted_sum_features': {}, 'shadow_baseline_delta': {'beta': 0.2, ...}, ...}
        all_strategy_specific_params_dict = reward_strategy_config.get('strategy_params', {}) # 'all_strategy_specific_params_dict'
        if not isinstance(all_strategy_specific_params_dict, dict):
            # config_loader debería validar esto.
            raise TypeError("'reward_strategy_config.strategy_params' must be a dictionary.")

        params_for_this_specific_strategy = all_strategy_specific_params_dict.get(strategy_type_name_from_config, {}) # 'params_for_this_specific_strategy'
        if not isinstance(params_for_this_specific_strategy, dict):
            logger.warning(f"[RewardFactory:create_reward_strategy] Params for strategy '{strategy_type_name_from_config}' (under 'strategy_params.{strategy_type_name_from_config}') is not a dictionary. Using {{}}.")
            params_for_this_specific_strategy = {}
        
        logger.info(f"[RewardFactory:create_reward_strategy] Creating RewardStrategy: {strategy_type_name_from_config} with params: {list(params_for_this_specific_strategy.keys())}")
        
        try:
            # El constructor de la estrategia es responsable de validar sus params_for_this_specific_strategy.
            strategy_instance_created = strategy_creator_func(**params_for_this_specific_strategy) # 'strategy_instance_created'
            logger.info(f"[RewardFactory:create_reward_strategy] RewardStrategy '{type(strategy_instance_created).__name__}' (type: {strategy_type_name_from_config}) created successfully.")
            return strategy_instance_created
        except (ValueError, TypeError) as e_constr_strat_factory: # 'e_constr_strat_factory'
            logger.error(f"[RewardFactory:create_reward_strategy] Error constructing RewardStrategy '{strategy_type_name_from_config}': {e_constr_strat_factory}", exc_info=True)
            raise RuntimeError(f"Failed to create RewardStrategy '{strategy_type_name_from_config}': {e_constr_strat_factory}") from e_constr_strat_factory
        except Exception as e_unexp_strat_factory: # 'e_unexp_strat_factory'
            logger.error(f"[RewardFactory:create_reward_strategy] Unexpected error creating RewardStrategy '{strategy_type_name_from_config}': {e_unexp_strat_factory}", exc_info=True)
            raise RuntimeError(f"Unexpected error creating RewardStrategy '{strategy_type_name_from_config}'") from e_unexp_strat_factory