# factories/reward_factory.py
import logging
from typing import Dict, Any, Optional, Callable

from interfaces.stability_calculator import BaseStabilityCalculator
from interfaces.reward_function import RewardFunction
from interfaces.reward_strategy import RewardStrategy

# Estrategias Base (Se registran para disponibilidad fallback)
from components.reward_strategies.base_reward_strategy import BaseRewardStrategy
from components.reward_strategies.shadow_baseline_reward_strategy import ShadowBaselineRewardStrategy
from components.reward_strategies.echo_baseline_reward_strategy import EchoBaselineRewardStrategy

logger = logging.getLogger(__name__)

class NullStabilityCalculator(BaseStabilityCalculator):
    """Implementación nula de BaseStabilityCalculator."""
    def __init__(self, params: Optional[Dict]=None): # Acepta params pero se ignoran
        logger.info("[NullStabilityCalculator] Instance created. No stability calculations will be performed.")
    def calculate_instantaneous_stability(self, state: Any) -> float: 
        return 1.0 # Neutral stability
    def calculate_stability_based_reward(self, state: Any) -> float: 
        return 0.0 # Neutral reward
    def update_calculator_stats(self, episode_metrics_dict: Dict, current_episode: int): 
        return None
    def get_current_adaptive_stats(self) -> Dict: 
        return {}

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
            error_msg = f"Unknown reward_strategy type: '{strategy_type_name}'. Available: {list(self._reward_strategy_creators.keys())}"
            logger.error(f"[RewardFactory] {error_msg}")
            raise ValueError(error_msg)
        # Obtener el bloque de parámetros específico para la estrategia elegida.
        try:
            # Desempaquetar los parámetros directamente en el constructor de la estrategia
            return strategy_creator_func(**strategy_params)
        except (ValueError, TypeError) as e:
            logger.error(f"Error constructing RewardStrategy '{strategy_type_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to create RewardStrategy '{strategy_type_name}': {e}") from e