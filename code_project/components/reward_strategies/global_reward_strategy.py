# components/reward_strategies/global_reward_strategy.py
# (Anteriormente, este podría haber sido WeightedSumFeaturesRewardStrategy si la recompensa se construía aquí.
#  Ahora, con InstantaneousRewardCalculator, esta estrategia es la más simple.)

from interfaces.reward_strategy import RewardStrategy
from typing import Dict, Any, Optional, TYPE_CHECKING, Tuple, List # List
import logging
import numpy as np # Para np.nan, np.isfinite
import pandas as pd # Para pd.notna

if TYPE_CHECKING:
    from interfaces.rl_agent import RLAgent
    from interfaces.controller import Controller

logger = logging.getLogger(__name__)

class GlobalRewardStrategy(RewardStrategy):
    # Atributos declarativos
    needs_virtual_simulation: bool = False
    required_auxiliary_tables: List[str] = [] # Global no usa tablas auxiliares del agente

    def __init__(self, **strategy_specific_params: Any):
        # No espera parámetros específicos, pero el **kwargs permite flexibilidad si se añade algo en el futuro.
        # logger.info("[GlobalRewardStrategy] Initialized.")
        if strategy_specific_params:
            logger.warning(f"[GlobalRewardStrategy] Received unexpected parameters: {list(strategy_specific_params.keys())} which are not used by this strategy.")

    def compute_reward_for_learning(
        self, 
        gain_id: str,
        agent_instance: 'RLAgent',
        controllers_dict: Dict[str, 'Controller'],
        current_agent_s_dict: Dict[str, Any],
        current_s_indices: tuple, # Tupla de índices para S
        actions_taken_map: Dict[str, int],
        action_idx_for_gain: int, 
        real_interval_reward: float, # R_real acumulada del intervalo
        avg_interval_stability_score: float, # W_stab promedio del intervalo
        differential_rewards_map: Optional[Dict[str, float]], # Para Echo, aquí sería None
        **kwargs: Any # Futura extensibilidad
    ) -> float:
        # Para GlobalRewardStrategy, R_learn es simplemente la recompensa real del intervalo.
        # Se asume que real_interval_reward es un float. Si es NaN/inf, se devuelve 0.0.
        if pd.notna(real_interval_reward) and np.isfinite(real_interval_reward):
            return float(real_interval_reward)
        else:
            # logger.warning(f"[GlobalRewardStrategy:compute_reward] Invalid real_interval_reward ({real_interval_reward}) for gain '{gain_id}'. Using 0.0 for R_learn.")
            return 0.0