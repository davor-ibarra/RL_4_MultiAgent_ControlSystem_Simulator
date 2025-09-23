# components/reward_strategies/global_reward_strategy.py
from interfaces.reward_strategy import RewardStrategy
from typing import Dict, Any, Optional, TYPE_CHECKING, Tuple, Union
import logging
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from interfaces.rl_agent import RLAgent
    from interfaces.controller import Controller

logger = logging.getLogger(__name__) # Logger específico del módulo

class GlobalRewardStrategy(RewardStrategy):
    # Atributo declarativo para SimulationManager
    needs_virtual_simulation: bool = False

    def __init__(self, **strategy_specific_params: Any):
        # strategy_specific_params sería config['environment']['reward_setup']['reward_strategy']['strategy_params']['global']
        logger.info("[GlobalRewardStrategy] Initialized.")
        if strategy_specific_params: # Global no espera params específicos
            logger.warning(f"[GlobalRewardStrategy] Received unexpected parameters: {strategy_specific_params.keys()}")

    def compute_reward_for_learning(
        self, gain: str, agent: 'RLAgent', controller: 'Controller',
        current_agent_state_dict: Dict[str, Any], current_state_indices: tuple,
        actions_dict: Dict[str, int], action_taken_idx: int,
        interval_reward: float, avg_w_stab: float,
        reward_dict: Optional[Dict[str, float]],
        **kwargs
    ) -> float:
        #logger.debug(f"[GlobalRewardStrategy:compute_reward] R_learn = R_real = {interval_reward:.4f} for gain '{gain}'.")
        if pd.notna(interval_reward) and np.isfinite(interval_reward):
            return float(interval_reward)
        else:
            logger.warning(f"[GlobalRewardStrategy:compute_reward] Invalid interval_reward ({interval_reward}). Using 0.0 for R_learn.")
            return 0.0