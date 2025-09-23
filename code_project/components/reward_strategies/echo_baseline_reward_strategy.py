# components/reward_strategies/echo_baseline_reward_strategy.py
from interfaces.reward_strategy import RewardStrategy
from typing import Dict, Any, Optional, TYPE_CHECKING, Tuple, Union
import logging
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from interfaces.rl_agent import RLAgent
    from interfaces.controller import Controller

logger = logging.getLogger(__name__) # Logger específico del módulo

class EchoBaselineRewardStrategy(RewardStrategy):
    # Atributo declarativo para SimulationManager
    needs_virtual_simulation: bool = True

    def __init__(self, **strategy_specific_params: Any):
        # strategy_specific_params sería config['...']['strategy_params']['echo_baseline']
        logger.info("[EchoBaselineStrategy] Initialized.")
        if strategy_specific_params: # Echo no espera params específicos
            logger.warning(f"[EchoBaselineStrategy] Received unexpected parameters: {strategy_specific_params.keys()}")

    def compute_reward_for_learning(
        self, gain: str, agent: 'RLAgent', controller: 'Controller',
        current_agent_state_dict: Dict[str, Any], current_state_indices: tuple,
        actions_dict: Dict[str, int], action_taken_idx: int,
        interval_reward: float, avg_w_stab: float, # Ignorados, R_diff se usa
        reward_dict: Optional[Dict[str, float]],  # R_diff_dict = {kp: R_real-R_cf_kp, ...}
        **kwargs
    ) -> float:
        if reward_dict is None:
            logger.error(f"[EchoBaselineStrategy:compute_reward] R_diff dict is None for gain '{gain}'. Cannot compute R_learn. Returning 0.0.")
            return 0.0
        if not isinstance(reward_dict, dict):
             logger.error(f"[EchoBaselineStrategy:compute_reward] R_diff (reward_dict) is not a dict ({type(reward_dict)}). Using 0.0."); return 0.0

        diff_reward_for_gain = reward_dict.get(gain)

        if diff_reward_for_gain is None:
            logger.warning(f"[EchoBaselineStrategy:compute_reward] R_diff for gain '{gain}' not found in reward_dict. Keys: {list(reward_dict.keys())}. Using 0.0.")
            return 0.0
        elif pd.isna(diff_reward_for_gain) or not np.isfinite(diff_reward_for_gain):
             logger.warning(f"[EchoBaselineStrategy:compute_reward] R_diff for gain '{gain}' is invalid ({diff_reward_for_gain}). Using 0.0.")
             return 0.0
        else:
            #logger.debug(f"[EchoBaselineStrategy:compute_reward] R_learn = R_diff = {diff_reward_for_gain:.4f} for gain '{gain}'.")
            return float(diff_reward_for_gain)