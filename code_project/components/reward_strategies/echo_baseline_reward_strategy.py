# components/reward_strategies/echo_baseline_reward_strategy.py

from interfaces.reward_strategy import RewardStrategy
from typing import Dict, Any, Optional, TYPE_CHECKING, Tuple, List
import logging
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from interfaces.rl_agent import RLAgent
    from interfaces.controller import Controller

logger = logging.getLogger(__name__)

class EchoBaselineRewardStrategy(RewardStrategy):
    # Atributos declarativos
    needs_virtual_simulation: bool = True # Esta estrategia SÍ necesita simulación virtual
    required_auxiliary_tables: List[str] = [] # Echo no usa tablas auxiliares del agente

    def __init__(self, **strategy_specific_params: Any):
        # logger.info("[EchoBaselineStrategy] Initialized.")
        if strategy_specific_params:
            logger.warning(f"[EchoBaselineStrategy] Received unused parameters: {list(strategy_specific_params.keys())}")

    def compute_reward_for_learning(
        self, 
        gain_id: str,
        agent_instance: 'RLAgent',
        controller_instance: 'Controller',
        current_agent_s_dict: Dict[str, Any],
        current_s_indices: tuple,
        actions_taken_map: Dict[str, int],
        action_idx_for_gain: int,
        real_interval_reward: float, # R_real del intervalo (usado por SimMan para calcular R_diff)
        avg_interval_stability_score: float, # W_stab promedio del intervalo (no usado directamente aquí)
        differential_rewards_map: Optional[Dict[str, float]], # Contiene R_diff[gain] = R_real - R_cf_gain
        **kwargs: Any
    ) -> float:
        
        if differential_rewards_map is None:
            # logger.error(f"[EchoBaselineStrategy:compute_reward] Differential rewards map is None for gain '{gain_id}'. Cannot compute R_learn. Returning 0.0.")
            return 0.0
        
        # R_learn para Echo es R_diff[gain_id]
        r_diff_for_this_gain = differential_rewards_map.get(gain_id)

        if r_diff_for_this_gain is None:
            # logger.warning(f"[EchoBaselineStrategy:compute_reward] R_diff for gain '{gain_id}' not found in differential_rewards_map. Keys: {list(differential_rewards_map.keys())}. Using 0.0.")
            return 0.0
        elif not (pd.notna(r_diff_for_this_gain) and np.isfinite(r_diff_for_this_gain)):
            # logger.warning(f"[EchoBaselineStrategy:compute_reward] R_diff for gain '{gain_id}' is invalid ({r_diff_for_this_gain}). Using 0.0.")
            return 0.0
        else:
            # logger.debug(f"[EchoBaselineStrategy:compute_reward] R_learn = R_diff = {r_diff_for_this_gain:.4f} for gain '{gain_id}'.")
            return float(r_diff_for_this_gain)