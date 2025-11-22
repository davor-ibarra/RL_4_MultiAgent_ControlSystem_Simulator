# components/reward_strategies/echo_baseline_reward_strategy.py

from interfaces.reward_strategy import RewardStrategy
from typing import Dict, Any, Optional, Tuple, List
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class EchoBaselineRewardStrategy(RewardStrategy):
    """
    Estrategia Echo Baseline (Contrafactual).
    Se usa un simulador virtual para calcular qué hubiera pasado si NO se cambiaba la ganancia.
    R_learn = R_real - R_virtual (Counterfactual)
    """
    needs_virtual_simulation: bool = True       # Activa el VirtualSimulator en SimulationManager
    required_auxiliary_tables: List[str] = []   # No usa tablas auxiliares del agente

    def __init__(self, **strategy_specific_params: Any):
        # logger.info("[EchoBaselineStrategy] Initialized.")
        if strategy_specific_params:
            logger.warning(f"[EchoBaselineStrategy] Received unused parameters: {list(strategy_specific_params.keys())}")

    def compute_reward_for_learning(
        self, 
        gain_id: str,
        agent_instance: Dict[str, Any],
        controllers_dict: Dict[str, Any],
        current_agent_s_dict: Dict[str, Any],
        current_s_indices: tuple,
        actions_taken_map: Dict[str, int],  
        action_idx_for_gain: int,                                           # ???
        real_interval_reward: float,                                        # R_real del intervalo (usado por SimMan para calcular R_diff)
        avg_interval_stability_score: float,                                # W_stab promedio del intervalo (no usado directamente aquí)
        differential_rewards_map: Optional[Dict[str, float]]=None,          # Contiene R_diff[gain] = R_real - R_cf_gain
        **kwargs: Any
    ) -> float:

        if differential_rewards_map:
            for key, val in differential_rewards_map.items():
                if key in gain_id:
                    return val
        # Fallback al reward real si no hay contrafactual
        return float(real_interval_reward)