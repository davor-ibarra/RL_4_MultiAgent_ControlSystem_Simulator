# components/reward_strategies/echo_baseline_reward_strategy.py
from interfaces.reward_strategy import RewardStrategy # Importar Interfaz
from typing import Dict, Any, Optional, TYPE_CHECKING, Tuple, Union
import logging
import numpy as np
import pandas as pd

# Evitar importación circular
if TYPE_CHECKING:
    from interfaces.rl_agent import RLAgent # Usar interfaz genérica

logger = logging.getLogger(__name__)

class EchoBaselineRewardStrategy(RewardStrategy): # Implementar Interfaz RewardStrategy
    """
    Estrategia de Recompensa Echo Baseline.
    Utiliza recompensas diferenciales pre-calculadas (R_real - R_counterfactual)
    proporcionadas por el Simulation Manager a través del `reward_dict`.
    R_learn = R_diff_g = R_real - R_cf_g
    """
    def __init__(self, **kwargs): # Aceptar kwargs por si se añaden params
        logger.info("EchoBaselineRewardStrategy inicializada.")
        if kwargs: logger.warning(f"EchoBaseline recibió params inesperados: {kwargs}")

    def compute_reward_for_learning(
        self,
        # --- Context ---
        gain: str,                          # Ganancia actual ('kp', 'ki', 'kd')
        agent: 'RLAgent',                   # Ignorado
        # --- State ---
        current_agent_state_dict: Dict[str, Any], # Ignorado
        current_state_indices: tuple,             # Ignorado
        # --- Action ---
        actions_dict: Dict[str, int],             # Ignorado
        action_taken_idx: int,                    # Ignorado
        # --- Raw Reward/Stability ---
        interval_reward: float,                   # R_real (Ignorado directamente, implícito en R_diff)
        avg_w_stab: float,                        # Ignorado
        # --- Pre-calculated Differential Rewards ---
        reward_dict: Optional[Dict[str, float]],  # R_diff = R_real - R_cf(maintain_g)
        # --- Optional ---
        **kwargs
    ) -> float:
        """
        Devuelve la recompensa diferencial pre-calculada (R_diff) para la ganancia específica.
        Implementa el método de la interfaz.
        """
        reward_for_q_update = 0.0 # Default

        if reward_dict is None:
            logger.warning(f"EchoBaseline: reward_dict (R_diff) es None para ganancia '{gain}'. Usando recompensa 0.")
            return 0.0
        if not isinstance(reward_dict, dict):
             logger.warning(f"EchoBaseline: reward_dict (R_diff) no es un dict para '{gain}'. Usando recompensa 0.")
             return 0.0

        # Extraer la recompensa diferencial específica para esta ganancia
        diff_reward = reward_dict.get(gain)

        if diff_reward is None:
            logger.warning(f"EchoBaseline: Clave '{gain}' no encontrada en reward_dict (R_diff): {reward_dict.keys()}. Usando 0.")
            reward_for_q_update = 0.0
        elif not isinstance(diff_reward, (float, int)) or pd.isna(diff_reward) or not np.isfinite(diff_reward):
             logger.warning(f"EchoBaseline: Valor R_diff para '{gain}' inválido ({diff_reward}). Usando 0.")
             reward_for_q_update = 0.0
        else:
            # El valor es válido, usarlo como R_learn
            reward_for_q_update = float(diff_reward)
            # logger.debug(f"EchoBaseline: Usando R_learn = R_diff = {reward_for_q_update:.4f} para '{gain}'.")

        return reward_for_q_update