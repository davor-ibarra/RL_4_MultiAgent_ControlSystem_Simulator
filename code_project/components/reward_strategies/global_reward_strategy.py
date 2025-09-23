from interfaces.reward_strategy import RewardStrategy # Importar Interfaz
from typing import Dict, Any, Optional, TYPE_CHECKING, Tuple, Union
import logging
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from interfaces.rl_agent import RLAgent
    from interfaces.controller import Controller # Añadir Controller

# 9.1: Usar logger específico del módulo
logger = logging.getLogger(__name__)

class GlobalRewardStrategy(RewardStrategy): # Implementar Interfaz RewardStrategy
    """
    Estrategia de Recompensa Global. R_learn = R_real.
    Implementa RewardStrategy.
    """
    def __init__(self):
        logger.info("GlobalRewardStrategy inicializada.")

    def compute_reward_for_learning(
        self,
        # --- Context ---
        gain: str,                          # Ignorado
        agent: 'RLAgent',                   # Ignorado
        controller: 'Controller',           # Ignorado
        # --- State ---
        current_agent_state_dict: Dict[str, Any], # Ignorado
        current_state_indices: tuple,             # Ignorado
        # --- Action ---
        actions_dict: Dict[str, int],             # Ignorado
        action_taken_idx: int,                    # Ignorado
        # --- Raw Reward/Stability ---
        interval_reward: float,                   # R_real -> Usado como R_learn
        avg_w_stab: float,                        # Ignorado
        # --- Pre-calculated Differential Rewards ---
        reward_dict: Optional[Dict[str, float]],  # Ignorado
        # --- Optional ---
        **kwargs
    ) -> float:
        """
        Devuelve la recompensa global del intervalo (R_real).
        """
        # logger.debug(f"GlobalReward: R_learn = R_real = {interval_reward:.4f} for gain '{gain}'.")

        # 9.2: Validar R_real (debe ser finito)
        if isinstance(interval_reward, (float, int)) and np.isfinite(interval_reward):
            return float(interval_reward)
        else:
            # Loguear advertencia y devolver 0 si es inválido
            # is_nan = isinstance(interval_reward, float) and pd.isna(interval_reward) # pd.isna maneja NaN
            logger.warning(f"GlobalReward: Recompensa del intervalo (R_real) inválida: {interval_reward}. Usando 0.0 para R_learn.")
            return 0.0