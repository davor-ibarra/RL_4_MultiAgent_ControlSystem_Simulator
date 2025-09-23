from interfaces.reward_strategy import RewardStrategy # Importar Interfaz
from typing import Dict, Any, Optional, TYPE_CHECKING, Tuple, Union
import logging
import numpy as np # Para NaN y isfinite
import pandas as pd # Para isnan

# Evitar importación circular
if TYPE_CHECKING:
    # from components.agents.pid_qlearning_agent import PIDQLearningAgent
    from interfaces.rl_agent import RLAgent # Usar interfaz genérica

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class GlobalRewardStrategy(RewardStrategy): # Implementar Interfaz RewardStrategy
    """
    Estrategia de Recompensa Global.
    Utiliza la recompensa total acumulada en el intervalo (R_real) directamente
    para la actualización de aprendizaje de *todas* las ganancias.
    """
    def __init__(self):
        logger.info("GlobalRewardStrategy inicializada.")

    def compute_reward_for_learning(
        self,
        # --- Context ---
        gain: str,                          # Ignorado
        agent: 'RLAgent',                   # Ignorado
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
        Implementa el método de la interfaz.
        """
        # logger.debug(f"GlobalReward: Usando R_real={interval_reward:.4f} para ganancia '{gain}'.")

        # Validar que la recompensa sea un número finito
        if isinstance(interval_reward, (float, int)) and np.isfinite(interval_reward):
             return float(interval_reward)
        else:
             # Usar pd.isnan para compatibilidad con numpy.nan
             is_nan = isinstance(interval_reward, float) and pd.isna(interval_reward)
             logger.warning(f"GlobalReward: Recompensa del intervalo inválida ({interval_reward}, NaN={is_nan}). Usando 0.0.")
             return 0.0