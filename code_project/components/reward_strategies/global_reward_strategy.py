from interfaces.reward_strategy import RewardStrategy
from typing import Dict, Any, Optional, TYPE_CHECKING
import logging
import numpy as np # Para NaN y isfinite

# Evitar importación circular
if TYPE_CHECKING:
    from components.agents.pid_qlearning_agent import PIDQLearningAgent

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class GlobalRewardStrategy(RewardStrategy):
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
        gain: str,                          # Ignorado (misma recompensa para todas)
        agent: 'PIDQLearningAgent',         # Ignorado aquí
        # --- State ---
        current_agent_state_dict: Dict[str, Any], # Ignorado aquí
        current_state_indices: tuple,             # Ignorado aquí
        # --- Action ---
        actions_dict: Dict[str, int],             # Ignorado aquí
        action_taken_idx: int,                    # Ignorado aquí
        # --- Raw Reward/Stability ---
        interval_reward: float,                   # R_real -> Usado como R_learn
        avg_w_stab: float,                        # Ignorado aquí
        # --- Pre-calculated Differential Rewards ---
        reward_dict: Optional[Dict[str, float]],  # Ignorado aquí
        # --- Optional ---
        **kwargs
    ) -> float:
        """
        Devuelve la recompensa global del intervalo (R_real).
        """
        # logger.debug(f"GlobalReward: Usando R_real={interval_reward:.4f} para ganancia '{gain}'.")

        # Validar que la recompensa sea un número finito
        if isinstance(interval_reward, (float, int)) and np.isfinite(interval_reward):
             return float(interval_reward)
        else:
             logger.warning(f"GlobalReward: Recompensa del intervalo inválida ({interval_reward}). Usando 0.0.")
             return 0.0